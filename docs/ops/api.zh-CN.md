# Ops 公共接口参考

## 架构边界

Forge Ops 分为四层：

```text
公开 ggml_ops_* 包装
        -> 语义契约与编码参数
        -> GGML 虚拟节点与能力探测
        -> CPU / CUDA / SYCL 注册 kernel
```

公共接口和 dispatcher 类型位于 `include/ops/ops.h`，契约位于 `include/ops/contracts/`，项目 kernel 位于 `src/ops/ops-cpu`、`ops-cuda`、`ops-sycl`。

MatMul/Linear、RoPE、普通 Softmax、基础逐元素运算等成熟功能应继续使用 GGML 原生实现。只有缺失语义、融合算子、直接量化 kernel，或原生组合会产生不可接受的中间内存时，才应进入 Forge Ops。

## 运行时生命周期

第一次 acquire hook 前，backend library 必须已经加载，并完成 Forge 注册。Category runtime 会管理这些步骤。独立程序通常先加载/初始化 backend，然后：

```cpp
ggml_ops_ext::acquire_ops_hook();

ggml_tensor* output = ggml_ops_mish(ctx, input, backend);
if (!output) {
    // backend/dtype/layout/shape 无效或不受支持。
}

ggml_cgraph* graph = ggml_new_graph(ctx);
ggml_build_forward_expand(graph, output);
ggml_status status =
    ggml_ops_ext::ops_backend_graph_compute(backend, graph);

ggml_ops_ext::release_ops_hook();
```

每次 acquire 必须对应一次 release。第一次 acquire 会冻结 kernel registry，之后再注册会失败。hook 是进程级且带引用计数。

`ops_backend_graph_compute` 只会串行化共享同一个 backend 实例的整图执行；不同 backend 实例和不同设备仍有独立执行 lane。Ops 不会自动回退到 CPU。

## 能力探测与分发接口

| 接口 | 含义 |
| --- | --- |
| `ggml_ops_backend_supports_op` | 使用源张量和可选编码参数进行构图期探测。 |
| `probe_ops_kernel(device, ...)` | 不创建 backend 实例，直接探测具体 device。 |
| `probe_ops_kernel(ops_request)` | 完整探测，可检查输出张量并返回 workspace/reason。 |
| `register_ops_backend` | 冷路径注册 backend 前缀和 kernel 表。 |
| `execute_ops_kernel` | 在 backend 上执行一个 Forge 虚拟节点，通常由 bridge hook 调用。 |
| `ops_new_virtual_node` | 完成语义和能力检查后创建虚拟节点。 |
| `ops_backend_graph_compute` | 使用 Forge lane 串行化规则执行计算图。 |

`ops_request` 包含 device、op id、输入、编码参数和可选输出。没有 output 的 probe 只能检查构图时已知内容；执行期 probe 还必须验证输出 shape 和 dtype。

`ops_probe_result` 包含 `supported`、可选 `workspace_bytes` 和诊断 `reason`。dispatcher 会在匹配当前 backend 前缀的实现中选择优先级最高且兼容的 kernel。

## 当前公共算子

Forge 目前公开 37 个 `ggml_ops_*` 构图接口。

### 索引

| 函数 | 语义 |
| --- | --- |
| `ggml_ops_embedding` | 共享二维 Embedding 表，支持最高三维 I32 索引，输出 `[embedding_width, ...index_shape]`。底层使用 GGML GET_ROWS，支持浮点或兼容量化行。 |

### Attention 与缓存

| 函数 | 语义 |
| --- | --- |
| `ggml_ops_attention` | 统一 MHA/GQA Attention，支持可选 bias、权重输出、滑动窗口、运行时 valid length 和 dependency。 |
| `ggml_ops_kv_cache_update` | 对 F32/F16/Q4_0/Q8_0 K/V Cache 进行逻辑原位更新。 |
| `ggml_ops_relative_pe_keys` | 相对位置 key score 增量。 |
| `ggml_ops_relative_pe_values` | 相对位置 value 增量。 |

Attention 按能力自动路由：

```text
GGML 原生 FlashAttention
  -> Forge streaming/fused Attention
  -> GGML 原生 MatMul/Softmax 组合
```

只有请求语义完全匹配时才使用 FlashAttention。通用逐 head bias、输出 attention weights、运行时 valid length 和缓存 dependency 可能要求 Forge 路径。路由不包含任何 GPU 型号判断。

### 卷积

| 函数 | 语义 |
| --- | --- |
| `ggml_ops_conv_1d`、`ggml_ops_conv_transpose_1d` | 1D 分组/深度卷积与转置卷积。 |
| `ggml_ops_conv_2d`、`ggml_ops_conv_transpose_2d` | 2D 直接卷积；转置卷积支持各轴 output padding。 |
| `ggml_ops_conv_3d`、`ggml_ops_conv_transpose_3d` | Packed 3D 卷积，逻辑 W/H/D 由 config 携带。 |

ConvND 支持 stride、非对称 padding、dilation、groups、bias、channel-row/flattened-row 权重布局和各后端支持的量化存储。Conv2D 使用 `[W,H,C,N]`；因为 GGML 只有四个物理维度，Conv3D 使用 `[W*H*D,C,N]`。

### Pool、Pad 与 Resize

| 函数 | 语义 |
| --- | --- |
| `ggml_ops_pool_1d/2d/3d` | Max/Average Pool；支持 kernel、stride、dilation、非对称 padding、ceil mode 和 average padding 策略。 |
| `ggml_ops_adaptive_pool_1d/2d/3d` | 输出到显式大小的 Adaptive Max/Average Pool。 |
| `ggml_ops_pad_1d/2d/3d` | Constant、Reflect、Replicate、Circular Padding。 |
| `ggml_ops_resize_1d/2d/3d` | Nearest 或 N-linear Resize，支持 `align_corners`。 |

这些 kernel 直接映射输出坐标，不生成 im2col 或索引/权重中间 workspace。3D 版本沿用 Conv3D 的 packed spatial 约定。

### 归一化

| 函数 | 实现 |
| --- | --- |
| `ggml_ops_layer_norm` | Forge CPU/CUDA/SYCL 融合 kernel。 |
| `ggml_ops_instance_norm` | Forge CPU/CUDA/SYCL 融合 kernel。 |
| `ggml_ops_ada_ln` | Forge kernel，并带原生 GGML 组合 fallback。 |
| `ggml_ops_rms_norm` | GGML 原生 RMSNorm，加可选 affine 乘法。 |
| `ggml_ops_group_norm` | GGML 原生 GroupNorm，加可选 affine。 |
| `ggml_ops_l2_normalize` | 支持指定 axis 的 GGML 原生组合。 |

### 激活与融合 block

| 函数 | 语义 |
| --- | --- |
| `ggml_ops_mish` | Mish。 |
| `ggml_ops_double_swish` | DoubleSwish。 |
| `ggml_ops_glu` | Sigmoid GLU，带组合 fallback。 |
| `ggml_ops_gated_activation` | 指定 axis 的 SwiGLU、GeGLU、ReGLU 或 identity gate。 |
| `ggml_ops_gated_tanh_sigmoid` | 融合 tanh/sigmoid gating。 |
| `ggml_ops_snake`、`ggml_ops_snake_beta` | 音频周期激活。 |
| `ggml_ops_alias_free_activation` | 融合 depthwise 2x 上采样滤波、SnakeBeta 和 2x 下采样滤波；CPU/CUDA/SYCL 都不会生成 2 倍长度中间张量。 |

## 配置契约

公开几何配置位于 contract 头文件：

- `ops_conv_nd_config`；
- `ops_pool_nd_config`、`ops_adaptive_pool_nd_config`；
- `ops_pad_nd_config`；
- `ops_resize_nd_config`。

公开 config 使用易读的 32 位字段。包装层负责验证并编码成存入 `ggml_tensor::op_params` 的紧凑结构，编码结构必须满足 GGML 的 64 字节限制。backend kernel 解码同一结构，不能重新解释公开 config 或独立猜测几何信息。

## 类型与量化

`ops_quantization_desc` 把 storage、compute、accumulation type、block size、量化 scheme、axis 和 weight layout 分开。应调用 `ops_describe_quantization`，不能按张量名称判断量化策略。

精确支持取决于 backend 和算子，能力 probe 是最终依据。总体策略是：

- activation 使用 F32/F16，部分 CPU/CUDA 路径支持 BF16；
- 归一化和对精度敏感的 reduction 使用 F32 累加；
- Conv/ConvTranspose 直接读取支持的低比特权重，不展开完整 F16/F32 权重；
- 量化行必须满足对应 GGML block size；
- 不支持的存储类型会明确返回 unsupported，不会偷偷转换或传输到 CPU。

ConvND 的详细存储支持和转换策略见 [runtime.md](runtime.md) 与 [NN 量化文档](../nn/quantization.md)。

## 返回值与错误

输入/config 无效，或者目标 backend 没有兼容直接 kernel 且不存在语义 fallback 时，公开构图函数返回 `nullptr`。加入 graph 前必须检查。

kernel 执行使用 `ops_status`：

- `success`；
- `not_handled`；
- `invalid_request`；
- `unsupported`；
- `execution_failed`。

backend kernel 应向对应 stream/queue 异步提交任务，不能随意执行全设备等待。同步应发生在计算图、数据传输或应用边界。
