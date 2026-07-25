# 扩展 Forge Ops

## 先判断是否真的需要新 Forge 算子

至少满足以下一项时才新增 Forge 虚拟算子：

- GGML 没有等价语义；
- 可以融合多个节点，消除大中间张量或多次 kernel launch；
- 推理需要 GGML 无法表达的直接量化路径；
- 常用操作需要统一的 CPU/CUDA/SYCL 行为和 shape 规则；
- 否则模型代码中会出现 backend 分支。

不能仅仅为了让名字看起来像 PyTorch，就给成熟的 GGML 原生算子增加虚拟 op。此时只需要通过 `nn::functional` 暴露。Embedding 就是一个有 Forge 公开语义包装、但刻意继续使用原生 GET_ROWS kernel 的例子。

## 涉及的文件

一个完整的新算子通常会修改：

```text
include/ops/contracts/<op>.h          语义契约
include/ops/ops.h                     虚拟 op id、验证路由、公共 API
src/ops/core_ops/ops_<op>.cpp         公开构图函数与 fallback
src/ops/ops-cpu/<op>.cpp              CPU kernel
src/ops/ops-cuda/<op>.cu              CUDA kernel
src/ops/ops-sycl/<op>.cpp             SYCL kernel
src/ops/ops-*/ops_*.{cpp,cu}          声明、probe、registry entry
src/ops/ops-*/CMakeLists.txt           backend 源文件列表
src/ops/CMakeLists.txt                 core wrapper 源文件
tests/ops_test/test_<op>.cpp           参考实现与后端测试
tests/ops_test/CMakeLists.txt           测试注册
include/nn/functional/...              可选 NN 接口
docs/ops/api*.md                       中英文 API 更新
```

算子可以有意使用原生 GGML，也可以暂时只支持部分后端，但这种决定和 fallback 行为必须明确并经过测试。

## 1. 定义语义与参数编码

Contract 是以下规则的唯一来源：

- 输入顺序和 optional 输入；
- 逻辑张量布局；
- 输出 shape 推导；
- 各后端共享的 dtype 和量化存储规则；
- 公开 config 验证；
- 紧凑 `op_params` 编码/解码；
- 输出验证。

```cpp
// include/ops/contracts/example.h
namespace ggml_ops_ext {

struct ops_example_config {
    int32_t axis = 0;
    float scale = 1.0f;
};

struct ops_example_encoded_params {
    int32_t axis = 0;
    float scale = 1.0f;
};
static_assert(sizeof(ops_example_encoded_params) <= 64);

inline bool ops_encode_example(
    const ops_example_config& config,
    ops_example_encoded_params& encoded) {
    if (config.axis < 0 || config.axis >= GGML_MAX_DIMS) return false;
    encoded = {config.axis, config.scale};
    return true;
}

inline ops_status ops_validate_example(
    const ops_request& request) {
    if (!request.srcs || request.n_srcs < 1 || !request.srcs[0]) {
        return ops_status::error(
            ops_status_code::invalid_request, "Example input is missing");
    }
    // 解码参数、检查输入 shape/type；request.output 非空时检查输出。
    return ops_status::ok();
}

} // namespace ggml_ops_ext
```

GGML 为 `op_params` 保留 64 字节。这里不能存指针、只在主机存在的对象、`size_t` 或依赖 ABI 的结构。使用固定宽度字段，并通过 `memcpy` 搬运 float/integer bit；缩窄前检查溢出。

所有后端必须调用共享 contract。CUDA/SYCL 不能悄悄接受 CPU 拒绝的 shape，也不能重新解释 axis 或 padding 顺序。

## 2. 分配虚拟算子 ID

在 `ops_virt_op_type` 的 `GGML_OP_OPS_VIRT_COUNT` 之前增加值。虚拟 ID 从 2000 开始，避免与 GGML core op 冲突。该值会跨共享库边界，修改 enum 后必须重编全部 Forge Ops backend。

属于通用验证路由的算子要加入 `ops_validate_request_contract`。ConvND、PoolND、PadND、ResizeND 这类几何家族可以使用专门 probe。

## 3. 实现公开构图函数

Core wrapper 负责验证输入/config、探测目标 backend、创建输出 shape、写入编码参数并定义 fallback。

```cpp
ggml_tensor* ggml_ops_example(
    ggml_context* ctx, ggml_tensor* input,
    const ggml_ops_ext::ops_example_config& config,
    ggml_backend_t backend) {
    if (!ctx || !input) return nullptr;

    ggml_ops_ext::ops_example_encoded_params params;
    if (!ggml_ops_ext::ops_encode_example(config, params)) return nullptr;

    ggml_tensor* sources[] = {input};
    if (ggml_ops_backend_supports_op(
            backend, ggml_ops_ext::GGML_OP_OPS_VIRT_EXAMPLE,
            sources, 1, &params, sizeof(params))) {
        int64_t shape[GGML_MAX_DIMS] = {
            input->ne[0], input->ne[1], input->ne[2], input->ne[3]};
        ggml_tensor* output = ggml_ops_ext::ops_new_virtual_node(
            ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_EXAMPLE,
            input->type, ggml_n_dims(input), shape, 1, sources);
        ggml_set_op_params(output, &params, sizeof(params));
        return output;
    }

    // 存在语义等价 GGML 组合时返回组合，否则返回 nullptr。
    // 不能偷偷把算子移动到 CPU。
    return nullptr;
}
```

Fallback 顺序属于这里，不能散落在模型代码中。Fallback 必须保持 shape、dtype、mask、mutation 和 dependency 语义。不能仅仅为了不返回 `nullptr`，就重新引入已知会产生巨大 workspace 的路径。

## 4. 实现后端 kernel

最简单的形式是由 `make_ops_kernel` 适配的 bool 入口：

```cpp
bool ops_cpu_op_example(ggml_backend_t backend, ggml_tensor* node) {
    if (!node) return false;
    ops_request request{
        ggml_backend_get_device(backend),
        GGML_OP_OPS_VIRT_EXAMPLE,
        node->src, 1, node->op_params, sizeof(node->op_params), node};
    if (!ops_validate_example(request)) return false;
    // 执行。
    return true;
}
```

需要更丰富错误或 `ops_execution_context` 资源时，直接实现返回 `ops_status` 的 `ops_kernel_execute_t`。

### CPU 规则

- 通过 `ggml_ops_ext::cpu::backend_thread_count(backend)` 获取线程数，不能使用进程级线程配置。
- 并行化外层独立任务，临时内存限制在每 worker/tile 的固定范围。
- 在热循环外完成 dtype 和量化 dispatch。
- 适用时复用现有 AVX2/F16C 和量化 block helper。
- 不能通过展开完整量化权重来“临时支持”某种格式。

### CUDA 规则

- 通过 `ops_cuda_common.cuh` 获取 device 和 stream。
- launch 前设置 device，并提交到 backend stream。
- 需要 workspace 时使用 stream-ordered 临时分配。
- 报告 launch error，但不能在算子内调用 `cudaDeviceSynchronize`。
- 按 capability/shape/type 路由，不能按产品名或 PCI ID 路由。

### SYCL 规则

- 通过 `ops_sycl.h` bridge 获取 queue。
- 异步 submit，不能在每个算子内部 `queue.wait()`。
- local memory 和 work-group size 必须受目标架构类别保证。
- 除非 kernel 明确请求并探测，否则不能依赖固定 subgroup。
- Forge kernel 留在 `ggml_ops_ext_sycl`，不能为了 Forge 专用算子修改上游 GGML。

所有后端都必须支持跨 tile 的非整齐长度，并精确实现首尾 tile 的 padding/有效域语义。

## 5. 增加 Probe 和 Registry Entry

每个 backend 的 `ops_cpu.cpp`、`ops_cuda.cu`、`ops_sycl.cpp` 需要：

1. kernel 入口声明；
2. probe 函数或共享 contract probe；
3. 命名 kernel 表项；
4. 在对应 CMake 源文件列表中加入实现。

```cpp
static ops_probe_result supports_example(const ops_request& request) {
    const ops_status valid = ops_validate_example(request);
    if (!valid) return {false, 0, valid.message};
    const ggml_type type = request.srcs[0]->type;
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16;
}

static const ops_kernel_entry CPU_KERNELS[] = {
    make_ops_kernel<ops_cpu_op_example>(
        GGML_OP_OPS_VIRT_EXAMPLE,
        "cpu.example", supports_example, 100),
};
```

Kernel 名称应是稳定的诊断标识。Priority 用于从多个兼容实现中选择；support check 不能进行基准测试或读取具体产品名称。

注册按 `CPU`、`CUDA`、`SYCL` 这样的 `backend_name_prefix` 匹配，同一个前缀不能注册两次。

## 6. 量化存储

算子接受权重时：

1. 定义语义 weight layout；
2. 在热循环外调用一次 `ops_describe_quantization`；
3. 在 contract/probe 中检查量化行 block 对齐；
4. 只解码当前计算需要的 tile/block；
5. 除非有明确策略，reduction 使用 F32 累加；
6. 新参数家族还要增加 converter 和 NN `Parameter::Usage` 支持。

“支持量化”意味着直接从 packed storage 执行。在模型加载或每次调用时展开全部权重不算量化支持。

## 7. 从 NN 暴露

通用模型代码不应包含后端 kernel 头文件。为算子增加 `nn::functional` 包装；涉及状态时再增加 `nn::Module` Layer。NN 负责方便的默认参数和模型侧布局，Ops 仍然是语义/后端边界。

## 8. 正确测试

新算子测试必须：

- 动态 backend 构建中调用 `ggml_backend_load_all`；
- 初始化链接的 Forge backend library 并 acquire Ops hook；
- 要求至少真正执行一个设备，禁止空设备枚举导致假通过；
- 枚举 CPU/CUDA/SYCL，但不能匹配具体 GPU 型号；
- 覆盖所有承诺的 dtype 和量化类型；
- 按适用情况测试极小、奇数、边界、跨 tile、batch、groups 和非默认参数；
- 对照独立主机参考或可信语义组合；
- 性能计时前后同步 backend；
- 容差按 dtype/量化设置，不能按设备型号设置；
- 融合算子要与旧组合路径比较性能，并拒绝明显退化。

测试要加入 `tests/ops_test/CMakeLists.txt` 和 `REGISTERED_OPS_TEST_TARGETS`。

## 9. 构建与验证

使用项目 preset 和专门的 Ops 开发目标：

```powershell
cmake --preset x64-windows-cuda-sycl-cpu-dl-release-f16
cmake --build --preset x64-windows-cuda-sycl-cpu-dl-release-f16-ops-dev --target test_<op> -j 32
ctest --preset x64-windows-cuda-sycl-cpu-dl-release-f16-ops-dev -R ops.test_<op> --output-on-failure
ctest --preset x64-windows-cuda-sycl-cpu-dl-release-f16-ops-dev --output-on-failure
git diff --check
```

修改公共 Ops 头文件会重编很多依赖。只修改一个 CUDA/SYCL 源文件时，应只重编较小的 Forge backend library，而不是上游 GGML。单元测试通过后，如果算子位于真实推理链路，还要运行对应 provider 端到端模型。

## 完成清单

- 写 kernel 前已经确定语义和 layout。
- 编码参数使用固定宽度且不超过 64 字节。
- 所有 backend 使用相同 contract 并验证输出。
- 不支持情况明确，不存在静默 CPU 传输。
- Fallback 语义和 workspace 成本合理。
- 量化权重保持 packed。
- Kernel 异步提交并使用 backend 所有的 stream/queue。
- 路由中没有具体设备型号检查。
- CPU/CUDA/SYCL 正确性和性能均已测量。
- NN 包装与中英文文档同步更新。
