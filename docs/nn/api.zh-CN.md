# NN 公共接口参考

## 范围与头文件

NN 位于 provider 模型和 Ops 之间：

```text
provider 模型 -> nn::layers / nn::functional -> Forge Ops 或原生 GGML
```

包含 `nn/nn.h` 可以使用完整接口，也可以单独包含 `include/nn/` 下的头文件以缩小依赖。`nn::F` 是 `nn::functional` 的别名。

NN 接口构建 GGML 计算图，不执行 eager 计算。张量维度遵循 GGML 顺序，第 0 维连续。传给 NN 函数的 backend 用于构图时的能力路由。backend、权重存储、计算图 context 和借用的输入数据都必须满足各自的生命周期要求。

## Context 与输入绑定

`nn::Context` 拥有或借用一个 `ggml_context`，跟踪由它创建的张量，并在执行前上传已绑定的输入。

| 接口 | 用途 |
| --- | --- |
| `Context(bytes, no_alloc)` | 拥有 GGML 元数据 context；后端执行通常使用 `no_alloc=true`。 |
| `Context::borrow(ctx)` | 包装现有 `ggml_context`，但不取得所有权。 |
| `empty(name, shape, type)` | 创建未初始化张量。 |
| `input<T>(...)` | 创建执行前必须绑定的数据输入。 |
| `constant<T>(..., data::copy(...))` | 把主机数据复制到 Context 管理的 staging 存储。 |
| `tensor<T>(..., data::borrow(...))` | 借用主机数据，不复制。 |
| `zeros<T>` / `full<T>` | 创建常量 staging 数据。 |
| `bind`、`write`、`read` | 绑定或传输带类型的张量数据。 |
| `view` / `view_bytes` | 创建经过检查的 view；量化张量必须使用字节偏移。 |
| `create_graph`、`build`、`expand` | 创建并填充前向计算图。 |
| `materialize()` | 上传 dirty 输入；`Executor::compute` 会自动调用。 |
| `reset()` | 使当前张量和绑定失效，然后重置 GGML context。 |

`nn::data::borrow` 不延长源数据生命周期，数据至少要保持到 `materialize()` 或 `compute()` 完成。临时容器应使用 `nn::data::copy`，接口明确禁止借用右值 vector。

```cpp
std::vector<float> values(256);
nn::Context context(2 * 1024 * 1024);
ggml_tensor* input = context.input<float>(
    "input", {64, 4}, nn::data::borrow(values));
ggml_tensor* output = ggml_sqr(context.native_handle(), input);
ggml_cgraph* graph = context.build(output);
```

## Executor

`nn::Executor` 为一个由应用管理的 backend 持有 GGML 图分配器。backend 必须比 Executor 活得更久。

```cpp
nn::Executor executor(backend);
executor.prepare(context, graph);
executor.compute(context, graph);
context.read(output, result.data(), result.size());
```

- `prepare` 分配计算图，并绑定确切的 `Context`/graph 组合。
- `compute` 会物化输入并调用 `ops_backend_graph_compute`，因此 Forge 虚拟算子可以正确分发。
- `synchronize` 等待后端完成尚未结束的工作。
- 重建计算图或调用 `Context::reset()` 后必须重新 `prepare`。
- `buffer_size()` 返回计算图分配器的 buffer 大小。

## Module 与参数路径

有状态层继承 `nn::Module<Derived>` 并实现 `forward`，`Module::operator()` 会直接转发到 `forward`。

参数和子模块必须通过受保护的 `parameter`、`submodule` 注册。注册名称组合成 schema 和加载器使用的规范点路径。

```cpp
class ProjectionBlock : public nn::Module<ProjectionBlock> {
public:
    nn::Linear& projection = submodule<nn::Linear>("projection");
    nn::Parameter& scale = parameter(
        "scale", nn::Parameter::optional(
            std::nullopt, nn::Parameter::Usage::scalar));

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* input) {
        ggml_tensor* value = projection(ctx, input);
        return scale.is_bound() ? ggml_mul(ctx, value, scale.tensor()) : value;
    }
};
```

子模块和参数名称只能是一个路径片段，不能包含 `.`。Module 不可复制、不可移动，因为注册成员保存稳定地址。动态集合应使用 `ModuleList<T>`、`ModuleDict<T>` 和 `ParameterList`。

`ModuleBase` 提供：

- `for_each_parameter`、`find_parameter`、`parameter_count`；
- `schema()`，生成加载器可见的 `ModelSchema`；
- `to(backend)`，向模块树传播 backend 指针；
- `attach_state_dict` 和 `state_dict`，管理不可变已加载权重的所有权。

## Parameter

`nn::Parameter` 把 required/optional、逻辑形状和存储用途与实际绑定张量分开描述。

```cpp
nn::Parameter::required(
    nn::Shape{768, 768}, nn::Parameter::Usage::linear_weight);
nn::Parameter::optional(std::nullopt, nn::Parameter::Usage::bias);
```

主要接口：

- `bind(tensor)` 或 `bind(tensor, logical_shape, layout)`；
- `tensor()`，解析 tied parameter 并返回绑定张量；
- `local_tensor()`，只查看本地绑定；
- `tie(other)`、`resolve_tie()`，用于共享权重；
- `supports_direct_storage(type)`、`storage_capability()`，用于加载检查；
- `logical_shape()`、`layout()`、`storage_type()`，区分制品逻辑和运行时存储。

`Parameter::Usage` 来自 Ops 量化契约。应使用最具体的用途，例如 `linear_weight`、`embedding_weight`、`conv1d_weight`、`conv2d_weight`、`norm_affine`、`bias`，因为用途决定允许的存储类型和量化布局。

## StateDict、Schema 与加载

`ModelSchema::from(module)` 或 `module.schema()` 会输出规范参数路径、required/optional、逻辑形状、存储能力和量化布局要求。`to_json()` 可供转换脚本和诊断使用。

`nn::io::Source` 是模型制品抽象，提供张量元数据、名称查找和分段读取；`GGUFSource` 是 GGUF 实现。

```cpp
nn::io::GGUFSource source(path);
nn::io::LoadResult loaded = nn::io::load_into(model, source, backend);
if (!loaded) {
    throw std::runtime_error(loaded.error);
}
```

加载是严格的：required 张量必须存在，多余张量会被拒绝，名称必须唯一，逻辑形状必须匹配，量化行必须满足 block 对齐，并且存储类型与布局必须可以直接执行。成功后 `StateDict` 拥有 backend buffer，并附着在 Module 上。

## Functional 接口

Functional 函数是无状态计算图构建器。它们可能创建 Forge 虚拟节点，也可能组合原生 GGML 节点，或者通过能力探测在两者之间路由。

| 头文件 | 公共函数 |
| --- | --- |
| `functional/linear.h` | `linear` |
| `functional/activation.h` | `mish`、`gated_tanh_sigmoid`、`alias_free_activation1d`、`swiglu`、`geglu`、`reglu` |
| `functional/attention.h` | `attention`、`relative_position_keys`、`relative_position_values` |
| `functional/convolution.h` | 1D、2D、packed 3D Conv/ConvTranspose |
| `functional/normalization.h` | `layer_norm`、`rms_norm`、`group_norm`、`l2_normalize` |
| `functional/pooling.h` | 1D/2D/3D Max/Avg 与 Adaptive Max/Avg Pool |
| `functional/padding.h` | 1D/2D/3D Pad |
| `functional/interpolation.h` | 旧插值接口和原生 Resize 1D/2D/3D |
| `functional/positional.h` | 正弦位置编码 |
| `functional/elementwise.h` | 标量逐元素辅助函数 |

只有目标函数存在原生 GGML 组合 fallback 时，传入 `backend=nullptr` 才有完整意义。只支持直接 Forge kernel 的算子在 backend、dtype、layout 或 shape 不支持时会返回 `nullptr`。

## Layer 接口

可复用的有状态层包括：

- `Linear`、`Embedding`；
- `Conv1d`、`ConvTranspose1d`、`Conv2d`、`ConvTranspose2d`、`Conv3d`、`ConvTranspose3d`；
- `LayerNorm`、`InstanceNorm`、`AdaLN`、`AdaLayerNormZero`；
- `Snake`、`PReLU`、`GLU`；
- `FeedForward`、`MultiHeadAttention`、`KVHeadAttention`；
- Encoder、Decoder 和 DiT Transformer block。

Layer 的配置字段有意保持公开。provider 负责在构图前设置 head 数、维度、stride、padding 和模型版本策略。

## KV Cache

`nn::KVCache` 拥有 backend 上的 K/V 张量。只要 `head_dim` 满足格式 block size，`allocate` 支持 F32、F16、Q8_0、Q4_0。`prefill_attention` 和 `decode_attention` 为指定 layer 构建缓存更新与 Attention 节点。KV Cache 和 backend 必须比引用它们的计算图活得更久。

## 错误与生命周期规则

- API 误用、分配失败、输入未绑定和加载校验错误通过异常或 `LoadResult::error` 报告。
- Functional/Ops 返回 `nullptr` 表示构图无效或不受支持，绝不能把它加入计算图。
- `ggml_context` 必须比其张量和 graph 活得更久。
- 模型 `StateDict`、可变 KV Cache、backend 和 graph buffer 是不同所有者，不能混为一个生命周期。
- NN 不会把不支持的操作静默转移到 CPU。
