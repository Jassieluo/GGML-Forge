# 扩展 NN

## 先确定正确层级

新增代码前先判断需求属于哪一层：

| 需求 | 放置位置 |
| --- | --- |
| 基于现有 NN/Ops/GGML 的无状态组合 | `include/nn/functional/`、`src/nn/functional/` |
| 带参数或子模块的可复用对象 | `include/nn/layers/`、`src/nn/layers/` |
| 可复用的可变执行状态 | `include/nn/runtime/`、`src/nn/runtime/` |
| 模型制品解析或权重绑定 | `include/nn/io/`、`src/nn/io/` 和 schema |
| 新融合语义或缺少高效后端实现 | 先扩展 Ops，再从 NN 暴露 |
| 模型版本、provider、tokenizer 或 pipeline 策略 | provider，而不是 NN |

不要仅仅为了给已有 GGML 算子换名字就创建 Forge 虚拟算子。GGML 已经具备正确且优化过的后端支持时，一层轻量 NN 包装就足够。

## 新增 Functional 函数

1. 在 `include/nn/functional/` 对应头文件中声明无状态构图函数。
2. 在 `src/nn/functional/` 中实现。
3. Forge 拥有语义或路由时调用公开 `ggml_ops_*`；否则组合原生 GGML 节点。
4. 底层必要调用不受支持时，原样返回 `nullptr`。
5. 新源文件加入 `src/nn/CMakeLists.txt`。
6. 通用头文件加入 `include/nn/nn.h`。

```cpp
// include/nn/functional/example.h
ggml_tensor* example(
    ggml_context* ctx, ggml_tensor* input,
    ggml_backend_t backend = nullptr);

// src/nn/functional/example.cpp
ggml_tensor* nn::functional::example(
    ggml_context* ctx, ggml_tensor* input, ggml_backend_t backend) {
    if (!ctx || !input) return nullptr;
    return ggml_ops_example(ctx, input, backend);
}
```

Functional 层负责公开 shape 约定和默认参数。kernel 细节、stream、同步以及设备名称判断都不能放在这里。

## 新增有状态 Layer

继承 `Module<Derived>`，把参数和子模块注册为成员引用，并通过 Functional 函数实现 `forward`。

```cpp
class AffineActivation : public nn::Module<AffineActivation> {
public:
    nn::Parameter& weight = parameter(
        "weight", nn::Parameter::required(
            std::nullopt, nn::Parameter::Usage::linear_weight));
    nn::Parameter& bias = parameter(
        "bias", nn::Parameter::optional(
            std::nullopt, nn::Parameter::Usage::bias));

    ggml_tensor* forward(
        ggml_context* ctx, ggml_tensor* input,
        ggml_backend_t backend = nullptr) {
        ggml_tensor* value = nn::F::linear(
            ctx, input, weight.tensor(), bias.local_tensor(), backend);
        return value ? ggml_relu(ctx, value) : nullptr;
    }
};
```

规则：

- 加上父模块前缀后，Parameter 名称必须与转换器/GGUF 张量路径一致。
- optional 参数使用 `local_tensor()`；required 或 tied 参数需要解析时使用 `tensor()`。
- 声明最具体的 `Parameter::Usage`，不能为了绕过加载检查而把卷积或 Embedding 权重标记成 generic。
- 不要用裸指针手工拥有子模块，`submodule` 同时完成所有权和注册。
- Module 必须保持不可复制、不可移动；运行时数量的集合使用模块容器。
- `forward` 只构建计算图。持久可变 buffer 应放在 runtime 对象里，不能放进临时 Context 或进程全局变量。

## 动态模块结构

重复编号 block 使用 `ModuleList<T>`，命名变体使用 `ModuleDict<T>`，重复参数使用 `ParameterList`。自动生成的名称会成为 state path，例如 `layers.0.*`、`experts.router.*`。

```cpp
nn::ModuleList<Block>& layers =
    submodule<nn::ModuleList<Block>>("layers");

for (int i = 0; i < count; ++i) {
    layers.emplace_back(/* constructor args */);
}
```

加载权重前必须构建最终模块树。`load_into` 之后改变模块树会破坏预期 schema，并让新参数处于未绑定状态。

## 扩展 Parameter 存储类型

Parameter 存储策略由 `include/ops/contracts/quantization.h` 中的 `ops_parameter_usage` 和 `ops_storage_capability` 定义。

新的参数家族需要独立布局或类型策略时：

1. 增加语义 usage，不能按张量名称匹配策略。
2. 定义允许直接执行的存储类型和量化布局。
3. 在 NN Layer 中使用该 usage。
4. 让转换器输出满足策略的 `storage_shape`、`logical_shape` 和 `Layout` 元数据。
5. 为允许和拒绝的制品分别增加严格加载测试。

加载器不会偷偷反量化不支持的权重，也不会猜测布局。后端不能直接消费某种存储时，应由转换选择其他类型，或者给 Ops 后端补上支持。

## 新增模型制品 Source

权重不是 GGUF 时实现 `nn::io::Source`。实现必须提供稳定的 `TensorInfo`、唯一名称查找、精确字节数和有边界的 `read`。格式和版本策略留在 provider 或转换器中，`load_into` 应继续只看到通用张量 Source。

## 测试

至少覆盖：

- 规范参数路径和 `ModelSchema` 输出；
- required、optional、tied、缺失、多余和重复参数；
- 逻辑形状、存储形状和布局检查；
- 与主机参考或可靠 GGML 组合一致的图输出；
- NN 功能暴露 Forge 算子时的 CPU/CUDA/SYCL 路由；
- borrowed/copied 输入及异步执行的生命周期。

NN 测试放在 `tests/nn_test/`。算子数值和后端测试放在 `tests/ops_test/`；不能只通过一个大型 provider 模型间接测试 kernel。

## 构建检查

只有 NN 改动时构建 NN library 和对应测试。涉及 Ops 时先使用专门的 Ops preset：

```powershell
cmake --preset x64-windows-cuda-sycl-cpu-dl-release-f16
cmake --build --preset x64-windows-cuda-sycl-cpu-dl-release-f16-ops-dev --target <test-target> -j 32
ctest --preset x64-windows-cuda-sycl-cpu-dl-release-f16-ops-dev --output-on-failure
```

修改真实模型使用的 Layer 后，还要运行对应 provider 的端到端测试。

## Review 清单

- 代码放在 NN 而不是 provider 或 backend 是否有明确理由？
- 参数路径是否稳定并与转换器兼容？
- 逻辑 shape 与存储 layout 是否分离？
- 不支持的构图是否返回 `nullptr` 或清晰错误？
- 模型和 Functional 代码中是否没有 backend/device-name 分支？
- 所有权和可变状态是否明确？
- 中英文 API/扩展文档是否同步更新？
