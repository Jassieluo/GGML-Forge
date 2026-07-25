# NN 文档

NN 子系统负责与后端无关的计算图组合、可复用模块、模型参数契约、权重加载和计算图执行。它不负责 provider 策略，也不实现后端 kernel。

- [公共接口参考](api.zh-CN.md)
- [扩展 NN](extending.zh-CN.md)
- [架构说明（英文）](architecture.md)
- [使用说明（英文）](usage.md)
- [量化运行时（英文）](quantization.md)
- [English index](README.md)

需要完整接口时包含 `nn/nn.h`。新的模型代码应优先使用 `nn::functional` 和 `nn::Module`；只有 GGML 基础节点无法高效表达目标语义时，才应新增 Forge 算子。
