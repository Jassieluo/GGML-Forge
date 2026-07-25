# Ops 文档

Forge Ops 定义与后端无关的算子契约，以及项目自有的 CPU、CUDA、SYCL kernel。它扩展 GGML，而不是替代 GGML 原生算子。

- [公共接口参考](api.zh-CN.md)
- [扩展 Ops](extending.zh-CN.md)
- [运行时架构与优化记录（英文）](runtime.md)
- [使用方法与张量布局（英文）](usage.md)
- [English index](README.md)

模型代码通常应调用 `nn::functional`。低层计算图构建器和算子测试可以直接调用 `ggml_ops_*`。
