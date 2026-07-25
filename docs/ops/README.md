# Ops documentation

Forge Ops defines backend-independent operator contracts and project-owned
CPU, CUDA, and SYCL kernels. It extends GGML; it does not replace native GGML
operators.

- [Public API reference](api.md)
- [Extending Ops](extending.md)
- [Runtime architecture and optimization notes](runtime.md)
- [Usage and tensor layouts](usage.md)
- [中文索引](README.zh-CN.md)

Model code should normally call `nn::functional`. Direct `ggml_ops_*` calls are
appropriate for low-level graph builders and operator tests.
