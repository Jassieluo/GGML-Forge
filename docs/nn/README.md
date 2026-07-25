# NN documentation

The NN subsystem provides backend-independent graph composition, reusable
modules, model parameter contracts, artifact loading, and graph execution. It
does not own provider policy or backend kernels.

- [Public API reference](api.md)
- [Extending NN](extending.md)
- [Architecture](architecture.md)
- [Usage guide](usage.md)
- [Quantization runtime](quantization.md)
- [中文索引](README.zh-CN.md)

Use `nn/nn.h` for the complete public surface. New model code should prefer
`nn::functional` and `nn::Module` over direct Ops calls. Add a Forge operator
only when GGML primitives cannot express the required semantics efficiently.
