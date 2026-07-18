# GGML-Forge

GGML-Forge is a C++17 inference foundation built around GGML. It separates
backend-independent operator semantics, reusable neural-network composition,
and domain-specific model providers so that new model families can be added
without expanding a single framework-wide pipeline.

```text
public category API
        |
category runtime -> provider -> loaded model -> session
        |                         |
        +---------- nn -----------+
                      |
                     ops
                      |
                    GGML
```

The repository is organized by responsibility:

- `ggml/` is the project GGML runtime synchronized from the managed llama.cpp
  revision and extended by Forge bridge patches.
- `include/ops/` and `src/ops/` define custom operator contracts and their
  CPU, CUDA, and SYCL implementations.
- `include/nn/` and `src/nn/` provide modules, layers, functional composition,
  model loading, schemas, and execution state.
- `include/categories/` and `src/categories/` contain domain APIs and provider
  implementations such as TTS, LLM, ASR, and visual generation.
- `scripts/conversion/` contains model conversion and artifact tooling.
- `scripts/maintenance/` owns dependency synchronization and Forge patches.
- `docs/` is the single home for project documentation, classified by
  architecture and subsystem.

Detailed documentation starts at [docs/README.md](docs/README.md).
