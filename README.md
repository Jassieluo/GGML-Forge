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
- `tools/server/` provides provider-neutral OpenAI, Anthropic, and Forge HTTP
  APIs.
- `ui/eui_neo/` contains the managed EUI-NEO upstream UI framework; Forge
  applications will remain separate under `apps/`.
- `docs/` is the single home for project documentation, classified by
  architecture and subsystem.

Detailed documentation starts at [docs/README.md](docs/README.md).

## Upstream projects

GGML-Forge builds on and learns from the following open-source projects:

- [llama.cpp](https://github.com/ggml-org/llama.cpp) provides the managed LLM
  upstream and the GGML revision used by the shared runtime.
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) provides the managed
  ASR upstream.
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
  provides the managed image and video generation upstream.
- [EUI-NEO](https://github.com/sudoevolve/EUI-NEO) is the managed C++17 UI
  framework reserved for Forge desktop applications.
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) is a primary reference
  for the GPT-SoVITS provider, model conversion, and compatibility work.
- [cppjieba](https://github.com/yanyiwu/cppjieba) provides some text preprocessing functions for Chinese for GPT-SoVITS.

Each upstream project and its bundled dependencies remain subject to their own
licenses and attribution requirements. Managed revisions are recorded under
`scripts/maintenance/dependencies/`.

## License

GGML-Forge is released under the [MIT License](LICENSE). Vendored and managed
upstream projects keep their own licenses as noted above.
