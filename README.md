# GGML-Forge

GGML-Forge is a C++17 inference foundation built on GGML. Model-independent
tensor operations and neural-network composition live in `ops` and `nn`;
domain code is organized by category, and each category discovers concrete
implementations through provider registries.

## Architecture

```text
include/, src/
├── ops/                    custom op contracts and CPU/CUDA/SYCL kernels
├── nn/                     modules, layers, execution, loading, and schemas
└── categories/
    └── tts/
        └── providers/      provider contract and implementations
            └── gpt_sovits/
                ├── frontend/ provider-owned text processing
                └── models/   provider-owned BERT, HuBERT, T2S, and VITS
```

A portable TTS composition selects a provider and its artifacts. Device,
thread, concurrency, and residency policies belong to `tts_runtime`; mutable
voice and request state belongs to `tts_session`.

```text
tts_runtime -> provider factory -> loaded tts_model -> tts_session -> audio
```

The public category API is `include/categories/tts/tts.h`. The older
`gpt_sovits_*` API remains available as a provider compatibility surface, but
new applications should use the category API and a model composition JSON.

## Build and test

On the configured Windows CUDA + SYCL + CPU toolchain:

```powershell
cmake --preset x64-windows-cuda-sycl-cpu-dl-release-f16 --fresh
cmake --build --preset x64-windows-cuda-sycl-cpu-dl-release-f16
ctest --test-dir build-x64-windows-cuda-sycl-cpu-dl-release-f16 --output-on-failure
```

Other configure/build presets are listed in `CMakePresets.json`.

The generic GPT-SoVITS example consumes one composition instead of individual
provider artifacts:

```powershell
build-x64-windows-cuda-sycl-cpu-dl-release-f16/bin/tts-gpt-sovits-example.exe `
  --model models/tts/gpt_sovits/configs/v3-q4.json `
  --ref-audio path/to/reference.wav `
  --ref-text "参考音频文本" `
  --text "需要合成的文本"
```

Model export tooling follows the same boundary: shared artifact machinery is
under `scripts/common`, while checkpoint adaptation is under
`scripts/categories/<category>/providers/<provider>`.
