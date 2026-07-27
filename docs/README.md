# GGML-Forge Documentation

Project documentation is classified by subsystem. Source, model, and script
directories should not contain project README files; third-party provenance
and generated upstream documentation are the only exceptions.

## Architecture

- [Repository layout](architecture/repository-layout.md)
- [Runtime assets](architecture/runtime-assets.md)

## Execution foundation

- [NN documentation index](nn/README.md) / [NN 中文索引](nn/README.zh-CN.md)
- [NN public API](nn/api.md) / [NN 公共接口](nn/api.zh-CN.md)
- [Extending NN](nn/extending.md) / [扩展 NN](nn/extending.zh-CN.md)
- [NN architecture](nn/architecture.md)
- [Using NN](nn/usage.md)
- [Quantization runtime](nn/quantization.md)
- [Ops documentation index](ops/README.md) / [Ops 中文索引](ops/README.zh-CN.md)
- [Ops public API](ops/api.md) / [Ops 公共接口](ops/api.zh-CN.md)
- [Extending Ops](ops/extending.md) / [扩展 Ops](ops/extending.zh-CN.md)
- [Ops runtime](ops/runtime.md)
- [Using Ops](ops/usage.md)

## Model categories

- [TTS architecture](tts/architecture.md)
- [Using TTS](tts/usage.md)
- [TTS runtime ownership](tts/runtime.md)
- [TTS providers](tts/providers/README.md)
- [LLM architecture](llm/architecture.md)
- [Using LLM](llm/usage.md)
- [llama.cpp integration](llm/providers/llama_cpp/integration.md)
- [llama.cpp conversion tools](llm/providers/llama_cpp/conversion.md)
- [ASR architecture](asr/architecture.md)
- [Using ASR](asr/usage.md)
- [whisper.cpp integration](asr/providers/whisper_cpp/integration.md)
- [whisper.cpp conversion tools](asr/providers/whisper_cpp/conversion.md)
- [Visual generation architecture](visual_generation/architecture.md)
- [Using visual generation](visual_generation/usage.md)
- [stable-diffusion.cpp integration](visual_generation/providers/stable_diffusion_cpp/integration.md)
- [stable-diffusion.cpp conversion tools](visual_generation/providers/stable_diffusion_cpp/conversion.md)
- [Object detection conversion and YOLOv8](object_detection/conversion.md)

## Tooling and maintenance

- [UI architecture](ui/architecture.md)
- [Server architecture](server/architecture.md)
- [Using forge-server](server/usage.md)
- [Server API compatibility](server/api.md)
- [Conversion architecture](conversion/architecture.md)
- [Using conversion tools](conversion/usage.md)
- [Maintenance architecture](maintenance/architecture.md)
- [Using dependency updates](maintenance/usage.md)
