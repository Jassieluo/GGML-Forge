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

## Supported model domains

GGML-Forge exposes provider-neutral category APIs while keeping model-family
details behind providers. The current tree includes:

- **Large language and multimodal models** through the managed llama.cpp
  provider, including streaming chat, embeddings, reranking, and image input
  when a compatible multimodal projector is supplied.
- **Speech synthesis** through the native GPT-SoVITS provider, with per-session
  reference voices, reusable model sessions, WAV input/output, and quantized
  model loading.
- **Automatic speech recognition** through whisper.cpp, including language
  detection, transcription, translation, and segment/token timestamps.
- **Visual generation** through stable-diffusion.cpp for text-to-image and the
  other generation capabilities exposed by the upstream runtime.
- **Visual perception**, implemented on Forge `nn` and `ops`, with compact Q4
  examples for the following tasks:

  | Category | Tasks | Example providers/models |
  | --- | --- | --- |
  | Image classification | Whole-image classification | YOLOv8 classification |
  | Instance perception | Object detection, instance segmentation, keypoint pose, rotated OBB | YOLOv8 detect/seg/pose/obb |
  | Semantic segmentation | Per-pixel semantic masks | LRASPP MobileNetV3 |
  | Depth estimation | Monocular relative depth and stereo disparity | FastDepth, StereoNet |

The visual-perception providers share image I/O, preprocessing, tensor
execution, quantized convolution, filtering, NMS, mask/keypoint decoding, and
rendering utilities without requiring OpenCV.

## Studio demo

`examples/ui/` contains the portable GGML-Forge Studio demonstration app. It
can discover packaged models at runtime and demonstrates local multimodal chat,
selectable/imported GPT-SoVITS reference voices, Whisper WAV transcription,
image generation, and every visual-perception task listed above. The interface
supports Chinese and English and can switch between CPU, CUDA, and SYCL where
the selected provider path is supported.

The repeatable Windows packaging entry point is
`examples/ui/package_release.ps1`. It stages the UI executable, runtime DLLs,
assets, selected demo models, notices, and a portable directory before creating
the release ZIP and SHA-256 checksum.

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


## If it helps, please give me a star
