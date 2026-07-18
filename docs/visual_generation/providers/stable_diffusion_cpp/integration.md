# stable-diffusion.cpp Integration

The provider embeds the official `stable-diffusion.cpp` repository at
`src/categories/visual_generation/providers/stable_diffusion_cpp/stable-diffusion.cpp`.
Its bundled GGML gitlink is retained for upstream provenance but is not built.
The parent CMake project supplies the root `ggml` target, which is synchronized
from the pinned llama.cpp revision.

All GGML consumers compile with `GGML_MAX_NAME=160`. This is an ABI contract,
not a provider-local tuning option: stable-diffusion.cpp requires the longer
name storage, so the root runtime and every consumer must agree on the same
structure layout. The provider bridge also supplies the repository root as a
private include root for upstream references such as `ggml/src/ggml-impl.h`.

WebP, WebM, examples, and the server frontend are disabled, so their nested
submodules do not need to be initialized. Their pinned gitlinks are still
recorded in dependency lock metadata. Stable-diffusion.cpp itself is built as
a shared library and the category wrapper links it privately.

The provider supports still-image generation, img2img, inpainting, ControlNet,
reference/identity inputs, LoRA, video and optional audio, hires, upscale,
ADetailer, Canny preprocessing, preview/progress, and cancellation. Actual
image/video availability is queried from the loaded upstream context; upscale
and ADetailer require their respective model paths.

Upstream progress and preview callbacks are process-global. The bridge installs
them only for a generation scope and guards the scope with a process-wide
mutex. Sessions sharing a loaded model also serialize model operations, while
cancellation bypasses that model mutex so another thread can stop generation.
