# Visual Generation Architecture

`visual_generation` is the category for generated or transformed visual media.
It deliberately covers still images, video frames, optional generated audio,
upscaling, detailing, and preprocessing instead of assuming every provider is
an image-only diffusion pipeline.

```text
visual_runtime
  -> provider
     -> loaded model       shared weights and optional auxiliary models
        -> session         callbacks, cancellation, and request execution
```

The public C API owns all returned image, frame, and audio buffers. Provider
objects translate generic requests into their native API and copy results
before returning. A provider may serialize access to model resources when its
upstream runtime or callback hooks are process-global.

Capabilities are queried from a loaded model. Callers must not infer support
from a provider name: image generation, video, audio, upscale, ADetailer,
preview, and cancellation can vary with the loaded artifacts.

The first provider is `stable_diffusion_cpp`. Provider-specific model paths,
sampler mapping, GGML bridge details, and upstream callbacks remain below
`providers/stable_diffusion_cpp`; reusable category lifecycle and buffer
ownership remain in `visual_generation`.
