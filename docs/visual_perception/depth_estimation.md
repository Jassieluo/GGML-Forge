# Depth estimation

The depth category exposes one dense per-pixel API for monocular and rectified
stereo models. It has no OpenCV dependency: image loading, resizing, and PNG/raw
output use `src/common/media`.

## Demonstration models

| Architecture | Input | Output | Source checkpoint |
| --- | --- | --- | --- |
| `fastdepth_mobilenet_v1` | one RGB image, 224 x 224 | metric depth in meters | pruned FastDepth TFLite from ST's model zoo |
| `stereonet` | rectified grayscale pair, longest edge at most 625 | horizontal disparity in pixels | public KeystoneDepth StereoNet checkpoint |

FastDepth is the small monocular example. StereoNet demonstrates the distinct
stereo pipeline: shared features, a 32-plane 3D cost volume representing 256
candidate disparities, soft argmin, and three edge-aware refiners.

StereoNet returns disparity because metric depth depends on camera calibration.
Use `depth_disparity_to_metric` with horizontal focal length in pixels and the
baseline in meters; it applies `Z = focal_length_px * baseline_m / disparity`.

## Convert

```powershell
python scripts/conversion/categories/visual_perception/depth_estimation/providers/fastdepth/process.py `
  fastdepth_224_int8.tflite models/fastdepth-f16.gguf --quantize F16

python scripts/conversion/categories/visual_perception/depth_estimation/providers/stereonet/process.py `
  stereonet.ckpt models/stereonet-f16.gguf --quantize F16

# StereoNet's 32-channel layers fit Q4_0 rows; Q4_K requires 256-value rows.
python scripts/conversion/categories/visual_perception/depth_estimation/providers/stereonet/process.py `
  stereonet.ckpt models/stereonet-q4_0.gguf --quantize Q4_0
```

Converters fuse batch normalization and validate tensor names against the C++
module schema. Three-dimensional convolution weights are stored as flattened
matrices because GGML tensors have a maximum physical rank of four; logical
five-dimensional shapes remain in GGUF metadata.

## Run

```powershell
# Monocular; output can be .png or row-major float32 .f32.
estimate-depth fastdepth-f16.gguf image.png depth.png CUDA0

# Stereo; left and right images must be rectified and have equal dimensions.
estimate-depth stereonet-q4_0.gguf left.png disparity.f32 SYCL0 right.png
```

The same artifacts run on CPU, CUDA, and SYCL through the project backends.

## GGUF metadata

All depth artifacts provide:

| Key | Type | Meaning |
| --- | --- | --- |
| `depth.task.monocular` | bool | accepts one image |
| `depth.task.stereo` | bool | accepts a rectified pair |
| `depth.output.kind` | string | `relative`, `metric`, or `disparity` |
| `depth.input.normalization` | string | provider preprocessing contract |

Fixed-input models use `depth.input.width` and `depth.input.height`. Dynamic
models may instead describe their resize limit, such as StereoNet's
`depth.input.max_side`.

## References

- [FastDepth project and paper](https://fastdepth.mit.edu/)
- [StereoNet paper](https://arxiv.org/abs/1807.08865)
- [Public StereoNet PyTorch checkpoint implementation](https://github.com/andrewlstewart/StereoNet_PyTorch)
