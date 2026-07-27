# Instance perception conversion

Instance-perception converters live below
`scripts/conversion/categories/visual_perception/providers/<provider>/`. The
YOLO provider has one family entry point and a version adapter for each
supported network generation:

```text
yolo/
  common.py            checkpoint loading and family-wide helpers
  process.py           stable command-line entry point
  versions/
    v8.py              YOLOv8 topology and tensor adapter
```

This split is intentional. A new YOLO generation adds a C++ network directory
and one Python adapter; GGUF writing, quantization, labels, validation, image
preprocessing, DFL decoding, and NMS remain shared when their contracts match.

## Convert YOLOv8

The input is a trusted Ultralytics `.pt` checkpoint. Conversion requires
Python packages `torch`, `ultralytics`, and `numpy`; runtime inference does not.

```powershell
python scripts/conversion/categories/visual_perception/providers/yolo/process.py `
  yolov8n.pt yolov8n.gguf --version v8 --quantize F16
```

The same entry point accepts an Ultralytics `Segment` checkpoint and can
quantize its convolution weights:

```powershell
python scripts/conversion/categories/visual_perception/providers/yolo/process.py `
  yolov8n-seg.pt yolov8n-seg-q4_0.gguf --version v8 --quantize Q4_0
```

Pose checkpoints use the same adapter and retain their keypoint shape:

```powershell
python scripts/conversion/categories/visual_perception/providers/yolo/process.py `
  yolov8n-pose.pt yolov8n-pose-q4_0.gguf --version v8 --quantize Q4_0
```

Oriented-box checkpoints also share the v8 adapter:

```powershell
python scripts/conversion/categories/visual_perception/providers/yolo/process.py `
  yolov8n-obb.pt yolov8n-obb-q4_0.gguf --version v8 --quantize Q4_0
```

`--input-width` and `--input-height` default to 640 and must be positive
multiples of 32. The converter fuses the Ultralytics model, derives the exact
channel counts and C2f repeat counts from its tensors, checks them against the
C++ model schema, then validates the completed GGUF artifact.

The v8 adapter accepts axis-aligned `Detect`, instance-segmentation `Segment`,
keypoint `Pose`, and oriented-box `OBB` checkpoints.

## GGUF contract

`general.architecture` selects the concrete implementation: `yolo_v8` for a
Detect head, `yolo_v8_seg` for a Segment head, `yolo_v8_pose` for a Pose head,
and `yolo_v8_obb` for an OBB head. Shared category metadata uses `instance.*`;
family metadata uses `yolo.*`.

| Key | Type | Meaning |
| --- | --- | --- |
| `yolo.version` | string | version adapter, currently `v8` |
| `yolo.reg_max` | uint32 | DFL bins per box side |
| `yolo.strides` | integer array | output strides, currently `[8, 16, 32]` |
| `yolo.mask_count` | uint32 | Segment prototype/mask coefficient count; omitted for Detect |
| `yolo.keypoint_dimensions` | uint32 | Pose values per point: 2 for coordinates or 3 with confidence |
| `yolo.angle_count` | uint32 | OBB angle values per anchor, currently 1 |
| `instance.task.boxes` | bool | axis-aligned boxes are supported |
| `instance.task.oriented_boxes` | bool | rotated boxes are supported |
| `instance.task.masks` | bool | instance masks are supported |
| `instance.task.keypoints` | bool | pose keypoints are supported |
| `instance.class_count` | uint32 | number of classes |
| `instance.keypoint_count` | uint32 | Pose keypoints per instance, otherwise zero |
| `instance.input.width` | uint32 | fixed inference width |
| `instance.input.height` | uint32 | fixed inference height |
| `instance.input.normalization` | string | currently `zero_to_one` |
| `instance.labels` | string array | labels in class-id order |

Tensor names exactly match the fused Ultralytics module tree. Biases remain
F32; matrix and convolution weights default to F16 or follow the selected
project quantization policy.

## Run an image

When examples are enabled, `detect-image` reads PNG/JPEG/BMP and other formats
supported by the common image module and prints source-image coordinates:

```powershell
build/bin/detect-image yolov8n.gguf image.jpg
```

An optional third argument selects a GGML device; otherwise the runtime uses
`auto`. For a Segment model, the example automatically requests instance masks
and reports each box-local mask's dimensions and foreground-pixel count. For a
Pose model, it requests keypoints and prints source-image x/y coordinates plus
per-point confidence. For an OBB model, it requests rotated boxes and prints
center, width, height, and the image-coordinate angle in radians. Rotated
Fast-NMS uses probabilistic IoU. The Forge C++ inference and image-loading path
does not link OpenCV; conversion runs in the upstream Ultralytics Python
environment.
