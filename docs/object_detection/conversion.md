# Object detection conversion

Object-detection converters live below
`scripts/conversion/categories/object_detection/providers/<provider>/`. The
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
python scripts/conversion/categories/object_detection/providers/yolo/process.py `
  yolov8n.pt yolov8n.gguf --version v8 --quantize F16
```

The same entry point accepts an Ultralytics `Segment` checkpoint and can
quantize its convolution weights:

```powershell
python scripts/conversion/categories/object_detection/providers/yolo/process.py `
  yolov8n-seg.pt yolov8n-seg-q4_0.gguf --version v8 --quantize Q4_0
```

`--input-width` and `--input-height` default to 640 and must be positive
multiples of 32. The converter fuses the Ultralytics model, derives the exact
channel counts and C2f repeat counts from its tensors, checks them against the
C++ model schema, then validates the completed GGUF artifact.

The v8 adapter accepts axis-aligned `Detect` and instance-segmentation
`Segment` checkpoints. Pose and oriented-box heads still need separate task
adapters because their output contracts differ.

## GGUF contract

`general.architecture` selects the concrete implementation: `yolo_v8` for a
Detect head and `yolo_v8_seg` for a Segment head. Shared category metadata uses
`detection.*`; family metadata uses `yolo.*`.

| Key | Type | Meaning |
| --- | --- | --- |
| `yolo.version` | string | version adapter, currently `v8` |
| `yolo.reg_max` | uint32 | DFL bins per box side |
| `yolo.strides` | integer array | output strides, currently `[8, 16, 32]` |
| `yolo.mask_count` | uint32 | Segment prototype/mask coefficient count; omitted for Detect |
| `detection.task.boxes` | bool | axis-aligned boxes are supported |
| `detection.task.oriented_boxes` | bool | rotated boxes are supported |
| `detection.task.instance_masks` | bool | instance masks are supported |
| `detection.task.keypoints` | bool | pose keypoints are supported |
| `detection.class_count` | uint32 | number of classes |
| `detection.keypoint_count` | uint32 | keypoints per instance, otherwise zero |
| `detection.input.width` | uint32 | fixed inference width |
| `detection.input.height` | uint32 | fixed inference height |
| `detection.input.normalization` | string | currently `zero_to_one` |
| `detection.labels` | string array | labels in class-id order |

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
and reports each box-local mask's dimensions and foreground-pixel count. No
OpenCV dependency is used by either conversion or inference.
