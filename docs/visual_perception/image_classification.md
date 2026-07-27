# Image classification and YOLOv8-CLS

Image classification uses a whole-image Top-K result contract under
`visual_perception`. Its public C API is
`categories/visual_perception/image_classification.h`.

## Convert YOLOv8-CLS

The shared YOLO family converter accepts a trusted Ultralytics Classify
checkpoint. Classification defaults to a 224 by 224 center crop:

```powershell
python scripts/conversion/categories/visual_perception/providers/yolo/process.py `
  yolov8n-cls.pt yolov8n-cls-q4_0.gguf --version v8 --quantize Q4_0
```

`general.architecture` is `yolo_v8_cls`. Required metadata:

| Key | Type | Meaning |
| --- | --- | --- |
| `classification.class_count` | uint32 | number of output classes |
| `classification.input.width` | uint32 | square crop width, normally 224 |
| `classification.input.height` | uint32 | square crop height, normally 224 |
| `classification.input.resize` | string | `shortest_center_crop` |
| `classification.input.normalization` | string | `zero_to_one` |
| `classification.labels` | string array | labels in class-id order |

The C++ preprocessing path uses a lightweight antialiased triangle filter to
match the torchvision/Pillow shortest-edge resize and center crop. It does not
link OpenCV.

## Run an image

```powershell
build/bin/classify-image yolov8n-cls-f16.gguf image.jpg
```

The example prints Top-5 labels and softmax probabilities. F16 is the accuracy
reference. Q4_0 quantizes both convolution and linear weights and is much
smaller, but classification boundaries can move more than localization scores
on ambiguous images.
