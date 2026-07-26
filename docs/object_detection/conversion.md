# Object Detection Model Conversion

Normative GGUF contract for the `object_detection` category. Converters live
under `scripts/conversion/categories/object_detection/providers/<provider>/`
and must use the shared `conversion.common` toolkit (`ModelArtifact` or
`ModelDefinition` + `export_model`).

## Model file

One GGUF file per model. The C++ core selects the provider from
`general.architecture`; providers register the architecture values they
implement, so the value is the load-bearing contract:

- Format: snake_case `<provider>[_<variant>]`, e.g. `yolo_v8`, `yolo_v11`,
  `rt_detr`.
- A provider that loads several variants registers each value.

## Required metadata

Category-generic keys use the `detection.` prefix. Provider-specific keys use
the provider's own prefix (e.g. `yolo.`), mirroring `tts.` vs `gpt_sovits.`.

| Key | Type | Meaning |
| --- | --- | --- |
| `detection.task.boxes` | bool | model produces axis-aligned boxes |
| `detection.task.oriented_boxes` | bool | model produces rotated boxes |
| `detection.task.instance_masks` | bool | model produces instance masks |
| `detection.task.keypoints` | bool | model produces keypoints |
| `detection.class_count` | uint32 | number of classes |
| `detection.keypoint_count` | uint32 | keypoints per instance; required when `detection.task.keypoints` is true |
| `detection.input.width` | uint32 | model input width; 0 = dynamic |
| `detection.input.height` | uint32 | model input height; 0 = dynamic |
| `detection.input.normalization` | string | `zero_to_one` or `imagenet` |

Optional:

| Key | Type | Meaning |
| --- | --- | --- |
| `detection.labels` | string array | class names in id order, length `detection.class_count`; surfaced through `detection_model_get_label` |

The task flags map one-to-one onto `detection_capabilities`; at least one must
be true. Every converter must finish with `validate_artifact` against an
`ArtifactContract` listing the keys above.

## Tensors

Tensor names are the C++ Module tree paths (dot-separated, ≤ 63 UTF-8 bytes),
exactly as for GPT-SoVITS. Once the C++ model class exists, converters must
validate names, shapes, and storage types against `load_cpp_schema(<arch>)` —
quantized export requires that schema.

Precision policy (uniform across the project): tensors with `ndim <= 1` or
names ending in `.bias` stay F32; everything else defaults to F16.
Quantization targets and fallback rules come from
`conversion.common.quantization` and an optional `--quant-policy` JSON.

## Entry point

`process.py` in the provider directory, invoked by script path with
kebab-case flags, minimally:

```powershell
python scripts/conversion/categories/object_detection/providers/<provider>/process.py `
  --src path/to/source.pt --output path/to/model.gguf --quantize F16
```
