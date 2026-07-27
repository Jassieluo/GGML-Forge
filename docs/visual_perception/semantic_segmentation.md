# Semantic Segmentation Model Conversion

Normative GGUF contract for the `semantic_segmentation` category. Converters
live under
`scripts/conversion/categories/visual_perception/semantic_segmentation/providers/<provider>/`
and must use the shared `conversion.common` toolkit.

## Model file

One GGUF file per model; the C++ core selects the provider from
`general.architecture` (snake_case `<provider>[_<variant>]`, e.g.
`segformer_b0`).

## Required metadata

Category-generic keys use the `segmentation.` prefix; provider-specific keys
use the provider's own prefix (e.g. `segformer.`).

| Key | Type | Meaning |
| --- | --- | --- |
| `segmentation.class_count` | uint32 | number of classes |
| `segmentation.confidence` | bool | model can report per-pixel confidence |
| `segmentation.input.width` | uint32 | model input width; 0 = dynamic |
| `segmentation.input.height` | uint32 | model input height; 0 = dynamic |
| `segmentation.input.normalization` | string | `zero_to_one` or `imagenet` |

Optional:

| Key | Type | Meaning |
| --- | --- | --- |
| `segmentation.labels` | string array | class names in id order, length `segmentation.class_count`; surfaced through `segmentation_model_get_label` |
| `segmentation.palette` | uint8 array | RGB triplets in id order, length `3 * segmentation.class_count`; surfaced through `segmentation_model_get_color` |

The keys map onto `segmentation_capabilities`. Every converter must finish
with `validate_artifact` against an `ArtifactContract` listing the required
keys.

## Tensors

Tensor names are the C++ Module tree paths (dot-separated, ≤ 63 UTF-8 bytes).
Validate against `load_cpp_schema(<arch>)` once the C++ model class exists;
quantized export requires that schema. Precision policy: `ndim <= 1` or
`.bias` → F32, otherwise F16 by default; quantization via
`conversion.common.quantization` and an optional `--quant-policy` JSON.

## Entry point

```powershell
python scripts/conversion/categories/visual_perception/semantic_segmentation/providers/<provider>/process.py `
  --src path/to/source.pt --output path/to/model.gguf --quantize F16
```
