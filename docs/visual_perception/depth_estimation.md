# Depth Estimation Model Conversion

Normative GGUF contract for the `depth_estimation` category. Converters live
under `scripts/conversion/categories/visual_perception/depth_estimation/providers/<provider>/`
and must use the shared `conversion.common` toolkit.

## Model file

One GGUF file per model; the C++ core selects the provider from
`general.architecture` (snake_case `<provider>[_<variant>]`, e.g.
`depth_anything_v2`).

## Required metadata

Category-generic keys use the `depth.` prefix; provider-specific keys use the
provider's own prefix (e.g. `depth_anything.`).

| Key | Type | Meaning |
| --- | --- | --- |
| `depth.task.monocular` | bool | consumes a single image |
| `depth.task.stereo` | bool | consumes a rectified pair |
| `depth.output.kind` | string | `relative`, `metric`, or `disparity` |
| `depth.input.width` | uint32 | model input width; 0 = dynamic |
| `depth.input.height` | uint32 | model input height; 0 = dynamic |
| `depth.input.normalization` | string | `zero_to_one` or `imagenet` |

`depth.task.*` maps onto `depth_capabilities` (at least one true);
`depth.output.kind` maps onto `depth_map_kind` (`metric` also sets the
`metric` capability). Every converter must finish with `validate_artifact`
against an `ArtifactContract` listing the keys above.

## Tensors

Tensor names are the C++ Module tree paths (dot-separated, ≤ 63 UTF-8 bytes).
Validate against `load_cpp_schema(<arch>)` once the C++ model class exists;
quantized export requires that schema. Precision policy: `ndim <= 1` or
`.bias` → F32, otherwise F16 by default; quantization via
`conversion.common.quantization` and an optional `--quant-policy` JSON.

## Entry point

```powershell
python scripts/conversion/categories/visual_perception/depth_estimation/providers/<provider>/process.py `
  --src path/to/source.pt --output path/to/model.gguf --quantize F16
```
