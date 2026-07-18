"""The only supported path from a canonical ModelDefinition to GGUF."""

import os
from typing import Optional

import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import quantize

from .artifact import ArtifactBuilder
from .layout import Layout, write_layout_metadata, write_logical_shape_metadata
from .quantization import QuantizationPolicy, QuantizationReport, TensorSpec, quantize_q4_k
from .schema import ModelDefinition, infer_role
from .validation import ArtifactContract, validate_artifact


_QTYPES = {
    "Q4_0": GGMLQuantizationType.Q4_0,
    "Q4_K": GGMLQuantizationType.Q4_K,
    "Q8_0": GGMLQuantizationType.Q8_0,
    "F16": GGMLQuantizationType.F16,
    "F32": GGMLQuantizationType.F32,
}


def _tensor_spec(parameter) -> TensorSpec:
    shape = tuple(reversed(parameter.data.shape))
    contract = parameter._contract
    if contract is None:
        role = infer_role(parameter.name, parameter.data).value
        return TensorSpec(
            parameter.name,
            shape,
            role=role,
            allowed_types=("F32",) if parameter.data.ndim <= 1 else ("F16", "F32"),
        )
    channel_rows = contract.quantized_layout == "channel_rows"
    flattened_rows = contract.quantized_layout == "flattened_rows"
    return TensorSpec(
        parameter.name,
        shape,
        role=contract.usage,
        sensitivity="high" if contract.usage == "embedding_weight" else "normal",
        allowed_types=contract.direct_storage_types,
        quant_shape=(shape[1], shape[0], *shape[2:]) if channel_rows else
                    ((int(np.prod(shape[:-1])), shape[-1]) if flattened_rows else None),
        transform="channel_rows" if channel_rows else
                  ("flattened_rows" if flattened_rows else None),
    )


def export_model(
    definition: ModelDefinition,
    output_path: str,
    target_type: str = "F16",
    policy: Optional[QuantizationPolicy] = None,
) -> QuantizationReport:
    if target_type != "F16" and definition.schema is None:
        raise ValueError("Quantized export requires a schema emitted by the C++ model")
    policy = policy or QuantizationPolicy(target_type)
    if policy.target_type != target_type:
        raise ValueError("Quantization policy target does not match requested target")

    builder = ArtifactBuilder(output_path, definition.architecture)
    report = QuantizationReport()
    layouts = {}
    logical_shapes = {}
    try:
        for name, entry in definition.metadata.items():
            method = "add_bool" if entry.kind == "bool" else f"add_{entry.kind}"
            getattr(builder, method)(name, entry.value)

        for parameter in definition.parameters:
            spec = _tensor_spec(parameter)
            written = False
            written_type = None
            transformed = None
            for candidate in policy.candidates(spec):
                if candidate == "F32":
                    data = parameter.data.astype(np.float32, copy=False)
                elif candidate == "F16":
                    data = parameter.data.astype(np.float16, copy=False)
                else:
                    if transformed is None:
                        transformed = parameter.data.astype(np.float32, copy=False)
                        if spec.transform == "channel_rows":
                            transformed = np.swapaxes(transformed, -1, -2)
                        elif spec.transform == "flattened_rows":
                            transformed = transformed.reshape(parameter.data.shape[0], -1)
                        transformed = transformed.reshape(spec.row_shape[::-1])
                    try:
                        data = quantize_q4_k(transformed) if candidate == "Q4_K" else \
                            quantize(transformed, _QTYPES[candidate])
                    except (ValueError, NotImplementedError) as error:
                        print(f"  [fallback] {parameter.name}: {candidate} unavailable ({error})")
                        continue
                builder.add_tensor(parameter.name, data, _QTYPES[candidate])
                report.counts[candidate] += 1
                report.record_transform(spec.transform, parameter.name)
                written = True
                written_type = candidate
                break
            if not written:
                raise RuntimeError(f"No usable storage type for {parameter.name}")
            if spec.transform == "channel_rows" and written_type not in ("F16", "F32"):
                if not parameter.layout.is_identity:
                    raise ValueError(f"Parameter {parameter.name} combines logical and quantized layouts")
                layouts[parameter.name] = Layout.permuted((1, 0, 2))
            elif spec.transform == "flattened_rows" and written_type not in ("F16", "F32"):
                if not parameter.layout.is_identity:
                    raise ValueError(f"Parameter {parameter.name} combines logical and flattened layouts")
                logical_shapes[parameter.name] = tuple(reversed(parameter.data.shape))
            elif not parameter.layout.is_identity:
                layouts[parameter.name] = parameter.layout

        if target_type in ("Q4_K", "Q4_K_M") and report.counts["Q4_K"] == 0:
            raise RuntimeError("Q4_K target produced no Q4_K tensors")
        if definition.schema is not None:
            definition.schema.validate_complete(parameter.name for parameter in definition.parameters)
        write_layout_metadata(builder, layouts)
        write_logical_shape_metadata(builder, logical_shapes)
        builder.write()
    except Exception:
        builder.abort()
        raise

    validate_artifact(
        output_path,
        ArtifactContract.create(
            metadata=definition.metadata.keys(),
            tensors=(parameter.name for parameter in definition.parameters),
        ),
    )
    size = os.path.getsize(output_path) / (1024 ** 2)
    print(f"Exported {len(definition.parameters)} canonical parameters to {output_path} ({size:.1f} MB)")
    return report
