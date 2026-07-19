"""Model-independent GGUF quantization policy and execution."""

import os
import json
import re
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from gguf.quants import quantize


QUANTIZED_TARGETS = (
    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q4_K", "Q8_0", "MXFP4",
)
SUPPORTED_TARGETS = frozenset({"F16", "Q4_K_M", *QUANTIZED_TARGETS})
POLICY_STORAGE_TYPES = frozenset({"F32", "F16", *QUANTIZED_TARGETS})
BLOCK_SIZES = {
    "Q4_0": 32, "Q4_1": 32, "Q5_0": 32, "Q5_1": 32,
    "Q4_K": 256, "Q8_0": 32, "MXFP4": 32,
}
FLOAT_SOURCE_TYPES = frozenset({GGMLQuantizationType.F32, GGMLQuantizationType.F16})


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Tuple[int, ...]
    role: str = "weight"
    sensitivity: str = "normal"
    allowed_types: Tuple[str, ...] = ("Q4_K", "Q4_0", "Q8_0", "F16", "F32")
    quant_shape: Optional[Tuple[int, ...]] = None
    transform: Optional[str] = None

    @property
    def row_shape(self) -> Tuple[int, ...]:
        return self.quant_shape or self.shape


class QuantizationPolicy:
    """Apply GGML block constraints to provider-declared tensor semantics."""

    def __init__(self, target_type: str, rules: Optional[List[Dict[str, object]]] = None, name: str = ""):
        if target_type not in SUPPORTED_TARGETS:
            raise ValueError(f"Unsupported quantization target: {target_type}")
        self.target_type = target_type
        self.rules = rules or []
        self.name = name

    @classmethod
    def from_file(cls, path: str, target_type: str) -> "QuantizationPolicy":
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        if not isinstance(document, dict):
            raise ValueError("Quantization policy must be a JSON object")
        rules = document.get("rules", [])
        if not isinstance(rules, list):
            raise ValueError("Quantization policy 'rules' must be an array")
        for index, rule in enumerate(rules):
            if not isinstance(rule, dict) or not isinstance(rule.get("types"), list):
                raise ValueError(f"Quantization policy rule {index} must contain a 'types' array")
            for selector in ("role", "sensitivity"):
                if selector in rule and not isinstance(rule[selector], list):
                    raise ValueError(f"Quantization policy rule {index} '{selector}' must be an array")
            if "name" in rule and not isinstance(rule["name"], str):
                raise ValueError(f"Quantization policy rule {index} 'name' must be a regular expression string")
            if "name" in rule:
                re.compile(rule["name"])
            invalid = set(rule["types"]) - POLICY_STORAGE_TYPES
            if invalid:
                raise ValueError(f"Quantization policy rule {index} has unsupported types: {sorted(invalid)}")
        return cls(target_type, rules=rules, name=str(document.get("name", "")))

    @staticmethod
    def _row_compatible(storage_type: str, shape: Tuple[int, ...]) -> bool:
        if len(shape) < 2:
            return storage_type in ("F16", "F32")
        block = BLOCK_SIZES.get(storage_type)
        return block is None or shape[0] % block == 0

    def candidates(self, spec: TensorSpec) -> Tuple[str, ...]:
        allowed = set(spec.allowed_types)
        if allowed == {"F32"}:
            return ("F32",)

        preferred = None
        for rule in self.rules:
            roles = rule.get("role")
            sensitivities = rule.get("sensitivity")
            name_pattern = rule.get("name")
            role_match = roles is None or spec.role in roles
            sensitivity_match = sensitivities is None or spec.sensitivity in sensitivities
            name_match = name_pattern is None or re.search(name_pattern, spec.name) is not None
            if role_match and sensitivity_match and name_match:
                preferred = tuple(rule["types"])
                break

        if preferred is None:
            requested = "Q4_K" if self.target_type == "Q4_K_M" else self.target_type
            if requested == "F16":
                preferred = ("F16",)
            elif requested == "Q8_0":
                preferred = ("Q8_0", "F16")
            else:
                preferred = ("Q8_0", "F16") if spec.sensitivity == "high" else \
                    tuple(dict.fromkeys((requested, "Q8_0", "F16")))

        result = tuple(
            storage_type for storage_type in preferred
            if storage_type in allowed and self._row_compatible(storage_type, spec.row_shape)
        )
        if not result:
            raise ValueError(
                f"Tensor {spec.name} has no valid storage type for target {self.target_type}; "
                f"shape={list(spec.row_shape)}, allowed={list(spec.allowed_types)}"
            )
        return result


TensorDescriber = Callable[[str, str, Tuple[int, ...]], TensorSpec]
TensorTransform = Callable[[str, TensorSpec, np.ndarray], np.ndarray]
MetadataFilter = Callable[[str], bool]
FinalizeMetadata = Callable[[GGUFWriter, "QuantizationReport"], None]


@dataclass
class QuantizationReport:
    counts: Dict[str, int] = field(default_factory=lambda: {
        storage_type: 0 for storage_type in (*QUANTIZED_TARGETS, "F16", "F32")
    })
    transformed: Dict[str, List[str]] = field(default_factory=dict)

    def record_transform(self, tag: Optional[str], name: str) -> None:
        if tag:
            self.transformed.setdefault(tag, []).append(name)


def default_tensor_transform(architecture: str, spec: TensorSpec, data: np.ndarray) -> np.ndarray:
    del architecture
    return data.astype(np.float32).reshape(spec.shape[::-1])


def quantize_q4_k(data: np.ndarray) -> np.ndarray:
    """Encode GGML Q4_K blocks using per-32-value affine min/max fits."""
    source = data.astype(np.float32, copy=False)
    if source.shape[-1] % 256 != 0:
        raise ValueError(f"Q4_K row size must be divisible by 256, got {source.shape[-1]}")
    blocks = source.reshape(-1, 8, 32)
    mins = np.maximum(0.0, -blocks.min(axis=2))
    scales = (blocks.max(axis=2) + mins) / 15.0
    max_scale = scales.max(axis=1, keepdims=True)
    max_min = mins.max(axis=1, keepdims=True)
    scale_codes = np.rint(np.divide(scales * 63.0, max_scale, out=np.zeros_like(scales), where=max_scale != 0)).astype(np.uint8).clip(0, 63)
    min_codes = np.rint(np.divide(mins * 63.0, max_min, out=np.zeros_like(mins), where=max_min != 0)).astype(np.uint8).clip(0, 63)

    packed_scales = np.zeros((blocks.shape[0], 12), dtype=np.uint8)
    packed_scales[:, :4] = scale_codes[:, :4] | ((scale_codes[:, 4:] >> 4) << 6)
    packed_scales[:, 4:8] = min_codes[:, :4] | ((min_codes[:, 4:] >> 4) << 6)
    packed_scales[:, 8:12] = (scale_codes[:, 4:] & 0x0F) | ((min_codes[:, 4:] & 0x0F) << 4)

    d = (max_scale[:, 0] / 63.0).astype(np.float16)
    dmin = (max_min[:, 0] / 63.0).astype(np.float16)
    decoded_scale = d.astype(np.float32)[:, None] * scale_codes
    decoded_min = dmin.astype(np.float32)[:, None] * min_codes
    quants = np.rint(np.divide(
        blocks + decoded_min[:, :, None], decoded_scale[:, :, None],
        out=np.zeros_like(blocks), where=decoded_scale[:, :, None] != 0,
    )).astype(np.uint8).clip(0, 15)
    packed_quants = np.concatenate([
        quants[:, 0::2, :] | (quants[:, 1::2, :] << 4)
    ], axis=2).reshape(blocks.shape[0], 128)
    encoded = np.concatenate([
        d.view(np.uint8).reshape(-1, 2), dmin.view(np.uint8).reshape(-1, 2), packed_scales, packed_quants
    ], axis=1)
    return encoded.reshape((*source.shape[:-1], source.shape[-1] // 256 * 144))


def quantize_gguf(
    input_path: str,
    output_path: str,
    target_type: str,
    describe_tensor: TensorDescriber,
    transform_tensor: TensorTransform = default_tensor_transform,
    copy_metadata: MetadataFilter = lambda name: not name.startswith("GGUF."),
    finalize_metadata: Optional[FinalizeMetadata] = None,
    policy: Optional[QuantizationPolicy] = None,
) -> QuantizationReport:
    policy = policy or QuantizationPolicy(target_type)
    if policy.target_type != target_type:
        raise ValueError("Quantization policy target does not match requested target")
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("Input and output GGUF paths must be different")

    print(f"\nQuantizing model to {target_type}:")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    reader = GGUFReader(input_path)
    architecture = "model"
    for field in reader.fields.values():
        if field.name == "general.architecture":
            architecture = bytes(field.parts[-1]).decode("utf-8").strip("\x00")
            break

    output_dir = os.path.dirname(os.path.abspath(output_path)) or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(prefix=".model-quant-", suffix=".gguf", dir=output_dir)
    os.close(temp_fd)
    os.remove(temp_path)
    writer = None
    report = QuantizationReport()

    try:
        writer = GGUFWriter(temp_path, arch=architecture)
        for name, field in reader.fields.items():
            if name == "general.architecture" or not copy_metadata(name) or not field.types:
                continue
            writer.add_key_value(name, field.contents(), field.types[0])

        qtypes = {
            "Q4_0": GGMLQuantizationType.Q4_0,
            "Q4_1": GGMLQuantizationType.Q4_1,
            "Q5_0": GGMLQuantizationType.Q5_0,
            "Q5_1": GGMLQuantizationType.Q5_1,
            "Q4_K": GGMLQuantizationType.Q4_K,
            "Q8_0": GGMLQuantizationType.Q8_0,
            "MXFP4": GGMLQuantizationType.MXFP4,
        }
        for tensor in reader.tensors:
            shape = tuple(int(value) for value in tensor.shape)
            spec = describe_tensor(architecture, tensor.name, shape)
            if spec.name != tensor.name or spec.shape != shape:
                raise ValueError(f"Provider returned an inconsistent TensorSpec for {tensor.name}")
            if tensor.tensor_type not in FLOAT_SOURCE_TYPES:
                raise ValueError(
                    f"Tensor {tensor.name} uses {tensor.tensor_type.name}; "
                    "quantization input must contain only F32/F16 tensors"
                )

            transformed = None
            written = False
            for candidate in policy.candidates(spec):
                if candidate == "F32":
                    data = tensor.data if tensor.tensor_type == GGMLQuantizationType.F32 else tensor.data.astype(np.float32)
                    writer.add_tensor(tensor.name, data, raw_dtype=GGMLQuantizationType.F32)
                elif candidate == "F16":
                    data = tensor.data if tensor.tensor_type == GGMLQuantizationType.F16 else tensor.data.astype(np.float16)
                    writer.add_tensor(tensor.name, data, raw_dtype=GGMLQuantizationType.F16)
                else:
                    if transformed is None:
                        transformed = transform_tensor(architecture, spec, tensor.data)
                    try:
                        q_data = quantize_q4_k(transformed) if candidate == "Q4_K" else quantize(transformed, qtypes[candidate])
                    except (ValueError, NotImplementedError) as exc:
                        print(f"  [fallback] {tensor.name}: {candidate} unavailable ({exc})")
                        continue
                    writer.add_tensor(tensor.name, q_data, raw_dtype=qtypes[candidate])
                    report.record_transform(spec.transform, tensor.name)
                    suffix = f" transformed={list(spec.row_shape)}" if spec.transform else ""
                    print(f"  [{candidate}] {tensor.name} shape={list(shape)}{suffix}")
                report.counts[candidate] += 1
                written = True
                break

            if not written:
                raise RuntimeError(f"No usable storage type for tensor {tensor.name} shape={list(shape)}")

        if target_type in ("Q4_K", "Q4_K_M") and report.counts["Q4_K"] == 0:
            raise RuntimeError(
                "The installed GGUF Python quantizer produced no Q4_K tensors; "
                "refusing to write a mislabeled Q8/F16 fallback artifact"
            )

        if finalize_metadata:
            finalize_metadata(writer, report)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        writer = None
        os.replace(temp_path, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    src_size = os.path.getsize(input_path)
    dst_size = os.path.getsize(output_path)
    print("  Quantization finished successfully!")
    for key in (*QUANTIZED_TARGETS, "F16", "F32"):
        print(f"    {key:6s} tensors: {report.counts[key]}")
    print(f"    Size reduction: {100 * (src_size - dst_size) / src_size:.1f}% "
          f"({src_size/(1024**2):.1f}MB -> {dst_size/(1024**2):.1f}MB)")
    return report
