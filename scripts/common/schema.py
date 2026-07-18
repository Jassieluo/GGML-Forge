"""Canonical model declarations shared by every checkpoint provider."""

from dataclasses import dataclass, field, replace
from enum import Enum
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .artifact import GGML_MAX_NAME_BYTES
from .layout import Layout
from .model_schema import ModelSchema, ParameterContract


_PATH_SEGMENT = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+)\Z")


def validate_parameter_name(name: str) -> str:
    if not name or len(name.encode("utf-8")) > GGML_MAX_NAME_BYTES:
        raise ValueError(f"Invalid canonical parameter name: {name!r}")
    if any(not _PATH_SEGMENT.fullmatch(segment) for segment in name.split(".")):
        raise ValueError(f"Invalid canonical parameter path: {name}")
    return name


class ParameterRole(str, Enum):
    WEIGHT = "weight"
    EMBEDDING = "embedding"
    CONV = "conv"
    NORM = "norm"
    BIAS = "bias"
    SCALAR = "scalar"
    CONSTANT = "constant"


class Sensitivity(str, Enum):
    NORMAL = "normal"
    HIGH = "high"


def infer_role(name: str, data: np.ndarray) -> ParameterRole:
    leaf = name.rsplit(".", 1)[-1]
    lower = name.lower()
    if leaf == "bias":
        return ParameterRole.BIAS
    if data.ndim == 0 or data.size == 1:
        return ParameterRole.SCALAR
    if data.ndim <= 1:
        return ParameterRole.CONSTANT
    if any(token in lower for token in ("norm.", "layer_norm", "layernorm")):
        return ParameterRole.NORM
    if "embedding" in lower and leaf == "weight":
        return ParameterRole.EMBEDDING
    return ParameterRole.WEIGHT


@dataclass(frozen=True)
class Parameter:
    name: str
    data: np.ndarray
    layout: Layout = field(default_factory=Layout.identity)
    _contract: Optional[ParameterContract] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        validate_parameter_name(self.name)
        if not isinstance(self.data, np.ndarray) or self.data.size == 0:
            raise ValueError(f"Parameter {self.name} must contain a non-empty numpy array")


@dataclass(frozen=True)
class MetadataEntry:
    kind: str
    value: Any


class ModelDefinition:
    def __init__(self, architecture: str, schema: Optional[ModelSchema] = None):
        if not _PATH_SEGMENT.fullmatch(architecture):
            raise ValueError(f"Invalid model architecture: {architecture!r}")
        self.architecture = architecture
        self.schema = schema
        self.metadata: Dict[str, MetadataEntry] = {}
        self.parameters: List[Parameter] = []
        self._parameter_names = set()

    def _metadata(self, name: str, kind: str, value: Any) -> None:
        if not name or name in self.metadata or name == "general.architecture":
            raise ValueError(f"Invalid or duplicate metadata key: {name!r}")
        self.metadata[name] = MetadataEntry(kind, value)

    def string(self, name: str, value: str) -> None:
        self._metadata(name, "string", str(value))

    def uint32(self, name: str, value: int) -> None:
        if value < 0 or value > 0xFFFFFFFF:
            raise ValueError(f"Metadata {name} is outside uint32 range")
        self._metadata(name, "uint32", int(value))

    def float32(self, name: str, value: float) -> None:
        self._metadata(name, "float32", float(value))

    def boolean(self, name: str, value: bool) -> None:
        self._metadata(name, "bool", bool(value))

    def array(self, name: str, value: Iterable[Any]) -> None:
        self._metadata(name, "array", list(value))

    def parameter(self, parameter: Parameter) -> None:
        if parameter.name in self._parameter_names:
            raise ValueError(f"Duplicate canonical parameter: {parameter.name}")
        if self.schema is not None:
            contract = self.schema.parameter(parameter.name)
            logical_shape = tuple(reversed(parameter.data.shape))
            if contract.logical_shape is not None and contract.logical_shape != logical_shape:
                raise ValueError(
                    f"Parameter {parameter.name} shape {logical_shape} does not match "
                    f"the C++ model shape {contract.logical_shape}"
                )
            parameter = replace(parameter, _contract=contract)
        self._parameter_names.add(parameter.name)
        self.parameters.append(parameter)
