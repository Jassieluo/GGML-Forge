"""Read the canonical parameter contract emitted by the C++ Module tree."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


_STORAGE_TYPES = frozenset({"Q4_K", "Q4_0", "Q8_0", "F16", "F32"})
_QUANTIZED_LAYOUTS = frozenset({"native", "channel_rows", "flattened_rows"})


@dataclass(frozen=True)
class ParameterContract:
    path: str
    required: bool
    usage: str
    direct_storage_types: Tuple[str, ...]
    quantized_layout: str
    logical_shape: Optional[Tuple[int, ...]] = None

    @classmethod
    def parse(cls, document: object) -> "ParameterContract":
        if not isinstance(document, dict):
            raise ValueError("Parameter schema entry must be an object")
        path = document.get("path")
        storage = document.get("direct_storage_types")
        layout = document.get("quantized_layout")
        if not isinstance(path, str) or not path:
            raise ValueError("Parameter schema path must be a non-empty string")
        if not isinstance(document.get("required"), bool):
            raise ValueError(f"Parameter schema {path} has an invalid required flag")
        if not isinstance(document.get("usage"), str):
            raise ValueError(f"Parameter schema {path} has an invalid usage")
        if not isinstance(storage, list) or not storage or any(not isinstance(item, str) for item in storage):
            raise ValueError(f"Parameter schema {path} has invalid direct storage types")
        invalid = set(storage) - _STORAGE_TYPES
        if invalid:
            raise ValueError(f"Parameter schema {path} has unsupported storage types: {sorted(invalid)}")
        if layout not in _QUANTIZED_LAYOUTS:
            raise ValueError(f"Parameter schema {path} has an invalid quantized layout")
        shape = document.get("logical_shape")
        if shape is not None and (not isinstance(shape, list) or
                                  any(not isinstance(dim, int) or dim <= 0 for dim in shape)):
            raise ValueError(f"Parameter schema {path} has an invalid logical shape")
        return cls(
            path=path,
            required=document["required"],
            usage=document["usage"],
            direct_storage_types=tuple(storage),
            quantized_layout=layout,
            logical_shape=tuple(shape) if shape is not None else None,
        )


class ModelSchema:
    def __init__(self, parameters: Tuple[ParameterContract, ...]):
        self.parameters = parameters
        self._by_path: Dict[str, ParameterContract] = {}
        for parameter in parameters:
            if parameter.path in self._by_path:
                raise ValueError(f"Duplicate parameter schema path: {parameter.path}")
            self._by_path[parameter.path] = parameter

    @classmethod
    def from_json(cls, text: str) -> "ModelSchema":
        document = json.loads(text)
        if not isinstance(document, dict) or not isinstance(document.get("parameters"), list):
            raise ValueError("Model schema must contain a parameters array")
        return cls(tuple(ParameterContract.parse(item) for item in document["parameters"]))

    @classmethod
    def from_file(cls, path: str) -> "ModelSchema":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def parameter(self, path: str) -> ParameterContract:
        try:
            return self._by_path[path]
        except KeyError as error:
            raise ValueError(f"Parameter is not declared by the C++ model: {path}") from error

    def validate_complete(self, provided_paths) -> None:
        provided = set(provided_paths)
        unexpected = sorted(provided - self._by_path.keys())
        missing = sorted(
            path for path, parameter in self._by_path.items()
            if parameter.required and path not in provided
        )
        if unexpected:
            raise ValueError(f"Parameters are not declared by the C++ model: {unexpected}")
        if missing:
            raise ValueError(f"Required C++ parameters are missing: {missing}")
