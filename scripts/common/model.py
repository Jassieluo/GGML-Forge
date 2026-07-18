"""Canonical, provider-facing model artifact declaration."""

from typing import Any, Iterable, Optional

import numpy as np
from .exporter import export_model
from .layout import Layout
from .schema import ModelDefinition, Parameter
from .model_schema import ModelSchema
from .quantization import QuantizationPolicy


class ModelArtifact:
    """Collect canonical parameters and metadata before an atomic GGUF commit."""

    def __init__(
        self,
        output_path: str,
        architecture: str,
        schema: Optional[ModelSchema] = None,
        target_type: str = "F16",
        policy: Optional[QuantizationPolicy] = None,
    ):
        self._output_path = output_path
        self._definition = ModelDefinition(architecture, schema)
        self._target_type = target_type
        self._policy = policy

    def add_string(self, name: str, value: str) -> None:
        self._definition.string(name, value)

    def add_uint32(self, name: str, value: int) -> None:
        self._definition.uint32(name, value)

    def add_float32(self, name: str, value: float) -> None:
        self._definition.float32(name, value)

    def add_bool(self, name: str, value: bool) -> None:
        self._definition.boolean(name, value)

    def add_array(self, name: str, value: Iterable[Any]) -> None:
        self._definition.array(name, value)

    def parameter(
        self,
        name: str,
        data: np.ndarray,
        layout: Layout = Layout.identity(),
    ) -> None:
        self._definition.parameter(Parameter(name, data, layout=layout))

    def write(self) -> None:
        export_model(self._definition, self._output_path, self._target_type, self._policy)

    def abort(self) -> None:
        # No writer exists before write(); export_model owns transaction cleanup.
        pass
