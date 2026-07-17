"""Canonical, provider-facing model artifact declaration."""

from typing import Any, Iterable, Optional

import numpy as np
from gguf import GGMLQuantizationType

from .artifact import ArtifactBuilder
from .layout import Layout, write_layout_metadata


class ModelArtifact:
    """Collect canonical parameters and metadata before an atomic GGUF commit."""

    def __init__(self, output_path: str, architecture: str):
        if not architecture:
            raise ValueError("Model architecture cannot be empty")
        self._builder = ArtifactBuilder(output_path, architecture)
        self._layouts = {}

    def add_string(self, name: str, value: str) -> None:
        self._builder.add_string(name, value)

    def add_uint32(self, name: str, value: int) -> None:
        self._builder.add_uint32(name, value)

    def add_float32(self, name: str, value: float) -> None:
        self._builder.add_float32(name, value)

    def add_bool(self, name: str, value: bool) -> None:
        self._builder.add_bool(name, value)

    def add_array(self, name: str, value: Iterable[Any]) -> None:
        self._builder.add_array(name, value)

    def parameter(
        self,
        name: str,
        data: np.ndarray,
        raw_dtype: Optional[GGMLQuantizationType] = None,
        layout: Layout = Layout.identity(),
    ) -> None:
        self._builder.add_tensor(name, data, raw_dtype)
        if not layout.is_identity:
            self._layouts[name] = layout

    def write(self) -> None:
        write_layout_metadata(self._builder, self._layouts)
        self._builder.write()

    def abort(self) -> None:
        self._builder.abort()
