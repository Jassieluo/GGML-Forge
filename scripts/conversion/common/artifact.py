"""Transactional GGUF artifact construction for provider exporters."""

import os
import tempfile
from typing import Any, Iterable, Optional

import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

GGML_MAX_NAME_BYTES = 63


class ArtifactBuilder:
    def __init__(self, output_path: str, architecture: str):
        self.output_path = os.path.abspath(output_path)
        output_dir = os.path.dirname(self.output_path) or os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        fd, self.temp_path = tempfile.mkstemp(prefix=".model-export-", suffix=".gguf", dir=output_dir)
        os.close(fd)
        os.remove(self.temp_path)
        self.writer = GGUFWriter(self.temp_path, architecture)
        self.metadata_names = {"general.architecture"}
        self.tensor_names = set()
        self.committed = False
        self.closed = False

    def _metadata(self, name: str, method: str, value: Any) -> None:
        if name in self.metadata_names:
            raise ValueError(f"Duplicate GGUF metadata key: {name}")
        self.metadata_names.add(name)
        getattr(self.writer, method)(name, value)

    def add_string(self, name: str, value: str) -> None:
        self._metadata(name, "add_string", value)

    def add_uint32(self, name: str, value: int) -> None:
        self._metadata(name, "add_uint32", value)

    def add_float32(self, name: str, value: float) -> None:
        self._metadata(name, "add_float32", value)

    def add_bool(self, name: str, value: bool) -> None:
        self._metadata(name, "add_bool", value)

    def add_array(self, name: str, value: Iterable[Any]) -> None:
        self._metadata(name, "add_array", value)

    def add_tensor(
        self,
        name: str,
        data: np.ndarray,
        raw_dtype: Optional[GGMLQuantizationType] = None,
    ) -> None:
        name_size = len(name.encode("utf-8"))
        if name_size > GGML_MAX_NAME_BYTES:
            raise ValueError(
                f"GGUF tensor name exceeds {GGML_MAX_NAME_BYTES} UTF-8 bytes: {name}"
            )
        if name in self.tensor_names:
            raise ValueError(f"Duplicate GGUF tensor: {name}")
        if not isinstance(data, np.ndarray) or data.size == 0:
            raise ValueError(f"Tensor {name} must be a non-empty numpy array")
        self.tensor_names.add(name)
        self.writer.add_tensor(name, data, raw_dtype=raw_dtype)

    def write(self) -> None:
        if self.committed:
            raise RuntimeError("Artifact has already been committed")
        try:
            self.writer.write_header_to_file()
            self.writer.write_kv_data_to_file()
            self.writer.write_tensors_to_file()
            self.writer.close()
            self.closed = True
            os.replace(self.temp_path, self.output_path)
            self.committed = True
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        if self.committed:
            return
        if not self.closed:
            self.writer.close()
            self.closed = True
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)

    def __del__(self):
        try:
            self.abort()
        except Exception:
            pass
