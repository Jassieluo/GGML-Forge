"""Invoke the C++ Module-tree schema generator."""

import os
from pathlib import Path
import subprocess
from typing import Iterable

from .model_schema import ModelSchema


def _find_schema_tool() -> Path:
    configured = os.environ.get("NN_MODEL_SCHEMA_TOOL")
    if configured:
        path = Path(configured)
        if path.is_file():
            return path
        raise FileNotFoundError(f"NN_MODEL_SCHEMA_TOOL does not exist: {path}")

    root = Path(__file__).resolve().parents[2]
    executable = "nn-model-schema.exe" if os.name == "nt" else "nn-model-schema"
    preferred = root / "build-x64-windows-cuda-sycl-cpu-dl-release-f16" / "bin" / executable
    if preferred.is_file():
        return preferred
    candidates = sorted(root.glob(f"build-*/bin/{executable}"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        "nn-model-schema is not built; build the nn-model-schema CMake target first"
    )


def load_cpp_schema(architecture: str, topology: Iterable[object] = ()) -> ModelSchema:
    command = [str(_find_schema_tool()), architecture]
    command.extend(str(value) for value in topology)
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nn-model-schema failed")
    return ModelSchema.from_json(result.stdout)
