from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def find_repository_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "CMakeLists.txt").is_file() and (
            candidate / "scripts" / "conversion" / "common"
        ).is_dir():
            return candidate
    raise RuntimeError(f"cannot locate repository root from {path}")


def ordered_labels(names, class_count: int) -> List[str]:
    if isinstance(names, dict):
        labels = [str(names[index]) for index in range(class_count)]
    elif isinstance(names, (list, tuple)):
        labels = [str(value) for value in names]
    else:
        raise ValueError("Ultralytics checkpoint has no usable class-name mapping")
    if len(labels) != class_count:
        raise ValueError(
            f"class-name count {len(labels)} does not match detector classes {class_count}"
        )
    return labels


def tensor_to_numpy(tensor, force_float32: bool) -> np.ndarray:
    value = tensor.detach().cpu()
    value = value.float() if force_float32 else value.half()
    return value.contiguous().numpy()


def csv(values: Iterable[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def load_ultralytics_checkpoint(path: Path):
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("conversion requires torch and ultralytics") from error

    original_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = trusted_load
    try:
        model = YOLO(str(path))
    finally:
        torch.load = original_load
    model.model.eval()
    model.model.fuse()
    return model
