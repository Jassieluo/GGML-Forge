import sys
from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch

# Add path to the official Python repository to resolve custom checkpoint objects like utils.HParams
sys.path.append(r"D:\Projects\PycharmProjects\GPT-SoVITS-main")
sys.path.append(r"D:\Projects\PycharmProjects\GPT-SoVITS-main\GPT_SoVITS")

# Add gguf-py to path
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(1, str(_SCRIPT_DIR / ".." / ".." / "gguf-py"))

from gguf import GGUFWriter, GGMLQuantizationType

def is_vits_3d_conv_weight(name: str) -> bool:
    """
    Disabled weight transposition. Returns False.
    """
    return False

def transpose_conv_weight_vits(t: torch.Tensor) -> torch.Tensor:
    """
    Transpose a VITS 3D conv weight:
      PyTorch [out_channels, in_channels, kernel_size]
        → ggml [kernel_size, in_channels, out_channels]
    """
    return t.permute(2, 1, 0).contiguous()

def extract_state_dict(obj: Any, path: str, depth: int = 0) -> dict[str, torch.Tensor]:
    """
    Recursively extract a state_dict (str→Tensor mapping) from a checkpoint.
    """
    indent = "  " + "  " * depth

    if not isinstance(obj, dict):
        raise ValueError(f"Cannot extract state_dict from {type(obj)}")

    tensor_count = sum(1 for v in obj.values() if isinstance(v, torch.Tensor))
    if tensor_count > 0 and tensor_count >= len(obj) * 0.5:
        print(f"{indent}→ state_dict ({len(obj)} keys, {tensor_count} tensors)")
        return obj

    if "state_dict" in obj:
        inner = obj["state_dict"]
        if isinstance(inner, dict):
            print(f"{indent}→ found 'state_dict' key")
            return extract_state_dict(inner, path, depth + 1)

    if "weight" in obj and isinstance(obj["weight"], dict):
        print(f"{indent}→ found 'weight' key (custom format)")
        inner = obj["weight"]
        if any(isinstance(v, torch.Tensor) for v in inner.values()):
            return extract_state_dict(inner, path, depth + 1)
        return extract_state_dict(inner, path, depth + 1)

    for key in ("model", "generator", "net_g", "net", "ar_model"):
        if key in obj and isinstance(obj[key], dict):
            print(f"{indent}→ recursing into '{key}'")
            return extract_state_dict(obj[key], path, depth + 1)

    candidates = []
    for k, v in obj.items():
        if isinstance(v, dict):
            tc = sum(1 for vv in v.values() if isinstance(vv, torch.Tensor))
            if tc > 0:
                candidates.append((tc, k, v))
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        tc, k, v = candidates[0]
        print(f"{indent}→ auto-selected '{k}' ({tc} tensors)")
        return v

    print(f"    Top-level keys: {list(obj.keys())[:20]}")
    raise ValueError(f"Cannot extract state_dict from {path}")

def load_checkpoint(path: str) -> dict[str, torch.Tensor]:
    """Load a PyTorch checkpoint and extract its state_dict."""
    print(f"  Loading: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return extract_state_dict(checkpoint, path)

def tensor_to_fp16(t: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to float16 numpy array."""
    t = t.detach().cpu()
    if t.dtype != torch.float16:
        t = t.half()
    return t.contiguous().numpy()

def tensor_to_fp32(t: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to float32 numpy array."""
    t = t.detach().cpu()
    if t.dtype != torch.float32:
        t = t.float()
    return t.contiguous().numpy()

def add_tensor_fp16(gguf_writer: GGUFWriter, name: str, tensor: torch.Tensor) -> None:
    """Add a tensor to the GGUF writer in FP16 format."""
    arr = tensor_to_fp16(tensor)
    gguf_writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F16)

def add_vits_tensor(gguf_writer: GGUFWriter, name: str, tensor: torch.Tensor,
                    pre_transpose: bool, dry_run: bool) -> None:
    """Add a VITS tensor to the GGUF writer."""
    is_f32 = ("bias" in name.lower() or "norm" in name.lower() or name == "ssl_proj.bias")

    if pre_transpose:
        original_shape = list(tensor.shape)
        tensor = transpose_conv_weight_vits(tensor)
        new_shape = list(tensor.shape)
        if dry_run:
            print(f"    + {name}: {original_shape} → TRANSPOSE → {new_shape}  (fp16)")
            return
        arr = tensor_to_fp16(tensor)
        print(f"    + {name}: {original_shape} → TRANSPOSE → {new_shape}  (fp16, {arr.nbytes} bytes)")
        gguf_writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F16)
    else:
        if is_f32:
            if dry_run:
                print(f"    + {name}: {list(tensor.shape)}  (f32)")
                return
            arr = tensor_to_fp32(tensor)
            print(f"    + {name}: {list(tensor.shape)}  (f32, {arr.nbytes} bytes)")
            gguf_writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            if dry_run:
                print(f"    + {name}: {list(tensor.shape)}  (fp16)")
                return
            arr = tensor_to_fp16(tensor)
            print(f"    + {name}: {list(tensor.shape)}  (fp16, {arr.nbytes} bytes)")
            gguf_writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F16)
