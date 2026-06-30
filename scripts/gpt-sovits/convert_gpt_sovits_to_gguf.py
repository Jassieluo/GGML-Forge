#!/usr/bin/env python3
"""
Convert GPT-SoVITS v2 pretrained models to GGUF format (FP16).

Source: GPT-SoVITS v2 final pretrained models
  - s1 (T2S):  s1bert25hz-5kh-longer-epoch3D369668.ckpt
  - s2D (VITS decoder):  s2D2333k.pth
  - s2G (VITS generator): s2G2333k.pth

Output:
  - models/speech/t2s/t2s_fp16.gguf
  - models/speech/vits/vits_fp16.gguf

Key design decisions:
  - All weights stored as FP16 (except alpha scalars → F32)
  - VITS 3D conv weights are PRE-TRANSPOSED from PyTorch [oc,ic,k] → ggml [k,ic,oc]
    so the C++ backend does NOT need runtime CPU transposition.
  - T2S attention head count: 16 (v2base model, detected from filename)
  - VITS encoder attention head count: 2 (192 / 96 d_k)

Usage:
  python convert_gpt_sovits_to_gguf.py --dry-run          # inspect keys first
  python convert_gpt_sovits_to_gguf.py                     # full conversion
  python convert_gpt_sovits_to_gguf.py --t2s-only          # T2S only
  python convert_gpt_sovits_to_gguf.py --vits-only         # VITS only
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure standard output and error use UTF-8 to avoid encoding errors on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch

# Add gguf-py to path
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(1, str(_SCRIPT_DIR / ".." / ".." / "gguf-py"))

from gguf import GGUFWriter, GGMLQuantizationType  # noqa: E402


# ============================================================================
# VITS conv weight transposition helpers
# ============================================================================

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

    This mirrors the C++ load-time transpose in gpt_sovits.cpp:537-549.
    By doing it at conversion time we avoid the slow CPU transpose at load.
    """
    # t.shape = [out_channels, in_channels, kernel_size]
    # Target:   [kernel_size, in_channels, out_channels] via permute(2,1,0)
    return t.permute(2, 1, 0).contiguous()


# ============================================================================
# Tensor name mapping: PyTorch state_dict key → GGUF tensor name
# ============================================================================

def map_t2s_key(pt_key: str) -> Optional[str]:
    """
    Map a PyTorch T2S (s1) state_dict key to the GGUF tensor name
    expected by ggml_t2s.cpp.

    Returns:
        - GGUF tensor name string for a direct 1:1 mapping
        - "__SPLIT_QKV__" for fused in_proj_weight (needs splitting into q/k/v)
        - "__SPLIT_QKV_BIAS__" for fused in_proj_bias (needs splitting into q/k/v)
        - None if the key cannot be mapped
    """
    import re

    clean = pt_key

    # Strip common prefixes
    for prefix in ("model.", "ar_model."):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]

    # ---- Known direct mappings ----

    # Text / audio embeddings
    if clean == "ar_text_embedding.word_embeddings.weight":
        return "ar_text_embedding.word_embeddings.weight"
    if clean == "ar_audio_embedding.word_embeddings.weight":
        return "ar_audio_embedding.word_embeddings.weight"

    # BERT projection
    if clean in ("bert_proj.weight", "bert_proj.bias"):
        return clean

    # Positional alpha scalars
    if clean in ("ar_text_position.alpha", "ar_text_position_alpha"):
        return "ar_text_position.alpha"
    if clean in ("ar_audio_position.alpha", "ar_audio_position_alpha"):
        return "ar_audio_position.alpha"

    # Final prediction layer
    if clean == "ar_predict_layer.weight":
        return "ar_predict_layer.weight"

    # ---- Fused QKV projections (PyTorch MHA uses combined in_proj) ----
    # h.layers.{N}.self_attn.in_proj_weight [3*hidden, hidden] → split into q/k/v
    if re.match(r"h\.layers\.(\d+)\.self_attn\.in_proj_weight", clean):
        return "__SPLIT_QKV__"
    if re.match(r"h\.layers\.(\d+)\.self_attn\.in_proj_bias", clean):
        return "__SPLIT_QKV_BIAS__"

    # ---- Transformer layers (h.layers.{N}.xxx pattern) ----

    # h.layers.{N}.norm1.{weight,bias}
    m = re.match(r"h\.layers\.(\d+)\.norm1\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.norm1.{m.group(2)}"

    # h.layers.{N}.norm2.{weight,bias}
    m = re.match(r"h\.layers\.(\d+)\.norm2\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.norm2.{m.group(2)}"

    # h.layers.{N}.linear1.{weight,bias}
    m = re.match(r"h\.layers\.(\d+)\.linear1\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.linear1.{m.group(2)}"

    # h.layers.{N}.linear2.{weight,bias}
    m = re.match(r"h\.layers\.(\d+)\.linear2\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.linear2.{m.group(2)}"

    # h.layers.{N}.self_attn.out_proj.{weight,bias}
    m = re.match(r"h\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.self_attn.out_proj.{m.group(2)}"

    # ---- Fallback: h.{N}.xxx patterns (some older models) ----

    # h.{N}.self_attn.{q|k|v}_proj.{weight|bias}
    m = re.match(r"h\.(\d+)\.self_attn\.(q|k|v)_proj\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.self_attn.{m.group(2)}.{m.group(3)}"

    # h.{N}.self_attn.out_proj.{weight|bias}
    m = re.match(r"h\.(\d+)\.self_attn\.out_proj\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.self_attn.out_proj.{m.group(2)}"

    # h.{N}.attn.{q|k|v}_proj.{weight|bias}
    m = re.match(r"h\.(\d+)\.attn\.(q|k|v)_proj\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.self_attn.{m.group(2)}.{m.group(3)}"

    # h.{N}.attn.out_proj.{weight|bias}
    m = re.match(r"h\.(\d+)\.attn\.out_proj\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.self_attn.out_proj.{m.group(2)}"

    # h.{N}.norm1.{weight|bias}
    m = re.match(r"h\.(\d+)\.norm1\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.norm1.{m.group(2)}"

    # h.{N}.norm2.{weight|bias}
    m = re.match(r"h\.(\d+)\.norm2\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.norm2.{m.group(2)}"

    # h.{N}.ln_1.{weight|bias}
    m = re.match(r"h\.(\d+)\.ln_1\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.norm1.{m.group(2)}"

    # h.{N}.ln_2.{weight|bias}
    m = re.match(r"h\.(\d+)\.ln_2\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.norm2.{m.group(2)}"

    # h.{N}.linear1.{weight|bias}
    m = re.match(r"h\.(\d+)\.linear1\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.linear1.{m.group(2)}"

    # h.{N}.linear2.{weight|bias}
    m = re.match(r"h\.(\d+)\.linear2\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.linear2.{m.group(2)}"

    # h.{N}.mlp.c_fc.{weight|bias}
    m = re.match(r"h\.(\d+)\.mlp\.c_fc\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.linear1.{m.group(2)}"

    # h.{N}.mlp.c_proj.{weight|bias}
    m = re.match(r"h\.(\d+)\.mlp\.c_proj\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.linear2.{m.group(2)}"

    return None


def map_vits_key(pt_key: str) -> Optional[str]:
    """
    Map a PyTorch VITS (s2D/s2G) state_dict key to the GGUF tensor name.

    The VITS model typically stores weights with a "net_g." prefix or directly.
    We strip known wrapper prefixes and keep the core name which matches
    what ggml_vits.cpp expects (see scratch_vits_keys.txt).
    """
    clean = pt_key

    # Strip known wrapper prefixes
    for prefix in ("model.", "generator.", "net_g.", "vits.", "net."):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]

    # The GGUF backend expects these top-level prefixes
    known_prefixes = (
        "enc_p.", "dec.", "flow.", "ref_enc.",
        "ssl_proj.", "quantizer.", "enc_q.",
        "sv_emb.", "ge_to512.", "prelu.",
    )
    if any(clean.startswith(p) for p in known_prefixes):
        return clean

    return None


# ============================================================================
# Checkpoint loading
# ============================================================================

def extract_state_dict(obj: Any, path: str, depth: int = 0) -> dict[str, torch.Tensor]:
    """
    Recursively extract a state_dict (str→Tensor mapping) from a checkpoint.

    Handles:
      - Plain state_dict (all values are Tensors)
      - Lightning checkpoint (has "state_dict" key)
      - Model save with 'weight'/'config'/'info' keys
      - Nested dicts with a single model wrapper
    """
    indent = "  " + "  " * depth

    if not isinstance(obj, dict):
        raise ValueError(f"Cannot extract state_dict from {type(obj)}")

    # Already a state_dict (all or mostly Tensor values)
    tensor_count = sum(1 for v in obj.values() if isinstance(v, torch.Tensor))
    if tensor_count > 0 and tensor_count >= len(obj) * 0.5:
        print(f"{indent}→ state_dict ({len(obj)} keys, {tensor_count} tensors)")
        return obj

    # Lightning / standard checkpoint: "state_dict" key
    if "state_dict" in obj:
        inner = obj["state_dict"]
        if isinstance(inner, dict):
            print(f"{indent}→ found 'state_dict' key")
            return extract_state_dict(inner, path, depth + 1)

    # Model save with 'weight' key (custom GPT-SoVITS format)
    if "weight" in obj and isinstance(obj["weight"], dict):
        print(f"{indent}→ found 'weight' key (custom format)")
        inner = obj["weight"]
        # Sometimes 'weight' is itself a state_dict
        if any(isinstance(v, torch.Tensor) for v in inner.values()):
            return extract_state_dict(inner, path, depth + 1)
        # Or it may have nested structure
        return extract_state_dict(inner, path, depth + 1)

    # Try common nested keys
    for key in ("model", "generator", "net_g", "net", "ar_model"):
        if key in obj and isinstance(obj[key], dict):
            print(f"{indent}→ recursing into '{key}'")
            return extract_state_dict(obj[key], path, depth + 1)

    # Last resort: enumerate top-level keys and return the largest dict of tensors
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


# ============================================================================
# Tensor conversion helpers
# ============================================================================

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
    """
    Add a VITS tensor to the GGUF writer.

    If pre_transpose is True, the tensor is a 3D conv weight that needs
    PyTorch [oc,ic,k] → ggml [k,ic,oc] transposition.
    """
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


# ============================================================================
# Convert T2S (s1)
# ============================================================================

def convert_t2s(src_path: str, dst_path: str, is_v2: bool = True, dry_run: bool = False) -> None:
    """Convert the s1 T2S checkpoint to GGUF."""
    print(f"\n{'=' * 70}")
    print(f"Converting T2S (s1) model")
    print(f"  Source: {src_path}")
    print(f"  Output: {dst_path}")
    print(f"{'=' * 70}")

    sd = load_checkpoint(src_path)

    # Auto-detect n_heads from fused in_proj_weight or separate q_proj
    # v2base model: 16 heads (hidden_dim=512, head_dim=32; or 1024, head_dim=64)
    # v1 model: 8 heads (hidden_dim=512, head_dim=64)
    is_v2_detected = is_v2 or any(x in src_path.lower() for x in ["v2", "base", "s1bert", "369668", "firekeeper"])
    
    n_heads = 16 if is_v2_detected else 8
    for key in sd.keys():
        if "self_attn" in key and "weight" in key:
            t = sd[key]
            if t.ndim == 2:
                if "in_proj_weight" in key:
                    # Fused: [3*hidden, hidden] → hidden = shape[1]
                    hidden_dim = t.shape[1]
                else:
                    hidden_dim = t.shape[0]
                if is_v2_detected:
                    n_heads = 16
                else:
                    n_heads = hidden_dim // 64
                print(f"  Auto-detected n_heads={n_heads} (hidden_dim={hidden_dim}, is_v2={is_v2_detected})")
                break

    if dry_run:
        print(f"\n  [DRY RUN] n_heads={n_heads}")
        print("  Inspecting keys and mapping...")
        mapped = 0
        unmapped = []
        for key in sorted(sd.keys()):
            gguf_name = map_t2s_key(key)
            if gguf_name is None:
                unmapped.append(key)
            elif gguf_name == "__SPLIT_QKV__":
                t = sd[key]
                m = re.match(r"(?:model\.)?h\.layers\.(\d+)\.self_attn\.in_proj_weight", key)
                layer = m.group(1) if m else "?"
                hidden = t.shape[1]
                print(f"    {key}  →  SPLIT → q/k/v.weight [{hidden},{hidden}] ×3  (layer {layer})")
                mapped += 3
            elif gguf_name == "__SPLIT_QKV_BIAS__":
                t = sd[key]
                m = re.match(r"(?:model\.)?h\.layers\.(\d+)\.self_attn\.in_proj_bias", key)
                layer = m.group(1) if m else "?"
                hidden = t.shape[0] // 3
                print(f"    {key}  →  SPLIT → q/k/v.bias [{hidden}] ×3  (layer {layer})")
                mapped += 3
            else:
                t = sd[key]
                print(f"    {key}  →  {gguf_name}  shape={list(t.shape)} dtype={t.dtype}")
                mapped += 1
        print(f"\n  Mapped: {mapped} tensors (from {len(sd)} keys), Unmapped: {len(unmapped)} keys")
        if unmapped:
            print("  Skipped keys:")
            for key in unmapped:
                t = sd[key]
                print(f"    ? {key}  shape={list(t.shape)}  dtype={t.dtype}")
        return

    gguf_writer = GGUFWriter(dst_path, "gpt_sovits_t2s")

    # Metadata
    gguf_writer.add_string("general.name", "GPT-SoVITS v2 T2S")
    gguf_writer.add_string("general.description",
                           "GPT-SoVITS v2 final Text-to-Semantic model")
    gguf_writer.add_uint32("attention.head_count", n_heads)

    mapped_count = 0
    unmapped_keys = []
    for key in sorted(sd.keys()):
        gguf_name = map_t2s_key(key)
        if gguf_name is None:
            unmapped_keys.append(key)
            continue

        # --- Fused QKV: split in_proj_weight [3*hidden, hidden] → q/k/v.weight [hidden, hidden] ---
        if gguf_name == "__SPLIT_QKV__":
            t = sd[key]
            # Extract layer number from key like "model.h.layers.23.self_attn.in_proj_weight"
            m = re.match(r"(?:model\.)?h\.layers\.(\d+)\.self_attn\.in_proj_weight", key)
            if not m:
                print(f"    ⚠ Cannot parse layer number from: {key}")
                continue
            layer = m.group(1)
            hidden = t.shape[1]  # [3*hidden, hidden]
            q_w, k_w, v_w = t[:hidden], t[hidden:2*hidden], t[2*hidden:3*hidden]
            for suffix, tw in [("q", q_w), ("k", k_w), ("v", v_w)]:
                name = f"h.layers.{layer}.self_attn.{suffix}.weight"
                add_tensor_fp16(gguf_writer, name, tw)
                mapped_count += 1
            continue

        # --- Fused QKV bias: split in_proj_bias [3*hidden] → q/k/v.bias [hidden] ---
        if gguf_name == "__SPLIT_QKV_BIAS__":
            t = sd[key]
            m = re.match(r"(?:model\.)?h\.layers\.(\d+)\.self_attn\.in_proj_bias", key)
            if not m:
                print(f"    ⚠ Cannot parse layer number from: {key}")
                continue
            layer = m.group(1)
            hidden = t.shape[0] // 3  # [3*hidden]
            q_b, k_b, v_b = t[:hidden], t[hidden:2*hidden], t[2*hidden:3*hidden]
            for suffix, tb in [("q", q_b), ("k", k_b), ("v", v_b)]:
                name = f"h.layers.{layer}.self_attn.{suffix}.bias"
                arr = tensor_to_fp32(tb)
                print(f"    + {name}: {list(tb.shape)} (f32, {arr.nbytes} bytes)")
                gguf_writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F32)
                mapped_count += 1
            continue

        t = sd[key]

        # Alpha scalars → F32
        if gguf_name.endswith(".alpha"):
            if t.ndim == 0:
                t = t.unsqueeze(0)
            arr = tensor_to_fp32(t)
            print(f"    + {gguf_name}: scalar (f32, {arr.nbytes} bytes)")
            gguf_writer.add_tensor(gguf_name, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            if t.ndim <= 1 or "bias" in gguf_name.lower() or "norm" in gguf_name.lower():
                arr = tensor_to_fp32(t)
                print(f"    + {gguf_name}: {list(t.shape)} (f32, {arr.nbytes} bytes)")
                gguf_writer.add_tensor(gguf_name, arr, raw_dtype=GGMLQuantizationType.F32)
            else:
                add_tensor_fp16(gguf_writer, gguf_name, t)
        mapped_count += 1

    print(f"\n  Writing GGUF file ({mapped_count} tensors)...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    if unmapped_keys:
        print(f"\n  ⚠ {len(unmapped_keys)} keys were SKIPPED (not mapped):")
        for key in unmapped_keys:
            print(f"    ? {key}  shape={list(sd[key].shape)}  dtype={sd[key].dtype}")
    else:
        print(f"\n  ✓ All {mapped_count} keys mapped successfully.")

    print(f"  ✓ T2S GGUF saved to: {dst_path}")


# ============================================================================
# Convert VITS (s2D + s2G → single GGUF)
# ============================================================================

def convert_vits(src_dir: str, dst_path: str, dry_run: bool = False) -> None:
    """Convert the s2 VITS checkpoints (s2D + s2G merged) to a single GGUF file."""
    print(f"\n{'=' * 70}")
    print(f"Converting VITS (s2) model")
    print(f"  Source dir: {src_dir}")
    print(f"  Output:     {dst_path}")
    print(f"{'=' * 70}")

    src_dir_p = Path(src_dir)
    s2d_files = sorted(src_dir_p.glob("s2D*.pth"))
    s2g_files = sorted(src_dir_p.glob("s2G*.pth"))

    if not s2d_files:
        raise FileNotFoundError(f"No s2D*.pth found in {src_dir}")
    if not s2g_files:
        raise FileNotFoundError(f"No s2G*.pth found in {src_dir}")

    print(f"\n  s2D: {s2d_files[0].name}")
    sd_d = load_checkpoint(str(s2d_files[0]))
    print(f"  s2G: {s2g_files[0].name}")
    sd_g = load_checkpoint(str(s2g_files[0]))

    # Merge (s2G wins on conflict)
    merged_sd = {**sd_d, **sd_g}
    print(f"\n  Merged: {len(merged_sd)} keys "
          f"({len(sd_d)} s2D + {len(sd_g)} s2G)")

    # Fold Weight Normalization parameters into standard weight tensors
    print("\n  Folding Weight Normalization parameters...")
    folded_keys = []
    keys_to_remove = []
    for key in list(merged_sd.keys()):
        if key.endswith(".weight_v"):
            base_name = key[:-len(".weight_v")]
            g_key = base_name + ".weight_g"
            if g_key in merged_sd:
                v = merged_sd[key]
                g = merged_sd[g_key]
                # Compute L2 norm along all dimensions except output channel (dim 0)
                norm_dims = list(range(1, v.ndim))
                norm = torch.norm(v, p=2, dim=norm_dims, keepdim=True)
                w = g * (v / (norm + 1e-12))
                merged_sd[base_name + ".weight"] = w
                keys_to_remove.extend([key, g_key])
                folded_keys.append(base_name + ".weight")
    
    for key in keys_to_remove:
        del merged_sd[key]
        
    print(f"  Successfully folded {len(folded_keys)} WeightNorm layers.")

    # Determine VITS encoder head count
    # d_k=96, hidden=192 → 192/96 = 2 heads
    n_heads_vits = 2

    if dry_run:
        print(f"\n  [DRY RUN] n_heads={n_heads_vits}")
        print("  Disabling pre-transposition in Python (handled at C++ load-time)")
        print("  Inspecting keys and mapping...")
        mapped = 0
        unmapped = []
        for key in sorted(merged_sd.keys()):
            gguf_name = map_vits_key(key)
            if gguf_name:
                t = merged_sd[key]
                needs_transpose = (t.ndim == 3 and is_vits_3d_conv_weight(gguf_name))
                add_vits_tensor(None, gguf_name, t, needs_transpose, dry_run=True)
                mapped += 1
            else:
                unmapped.append(key)
        print(f"\n  Mapped: {mapped} keys, Unmapped: {len(unmapped)} keys")
        if unmapped:
            print("  Skipped keys:")
            for key in unmapped:
                t = merged_sd[key]
                print(f"    ? {key}  shape={list(t.shape)}  dtype={t.dtype}")
        return

    gguf_writer = GGUFWriter(dst_path, "gpt_sovits_vits")

    # Metadata
    gguf_writer.add_string("general.name", "GPT-SoVITS v2 VITS")
    gguf_writer.add_string("general.description",
                           "GPT-SoVITS v2 final VITS vocoder (s2D + s2G merged)")
    gguf_writer.add_uint32("attention.head_count", n_heads_vits)

    mapped_count = 0
    unmapped_keys = []
    for key in sorted(merged_sd.keys()):
        gguf_name = map_vits_key(key)
        if gguf_name is None:
            unmapped_keys.append(key)
            continue
        t = merged_sd[key]
        needs_transpose = (t.ndim == 3 and is_vits_3d_conv_weight(gguf_name))
        add_vits_tensor(gguf_writer, gguf_name, t, needs_transpose, dry_run=False)
        mapped_count += 1

    print(f"\n  Writing GGUF file ({mapped_count} tensors)...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    if unmapped_keys:
        print(f"\n  ⚠ {len(unmapped_keys)} keys were SKIPPED (not mapped):")
        for key in unmapped_keys:
            print(f"    ? {key}  shape={list(merged_sd[key].shape)}  dtype={merged_sd[key].dtype}")
    else:
        print(f"\n  ✓ All {mapped_count} keys mapped successfully.")

    print(f"  ✓ VITS GGUF saved to: {dst_path}")


# ============================================================================
# Main
# ============================================================================

# ============================================================================
# Convert BERT (chinese-roberta-wwm-ext-large)
# ============================================================================

def convert_bert(src_path: str, dst_path: str, dry_run: bool = False) -> None:
    """
    Convert HuggingFace Chinese RoBERTa BERT model to GGUF.
    All tensor names match the existing models/speech/bert/bert.gguf convention.
    """
    print(f"\n{'=' * 70}")
    print(f"Converting BERT model")
    print(f"  Source: {src_path}")
    print(f"  Output: {dst_path}")
    print(f"{'=' * 70}")

    sd = torch.load(src_path, map_location="cpu", weights_only=False)
    print(f"  Loaded {len(sd)} tensors")

    if dry_run:
        print("\n  [DRY RUN] Inspecting keys...")
        for key in sorted(sd.keys()):
            t = sd[key]
            print(f"    {key}: shape={list(t.shape)} dtype={t.dtype}")
        return

    gguf_writer = GGUFWriter(dst_path, "gpt-sovits-bert")
    gguf_writer.add_string("general.name", "Chinese RoBERTa WWM Ext Large")
    gguf_writer.add_string("general.description",
                           "Chinese RoBERTa BERT model for GPT-SoVITS text encoding")

    mapped_count = 0
    for key in sorted(sd.keys()):
        t = sd[key]
        if key == "bert.embeddings.position_ids":
            # Keep as [512] (inference creates its own position_ids, this is unused metadata)
            # Cast to float() to ensure it is represented as 4-byte FP32, matching raw_dtype F32
            t = t.squeeze(0).float()
            arr = t.contiguous().numpy()
            print(f"    + {key}: {list(t.shape)} (F32, {arr.nbytes} bytes)")
            gguf_writer.add_tensor(key, arr, raw_dtype=GGMLQuantizationType.F32)
        elif t.ndim <= 1 or "bias" in key or "norm" in key:
            arr = tensor_to_fp32(t)
            gguf_writer.add_tensor(key, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            add_tensor_fp16(gguf_writer, key, t)
        mapped_count += 1

    print(f"\n  Writing GGUF file ({mapped_count} tensors)...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    print(f"  ✓ BERT GGUF saved to: {dst_path}")


# ============================================================================
# Convert HuBERT (chinese-hubert-base)
# ============================================================================

def convert_hubert(src_path: str, dst_path: str, dry_run: bool = False) -> None:
    """
    Convert HuggingFace Chinese HuBERT model to GGUF.
    All tensor names match the existing models/speech/cnhubert/cnhubert.gguf convention.
    Uses standard ggml_conv_1d (not VITS) so no weight transposition is needed.
    """
    print(f"\n{'=' * 70}")
    print(f"Converting HuBERT model")
    print(f"  Source: {src_path}")
    print(f"  Output: {dst_path}")
    print(f"{'=' * 70}")

    sd = torch.load(src_path, map_location="cpu", weights_only=False)
    print(f"  Loaded {len(sd)} tensors")

    if dry_run:
        print("\n  [DRY RUN] Inspecting keys...")
        for key in sorted(sd.keys()):
            t = sd[key]
            print(f"    {key}: shape={list(t.shape)} dtype={t.dtype}")
        return

    gguf_writer = GGUFWriter(dst_path, "gpt-sovits-hubert")
    gguf_writer.add_string("general.name", "Chinese HuBERT Base")
    gguf_writer.add_string("general.description",
                           "Chinese HuBERT Base model for GPT-SoVITS speech feature extraction")

    mapped_count = 0
    for key in sorted(sd.keys()):
        t = sd[key]
        if t.ndim <= 1 or "bias" in key or "norm" in key:
            arr = tensor_to_fp32(t)
            gguf_writer.add_tensor(key, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            add_tensor_fp16(gguf_writer, key, t)
        mapped_count += 1

    print(f"\n  Writing GGUF file ({mapped_count} tensors)...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    print(f"  ✓ HuBERT GGUF saved to: {dst_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert GPT-SoVITS v2 pretrained models to GGUF (FP16)")
    parser.add_argument(
        "--src-dir", type=str,
        default=r"D:\Projects\PycharmProjects\GPT-SoVITS-main\GPT_SoVITS\pretrained_models\gsv-v2final-pretrained",
        help="Directory containing s1 and s2 PyTorch model files")
    parser.add_argument(
        "--dst-dir", type=str, default="models/speech",
        help="Output directory for GGUF files (default: models/speech) — relative to project root")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Inspect keys and mapping without writing files")
    parser.add_argument(
        "--v1", action="store_true",
        help="Convert as GPT-SoVITS v1 model (8 heads for T2S)")
    parser.add_argument(
        "--t2s-only", action="store_true", help="Convert only T2S (s1)")
    parser.add_argument(
        "--vits-only", action="store_true", help="Convert only VITS (s2)")
    parser.add_argument(
        "--bert-only", action="store_true", help="Convert only BERT")
    parser.add_argument(
        "--hubert-only", action="store_true", help="Convert only HuBERT")
    parser.add_argument(
        "--bert-src", type=str,
        default=r"D:\Projects\PycharmProjects\GPT-SoVITS-main\GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large\pytorch_model.bin",
        help="Path to BERT pytorch_model.bin")
    parser.add_argument(
        "--hubert-src", type=str,
        default=r"D:\Projects\PycharmProjects\GPT-SoVITS-main\GPT_SoVITS\pretrained_models\chinese-hubert-base\pytorch_model.bin",
        help="Path to HuBERT pytorch_model.bin")
    parser.add_argument(
        "--bert-out", type=str, default="bert_fp16.gguf",
        help="BERT output filename")
    parser.add_argument(
        "--hubert-out", type=str, default="cnhubert_fp16.gguf",
        help="HuBERT output filename")
    parser.add_argument(
        "--s1-file", type=str, default=None,
        help="Specific s1 checkpoint filename")
    parser.add_argument(
        "--t2s-out", type=str, default="t2s_fp16.gguf",
        help="T2S output filename")
    parser.add_argument(
        "--vits-out", type=str, default="vits_fp16.gguf",
        help="VITS output filename")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    if not src_dir.exists():
        print(f"ERROR: Source directory does not exist: {src_dir}")
        print("Use --src-dir to specify the correct path.")
        sys.exit(1)

    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    t2s_dst_dir = dst_dir / "t2s"
    t2s_dst_dir.mkdir(parents=True, exist_ok=True)
    t2s_dst = t2s_dst_dir / args.t2s_out

    vits_dst_dir = dst_dir / "vits"
    vits_dst_dir.mkdir(parents=True, exist_ok=True)
    vits_dst = vits_dst_dir / args.vits_out

    # Determine which models to run
    run_all = not (args.t2s_only or args.vits_only or args.bert_only or args.hubert_only)

    # ── T2S ──
    if run_all or args.t2s_only:
        if args.s1_file:
            s1_path = src_dir / args.s1_file
        else:
            s1_candidates = sorted(src_dir.glob("s1*.ckpt")) + sorted(src_dir.glob("s1*.pth"))
            if not s1_candidates:
                print("ERROR: No s1 checkpoint found. Use --s1-file.")
                sys.exit(1)
            s1_path = s1_candidates[0]
        if not s1_path.exists():
            print(f"ERROR: {s1_path} not found")
            sys.exit(1)
        convert_t2s(str(s1_path), str(t2s_dst), is_v2=not args.v1, dry_run=args.dry_run)

    # ── VITS ──
    if run_all or args.vits_only:
        convert_vits(str(src_dir), str(vits_dst), dry_run=args.dry_run)

    # ── BERT ──
    if run_all or args.bert_only:
        bert_src = Path(args.bert_src)
        if not bert_src.exists():
            print(f"ERROR: BERT source not found: {bert_src}")
            sys.exit(1)
        bert_dst_dir = dst_dir / "bert"
        bert_dst_dir.mkdir(parents=True, exist_ok=True)
        convert_bert(str(bert_src), str(bert_dst_dir / args.bert_out), dry_run=args.dry_run)

    # ── HuBERT ──
    if run_all or args.hubert_only:
        hubert_src = Path(args.hubert_src)
        if not hubert_src.exists():
            print(f"ERROR: HuBERT source not found: {hubert_src}")
            sys.exit(1)
        hubert_dst_dir = dst_dir / "cnhubert"
        hubert_dst_dir.mkdir(parents=True, exist_ok=True)
        convert_hubert(str(hubert_src), str(hubert_dst_dir / args.hubert_out), dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\n{'=' * 70}")
        print("Conversion complete!")
        if run_all or args.t2s_only:
            print(f"  T2S:  {t2s_dst}")
        if run_all or args.vits_only:
            print(f"  VITS: {vits_dst}")
        if run_all or args.bert_only:
            print(f"  BERT: {dst_dir / 'bert' / args.bert_out}")
        if run_all or args.hubert_only:
            print(f"  HuBERT: {dst_dir / 'cnhubert' / args.hubert_out}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
