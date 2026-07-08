#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified GPT-SoVITS Model Processing Script: Conversion & Quantization.
Processes one model type at a time.

Supported Model Types:
  - t2s: Text-to-Semantic Autoregressive GPT Model
  - vits: Acoustic Decoder & Vocoder Model (s2D + s2G merged)
  - bert: Chinese RoBERTa WWM Ext Large Text Encoder
  - hubert: Chinese HuBERT Speech Feature Extractor

Quantization Rules:
  - bert: Supports F16 and Q4_0 quantization.
  - hubert / t2s / vits: Defaults to F16. Quantization lower than F16 is disabled to preserve quality.
"""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Dict, Tuple

# Ensure standard output and error use UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch

# Add gguf-py to path
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(1, str(_SCRIPT_DIR / ".." / ".." / "gguf-py"))

try:
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
    from gguf.quants import quantize
except ImportError:
    print("ERROR: gguf-py dependency not found. Please ensure gguf-py submodule is present.")
    sys.exit(1)


# ============================================================================
# Helpers
# ============================================================================

def load_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    """Load a PyTorch checkpoint and extract its state_dict."""
    print(f"  Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Cannot extract state_dict from {type(checkpoint)}")

    # Already a state_dict (all or mostly Tensor values)
    tensor_count = sum(1 for v in checkpoint.values() if isinstance(v, torch.Tensor))
    if tensor_count > 0 and tensor_count >= len(checkpoint) * 0.5:
        return checkpoint

    # Lightning / standard checkpoint
    if "state_dict" in checkpoint:
        inner = checkpoint["state_dict"]
        if isinstance(inner, dict):
            return inner

    # Custom model save with 'weight' key
    if "weight" in checkpoint and isinstance(checkpoint["weight"], dict):
        return checkpoint["weight"]

    # Common nested keys
    for key in ("model", "generator", "net_g", "net", "ar_model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    raise ValueError(f"Cannot extract state_dict from {path}")


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


# ============================================================================
# Auto-Detection of Model Version & Parameters
# ============================================================================

def auto_detect_t2s_params(sd: Dict[str, torch.Tensor], file_path: str) -> Tuple[str, int, int, int]:
    """
    Detect T2S hyperparameters from weights.
    Returns: (version, n_layers, n_heads, hidden_dim)
    """
    # 1. Detect layer count
    max_layer = -1
    for key in sd.keys():
        m = re.match(r"(?:model\.)?h\.layers\.(\d+)\.", key)
        if m:
            max_layer = max(max_layer, int(m.group(1)))
        else:
            m2 = re.match(r"(?:model\.)?h\.(\d+)\.", key)
            if m2:
                max_layer = max(max_layer, int(m2.group(1)))
    n_layers = max_layer + 1 if max_layer >= 0 else 24

    # 2. Detect hidden dimension
    hidden_dim = 512
    for key in ["ar_text_embedding.word_embeddings.weight", "model.ar_text_embedding.word_embeddings.weight"]:
        if key in sd:
            hidden_dim = sd[key].shape[1]
            break

    # 3. Detect version
    file_lower = Path(file_path).name.lower()
    if "v1" in file_lower:
        version = "v1"
    elif "v3" in file_lower:
        version = "v3"
    elif "v4" in file_lower:
        version = "v4"
    elif "v2pro" in file_lower:
        version = "v2pro"
    else:
        # Heuristics based on layers & dimension
        if n_layers == 12:
            version = "v1"
        else:
            version = "v2"

    # 4. Head count heuristics
    n_heads = 16 if version != "v1" else 8
    
    return version, n_layers, n_heads, hidden_dim


def auto_detect_vits_params(sd: Dict[str, torch.Tensor], file_path: str) -> Tuple[str, int, int]:
    """
    Detect VITS hyperparameters from weights.
    Returns: (version, n_heads, hidden_dim)
    """
    # Look for presence of v2-only keys
    has_ssl_proj = any("ssl_proj" in k for k in sd.keys())
    
    # Version heuristic
    file_lower = Path(file_path).name.lower()
    if "v1" in file_lower:
        version = "v1"
    elif "v3" in file_lower:
        version = "v3"
    elif "v4" in file_lower:
        version = "v4"
    elif "v2pro" in file_lower:
        version = "v2pro"
    else:
        version = "v2" if has_ssl_proj else "v1"

    # Encoder head count (default is 2)
    n_heads = 2
    
    # Hidden dimension (default is 192)
    hidden_dim = 192
    for k in sd.keys():
        if "enc_p.proj.weight" in k:
            hidden_dim = sd[k].shape[1] // 2  # PyTorch shape is [384, 192]
            break

    return version, n_heads, hidden_dim


# ============================================================================
# Key Mapping & Conversion Implementations
# ============================================================================

def map_t2s_key(pt_key: str) -> Optional[str]:
    """Map a PyTorch T2S key to standard GGUF name."""
    clean = pt_key
    for prefix in ("model.", "ar_model."):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]

    if clean == "ar_text_embedding.word_embeddings.weight":
        return "ar_text_embedding.word_embeddings.weight"
    if clean == "ar_audio_embedding.word_embeddings.weight":
        return "ar_audio_embedding.word_embeddings.weight"
    if clean in ("bert_proj.weight", "bert_proj.bias"):
        return clean
    if clean in ("ar_text_position.alpha", "ar_text_position_alpha"):
        return "ar_text_position.alpha"
    if clean in ("ar_audio_position.alpha", "ar_audio_position_alpha"):
        return "ar_audio_position.alpha"
    if clean == "ar_predict_layer.weight":
        return "ar_predict_layer.weight"

    # Fused QKV
    if re.match(r"h\.layers\.(\d+)\.self_attn\.in_proj_weight", clean):
        return "__SPLIT_QKV__"
    if re.match(r"h\.layers\.(\d+)\.self_attn\.in_proj_bias", clean):
        return "__SPLIT_QKV_BIAS__"

    # Transformer layers
    for norm in ("norm1", "norm2", "ln_1", "ln_2"):
        m = re.match(rf"h\.layers\.(\d+)\.{norm}\.(weight|bias)", clean)
        if m:
            layer = m.group(1)
            suffix = m.group(2)
            target_norm = "norm1" if norm in ("norm1", "ln_1") else "norm2"
            return f"h.layers.{layer}.{target_norm}.{suffix}"
        
        m = re.match(rf"h\.(\d+)\.{norm}\.(weight|bias)", clean)
        if m:
            layer = m.group(1)
            suffix = m.group(2)
            target_norm = "norm1" if norm in ("norm1", "ln_1") else "norm2"
            return f"h.layers.{layer}.{target_norm}.{suffix}"

    for fc in ("linear1", "linear2", "mlp.c_fc", "mlp.c_proj"):
        m = re.match(rf"h\.layers\.(\d+)\.{fc}\.(weight|bias)", clean)
        if m:
            layer = m.group(1)
            suffix = m.group(2)
            target_fc = "linear1" if fc in ("linear1", "mlp.c_fc") else "linear2"
            return f"h.layers.{layer}.{target_fc}.{suffix}"
            
        m = re.match(rf"h\.(\d+)\.{fc}\.(weight|bias)", clean)
        if m:
            layer = m.group(1)
            suffix = m.group(2)
            target_fc = "linear1" if fc in ("linear1", "mlp.c_fc") else "linear2"
            return f"h.layers.{layer}.{target_fc}.{suffix}"

    # Self-attention out-projection
    m = re.match(r"h\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.self_attn.out_proj.{m.group(2)}"
    m = re.match(r"h\.(\d+)\.self_attn\.out_proj\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.self_attn.out_proj.{m.group(2)}"
    m = re.match(r"h\.(\d+)\.attn\.out_proj\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.self_attn.out_proj.{m.group(2)}"

    # Projection splits fallback
    m = re.match(r"h\.(\d+)\.self_attn\.(q|k|v)_proj\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.self_attn.{m.group(2)}.{m.group(3)}"
    m = re.match(r"h\.(\d+)\.attn\.(q|k|v)_proj\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.self_attn.{m.group(2)}.{m.group(3)}"

    return None


def map_vits_key(pt_key: str) -> Optional[str]:
    """Map a PyTorch VITS key to standard GGUF name."""
    clean = pt_key
    for prefix in ("model.", "generator.", "net_g.", "vits.", "net."):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]

    known_prefixes = (
        "enc_p.", "dec.", "flow.", "ref_enc.",
        "ssl_proj.", "quantizer.", "enc_q.",
        "sv_emb.", "ge_to512.", "prelu.",
        "cfm.", "wns1.", "linear_mel.", "bridge.",
    )
    if any(clean.startswith(p) for p in known_prefixes):
        return clean
    return None


# ============================================================================
# Main Conversions
# ============================================================================

def convert_t2s(src_path: str, dst_path: str, opt_version: Optional[str] = None,
                opt_heads: Optional[int] = None, opt_layers: Optional[int] = None) -> None:
    print(f"\nConverting T2S Autoregressive GPT...")
    sd = load_checkpoint(src_path)

    # Detect parameters
    det_version, det_layers, det_heads, det_dim = auto_detect_t2s_params(sd, src_path)
    version = opt_version or det_version
    n_layers = opt_layers or det_layers
    n_heads = opt_heads or det_heads
    hidden_dim = det_dim

    print(f"  Configuration Detected/Selected:")
    print(f"    Version:           {version}")
    print(f"    Layers (n_layers): {n_layers}")
    print(f"    Heads (n_heads):   {n_heads}")
    print(f"    Hidden Dimension:  {hidden_dim}")

    writer = GGUFWriter(dst_path, "gpt_sovits_t2s")
    writer.add_string("general.architecture", "gpt_sovits_t2s")
    writer.add_string("general.name", f"GPT-SoVITS T2S ({version})")
    writer.add_string("gpt_sovits.version", version)
    writer.add_uint32("attention.head_count", n_heads)
    writer.add_uint32("gpt_sovits.t2s.n_layers", n_layers)
    writer.add_uint32("gpt_sovits.t2s.hidden_dim", hidden_dim)

    # Embed name map JSON in GGUF metadata
    import json
    name_map = {}
    name_map["word_embeddings.weight"] = "ar_text_embedding.word_embeddings.weight"
    name_map["audio_embeddings.weight"] = "ar_audio_embedding.word_embeddings.weight"
    name_map["bert_proj.weight"] = "bert_proj.weight"
    name_map["bert_proj.bias"] = "bert_proj.bias"
    name_map["predict.weight"] = "ar_predict_layer.weight"
    
    for i in range(n_layers):
        cpp = f"layers.{i}."
        gguf = f"h.layers.{i}."
        name_map[cpp + "self_attn.q_proj.weight"] = gguf + "self_attn.q.weight"
        name_map[cpp + "self_attn.q_proj.bias"] = gguf + "self_attn.q.bias"
        name_map[cpp + "self_attn.k_proj.weight"] = gguf + "self_attn.k.weight"
        name_map[cpp + "self_attn.k_proj.bias"] = gguf + "self_attn.k.bias"
        name_map[cpp + "self_attn.v_proj.weight"] = gguf + "self_attn.v.weight"
        name_map[cpp + "self_attn.v_proj.bias"] = gguf + "self_attn.v.bias"
        name_map[cpp + "self_attn.out_proj.weight"] = gguf + "self_attn.out_proj.weight"
        name_map[cpp + "self_attn.out_proj.bias"] = gguf + "self_attn.out_proj.bias"
        
        name_map[cpp + "ln1.weight"] = gguf + "norm1.weight"
        name_map[cpp + "ln1.bias"] = gguf + "norm1.bias"
        name_map[cpp + "ln2.weight"] = gguf + "norm2.weight"
        name_map[cpp + "ln2.bias"] = gguf + "norm2.bias"
        
        name_map[cpp + "ffn.w1.weight"] = gguf + "linear1.weight"
        name_map[cpp + "ffn.w1.bias"] = gguf + "linear1.bias"
        name_map[cpp + "ffn.w2.weight"] = gguf + "linear2.weight"
        name_map[cpp + "ffn.w2.bias"] = gguf + "linear2.bias"
        
    writer.add_string("gpt_sovits.t2s.name_map", json.dumps(name_map))

    mapped_count = 0
    for key in sorted(sd.keys()):
        gguf_name = map_t2s_key(key)
        if gguf_name is None:
            continue

        t = sd[key]

        if gguf_name == "__SPLIT_QKV__":
            m = re.match(r"(?:model\.)?h\.layers\.(\d+)\.self_attn\.in_proj_weight", key)
            if not m:
                continue
            layer = m.group(1)
            hidden = t.shape[1]
            q_w, k_w, v_w = t[:hidden], t[hidden:2*hidden], t[2*hidden:3*hidden]
            for suffix, tw in [("q", q_w), ("k", k_w), ("v", v_w)]:
                name = f"h.layers.{layer}.self_attn.{suffix}.weight"
                arr = tensor_to_fp16(tw)
                writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F16)
                mapped_count += 1
            continue

        if gguf_name == "__SPLIT_QKV_BIAS__":
            m = re.match(r"(?:model\.)?h\.layers\.(\d+)\.self_attn\.in_proj_bias", key)
            if not m:
                continue
            layer = m.group(1)
            hidden = t.shape[0] // 3
            q_b, k_b, v_b = t[:hidden], t[hidden:2*hidden], t[2*hidden:3*hidden]
            for suffix, tb in [("q", q_b), ("k", k_b), ("v", v_b)]:
                name = f"h.layers.{layer}.self_attn.{suffix}.bias"
                arr = tensor_to_fp32(tb)
                writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F32)
                mapped_count += 1
            continue

        # Alpha scalar (f32)
        if gguf_name.endswith(".alpha"):
            if t.ndim == 0:
                t = t.unsqueeze(0)
            arr = tensor_to_fp32(t)
            writer.add_tensor(gguf_name, arr, raw_dtype=GGMLQuantizationType.F32)
        # Biases and layer norms as float32, weights as float16
        elif t.ndim <= 1 or "bias" in gguf_name.lower() or "norm" in gguf_name.lower():
            arr = tensor_to_fp32(t)
            writer.add_tensor(gguf_name, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            arr = tensor_to_fp16(t)
            writer.add_tensor(gguf_name, arr, raw_dtype=GGMLQuantizationType.F16)
        mapped_count += 1

    print(f"  Writing GGUF output with {mapped_count} tensors...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  ✓ Saved T2S GGUF to: {dst_path}")


def convert_vits(src_dir_or_files: str, dst_path: str, opt_version: Optional[str] = None,
                 opt_heads: Optional[int] = None) -> None:
    print(f"\nConverting VITS (s2D + s2G merged)...")
    
    # Locate s2D and s2G checkpoints
    src_path = Path(src_dir_or_files)
    if src_path.is_dir():
        s2d_files = sorted(src_path.glob("s2D*.pth"))
        s2g_files = sorted(src_path.glob("s2G*.pth"))
        if not s2d_files or not s2g_files:
            raise FileNotFoundError(f"Missing s2D*.pth or s2G*.pth checkpoints inside directory '{src_dir_or_files}'")
        s2d_path = str(s2d_files[0])
        s2g_path = str(s2g_files[0])
    else:
        # User passed a single checkpoint path, check if they are comma-separated
        parts = src_dir_or_files.split(",")
        if len(parts) == 2:
            s2d_path, s2g_path = parts[0].strip(), parts[1].strip()
        else:
            raise ValueError("VITS requires both s2D and s2G checkpoints. Pass a directory or two comma-separated file paths.")

    sd_d = load_checkpoint(s2d_path)
    sd_g = load_checkpoint(s2g_path)
    
    # Merge checkpoints (s2G generator overrides s2D if duplicates exist)
    merged_sd = {**sd_d, **sd_g}
    print(f"  Merged s2D ({len(sd_d)} keys) + s2G ({len(sd_g)} keys) -> Total {len(merged_sd)} keys.")

    # Auto-detect parameters
    det_version, det_heads, det_dim = auto_detect_vits_params(merged_sd, s2g_path)
    version = opt_version or det_version
    n_heads = opt_heads or det_heads
    hidden_dim = det_dim

    print(f"  Configuration Detected/Selected:")
    print(f"    Version:           {version}")
    print(f"    Encoder Heads:     {n_heads}")
    print(f"    Hidden Dimension:  {hidden_dim}")

    # Fold Weight Normalization parameters (.weight_g & .weight_v -> .weight)
    print("  Folding Weight Normalization layers...")
    folded_keys = []
    keys_to_remove = []
    for key in list(merged_sd.keys()):
        if key.endswith(".weight_v"):
            base_name = key[:-len(".weight_v")]
            g_key = base_name + ".weight_g"
            if g_key in merged_sd:
                v = merged_sd[key]
                g = merged_sd[g_key]
                norm_dims = list(range(1, v.ndim))
                norm = torch.norm(v, p=2, dim=norm_dims, keepdim=True)
                w = g * (v / (norm + 1e-12))
                merged_sd[base_name + ".weight"] = w
                keys_to_remove.extend([key, g_key])
                folded_keys.append(base_name + ".weight")
    
    for key in keys_to_remove:
        del merged_sd[key]
    print(f"    Folded {len(folded_keys)} WeightNorm tensors.")

    writer = GGUFWriter(dst_path, "gpt_sovits_vits")
    writer.add_string("general.architecture", "gpt_sovits_vits")
    writer.add_string("general.name", f"GPT-SoVITS VITS ({version})")
    writer.add_string("gpt_sovits.version", version)
    writer.add_uint32("gpt_sovits.vits.n_heads", n_heads)
    writer.add_uint32("gpt_sovits.vits.hidden_dim", hidden_dim)
    
    # Crucial Metadata: Tells C++ that Conv1D weights are stored in GGML format directly [k, ic, oc]
    # No permutation inside python is required because PyTorch [oc, ic, k] is naturally reversed to [k, ic, oc] by GGUFWriter.
    writer.add_bool("gpt_sovits.vits.pre_transposed", True)

    mapped_count = 0
    for key in sorted(merged_sd.keys()):
        gguf_name = map_vits_key(key)
        if gguf_name is None:
            continue

        t = merged_sd[key]
        is_f32 = ("bias" in gguf_name.lower() or "norm" in gguf_name.lower() or gguf_name == "ssl_proj.bias" or t.ndim <= 1)

        if is_f32:
            arr = tensor_to_fp32(t)
            writer.add_tensor(gguf_name, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            arr = tensor_to_fp16(t)
            writer.add_tensor(gguf_name, arr, raw_dtype=GGMLQuantizationType.F16)
        mapped_count += 1

    print(f"  Writing GGUF output with {mapped_count} tensors...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  ✓ Saved VITS GGUF to: {dst_path}")


def convert_bert(src_path: str, dst_path: str, opt_version: Optional[str] = None,
                 opt_heads: Optional[int] = None) -> None:
    print(f"\nConverting Chinese RoBERTa BERT...")
    sd = load_checkpoint(src_path)

    # Detect heads
    is_bert_large = any("encoder.layer.23" in k for k in sd.keys()) or "large" in src_path.lower()
    n_heads = opt_heads or (16 if is_bert_large else 8)
    version = opt_version or "v2"

    print(f"  Configuration Detected/Selected:")
    print(f"    Version:     {version}")
    print(f"    Heads:       {n_heads}")

    writer = GGUFWriter(dst_path, "gpt_sovits_bert")
    writer.add_string("general.architecture", "gpt_sovits_bert")
    writer.add_string("general.name", f"GPT-SoVITS BERT ({version})")
    writer.add_string("gpt_sovits.version", version)
    writer.add_uint32("attention.head_count", n_heads)

    mapped_count = 0
    for key in sorted(sd.keys()):
        t = sd[key]
        if key == "bert.embeddings.position_ids":
            # Cast position_ids to float32 [512] for GGML compatibility
            t = t.squeeze(0).float()
            arr = t.contiguous().numpy()
            writer.add_tensor(key, arr, raw_dtype=GGMLQuantizationType.F32)
        elif t.ndim <= 1 or "bias" in key or "norm" in key:
            arr = tensor_to_fp32(t)
            writer.add_tensor(key, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            arr = tensor_to_fp16(t)
            writer.add_tensor(key, arr, raw_dtype=GGMLQuantizationType.F16)
        mapped_count += 1

    print(f"  Writing GGUF output with {mapped_count} tensors...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  ✓ Saved BERT GGUF to: {dst_path}")


def convert_hubert(src_path: str, dst_path: str, opt_version: Optional[str] = None) -> None:
    print(f"\nConverting Chinese HuBERT Speech Extractor...")
    sd = load_checkpoint(src_path)
    version = opt_version or "v2"

    print(f"  Configuration Detected/Selected:")
    print(f"    Version:     {version}")

    writer = GGUFWriter(dst_path, "gpt_sovits_hubert")
    writer.add_string("general.architecture", "gpt_sovits_hubert")
    writer.add_string("general.name", f"GPT-SoVITS HuBERT ({version})")
    writer.add_string("gpt_sovits.version", version)

    mapped_count = 0
    for key in sorted(sd.keys()):
        t = sd[key]
        if t.ndim <= 1 or "bias" in key or "norm" in key:
            arr = tensor_to_fp32(t)
            writer.add_tensor(key, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            arr = tensor_to_fp16(t)
            writer.add_tensor(key, arr, raw_dtype=GGMLQuantizationType.F16)
        mapped_count += 1

    print(f"  Writing GGUF output with {mapped_count} tensors...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  ✓ Saved HuBERT GGUF to: {dst_path}")


# ============================================================================
# Quantization Pipeline
# ============================================================================

SKIP_PATTERNS = ["emb", "bias", "norm", "alpha", "position_ids", "masked_spec_embed"]

def should_quantize(name: str) -> bool:
    name_lower = name.lower()
    return not any(pat in name_lower for pat in SKIP_PATTERNS)


def quantize_gguf(input_path: str, output_path: str, target_type: str) -> None:
    """Quantize an existing GGUF model using gguf-py's reference quantization engine."""
    qtype = {
        "F16": GGMLQuantizationType.F16,
        "Q4_0": GGMLQuantizationType.Q4_0,
        "Q8_0": GGMLQuantizationType.Q8_0,
    }.get(target_type)

    if qtype is None:
        raise ValueError(f"Unsupported quantization target: {target_type}")

    print(f"\nQuantizing model to {target_type}:")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    reader = GGUFReader(input_path)
    
    # Extract architecture
    arch = "gpt-sovits"
    for field in reader.fields.values():
        if field.name == "general.architecture":
            arch = bytes(field.parts[-1]).decode("utf-8").strip("\x00")
            break

    writer = GGUFWriter(output_path, arch=arch)

    # Copy metadata
    for name, field in reader.fields.items():
        if name.startswith("GGUF."):
            continue
        if not field.types:
            continue
        val = field.contents()
        writer.add_key_value(name, val, field.types[0])

    quantized_count = 0
    f16_count = 0
    kept_count = 0

    for tensor in reader.tensors:
        name = tensor.name
        tensor_type = tensor.tensor_type
        shape = tensor.shape
        data = tensor.data

        do_quantize = (
            target_type in ("Q4_0", "Q8_0")
            and should_quantize(name)
            and len(shape) >= 2
            and shape[0] % 32 == 0  # Block size alignment
        )

        if do_quantize:
            arr = data.astype(np.float32).reshape(shape[::-1])
            q_data = quantize(arr, qtype)
            writer.add_tensor(name, q_data, raw_dtype=qtype)
            quantized_count += 1
            print(f"  [{target_type}] {name} shape={list(shape)}")
        elif tensor_type == GGMLQuantizationType.F32 and target_type in ("F16", "Q4_0", "Q8_0") and not ("bias" in name.lower() or "norm" in name.lower() or "alpha" in name.lower() or len(shape) <= 1):
            arr = data.astype(np.float16)
            writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F16)
            f16_count += 1
        else:
            writer.add_tensor(name, data, raw_dtype=tensor_type)
            kept_count += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    src_sz = os.path.getsize(input_path)
    dst_sz = os.path.getsize(output_path)
    print(f"  Quantization finished successfully!")
    print(f"    Quantized tensors: {quantized_count} ({target_type})")
    print(f"    Cast F16 tensors:  {f16_count}")
    print(f"    Kept as-is:        {kept_count}")
    print(f"    Size reduction:    {100 * (src_sz - dst_sz) / src_sz:.1f}% ({src_sz/(1024**2):.1f}MB -> {dst_sz/(1024**2):.1f}MB)")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified GPT-SoVITS Conversion & Quantization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert T2S checkpoint (automatically detects version, heads, layers)
  python process.py --model-type t2s --src s1bert25hz.ckpt --output t2s_fp16.gguf

  # Convert VITS directory containing s2D/s2G (defaults to FP16)
  python process.py --model-type vits --src path/to/vits_checkpoints --output vits_fp16.gguf

  # Convert and quantize BERT model to Q4_0 (highly recommended)
  python process.py --model-type bert --src pytorch_model.bin --output bert_q4_0.gguf --quantize Q4_0
"""
    )
    
    parser.add_argument("--model-type", required=True, choices=["t2s", "vits", "bert", "hubert"],
                        help="The type of model to process.")
    parser.add_argument("--src", required=True,
                        help="Path to source PyTorch file (or directory for VITS). For VITS, two paths can also be comma-separated: 's2D.pth,s2G.pth'")
    parser.add_argument("--output", required=True,
                        help="Path to save the final GGUF file.")
    parser.add_argument("--quantize", choices=["F16", "Q4_0"], default="F16",
                        help="Target precision/quantization (default: F16). Only 'bert' supports Q4_0.")
    parser.add_argument("--version", choices=["v1", "v2", "v2pro", "v3", "v4"], default=None,
                        help="Explicitly override model version (default: auto-detect).")
    parser.add_argument("--n-heads", type=int, default=None,
                        help="Explicitly override Attention head count (default: auto-detect).")
    parser.add_argument("--n-layers", type=int, default=None,
                        help="Explicitly override T2S layer count (default: auto-detect).")
    
    args = parser.parse_args()

    # Safety constraint: Only BERT is allowed to quantize below F16
    if args.quantize == "Q4_0" and args.model_type != "bert":
        print(f"WARNING: Quantization type 'Q4_0' is not supported for model_type '{args.model_type}' because it degrades speech synthesis quality.")
        print("Falling back to target precision F16.")
        args.quantize = "F16"

    # Define intermediate path if quantization is required
    needs_quantize_step = (args.quantize != "F16")
    
    # We will write the initial FP16 GGUF file
    if needs_quantize_step:
        # Create a temporary file for the FP16 GGUF
        temp_fd, temp_fp16_path = tempfile.mkstemp(suffix=".gguf")
        os.close(temp_fd)
    else:
        temp_fp16_path = args.output

    try:
        # 1. Conversion step
        if args.model_type == "t2s":
            convert_t2s(args.src, temp_fp16_path, args.version, args.n_heads, args.n_layers)
        elif args.model_type == "vits":
            convert_vits(args.src, temp_fp16_path, args.version, args.n_heads)
        elif args.model_type == "bert":
            convert_bert(args.src, temp_fp16_path, args.version, args.n_heads)
        elif args.model_type == "hubert":
            convert_hubert(args.src, temp_fp16_path, args.version)

        # 2. Quantization step (if required)
        if needs_quantize_step:
            quantize_gguf(temp_fp16_path, args.output, args.quantize)
            
    finally:
        # Cleanup temporary file if it was created
        if needs_quantize_step and os.path.exists(temp_fp16_path):
            os.remove(temp_fp16_path)

    print(f"\n✓ Finished processing. Output saved to: {args.output}\n")


if __name__ == "__main__":
    main()
