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
  - All artifacts support F16, Q4_0, Q4_K/Q4_K_M, and Q8_0 export.
  - Quantized targets fall back through Q8_0 to F16 when row constraints require it.
  - Small normalization parameters, biases, and scalar parameters stay in F32.
"""

import argparse
import os
import re
import sys
import tempfile
import types
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
_REPOSITORY_ROOT = _SCRIPT_DIR.parents[3]
sys.path.insert(1, str(_REPOSITORY_ROOT / "gguf-py"))
sys.path.insert(1, str(_REPOSITORY_ROOT / "scripts"))

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("ERROR: gguf-py dependency not found. Please ensure gguf-py submodule is present.")
    sys.exit(1)

from model_version import normalize_model_version, t2s_family_for_version
from common.model import ModelArtifact
from common.layout import Layout, write_layout_metadata
from common.quantization import QuantizationPolicy, QuantizationReport, TensorSpec, quantize_gguf as run_quantization
from common.validation import ArtifactContract, validate_artifact

VITS_ARTIFACT_CONTRACTS = {
    "v1": dict(api_version=1, symbol_version=1, family=1, semantic_stride=1,
               implementation="classic", vocoder="sovits", requires_sv=False, output_sr=32000,
               reference_sr=32000, prompt_sr=0, filter_length=2048, hop_length=640,
               window_length=2048, mel_channels=0, default_steps=0,
               feature_rate_scale=0.0, max_prompt_frames=0, upsample_rates=[10, 8, 2, 2, 2]),
    "v2": dict(api_version=2, symbol_version=2, family=2, semantic_stride=2,
               implementation="classic", vocoder="sovits", requires_sv=False, output_sr=32000,
               reference_sr=32000, prompt_sr=0, filter_length=2048, hop_length=640,
               window_length=2048, mel_channels=0, default_steps=0,
               feature_rate_scale=0.0, max_prompt_frames=0, upsample_rates=[10, 8, 2, 2, 2]),
    "v2Pro": dict(api_version=2, symbol_version=2, family=3, semantic_stride=2,
                  implementation="classic", vocoder="sovits", requires_sv=True, output_sr=32000,
                  reference_sr=32000, prompt_sr=0, filter_length=2048, hop_length=640,
                  window_length=2048, mel_channels=0, default_steps=0,
                  feature_rate_scale=0.0, max_prompt_frames=0, upsample_rates=[10, 8, 2, 2, 2]),
    "v2ProPlus": dict(api_version=2, symbol_version=2, family=3, semantic_stride=2,
                      implementation="classic", vocoder="sovits", requires_sv=True, output_sr=32000,
                      reference_sr=32000, prompt_sr=0, filter_length=2048, hop_length=640,
                      window_length=2048, mel_channels=0, default_steps=0,
                      feature_rate_scale=0.0, max_prompt_frames=0, upsample_rates=[10, 8, 2, 2, 2]),
    "v3": dict(api_version=3, symbol_version=2, family=3, semantic_stride=2,
               implementation="cfm", vocoder="bigvgan-v2", requires_sv=False, output_sr=24000,
               reference_sr=32000, prompt_sr=24000, filter_length=1024, hop_length=256,
               window_length=1024, mel_channels=100, default_steps=32,
               feature_rate_scale=1.875, max_prompt_frames=468, upsample_rates=[4, 4, 2, 2, 2, 2]),
    "v4": dict(api_version=4, symbol_version=2, family=3, semantic_stride=2,
               implementation="cfm", vocoder="hifigan", requires_sv=False, output_sr=48000,
               reference_sr=32000, prompt_sr=32000, filter_length=1280, hop_length=320,
               window_length=1280, mel_channels=100, default_steps=8,
               feature_rate_scale=2.0, max_prompt_frames=500, upsample_rates=[10, 6, 2, 2, 2]),
}


def write_vits_artifact_contract(writer: GGUFWriter, version: str) -> None:
    contract = VITS_ARTIFACT_CONTRACTS[version]
    writer.add_string("tts.artifact.kind", "audio-decoder")
    writer.add_string("tts.artifact.implementation", contract["implementation"])
    writer.add_string("tts.vocoder.architecture", contract["vocoder"])
    writer.add_uint32("gpt_sovits.vits.api_version", contract["api_version"])
    writer.add_uint32("gpt_sovits.frontend.symbol_version", contract["symbol_version"])
    writer.add_uint32("gpt_sovits.t2s.expected_family", contract["family"])
    writer.add_uint32("gpt_sovits.vits.semantic_frame_stride", contract["semantic_stride"])
    writer.add_bool("gpt_sovits.vits.requires_speaker_embedding", contract["requires_sv"])
    writer.add_uint32("tts.audio.output_sample_rate", contract["output_sr"])
    writer.add_uint32("tts.audio.reference_sample_rate", contract["reference_sr"])
    writer.add_uint32("tts.audio.prompt_sample_rate", contract["prompt_sr"])
    writer.add_uint32("tts.audio.filter_length", contract["filter_length"])
    writer.add_uint32("tts.audio.hop_length", contract["hop_length"])
    writer.add_uint32("tts.audio.window_length", contract["window_length"])
    writer.add_uint32("tts.audio.mel_channels", contract["mel_channels"])
    writer.add_uint32("tts.inference.default_steps", contract["default_steps"])
    writer.add_float32("tts.acoustic.feature_rate_scale", contract["feature_rate_scale"])
    writer.add_uint32("tts.acoustic.max_prompt_frames", contract["max_prompt_frames"])
    writer.add_array("tts.vocoder.upsample_rates", contract["upsample_rates"])


def validate_vits_gguf(path: str, version: str) -> None:
    required_metadata = {
        "gpt_sovits.version",
        "tts.artifact.kind",
        "tts.artifact.implementation",
        "tts.vocoder.architecture",
        "gpt_sovits.frontend.symbol_version",
        "gpt_sovits.t2s.expected_family",
        "tts.audio.output_sample_rate",
        "tts.acoustic.feature_rate_scale",
        "tts.acoustic.max_prompt_frames",
        "tts.vocoder.upsample_rates",
    }
    required_tensors = {
        "enc_p.text_embedding.weight",
        "ref_enc.spectral.0.fc.weight",
        "dec.conv_pre.weight",
        "dec.conv_post.weight",
    }
    if version in ("v3", "v4"):
        required_tensors.add("cfm.estimator.input_embed.proj.weight")
    validate_artifact(path, ArtifactContract.create(required_metadata, required_tensors))
    print(f"  Artifact contract validated: {path}")


# ============================================================================
# Helpers
# ============================================================================

def extract_state_dict(checkpoint: Any, path: str) -> Dict[str, torch.Tensor]:
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


class _CheckpointHParams:
    """Minimal stand-in for legacy checkpoints pickled with utils.HParams."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = _CheckpointHParams(**value)
            setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def load_torch_checkpoint(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as error:
        if error.name != "utils" or "utils" in sys.modules:
            raise

    legacy_utils = types.ModuleType("utils")
    legacy_utils.HParams = _CheckpointHParams
    previous_module = sys.modules.get("utils")
    sys.modules["utils"] = legacy_utils
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        if previous_module is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_module


def load_checkpoint_with_metadata(path: str) -> Tuple[Dict[str, torch.Tensor], Any]:
    print(f"  Loading checkpoint: {path}")
    checkpoint = load_torch_checkpoint(path)
    return extract_state_dict(checkpoint, path), checkpoint


def load_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    state_dict, _ = load_checkpoint_with_metadata(path)
    return state_dict


def checkpoint_config_value(checkpoint: Any, *keys: str) -> Any:
    value = checkpoint
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            value = getattr(value, key, None)
        if value is None:
            return None
    return value


def resolve_checkpoint_int(
    checkpoint: Any,
    config_path: Tuple[str, ...],
    override: Optional[int],
    label: str,
) -> int:
    detected = checkpoint_config_value(checkpoint, *config_path)
    if detected is not None:
        detected = int(detected)
        if detected <= 0:
            raise ValueError(f"Checkpoint {label} must be positive, got {detected}")
        if override is not None and override != detected:
            raise ValueError(
                f"--{label}={override} conflicts with checkpoint {'.'.join(config_path)}={detected}"
            )
        return detected
    if override is None:
        raise ValueError(
            f"Checkpoint does not declare {'.'.join(config_path)}; pass --{label} explicitly"
        )
    if override <= 0:
        raise ValueError(f"--{label} must be positive")
    return override


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


def fold_weight_norm_parameters(state_dict: Dict[str, torch.Tensor]) -> int:
    """Replace PyTorch weight-norm g/v pairs with canonical weight tensors."""
    folded = 0
    for v_key in [key for key in state_dict if key.endswith(".weight_v")]:
        base_name = v_key[:-len(".weight_v")]
        g_key = base_name + ".weight_g"
        if g_key not in state_dict:
            continue

        v = state_dict[v_key]
        g = state_dict[g_key]
        if g.ndim != v.ndim:
            matching_dims = [index for index, size in enumerate(v.shape) if size == g.numel()]
            if len(matching_dims) != 1:
                raise ValueError(
                    f"Cannot align WeightNorm scale for {base_name}: "
                    f"g={tuple(g.shape)}, v={tuple(v.shape)}"
                )
            aligned_shape = [1] * v.ndim
            aligned_shape[matching_dims[0]] = g.numel()
            g = g.reshape(aligned_shape)
        varying_dims = [index for index, size in enumerate(g.shape) if size != 1]
        if len(varying_dims) > 1:
            raise ValueError(
                f"Cannot infer WeightNorm dimension for {base_name}: "
                f"g={tuple(g.shape)}, v={tuple(v.shape)}"
            )
        norm_dim = varying_dims[0] if varying_dims else 0
        if g.shape[norm_dim] not in (1, v.shape[norm_dim]):
            raise ValueError(
                f"WeightNorm shape mismatch for {base_name}: "
                f"g={tuple(g.shape)}, v={tuple(v.shape)}"
            )

        reduce_dims = tuple(index for index in range(v.ndim) if index != norm_dim)
        v_float = v.float()
        norm = torch.linalg.vector_norm(v_float, ord=2, dim=reduce_dims, keepdim=True)
        state_dict[base_name + ".weight"] = g.float() * (v_float / (norm + 1e-12))
        del state_dict[v_key]
        del state_dict[g_key]
        folded += 1
    return folded


def exponentiate_snake_parameters(state_dict: Dict[str, torch.Tensor]) -> int:
    """Materialize BigVGAN SnakeBeta parameters from checkpoint log-space."""
    converted = 0
    for name, value in list(state_dict.items()):
        if name.endswith(".act.alpha") or name.endswith(".act.beta"):
            state_dict[name] = torch.exp(value.float())
            converted += 1
    return converted


def materialize_alias_free_filters(state_dict: Dict[str, torch.Tensor]) -> int:
    """Expand BigVGAN anti-alias filters to their final per-channel tensors."""
    suffixes = (
        (".upsample.filter", ".upsample.filter_repeated"),
        (".downsample.lowpass.filter", ".downsample.filter_repeated"),
    )
    converted = 0
    for source_name in list(state_dict):
        matched = next((pair for pair in suffixes if source_name.endswith(pair[0])), None)
        if matched is None:
            continue
        prefix = source_name[:-len(matched[0])]
        alpha_name = prefix + ".act.alpha"
        if alpha_name not in state_dict:
            raise ValueError(f"Missing channel parameter {alpha_name} for alias-free filter")
        channels = state_dict[alpha_name].numel()
        kernel = state_dict[source_name].detach().float().reshape(-1)
        state_dict[prefix + matched[1]] = kernel.reshape(1, 1, -1).repeat(channels, 1, 1)
        del state_dict[source_name]
        converted += 1
    return converted


# ============================================================================
# Topology Inspection
# ============================================================================

def inspect_t2s_topology(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
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
    if max_layer < 0:
        raise ValueError("T2S checkpoint contains no transformer layers")
    n_layers = max_layer + 1

    # 2. Detect hidden dimension
    hidden_dim = None
    for key in ["ar_text_embedding.word_embeddings.weight", "model.ar_text_embedding.word_embeddings.weight"]:
        if key in sd:
            hidden_dim = sd[key].shape[1]
            break

    if hidden_dim is None:
        raise ValueError("T2S checkpoint contains no text embedding tensor")
    return n_layers, hidden_dim


def inspect_vits_hidden_dim(sd: Dict[str, torch.Tensor]) -> int:
    hidden_dim = None
    for k in sd.keys():
        if "enc_p.proj.weight" in k:
            hidden_dim = sd[k].shape[1] // 2  # PyTorch shape is [384, 192]
            break

    if hidden_dim is None:
        raise ValueError("VITS checkpoint contains no enc_p.proj.weight tensor")
    return hidden_dim


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

    # Supported split-projection checkpoint layout
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
    sd, checkpoint = load_checkpoint_with_metadata(src_path)

    # Detect parameters
    det_layers, det_dim = inspect_t2s_topology(sd)
    if opt_version is None:
        raise ValueError("T2S conversion requires an explicit --version; checkpoint weights do not identify the paired model family")
    version = normalize_model_version(opt_version)
    n_layers = resolve_checkpoint_int(checkpoint, ("config", "model", "n_layer"), opt_layers, "n-layers")
    n_heads = resolve_checkpoint_int(checkpoint, ("config", "model", "head"), opt_heads, "n-heads")
    hidden_dim = det_dim
    config_hidden_dim = checkpoint_config_value(checkpoint, "config", "model", "hidden_dim")
    if n_layers != det_layers:
        raise ValueError(f"Checkpoint declares {n_layers} T2S layers but tensors contain {det_layers}")
    if config_hidden_dim is not None and int(config_hidden_dim) != hidden_dim:
        raise ValueError(
            f"Checkpoint declares hidden_dim={config_hidden_dim} but embedding tensor uses {hidden_dim}"
        )
    if hidden_dim % n_heads != 0:
        raise ValueError(f"T2S hidden_dim={hidden_dim} is not divisible by n_heads={n_heads}")

    print(f"  Configuration Detected/Selected:")
    print(f"    Version:           {version}")
    print(f"    Layers (n_layers): {n_layers}")
    print(f"    Heads (n_heads):   {n_heads}")
    print(f"    Hidden Dimension:  {hidden_dim}")

    writer = ModelArtifact(dst_path, "gpt_sovits_t2s")
    writer.add_string("general.name", f"GPT-SoVITS T2S ({version})")
    writer.add_string("gpt_sovits.version", version)
    writer.add_uint32("gpt_sovits.t2s.family", t2s_family_for_version(version))
    writer.add_uint32("attention.head_count", n_heads)
    writer.add_uint32("gpt_sovits.t2s.n_layers", n_layers)
    writer.add_uint32("gpt_sovits.t2s.hidden_dim", hidden_dim)

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
                writer.parameter(name, arr, raw_dtype=GGMLQuantizationType.F16)
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
                writer.parameter(name, arr, raw_dtype=GGMLQuantizationType.F32)
                mapped_count += 1
            continue

        # Alpha scalar (f32)
        if gguf_name.endswith(".alpha"):
            if t.ndim == 0:
                t = t.unsqueeze(0)
            arr = tensor_to_fp32(t)
            writer.parameter(gguf_name, arr, raw_dtype=GGMLQuantizationType.F32)
        # Scalar/vector parameters stay F32; matrix weights are at least F16.
        elif t.ndim <= 1 or gguf_name.lower().endswith(".bias"):
            arr = tensor_to_fp32(t)
            writer.parameter(gguf_name, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            arr = tensor_to_fp16(t)
            writer.parameter(gguf_name, arr, raw_dtype=GGMLQuantizationType.F16)
        mapped_count += 1

    print(f"  Writing GGUF output with {mapped_count} tensors...")
    writer.write()
    print(f"  ✓ Saved T2S GGUF to: {dst_path}")


def convert_vits(src_dir_or_files: str, dst_path: str, opt_version: Optional[str] = None,
                 opt_heads: Optional[int] = None) -> None:
    print(f"\nConverting VITS (s2D + s2G merged)...")
    
    # Locate s2D and s2G checkpoints
    if opt_version is None:
        raise ValueError("VITS conversion requires an explicit --version")
    version = normalize_model_version(opt_version)
    parts = src_dir_or_files.split(",")
    vocoder_path = None
    if len(parts) == 2 and version in ("v1", "v2", "v2Pro", "v2ProPlus"):
        s2d_path, s2g_path = parts[0].strip(), parts[1].strip()
    elif len(parts) == 2 and version in ("v3", "v4"):
        s2d_path = None
        s2g_path, vocoder_path = parts[0].strip(), parts[1].strip()
    else:
        raise ValueError("Pass 's2D.pth,s2G.pth' for classic VITS or 's2G.pth,vocoder' for V3/V4")

    sd_d = load_checkpoint(s2d_path) if s2d_path else {}
    sd_g, s2g_checkpoint = load_checkpoint_with_metadata(s2g_path)
    
    # Merge checkpoints (s2G generator overrides s2D if duplicates exist)
    merged_sd = {**sd_d, **sd_g}
    if vocoder_path:
        vocoder_sd = load_checkpoint(vocoder_path)
        merged_sd.update({f"dec.{key}": value for key, value in vocoder_sd.items()})
        print(f"  Merged vocoder ({len(vocoder_sd)} keys) from {vocoder_path}.")
    print(f"  Merged s2D ({len(sd_d)} keys) + s2G ({len(sd_g)} keys) -> Total {len(merged_sd)} keys.")

    n_heads = resolve_checkpoint_int(
        s2g_checkpoint, ("config", "model", "n_heads"), opt_heads, "n-heads"
    )
    hidden_dim = inspect_vits_hidden_dim(merged_sd)

    print(f"  Configuration Detected/Selected:")
    print(f"    Version:           {version}")
    print(f"    Encoder Heads:     {n_heads}")
    print(f"    Hidden Dimension:  {hidden_dim}")

    # Fold Weight Normalization parameters (.weight_g & .weight_v -> .weight)
    print("  Folding Weight Normalization layers...")
    folded_count = fold_weight_norm_parameters(merged_sd)
    print(f"    Folded {folded_count} WeightNorm tensors.")
    snake_count = exponentiate_snake_parameters(merged_sd) if version == "v3" else 0
    if snake_count:
        print(f"    Materialized {snake_count} SnakeBeta tensors from log-space.")
    filter_count = materialize_alias_free_filters(merged_sd) if version == "v3" else 0
    if filter_count:
        print(f"    Materialized {filter_count} alias-free filters.")

    mapped_names = {name for key in merged_sd if (name := map_vits_key(key)) is not None}
    required_names = {
        "enc_p.text_embedding.weight",
        "ref_enc.spectral.0.fc.weight",
        "dec.conv_pre.weight",
        "dec.conv_post.weight",
    }
    if version in ("v3", "v4"):
        required_names.add("cfm.estimator.input_embed.proj.weight")
    missing = sorted(required_names - mapped_names)
    for index in range(len(VITS_ARTIFACT_CONTRACTS[version]["upsample_rates"])):
        if (f"dec.ups.{index}.weight" not in mapped_names and
                f"dec.ups.{index}.0.weight" not in mapped_names):
            missing.append(f"dec.ups.{index}[.0].weight")
    if missing:
        raise ValueError(f"VITS source artifacts are incomplete; missing tensors: {', '.join(missing)}")

    writer = ModelArtifact(dst_path, "gpt_sovits_vits")
    writer.add_string("general.name", f"GPT-SoVITS VITS ({version})")
    writer.add_string("gpt_sovits.version", version)
    writer.add_uint32("gpt_sovits.vits.n_heads", n_heads)
    writer.add_uint32("gpt_sovits.vits.hidden_dim", hidden_dim)
    write_vits_artifact_contract(writer, version)
    
    # Crucial Metadata: Tells C++ that Conv1D weights are stored in GGML format directly [k, ic, oc]
    # No permutation inside python is required because PyTorch [oc, ic, k] is naturally reversed to [k, ic, oc] by GGUFWriter.
    writer.add_bool("gpt_sovits.vits.pre_transposed", True)

    mapped_count = 0
    for key in sorted(merged_sd.keys()):
        gguf_name = map_vits_key(key)
        if gguf_name is None:
            continue

        t = merged_sd[key]
        is_f32 = (gguf_name.lower().endswith(".bias") or t.ndim <= 1)

        if is_f32:
            arr = tensor_to_fp32(t)
            writer.parameter(gguf_name, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            arr = tensor_to_fp16(t)
            writer.parameter(gguf_name, arr, raw_dtype=GGMLQuantizationType.F16)
        mapped_count += 1

    print(f"  Writing GGUF output with {mapped_count} tensors...")
    writer.write()
    validate_vits_gguf(dst_path, version)
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

    writer = ModelArtifact(dst_path, "gpt_sovits_bert")
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
            writer.parameter(key, arr, raw_dtype=GGMLQuantizationType.F32)
        elif t.ndim <= 1 or key.lower().endswith(".bias"):
            arr = tensor_to_fp32(t)
            writer.parameter(key, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            arr = tensor_to_fp16(t)
            writer.parameter(key, arr, raw_dtype=GGMLQuantizationType.F16)
        mapped_count += 1

    print(f"  Writing GGUF output with {mapped_count} tensors...")
    writer.write()
    print(f"  ✓ Saved BERT GGUF to: {dst_path}")


def convert_hubert(src_path: str, dst_path: str, opt_version: Optional[str] = None) -> None:
    print(f"\nConverting Chinese HuBERT Speech Extractor...")
    sd = load_checkpoint(src_path)
    folded_count = fold_weight_norm_parameters(sd)
    version = opt_version or "v2"

    print(f"  Configuration Detected/Selected:")
    print(f"    Version:     {version}")
    print(f"    Folded WeightNorm tensors: {folded_count}")

    writer = ModelArtifact(dst_path, "gpt_sovits_hubert")
    writer.add_string("general.name", f"GPT-SoVITS HuBERT ({version})")
    writer.add_string("gpt_sovits.version", version)

    mapped_count = 0
    for key in sorted(sd.keys()):
        t = sd[key]
        if t.ndim <= 1 or key.lower().endswith(".bias"):
            arr = tensor_to_fp32(t)
            writer.parameter(key, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            arr = tensor_to_fp16(t)
            writer.parameter(key, arr, raw_dtype=GGMLQuantizationType.F16)
        mapped_count += 1

    print(f"  Writing GGUF output with {mapped_count} tensors...")
    writer.write()
    print(f"  ✓ Saved HuBERT GGUF to: {dst_path}")


# ============================================================================
# Quantization Pipeline
# ============================================================================

SENSITIVE_WEIGHT_PATTERNS = (
    re.compile(r"(^|[._])(emb|embed|embedding|embeddings)([._]|$)"),
    re.compile(r"(^|[._])(predict|prediction|lm_head)([._]|$)"),
    re.compile(r"(^|[._])pos_conv([._]|$)"),
)


def keep_f32_parameter(name: str, shape: Tuple[int, ...]) -> bool:
    """Keep only small, numerically sensitive parameters in F32."""
    leaf = name.lower().rsplit(".", 1)[-1]
    return len(shape) <= 1 or leaf in {
        "bias", "alpha", "beta", "gamma", "scale", "position_ids", "masked_spec_embed",
    }


def is_sensitive_weight(name: str) -> bool:
    """Weights that benefit from Q8 in an otherwise Q4 artifact."""
    name_lower = name.lower()
    return any(pattern.search(name_lower) for pattern in SENSITIVE_WEIGHT_PATTERNS)


def should_pack_conv_weight(architecture: str, name: str, shape: Tuple[int, ...]) -> bool:
    """Use channels as the quantization row for 1D convolution weights."""
    if not name.lower().endswith(".weight") or len(shape) != 3 or shape[1] % 32 != 0:
        return False
    if architecture == "gpt_sovits_vits":
        return True
    if architecture == "gpt_sovits_hubert":
        return name.startswith("feature_extractor.conv_layers.") and ".conv.weight" in name
    return False


def vits_convolution_role(name: str) -> str:
    if re.match(r"^dec\.ups\.\d+(?:\.0)?\.weight$", name):
        return "conv_transpose1d_weight"
    return "conv1d_weight"


def describe_gpt_sovits_tensor(architecture: str, name: str, shape: Tuple[int, ...]) -> TensorSpec:
    if keep_f32_parameter(name, shape):
        return TensorSpec(name=name, shape=shape, role="parameter", allowed_types=("F32",))

    pack_conv = should_pack_conv_weight(architecture, name, shape)
    is_embedding = SENSITIVE_WEIGHT_PATTERNS[0].search(name.lower()) is not None
    keep_floating = architecture == "gpt_sovits_vits" and (
        name == "ssl_proj.weight" or name.endswith("filter_repeated")
    )
    keep_floating = keep_floating or (
        architecture == "gpt_sovits_hubert" and name == "encoder.pos_conv_embed.conv.weight"
    )
    allowed_types = ("F16", "F32") if keep_floating else (
        "Q4_K", "Q4_0", "Q8_0", "F16", "F32"
    )
    return TensorSpec(
        name=name,
        shape=shape,
        role=vits_convolution_role(name) if pack_conv else ("embedding" if is_embedding else "weight"),
        sensitivity="high" if is_sensitive_weight(name) else "normal",
        allowed_types=allowed_types,
        quant_shape=(shape[1], shape[0], shape[2]) if pack_conv else None,
        transform="conv1d_channel_rows" if pack_conv else None,
    )


def prepare_gpt_sovits_tensor(architecture: str, spec: TensorSpec, data: np.ndarray) -> np.ndarray:
    del architecture
    array = data.astype(np.float32).reshape(spec.shape[::-1])
    return np.swapaxes(array, -1, -2) if spec.transform == "conv1d_channel_rows" else array


def finalize_gpt_sovits_metadata(writer: GGUFWriter, report: QuantizationReport) -> None:
    packed = report.transformed.get("conv1d_channel_rows", [])
    write_layout_metadata(writer, {name: Layout.permuted((1, 0, 2)) for name in packed})


def quantize_gguf(input_path: str, output_path: str, target_type: str, policy_path: Optional[str] = None) -> None:
    policy = QuantizationPolicy.from_file(policy_path, target_type) if policy_path else None
    run_quantization(
        input_path,
        output_path,
        target_type,
        describe_tensor=describe_gpt_sovits_tensor,
        transform_tensor=prepare_gpt_sovits_tensor,
        copy_metadata=lambda name: not name.startswith(("GGUF.", "nn.storage_layout.")),
        finalize_metadata=finalize_gpt_sovits_metadata,
        policy=policy,
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified GPT-SoVITS Conversion & Quantization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a V3 T2S checkpoint (version is an explicit model-package contract)
  python process.py --model-type t2s --version v3 --src s1v3.ckpt --output t2s_v3_fp16.gguf

  # Convert classic VITS from an explicit s2D/s2G pair (defaults to FP16)
  python process.py --model-type vits --version v2 --src s2D.pth,s2G.pth --output vits_v2_fp16.gguf

  # Convert V3 from explicit s2G and BigVGAN artifacts
  python process.py --model-type vits --version v3 --src s2Gv3.pth,bigvgan_generator.pt --output vits_v3_fp16.gguf

  # Convert and quantize BERT model to Q4_0 (highly recommended)
  python process.py --model-type bert --src pytorch_model.bin --output bert_q4_0.gguf --quantize Q4_0
"""
    )
    
    parser.add_argument("--model-type", required=True, choices=["t2s", "vits", "bert", "hubert"],
                        help="The type of model to process.")
    parser.add_argument("--src", required=True,
                        help="Explicit source artifacts: classic VITS uses 's2D.pth,s2G.pth'; V3/V4 use 's2G.pth,vocoder'.")
    parser.add_argument("--output", required=True,
                        help="Path to save the final GGUF file.")
    parser.add_argument("--quantize", choices=["F16", "Q4_0", "Q4_K", "Q4_K_M", "Q8_0"], default="F16",
                        help="Target precision/quantization (default: F16).")
    parser.add_argument("--quant-policy", default=None,
                        help="Optional machine-independent JSON precision policy.")
    parser.add_argument("--version", choices=["v1", "v2", "v2Pro", "v2ProPlus", "v3", "v4"], default=None,
                        help="Exact model version. Required for T2S conversion.")
    parser.add_argument("--n-heads", type=int, default=None,
                        help="Attention head override; normally read from checkpoint config.")
    parser.add_argument("--n-layers", type=int, default=None,
                        help="Explicit T2S layer count; by default it is read from checkpoint tensors.")
    
    args = parser.parse_args()

    if args.model_type in ("t2s", "vits") and args.version is None:
        parser.error("--version is required for T2S and VITS conversion")

    # Define intermediate path if quantization is required
    needs_quantize_step = args.quantize != "F16" or args.quant_policy is not None
    
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
            quantize_gguf(temp_fp16_path, args.output, args.quantize, args.quant_policy)
            if args.model_type == "vits":
                validate_vits_gguf(args.output, args.version)
            
    finally:
        # Cleanup temporary file if it was created
        if needs_quantize_step and os.path.exists(temp_fp16_path):
            os.remove(temp_fp16_path)

    print(f"\n✓ Finished processing. Output saved to: {args.output}\n")


if __name__ == "__main__":
    main()
