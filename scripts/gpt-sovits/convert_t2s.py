import re
import sys
from pathlib import Path
from typing import Optional

# Add path for utils and gguf-py
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(1, str(_SCRIPT_DIR / ".." / ".." / "gguf-py"))

from gguf import GGUFWriter, GGMLQuantizationType
from gguf_utils import load_checkpoint, add_tensor_fp16, tensor_to_fp32, tensor_to_fp16

def map_t2s_key(pt_key: str) -> Optional[str]:
    """Map a PyTorch T2S (s1) state_dict key to GGUF tensor name."""
    clean = pt_key

    # Strip common prefixes
    for prefix in ("model.", "ar_model."):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]

    # ---- Known direct mappings ----
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

    # ---- Fused QKV projections ----
    if re.match(r"h\.layers\.(\d+)\.self_attn\.in_proj_weight", clean):
        return "__SPLIT_QKV__"
    if re.match(r"h\.layers\.(\d+)\.self_attn\.in_proj_bias", clean):
        return "__SPLIT_QKV_BIAS__"

    # ---- Transformer layers ----
    m = re.match(r"h\.layers\.(\d+)\.norm1\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.norm1.{m.group(2)}"

    m = re.match(r"h\.layers\.(\d+)\.norm2\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.norm2.{m.group(2)}"

    m = re.match(r"h\.layers\.(\d+)\.linear1\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.linear1.{m.group(2)}"

    m = re.match(r"h\.layers\.(\d+)\.linear2\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.linear2.{m.group(2)}"

    m = re.match(r"h\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.self_attn.out_proj.{m.group(2)}"

    # ---- Fallbacks ----
    m = re.match(r"h\.(\d+)\.self_attn\.(q|k|v)_proj\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.self_attn.{m.group(2)}.{m.group(3)}"

    m = re.match(r"h\.(\d+)\.self_attn\.out_proj\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.self_attn.out_proj.{m.group(2)}"

    m = re.match(r"h\.(\d+)\.attn\.(q|k|v)_proj\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.self_attn.{m.group(2)}.{m.group(3)}"

    m = re.match(r"h\.(\d+)\.attn\.out_proj\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.self_attn.out_proj.{m.group(2)}"

    m = re.match(r"h\.(\d+)\.norm1\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.norm1.{m.group(2)}"

    m = re.match(r"h\.(\d+)\.norm2\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.norm2.{m.group(2)}"

    m = re.match(r"h\.(\d+)\.ln_1\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.norm1.{m.group(2)}"

    m = re.match(r"h\.(\d+)\.ln_2\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.norm2.{m.group(2)}"

    m = re.match(r"h\.(\d+)\.linear1\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.linear1.{m.group(2)}"

    m = re.match(r"h\.(\d+)\.linear2\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.linear2.{m.group(2)}"

    m = re.match(r"h\.(\d+)\.mlp\.c_fc\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.linear1.{m.group(2)}"

    m = re.match(r"h\.(\d+)\.mlp\.c_proj\.(weight|bias)", clean)
    if m:
        return f"h.layers.{m.group(1)}.linear2.{m.group(2)}"

    return None

def convert_t2s(src_path: str, dst_path: str, is_v2: bool = True, dry_run: bool = False) -> None:
    """Convert the s1 T2S checkpoint to GGUF."""
    print(f"\n{'=' * 70}")
    print(f"Converting T2S (s1) model")
    print(f"  Source: {src_path}")
    print(f"  Output: {dst_path}")
    print(f"{'=' * 70}")

    sd = load_checkpoint(src_path)

    is_v2_detected = is_v2 or any(x in src_path.lower() for x in ["v2", "base", "s1bert", "369668", "firekeeper"])

    n_heads = 16 if is_v2_detected else 8
    for key in sd.keys():
        if "self_attn" in key and "weight" in key:
            t = sd[key]
            if t.ndim == 2:
                if "in_proj_weight" in key:
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
    gguf_writer.add_string("general.name", "GPT-SoVITS v2 T2S" if is_v2_detected else "GPT-SoVITS v1 T2S")
    gguf_writer.add_string("general.description",
                           "GPT-SoVITS v2 final Text-to-Semantic model" if is_v2_detected else "GPT-SoVITS v1 Text-to-Semantic model")
    gguf_writer.add_string("gpt_sovits.version", "v2" if is_v2_detected else "v1")
    gguf_writer.add_uint32("attention.head_count", n_heads)

    # Embed name map JSON in GGUF metadata
    import json
    name_map = {}
    name_map["word_embeddings.weight"] = "ar_text_embedding.word_embeddings.weight"
    name_map["audio_embeddings.weight"] = "ar_audio_embedding.word_embeddings.weight"
    name_map["bert_proj.weight"] = "bert_proj.weight"
    name_map["bert_proj.bias"] = "bert_proj.bias"
    name_map["predict.weight"] = "ar_predict_layer.weight"
    
    for i in range(24):
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
        
    gguf_writer.add_string("gpt_sovits.t2s.name_map", json.dumps(name_map))

    mapped_count = 0
    unmapped_keys = []
    for key in sorted(sd.keys()):
        gguf_name = map_t2s_key(key)
        if gguf_name is None:
            unmapped_keys.append(key)
            continue

        if gguf_name == "__SPLIT_QKV__":
            t = sd[key]
            m = re.match(r"(?:model\.)?h\.layers\.(\d+)\.self_attn\.in_proj_weight", key)
            if not m:
                print(f"    ⚠ Cannot parse layer number from: {key}")
                continue
            layer = m.group(1)
            hidden = t.shape[1]
            q_w, k_w, v_w = t[:hidden], t[hidden:2*hidden], t[2*hidden:3*hidden]
            for suffix, tw in [("q", q_w), ("k", k_w), ("v", v_w)]:
                name = f"h.layers.{layer}.self_attn.{suffix}.weight"
                add_tensor_fp16(gguf_writer, name, tw)
                mapped_count += 1
            continue

        if gguf_name == "__SPLIT_QKV_BIAS__":
            t = sd[key]
            m = re.match(r"(?:model\.)?h\.layers\.(\d+)\.self_attn\.in_proj_bias", key)
            if not m:
                print(f"    ⚠ Cannot parse layer number from: {key}")
                continue
            layer = m.group(1)
            hidden = t.shape[0] // 3
            q_b, k_b, v_b = t[:hidden], t[hidden:2*hidden], t[2*hidden:3*hidden]
            for suffix, tb in [("q", q_b), ("k", k_b), ("v", v_b)]:
                name = f"h.layers.{layer}.self_attn.{suffix}.bias"
                arr = tensor_to_fp32(tb)
                print(f"    + {name}: {list(tb.shape)} (f32, {arr.nbytes} bytes)")
                gguf_writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F32)
                mapped_count += 1
            continue

        t = sd[key]
        is_f32 = ("bias" in gguf_name.lower() or "norm" in gguf_name.lower() or gguf_name.endswith(".alpha"))
        if is_f32:
            arr = tensor_to_fp32(t)
            print(f"    + {gguf_name}: {list(t.shape)} (f32, {arr.nbytes} bytes)")
            gguf_writer.add_tensor(gguf_name, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            arr = tensor_to_fp16(t)
            print(f"    + {gguf_name}: {list(t.shape)} (fp16, {arr.nbytes} bytes)")
            gguf_writer.add_tensor(gguf_name, arr, raw_dtype=GGMLQuantizationType.F16)
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
        print(f"\n  ✓ All {mapped_count} tensors mapped successfully.")
    print(f"  ✓ T2S GGUF saved to: {dst_path}")
