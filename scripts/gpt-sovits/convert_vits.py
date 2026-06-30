import sys
import torch
from pathlib import Path
from typing import Optional

# Add path for utils and gguf-py
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(1, str(_SCRIPT_DIR / ".." / ".." / "gguf-py"))

from gguf import GGUFWriter
from gguf_utils import load_checkpoint, add_vits_tensor

def map_vits_key(pt_key: str) -> Optional[str]:
    """Map a PyTorch VITS (s2D/s2G) state_dict key to the GGUF tensor name."""
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

def convert_vits(src_dir: str, dst_path: str, dry_run: bool = False, is_v2: bool = True) -> None:
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
    # For VITS, check encoder / decoders
    # We find all keys containing 'weight_g' or 'weight_v'
    g_keys = [k for k in merged_sd.keys() if k.endswith(".weight_g")]
    for g_key in g_keys:
        prefix = g_key[:-9] # strip '.weight_g'
        v_key = prefix + ".weight_v"
        w_key = prefix + ".weight"
        if v_key in merged_sd:
            print(f"    Folding WeightNorm for: {w_key}")
            g = merged_sd[g_key] # [out_channels, 1, 1] or similar
            v = merged_sd[v_key] # [out_channels, in_channels, ...]
            
            # Compute folded weight: w = g * (v / ||v||)
            # Find dimensions to normalize over: all except dim 0
            dims = list(range(1, v.ndim))
            norm = torch.linalg.vector_norm(v, ord=2, dim=dims, keepdim=True)
            w = g * (v / (norm + 1e-12))
            
            # Save folded weight, delete parameter weights
            merged_sd[w_key] = w
            del merged_sd[g_key]
            del merged_sd[v_key]

    # Pre-calculate attention heads (192 channels, head_dim=96 -> 2 heads)
    n_heads_vits = 2

    if dry_run:
        print(f"\n  [DRY RUN] Inspecting keys...")
        mapped = 0
        unmapped = []
        for key in sorted(merged_sd.keys()):
            gguf_name = map_vits_key(key)
            if gguf_name is None:
                unmapped.append(key)
            else:
                t = merged_sd[key]
                print(f"    {key}  →  {gguf_name}  shape={list(t.shape)}  dtype={t.dtype}")
                mapped += 1
        print(f"\n  Mapped: {mapped} keys, Unmapped: {len(unmapped)} keys")
        if unmapped:
            print("  Skipped keys:")
            for key in unmapped:
                t = merged_sd[key]
                print(f"    ? {key}  shape={list(t.shape)}  dtype={t.dtype}")
        return

    gguf_writer = GGUFWriter(dst_path, "gpt_sovits_vits")

    # Metadata
    gguf_writer.add_string("general.name", "GPT-SoVITS v2 VITS" if is_v2 else "GPT-SoVITS v1 VITS")
    gguf_writer.add_string("general.description",
                           "GPT-SoVITS v2 final VITS vocoder (s2D + s2G merged)" if is_v2 else "GPT-SoVITS v1 VITS vocoder (s2D + s2G merged)")
    gguf_writer.add_string("gpt_sovits.version", "v2" if is_v2 else "v1")
    gguf_writer.add_uint32("attention.head_count", n_heads_vits)

    mapped_count = 0
    unmapped_keys = []
    for key in sorted(merged_sd.keys()):
        gguf_name = map_vits_key(key)
        if gguf_name is None:
            unmapped_keys.append(key)
            continue
        t = merged_sd[key]
        pre_transpose = False
        add_vits_tensor(gguf_writer, gguf_name, t, pre_transpose, dry_run=False)
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
