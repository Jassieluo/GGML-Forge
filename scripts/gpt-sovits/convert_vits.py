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
        "cfm.", "wns1.", "linear_mel.", "bridge.",
    )
    if any(clean.startswith(p) for p in known_prefixes):
        return clean

    return None

def convert_vits(src_dir: str, dst_path: str, dry_run: bool = False, version: str = "v2") -> None:
    """Convert the s2 checkpoints to a single GGUF file."""
    print(f"\n{'=' * 70}")
    print(f"Converting VITS/Synthesis (s2) model")
    print(f"  Source dir: {src_dir}")
    print(f"  Output:     {dst_path}")
    print(f"  Version:    {version}")
    print(f"{'=' * 70}")

    src_dir_p = Path(src_dir)
    s2d_files = sorted(src_dir_p.glob("s2D*.pth"))
    s2g_files = sorted(src_dir_p.glob("s2G*.pth"))

    # Filter based on version
    if version == "v1":
        s2g_files = [f for f in s2g_files if "488k" in f.name]
        s2d_files = [f for f in s2d_files if "488k" in f.name]
    elif version == "v2":
        s2g_files = [f for f in s2g_files if "2333k" in f.name]
        s2d_files = [f for f in s2d_files if "2333k" in f.name]
    elif version == "v3":
        s2g_files = [f for f in s2g_files if "v3" in f.name.lower()]
        s2d_files = [f for f in s2d_files if "v3" in f.name.lower()]
    elif version == "v4":
        s2g_files = [f for f in s2g_files if "v4" in f.name.lower()]
        s2d_files = [f for f in s2d_files if "v4" in f.name.lower()]

    if not s2g_files:
        # Fallback to general list if no version-specific files are found
        s2g_files = sorted(src_dir_p.glob("s2G*.pth"))
        s2d_files = sorted(src_dir_p.glob("s2D*.pth"))

    if not s2g_files:
        raise FileNotFoundError(f"No s2G*.pth found in {src_dir}")

    # For v3/v4, ref_enc / quantizer / cfm are all packaged in s2G, no separate s2D is needed.
    if s2d_files and version in ("v1", "v2"):
        print(f"\n  s2D: {s2d_files[0].name}")
        sd_d = load_checkpoint(str(s2d_files[0]))
        print(f"  s2G: {s2g_files[0].name}")
        sd_g = load_checkpoint(str(s2g_files[0]))
        merged_sd = {**sd_d, **sd_g}
        print(f"\n  Merged: {len(merged_sd)} keys "
              f"({len(sd_d)} s2D + {len(sd_g)} s2G)")
    else:
        print(f"  s2G: {s2g_files[0].name}")
        merged_sd = load_checkpoint(str(s2g_files[0]))
        print(f"\n  Loaded: {len(merged_sd)} keys from s2G")

        # For v3/v4, load and merge vocoder weights as "dec." prefix
        vocoder_sd = None
        if version == "v3":
            # Look for bigvgan_generator.pt
            paths = [
                src_dir_p / "models--nvidia--bigvgan_v2_24khz_100band_256x" / "bigvgan_generator.pt",
                src_dir_p / "bigvgan_generator.pt",
                src_dir_p.parent / "models--nvidia--bigvgan_v2_24khz_100band_256x" / "bigvgan_generator.pt",
            ]
            for p in paths:
                if p.exists():
                    print(f"  Loading Vocoder (BigVGAN): {p}")
                    sd_raw = torch.load(str(p), map_location="cpu")
                    vocoder_sd = sd_raw.get("generator", sd_raw)
                    break
        elif version == "v4":
            # Look for vocoder.pth
            paths = [
                src_dir_p / "vocoder.pth",
                src_dir_p / "gsv-v4-pretrained" / "vocoder.pth",
                src_dir_p.parent / "gsv-v4-pretrained" / "vocoder.pth",
            ]
            for p in paths:
                if p.exists():
                    print(f"  Loading Vocoder (HiFi-GAN): {p}")
                    vocoder_sd = torch.load(str(p), map_location="cpu")
                    if isinstance(vocoder_sd, dict) and "generator" in vocoder_sd:
                        vocoder_sd = vocoder_sd["generator"]
                    break

        if vocoder_sd:
            # Merge with "dec." prefix
            merged_vocoder = {f"dec.{k}": v for k, v in vocoder_sd.items()}
            merged_sd = {**merged_sd, **merged_vocoder}
            print(f"  Merged {len(merged_vocoder)} vocoder keys as 'dec.'")
        else:
            print("  Warning: Vocoder weights not found, converted GGUF will NOT contain generator!")

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
    gguf_writer.add_string("general.name", f"GPT-SoVITS {version} Synthesis")
    gguf_writer.add_string("general.description", f"GPT-SoVITS {version} synthesis model")
    gguf_writer.add_string("gpt_sovits.version", version)
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
    print(f"  ✓ Synthesis GGUF saved to: {dst_path}")
