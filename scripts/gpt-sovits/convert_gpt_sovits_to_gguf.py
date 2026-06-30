#!/usr/bin/env python3
"""
Convert GPT-SoVITS pretrained models to GGUF format (FP16 or Quantized).

This is the main entry script that parses options and delegates to individual converters,
and performs quantization inline for a single-step PyTorch -> Quantized GGUF workflow.
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure standard output and error use UTF-8 to avoid encoding errors on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add the script's directory to the python path so it can import the sub-scripts
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

# Import converters from sub-scripts
from convert_t2s import convert_t2s
from convert_vits import convert_vits
from convert_bert import convert_bert
from convert_hubert import convert_hubert
from quantize_gpt_sovits import process_gguf

def run_conversion(convert_fn, src, dst, qtype, dry_run=False, **kwargs):
    """Wrapper that converts to FP16 first, then quantizes inline if requested."""
    if dry_run or qtype == "F16":
        convert_fn(src, dst, dry_run=dry_run, **kwargs)
    else:
        tmp_dst = str(dst) + ".tmp"
        try:
            # 1. Convert to FP16 temp file
            convert_fn(src, tmp_dst, dry_run=False, **kwargs)
            # 2. Quantize to final destination
            print(f"\n--- Quantizing to {qtype} ---")
            success = process_gguf(tmp_dst, str(dst), qtype)
            if not success:
                print(f"ERROR: Quantization failed for {dst}")
                sys.exit(1)
        finally:
            if os.path.exists(tmp_dst):
                try:
                    os.remove(tmp_dst)
                except Exception as e:
                    print(f"Warning: Failed to clean up temp file {tmp_dst}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert GPT-SoVITS pretrained models to GGUF (FP16 or Quantized)")
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
        "-q", "--quantization", choices=["F16", "Q4_0", "Q4_1", "Q4_K", "Q5_0", "Q5_K", "Q6_K", "Q8_0"], default="F16",
        help="Target quantization type (default: F16, i.e. no quantization)")
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

    qtype = args.quantization.upper()

    # Auto-adjust output filenames if they are default and quantization is enabled
    t2s_out = args.t2s_out
    vits_out = args.vits_out
    bert_out = args.bert_out
    hubert_out = args.hubert_out

    if qtype != "F16":
        if t2s_out == "t2s_fp16.gguf":
            t2s_out = f"t2s_{qtype.lower()}.gguf"
        if vits_out == "vits_fp16.gguf":
            vits_out = f"vits_{qtype.lower()}.gguf"
        if bert_out == "bert_fp16.gguf":
            bert_out = f"bert_{qtype.lower()}.gguf"
        if hubert_out == "cnhubert_fp16.gguf":
            hubert_out = f"cnhubert_{qtype.lower()}.gguf"

    src_dir = Path(args.src_dir)
    # Check source dir only if we are converting T2S or VITS
    run_all = not (args.t2s_only or args.vits_only or args.bert_only or args.hubert_only)
    if run_all or args.t2s_only or args.vits_only:
        if not src_dir.exists():
            print(f"ERROR: Source directory does not exist: {src_dir}")
            print("Use --src-dir to specify the correct path.")
            sys.exit(1)

    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    t2s_dst_dir = dst_dir / "t2s"
    t2s_dst_dir.mkdir(parents=True, exist_ok=True)
    t2s_dst = t2s_dst_dir / t2s_out

    vits_dst_dir = dst_dir / "vits"
    vits_dst_dir.mkdir(parents=True, exist_ok=True)
    vits_dst = vits_dst_dir / vits_out

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
        run_conversion(convert_t2s, str(s1_path), t2s_dst, qtype, dry_run=args.dry_run, is_v2=not args.v1)

    # ── VITS ──
    if run_all or args.vits_only:
        run_conversion(convert_vits, str(src_dir), vits_dst, qtype, dry_run=args.dry_run, is_v2=not args.v1)

    # ── BERT ──
    if run_all or args.bert_only:
        bert_src = Path(args.bert_src)
        if not bert_src.exists():
            print(f"ERROR: BERT source not found: {bert_src}")
            sys.exit(1)
        bert_dst_dir = dst_dir / "bert"
        bert_dst_dir.mkdir(parents=True, exist_ok=True)
        bert_dst = bert_dst_dir / bert_out
        run_conversion(convert_bert, str(bert_src), bert_dst, qtype, dry_run=args.dry_run)

    # ── HuBERT ──
    if run_all or args.hubert_only:
        hubert_src = Path(args.hubert_src)
        if not hubert_src.exists():
            print(f"ERROR: HuBERT source not found: {hubert_src}")
            sys.exit(1)
        hubert_dst_dir = dst_dir / "cnhubert"
        hubert_dst_dir.mkdir(parents=True, exist_ok=True)
        hubert_dst = hubert_dst_dir / hubert_out
        run_conversion(convert_hubert, str(hubert_src), hubert_dst, qtype, dry_run=args.dry_run)

    print("\nAll conversions completed successfully!")

if __name__ == "__main__":
    main()
