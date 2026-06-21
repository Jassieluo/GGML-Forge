#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPT-SoVITS GGUF Quantization Script
Pure Python — uses gguf-py's reference quantization engine.
Supports F16, Q4_0, Q8_0.

Recommended:
  BERT  (large)  → Q4_0  (~75% size reduction, minimal quality loss)
  HuBERT (base)  → Q8_0  (~50% size reduction, near-lossless)
  T2S / VITS     → F16   (keep as-is, already FP16)
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(1, str(_SCRIPT_DIR / ".." / ".." / "gguf-py"))

from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType  # noqa: E402
from gguf.quants import quantize  # noqa: E402


# Tensors to skip quantization (keep as-is for precision)
SKIP_PATTERNS = [
    "emb",       # embeddings
    "bias",      # biases
    "norm",      # layer norms
    "alpha",     # positional alpha scalars
    "position_ids",
    "masked_spec_embed",
]


def should_quantize(name: str) -> bool:
    """Return True if this tensor should be quantized."""
    name_lower = name.lower()
    for pat in SKIP_PATTERNS:
        if pat in name_lower:
            return False
    return True


def process_gguf(input_path: str, output_path: str, target_type: str) -> bool:
    if not os.path.exists(input_path):
        print(f"[Error] Input GGUF file '{input_path}' does not exist.")
        return False

    qtype = {
        "F16":  GGMLQuantizationType.F16,
        "Q4_0": GGMLQuantizationType.Q4_0,
        "Q8_0": GGMLQuantizationType.Q8_0,
    }.get(target_type)

    if qtype is None:
        print(f"[Error] Unsupported target type: {target_type}")
        return False

    print(f"Reading input model: {input_path}")
    reader = GGUFReader(input_path)

    # Extract GGUF architecture
    arch = "gpt-sovits"
    for field in reader.fields.values():
        if field.name == "general.architecture":
            arch = bytes(field.parts[-1]).decode("utf-8").strip("\x00")
            break

    writer = GGUFWriter(output_path, arch=arch)

    # Copy all metadata key-value pairs
    print("Copying metadata...")
    for name, field in reader.fields.items():
        if name.startswith("GGUF."):
            continue  # writer handles these automatically
        if not field.types:
            continue
        val = field.contents()
        writer.add_key_value(name, val, field.types[0])

    # Process tensors
    print(f"Processing tensors (target: {target_type})...")
    quantized_count = 0
    kept_count = 0
    f16_count = 0

    for tensor in reader.tensors:
        name = tensor.name
        tensor_type = tensor.tensor_type
        shape = tensor.shape
        data = tensor.data

        do_quantize = (
            target_type in ("Q4_0", "Q8_0")
            and should_quantize(name)
            and len(shape) >= 2
            and shape[0] % 32 == 0  # Q4_0/Q8_0 require ne0 (first dim in GGUF) divisible by block_size=32
        )

        if do_quantize:
            # Load as float32, reshape to original element shape, quantize
            arr = data.astype(np.float32).reshape(shape[::-1])
            q_data = quantize(arr, qtype)
            writer.add_tensor(name, q_data, raw_dtype=qtype)
            quantized_count += 1
            print(f"  [{target_type}] {name}  shape={list(shape)}")
        elif tensor_type in (GGMLQuantizationType.F32,) and target_type in ("F16", "Q4_0", "Q8_0") and not ("bias" in name.lower() or "norm" in name.lower() or "alpha" in name.lower() or len(shape) <= 1):
            # F32 → F16 cast for weights to save space
            arr = data.astype(np.float16)
            writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F16)
            f16_count += 1
            print(f"  [F16] {name}  shape={list(shape)}")
        else:
            writer.add_tensor(name, data, raw_dtype=tensor_type)
            kept_count += 1

    print(f"Writing output model to: {output_path}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    src_sz = os.path.getsize(input_path)
    dst_sz = os.path.getsize(output_path)
    print(f"\nQuantization complete!")
    print(f"  {quantized_count} tensors quantized to {target_type}")
    print(f"  {f16_count} tensors cast to F16")
    print(f"  {kept_count} tensors kept as-is")
    print(f"  Source:      {src_sz / (1024**2):.1f} MB")
    print(f"  Quantized:   {dst_sz / (1024**2):.1f} MB")
    print(f"  Reduction:   {100 * (src_sz - dst_sz) / src_sz:.1f}%")
    return True


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS GGUF Quantization Tool")
    parser.add_argument("-i", "--input", required=True, help="Path to the input GGUF model")
    parser.add_argument("-o", "--output", required=True, help="Path to save the quantized GGUF model")
    parser.add_argument("-t", "--type", choices=["F16", "Q4_0", "Q8_0"], default="F16",
                        help="Target quantization type (default: F16)")
    args = parser.parse_args()

    try:
        process_gguf(args.input, args.output, args.type)
    except Exception as e:
        print(f"[Error] Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
