import sys
import torch
from pathlib import Path

# Add path for utils and gguf-py
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(1, str(_SCRIPT_DIR / ".." / ".." / "gguf-py"))

from gguf import GGUFWriter, GGMLQuantizationType
from gguf_utils import tensor_to_fp16, tensor_to_fp32

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
    gguf_writer.add_string("gpt_sovits.version", "v1|v2|v2pro|v3|v4")

    mapped_count = 0
    for key in sorted(sd.keys()):
        t = sd[key]
        if key == "bert.embeddings.position_ids":
            # Keep as [512] (inference creates its own position_ids, this is unused metadata)
            arr = tensor_to_fp32(t)
            gguf_writer.add_tensor(key, arr, raw_dtype=GGMLQuantizationType.F32)
        elif t.ndim <= 1 or "bias" in key or "LayerNorm" in key:
            arr = tensor_to_fp32(t)
            gguf_writer.add_tensor(key, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            arr = tensor_to_fp16(t)
            gguf_writer.add_tensor(key, arr, raw_dtype=GGMLQuantizationType.F16)
        mapped_count += 1
        print(f"    + {key}: {list(t.shape)}  ({'f32' if t.ndim <= 1 or 'bias' in key or 'LayerNorm' in key else 'fp16'})")

    print(f"\n  Writing GGUF file ({mapped_count} tensors)...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"  ✓ BERT GGUF saved to: {dst_path}")
