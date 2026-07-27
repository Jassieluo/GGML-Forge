#!/usr/bin/env python3
"""Convert supported YOLO-family checkpoints to Forge GGUF artifacts."""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
from common import find_repository_root, load_ultralytics_checkpoint

_ROOT = find_repository_root(_SCRIPT_DIR)
sys.path.insert(1, str(_ROOT / "gguf-py"))
sys.path.insert(1, str(_ROOT / "scripts"))

from conversion.common.model import ModelArtifact
from conversion.common.quantization import QuantizationPolicy, SUPPORTED_TARGETS
from conversion.common.schema_tool import load_cpp_schema
from conversion.common.validation import ArtifactContract, validate_artifact
from versions import v8


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a YOLO checkpoint to GGUF")
    parser.add_argument("input", type=Path, help="trusted Ultralytics .pt checkpoint")
    parser.add_argument("output", type=Path, help="output .gguf artifact")
    parser.add_argument("--version", choices=("v8",), default="v8")
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--input-height", type=int, default=640)
    parser.add_argument("--quantize", choices=sorted(SUPPORTED_TARGETS), default="F16")
    parser.add_argument("--quant-policy", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.suffix.lower() != ".gguf":
        raise ValueError("output path must use the .gguf extension")
    if min(args.input_width, args.input_height) <= 0 or \
       args.input_width % 32 or args.input_height % 32:
        raise ValueError("input dimensions must be positive multiples of 32")

    source = load_ultralytics_checkpoint(args.input)
    network = source.model
    state = network.state_dict()
    adapter = v8
    topology = adapter.inspect(network, state)
    schema = load_cpp_schema(topology.architecture, topology.schema_arguments())
    policy = QuantizationPolicy.from_file(str(args.quant_policy), args.quantize) \
        if args.quant_policy else QuantizationPolicy(args.quantize)
    writer = ModelArtifact(
        str(args.output), topology.architecture, schema, args.quantize, policy)
    task_name = "Segment" if topology.task == "segment" else "Detect"
    writer.add_string("general.name", f"YOLOv8 {task_name} ({topology.class_count} classes)")
    writer.add_string("yolo.family", "yolo")
    writer.add_string("yolo.version", "v8")
    writer.add_uint32("yolo.reg_max", topology.reg_max)
    writer.add_array("yolo.strides", [8, 16, 32])
    writer.add_bool("detection.task.boxes", True)
    writer.add_bool("detection.task.oriented_boxes", False)
    writer.add_bool("detection.task.instance_masks", topology.task == "segment")
    writer.add_bool("detection.task.keypoints", False)
    writer.add_uint32("detection.class_count", topology.class_count)
    writer.add_uint32("detection.keypoint_count", 0)
    writer.add_uint32("detection.input.width", args.input_width)
    writer.add_uint32("detection.input.height", args.input_height)
    writer.add_string("detection.input.normalization", "zero_to_one")
    if topology.task == "segment":
        writer.add_uint32("yolo.mask_count", topology.mask_count)
    writer.add_array("detection.labels", adapter.labels(source, topology))
    for name, value in adapter.canonical_tensors(state):
        writer.parameter(name, value)
    writer.write()
    validate_artifact(
        str(args.output),
        ArtifactContract.create(
            metadata={
                "yolo.version", "yolo.reg_max", "yolo.strides",
                "detection.task.boxes", "detection.task.instance_masks",
                "detection.class_count",
                "detection.input.width", "detection.input.height",
                "detection.input.normalization", "detection.labels",
                *({"yolo.mask_count"} if topology.task == "segment" else set()),
            },
            tensors={
                "model.0.conv.weight", "model.22.cv2.0.2.weight",
                "model.22.cv3.0.2.weight",
                *({"model.22.proto.cv1.conv.weight", "model.22.cv4.0.2.weight"}
                  if topology.task == "segment" else set()),
            },
        ),
    )
    print(f"wrote YOLOv8 {task_name} GGUF: {args.output}")


if __name__ == "__main__":
    main()
