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
    parser.add_argument("--input-width", type=int)
    parser.add_argument("--input-height", type=int)
    parser.add_argument("--quantize", choices=sorted(SUPPORTED_TARGETS), default="F16")
    parser.add_argument("--quant-policy", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.suffix.lower() != ".gguf":
        raise ValueError("output path must use the .gguf extension")
    source = load_ultralytics_checkpoint(args.input)
    network = source.model
    state = network.state_dict()
    adapter = v8
    topology = adapter.inspect(network, state)
    default_size = 224 if topology.task == "classify" else 640
    input_width = args.input_width or default_size
    input_height = args.input_height or default_size
    if min(input_width, input_height) <= 0 or input_width % 32 or input_height % 32:
        raise ValueError("input dimensions must be positive multiples of 32")
    if topology.task == "classify" and input_width != input_height:
        raise ValueError("YOLOv8 Classify requires a square center crop")
    schema = load_cpp_schema(topology.architecture, topology.schema_arguments())
    policy = QuantizationPolicy.from_file(str(args.quant_policy), args.quantize) \
        if args.quant_policy else QuantizationPolicy(args.quantize)
    writer = ModelArtifact(
        str(args.output), topology.architecture, schema, args.quantize, policy)
    task_name = {"detect": "Detect", "segment": "Segment", "pose": "Pose",
                 "obb": "OBB", "classify": "Classify"}[topology.task]
    writer.add_string("general.name", f"YOLOv8 {task_name} ({topology.class_count} classes)")
    writer.add_string("yolo.family", "yolo")
    writer.add_string("yolo.version", "v8")
    if topology.task == "classify":
        writer.add_uint32("classification.class_count", topology.class_count)
        writer.add_uint32("classification.input.width", input_width)
        writer.add_uint32("classification.input.height", input_height)
        writer.add_string("classification.input.resize", "shortest_center_crop")
        writer.add_string("classification.input.normalization", "zero_to_one")
        writer.add_array("classification.labels", adapter.labels(source, topology))
    else:
        writer.add_uint32("yolo.reg_max", topology.reg_max)
        writer.add_array("yolo.strides", [8, 16, 32])
        writer.add_bool("instance.task.boxes", True)
        writer.add_bool("instance.task.oriented_boxes", topology.task == "obb")
        writer.add_bool("instance.task.masks", topology.task == "segment")
        writer.add_bool("instance.task.keypoints", topology.task == "pose")
        writer.add_uint32("instance.class_count", topology.class_count)
        writer.add_uint32("instance.keypoint_count", topology.keypoint_count)
        writer.add_uint32("instance.input.width", input_width)
        writer.add_uint32("instance.input.height", input_height)
        writer.add_string("instance.input.normalization", "zero_to_one")
        if topology.task == "segment":
            writer.add_uint32("yolo.mask_count", topology.mask_count)
        elif topology.task == "pose":
            writer.add_uint32("yolo.keypoint_dimensions", topology.keypoint_dimensions)
        elif topology.task == "obb":
            writer.add_uint32("yolo.angle_count", topology.angle_count)
        writer.add_array("instance.labels", adapter.labels(source, topology))
    for name, value in adapter.canonical_tensors(state):
        writer.parameter(name, value)
    writer.write()
    validate_artifact(
        str(args.output),
        ArtifactContract.create(
            metadata={
                "yolo.version",
                *({"classification.class_count", "classification.input.width",
                   "classification.input.height", "classification.input.resize",
                   "classification.input.normalization", "classification.labels"}
                  if topology.task == "classify" else
                  {"yolo.reg_max", "yolo.strides", "instance.task.boxes",
                   "instance.task.oriented_boxes", "instance.task.masks",
                   "instance.task.keypoints", "instance.keypoint_count",
                   "instance.class_count", "instance.input.width",
                   "instance.input.height", "instance.input.normalization",
                   "instance.labels"}),
                *({"yolo.mask_count"} if topology.task == "segment" else set()),
                *({"yolo.keypoint_dimensions"} if topology.task == "pose" else set()),
                *({"yolo.angle_count"} if topology.task == "obb" else set()),
            },
            tensors={
                "model.0.conv.weight",
                *({"model.9.conv.conv.weight", "model.9.linear.weight"}
                  if topology.task == "classify" else
                  {"model.22.cv2.0.2.weight", "model.22.cv3.0.2.weight"}),
                *({"model.22.proto.cv1.conv.weight", "model.22.cv4.0.2.weight"}
                  if topology.task == "segment" else set()),
                *({"model.22.cv4.0.0.conv.weight", "model.22.cv4.0.2.weight"}
                  if topology.task == "pose" else set()),
                *({"model.22.cv4.0.0.conv.weight", "model.22.cv4.0.2.weight"}
                  if topology.task == "obb" else set()),
            },
        ),
    )
    print(f"wrote YOLOv8 {task_name} GGUF: {args.output}")


if __name__ == "__main__":
    main()
