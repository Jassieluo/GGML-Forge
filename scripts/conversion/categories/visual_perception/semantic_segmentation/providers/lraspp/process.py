#!/usr/bin/env python3
"""Convert the official torchvision LR-ASPP MobileNetV3-Large state dict."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = next(parent for parent in _HERE.parents if (parent / "CMakeLists.txt").exists())
sys.path.insert(1, str(_ROOT / "gguf-py"))
sys.path.insert(1, str(_ROOT / "scripts"))

from conversion.common.model import ModelArtifact
from conversion.common.quantization import QuantizationPolicy, SUPPORTED_TARGETS
from conversion.common.schema_tool import load_cpp_schema
from conversion.common.validation import ArtifactContract, validate_artifact


LABELS = [
    "__background__", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def pascal_palette(count):
    result = []
    for class_id in range(count):
        value, red, green, blue, shift = class_id, 0, 0, 0, 7
        while value:
            red |= (value & 1) << shift
            green |= ((value >> 1) & 1) << shift
            blue |= ((value >> 2) & 1) << shift
            value >>= 3
            shift -= 1
        result.extend((red, green, blue))
    return result


def fuse_conv_bn(state, conv, batch_norm, eps):
    weight = state[f"{conv}.weight"].detach().float()
    gamma = state[f"{batch_norm}.weight"].detach().float()
    beta = state[f"{batch_norm}.bias"].detach().float()
    mean = state[f"{batch_norm}.running_mean"].detach().float()
    variance = state[f"{batch_norm}.running_var"].detach().float()
    scale = gamma / torch.sqrt(variance + eps)
    fused_weight = weight * scale.reshape(-1, 1, 1, 1)
    conv_bias = state.get(f"{conv}.bias")
    if conv_bias is None:
        conv_bias = torch.zeros_like(mean)
    fused_bias = beta + (conv_bias.detach().float() - mean) * scale
    return fused_weight.numpy(), fused_bias.numpy()


def numpy(state, name, force_float32=False):
    value = state[name].detach().cpu().numpy()
    return value.astype(np.float32) if force_float32 else value


def canonical_tensors(state):
    def fused(target, conv, bn, eps=1e-3):
        weight, bias = fuse_conv_bn(state, conv, bn, eps)
        yield f"{target}.weight", weight
        yield f"{target}.bias", bias

    yield from fused("backbone.stem.conv", "backbone.0.0", "backbone.0.1")
    se_blocks = {3, 4, 5, 10, 11, 12, 13, 14}
    for index in range(15):
        source = f"backbone.{index + 1}.block"
        target = f"backbone.blocks.{index}"
        has_expand = index != 0
        has_se = index in se_blocks
        position = 0
        if has_expand:
            yield from fused(f"{target}.expand.conv", f"{source}.0.0", f"{source}.0.1")
            position = 1
        yield from fused(f"{target}.depthwise.conv", f"{source}.{position}.0", f"{source}.{position}.1")
        position += 1
        if has_se:
            for layer in ("fc1", "fc2"):
                name = f"{source}.{position}.{layer}"
                yield f"{target}.se.{layer}.weight", numpy(state, f"{name}.weight")
                yield f"{target}.se.{layer}.bias", numpy(state, f"{name}.bias", True)
            position += 1
        yield from fused(f"{target}.project.conv", f"{source}.{position}.0", f"{source}.{position}.1")
    yield from fused("backbone.final.conv", "backbone.16.0", "backbone.16.1")
    yield from fused("classifier.cbr.conv", "classifier.cbr.0", "classifier.cbr.1", 1e-5)
    yield "classifier.scale.weight", numpy(state, "classifier.scale.1.weight")
    for name in ("low_classifier", "high_classifier"):
        yield f"classifier.{name}.weight", numpy(state, f"classifier.{name}.weight")
        yield f"classifier.{name}.bias", numpy(state, f"classifier.{name}.bias", True)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert torchvision LR-ASPP to GGUF")
    parser.add_argument("input", type=Path, help="official lraspp_mobilenet_v3_large .pth state dict")
    parser.add_argument("output", type=Path)
    parser.add_argument("--shortest-edge", type=int, default=520)
    parser.add_argument("--quantize", choices=sorted(SUPPORTED_TARGETS), default="F16")
    parser.add_argument("--quant-policy", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.suffix.lower() != ".gguf" or args.shortest_edge <= 0:
        raise ValueError("output must be .gguf and shortest edge must be positive")
    state = torch.load(args.input, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "backbone.0.0.weight" not in state:
        raise ValueError("input is not an LR-ASPP MobileNetV3-Large state dict")
    schema = load_cpp_schema("lraspp_mobilenet_v3_large", ())
    policy = QuantizationPolicy.from_file(str(args.quant_policy), args.quantize) \
        if args.quant_policy else QuantizationPolicy(args.quantize)
    writer = ModelArtifact(str(args.output), "lraspp_mobilenet_v3_large", schema,
                           args.quantize, policy)
    writer.add_string("general.name", "Torchvision LR-ASPP MobileNetV3-Large")
    writer.add_uint32("segmentation.class_count", len(LABELS))
    writer.add_bool("segmentation.confidence", True)
    writer.add_uint32("segmentation.input.width", 0)
    writer.add_uint32("segmentation.input.height", 0)
    writer.add_uint32("segmentation.input.shortest_edge", args.shortest_edge)
    writer.add_string("segmentation.input.resize", "shortest_edge")
    writer.add_string("segmentation.input.normalization", "imagenet")
    writer.add_array("segmentation.labels", LABELS)
    writer.add_array("segmentation.palette", pascal_palette(len(LABELS)))
    for name, value in canonical_tensors(state):
        writer.parameter(name, value)
    writer.write()
    validate_artifact(str(args.output), ArtifactContract.create(
        metadata={"segmentation.class_count", "segmentation.confidence",
                  "segmentation.input.width", "segmentation.input.height",
                  "segmentation.input.shortest_edge", "segmentation.input.resize",
                  "segmentation.input.normalization", "segmentation.labels",
                  "segmentation.palette"},
        tensors={"backbone.stem.conv.weight", "backbone.final.conv.weight",
                 "classifier.low_classifier.weight", "classifier.high_classifier.weight"},
    ))
    print(f"wrote LR-ASPP GGUF: {args.output}")


if __name__ == "__main__":
    main()
