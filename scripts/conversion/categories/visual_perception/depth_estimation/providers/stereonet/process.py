#!/usr/bin/env python3
"""Convert the public KeystoneDepth StereoNet Lightning checkpoint to Forge GGUF."""

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


def array(state, name, force_float32=False):
    value = state[name].detach().cpu().numpy()
    return value.astype(np.float32) if force_float32 else value


def fuse_conv_bn(state, conv, batch_norm, dimensions, eps=1e-5):
    weight = state[f"{conv}.weight"].detach().float()
    mean = state[f"{batch_norm}.running_mean"].detach().float()
    variance = state[f"{batch_norm}.running_var"].detach().float()
    gamma = state[f"{batch_norm}.weight"].detach().float()
    beta = state[f"{batch_norm}.bias"].detach().float()
    scale = gamma / torch.sqrt(variance + eps)
    bias = state.get(f"{conv}.bias")
    if bias is None:
        bias = torch.zeros_like(mean)
    shape = (-1,) + (1,) * (dimensions + 1)
    return ((weight * scale.reshape(shape)).numpy(),
            (beta + (bias.detach().float() - mean) * scale).numpy())


def canonical_tensors(state):
    feature = "feature_extractor.net"
    for index in range(3):
        source = f"{feature}.segment_0_conv_{index}"
        yield f"features.downsample.{index}.weight", array(state, f"{source}.weight")
        yield f"features.downsample.{index}.bias", array(state, f"{source}.bias", True)
    for index in range(6):
        source = f"{feature}.segment_1_res_{index}"
        for part in (1, 2):
            weight, bias = fuse_conv_bn(state, f"{source}.conv_{part}",
                                         f"{source}.batch_norm_{part}", 2)
            yield f"features.residual.{index}.{'first' if part == 1 else 'second'}.weight", weight
            yield f"features.residual.{index}.{'first' if part == 1 else 'second'}.bias", bias
    source = f"{feature}.segment_2_conv_0"
    yield "features.output.weight", array(state, f"{source}.weight")
    yield "features.output.bias", array(state, f"{source}.bias", True)

    cost = "cost_volumizer.net"
    for index in range(4):
        weight, bias = fuse_conv_bn(state, f"{cost}.segment_0_conv_{index}",
                                     f"{cost}.segment_0_bn_{index}", 3)
        yield f"cost.layers.{index}.weight", weight
        yield f"cost.layers.{index}.bias", bias
    yield "cost.output.weight", array(state, f"{cost}.segment_1_conv_0.weight")
    yield "cost.output.bias", array(state, f"{cost}.segment_1_conv_0.bias", True)

    for refiner in range(3):
        source = f"refiners.{refiner}.net"
        yield f"refiners.{refiner}.input.weight", array(state, f"{source}.segment_0_conv_0.weight")
        yield f"refiners.{refiner}.input.bias", array(state, f"{source}.segment_0_conv_0.bias", True)
        for index in range(6):
            block = f"{source}.segment_1_res_{index}"
            for part in (1, 2):
                weight, bias = fuse_conv_bn(state, f"{block}.conv_{part}",
                                             f"{block}.batch_norm_{part}", 2)
                target = "first" if part == 1 else "second"
                yield f"refiners.{refiner}.residual.{index}.{target}.weight", weight
                yield f"refiners.{refiner}.residual.{index}.{target}.bias", bias
        yield f"refiners.{refiner}.output.weight", array(state, f"{source}.segment_2_conv_0.weight")
        yield f"refiners.{refiner}.output.bias", array(state, f"{source}.segment_2_conv_0.bias", True)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert KeystoneDepth StereoNet to GGUF")
    parser.add_argument("input", type=Path, help="trusted public Lightning .ckpt")
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-side", type=int, default=625)
    parser.add_argument("--quantize", choices=sorted(SUPPORTED_TARGETS), default="F16")
    parser.add_argument("--quant-policy", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.suffix.lower() != ".gguf" or args.max_side < 256:
        raise ValueError("output must be .gguf and max-side must be at least 256")
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or "feature_extractor.net.segment_0_conv_0.weight" not in state:
        raise ValueError("input is not the supported StereoNet Lightning checkpoint")
    hparams = checkpoint.get("hyper_parameters", {})
    expected = {"in_channels": 1, "k_downsampling_layers": 3,
                "k_refinement_layers": 3, "candidate_disparities": 256,
                "feature_extractor_filters": 32, "cost_volumizer_filters": 32}
    for name, value in expected.items():
        if hparams.get(name) != value:
            raise ValueError(f"unsupported StereoNet topology: {name}={hparams.get(name)!r}")

    schema = load_cpp_schema("stereonet", ())
    policy = QuantizationPolicy.from_file(str(args.quant_policy), args.quantize) \
        if args.quant_policy else QuantizationPolicy(args.quantize)
    writer = ModelArtifact(str(args.output), "stereonet", schema, args.quantize, policy)
    writer.add_string("general.name", "StereoNet KeystoneDepth grayscale")
    writer.add_bool("depth.task.monocular", False)
    writer.add_bool("depth.task.stereo", True)
    writer.add_string("depth.output.kind", "disparity")
    writer.add_uint32("depth.input.max_side", args.max_side)
    writer.add_string("depth.input.resize", "shrink_longest_edge")
    writer.add_string("depth.input.normalization", "keystone_gray")
    writer.add_uint32("depth.candidate_disparities", 256)
    for name, value in canonical_tensors(state):
        writer.parameter(name, value)
    writer.write()
    validate_artifact(str(args.output), ArtifactContract.create(
        metadata={"depth.task.monocular", "depth.task.stereo", "depth.output.kind",
                  "depth.input.max_side", "depth.input.resize",
                  "depth.input.normalization", "depth.candidate_disparities"},
        tensors={"features.downsample.0.weight", "cost.layers.0.weight",
                 "cost.output.weight", "refiners.2.output.weight"},
    ))
    print(f"wrote StereoNet GGUF: {args.output}")


if __name__ == "__main__":
    main()
