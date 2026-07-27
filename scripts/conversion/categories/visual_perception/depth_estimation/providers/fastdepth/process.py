#!/usr/bin/env python3
"""Convert the ST/PINTO pruned FastDepth TFLite checkpoint to Forge GGUF."""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import tflite
except ImportError as error:
    raise SystemExit("FastDepth conversion requires the small 'tflite' Python package") from error

_HERE = Path(__file__).resolve().parent
_ROOT = next(parent for parent in _HERE.parents if (parent / "CMakeLists.txt").exists())
sys.path.insert(1, str(_ROOT / "gguf-py"))
sys.path.insert(1, str(_ROOT / "scripts"))

from conversion.common.model import ModelArtifact
from conversion.common.quantization import QuantizationPolicy, SUPPORTED_TARGETS
from conversion.common.schema_tool import load_cpp_schema
from conversion.common.validation import ArtifactContract, validate_artifact


DTYPES = {2: np.int32, 3: np.uint8, 9: np.int8}


def tensor_array(model, subgraph, index):
    tensor = subgraph.Tensors(index)
    shape = tuple(tensor.Shape(i) for i in range(tensor.ShapeLength()))
    dtype = DTYPES.get(tensor.Type())
    if dtype is None:
        raise ValueError(f"unsupported TFLite tensor type {tensor.Type()}")
    raw = model.Buffers(tensor.Buffer()).DataAsNumpy()
    if not isinstance(raw, np.ndarray):
        raise ValueError(f"TFLite tensor {index} has no constant buffer")
    values = np.frombuffer(raw.tobytes(), dtype=dtype).reshape(shape)
    quant = tensor.Quantization()
    scales = np.asarray([quant.Scale(i) for i in range(quant.ScaleLength())], np.float32)
    zeros = np.asarray([quant.ZeroPoint(i) for i in range(quant.ZeroPointLength())], np.float32)
    if scales.size == 0:
        return values.astype(np.float32)
    axis = quant.QuantizedDimension()
    broadcast = [1] * values.ndim
    broadcast[axis] = scales.size
    if zeros.size == 1 and scales.size > 1:
        zeros = np.repeat(zeros, scales.size)
    return (values.astype(np.float32) - zeros.reshape(broadcast)) * scales.reshape(broadcast)


def convolution(model, subgraph, operator_index, target, depthwise):
    operator = subgraph.Operators(operator_index)
    weight_index = operator.Inputs(1)
    bias_index = operator.Inputs(2)
    weight = tensor_array(model, subgraph, weight_index)
    bias = tensor_array(model, subgraph, bias_index).reshape(-1).astype(np.float32)
    if depthwise:
        if weight.shape[0] != 1:
            raise ValueError("FastDepth depthwise weight multiplier is not one")
        weight = np.transpose(weight[0], (2, 0, 1))[:, None, :, :]
    else:
        weight = np.transpose(weight, (0, 3, 1, 2))
    return ((f"{target}.weight", weight.astype(np.float32)),
            (f"{target}.bias", bias))


def canonical_tensors(model, subgraph):
    yield from convolution(model, subgraph, 2, "stem.conv", False)
    for index in range(13):
        yield from convolution(model, subgraph, 4 + index * 3,
                               f"encoder.{index}.depthwise.conv", True)
        yield from convolution(model, subgraph, 5 + index * 3,
                               f"encoder.{index}.pointwise.conv", False)
    decoder_ops = ((43, 44), (47, 48), (52, 53), (57, 58), (62, 63))
    for index, (depthwise_op, pointwise_op) in enumerate(decoder_ops):
        yield from convolution(model, subgraph, depthwise_op,
                               f"decoder.{index}.depthwise.conv", True)
        yield from convolution(model, subgraph, pointwise_op,
                               f"decoder.{index}.pointwise.conv", False)
    yield from convolution(model, subgraph, 65, "output.conv", False)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert pruned FastDepth TFLite to GGUF")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quantize", choices=sorted(SUPPORTED_TARGETS), default="F16")
    parser.add_argument("--quant-policy", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.suffix.lower() != ".gguf":
        raise ValueError("output path must use .gguf")
    buffer = args.input.read_bytes()
    model = tflite.Model.GetRootAsModel(buffer, 0)
    if model.SubgraphsLength() != 1:
        raise ValueError("FastDepth TFLite must contain one subgraph")
    subgraph = model.Subgraphs(0)
    if subgraph.OperatorsLength() != 67:
        raise ValueError("unsupported FastDepth TFLite graph revision")
    schema = load_cpp_schema("fastdepth_mobilenet_v1", ())
    policy = QuantizationPolicy.from_file(str(args.quant_policy), args.quantize) \
        if args.quant_policy else QuantizationPolicy(args.quantize)
    writer = ModelArtifact(str(args.output), "fastdepth_mobilenet_v1", schema,
                           args.quantize, policy)
    writer.add_string("general.name", "FastDepth MobileNetV1 pruned 224")
    writer.add_bool("depth.task.monocular", True)
    writer.add_bool("depth.task.stereo", False)
    writer.add_string("depth.output.kind", "metric")
    writer.add_uint32("depth.input.width", 224)
    writer.add_uint32("depth.input.height", 224)
    writer.add_string("depth.input.resize", "stretch")
    writer.add_string("depth.input.normalization", "zero_to_one")
    for name, value in canonical_tensors(model, subgraph):
        writer.parameter(name, value)
    writer.write()
    validate_artifact(str(args.output), ArtifactContract.create(
        metadata={"depth.task.monocular", "depth.task.stereo", "depth.output.kind",
                  "depth.input.width", "depth.input.height", "depth.input.resize",
                  "depth.input.normalization"},
        tensors={"stem.conv.weight", "encoder.0.depthwise.conv.weight",
                 "decoder.0.depthwise.conv.weight", "output.conv.weight"},
    ))
    print(f"wrote FastDepth GGUF: {args.output}")


if __name__ == "__main__":
    main()
