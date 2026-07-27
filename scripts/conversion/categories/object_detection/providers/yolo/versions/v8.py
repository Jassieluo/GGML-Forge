"""Ultralytics YOLOv8 Detect and Segment adapter."""

from dataclasses import dataclass
from typing import Dict, List

from common import csv, ordered_labels, tensor_to_numpy


@dataclass(frozen=True)
class Topology:
    task: str
    class_count: int
    reg_max: int
    box_channels: int
    class_channels: int
    out_channels: List[int]
    hidden_channels: List[int]
    repeats: List[int]
    mask_count: int = 0
    mask_channels: int = 0
    prototype_channels: int = 0

    @property
    def architecture(self):
        return "yolo_v8_seg" if self.task == "segment" else "yolo_v8"

    def schema_arguments(self):
        arguments = (
            self.class_count,
            self.reg_max,
            self.box_channels,
            self.class_channels,
            csv(self.out_channels),
            csv(self.hidden_channels),
            csv(self.repeats),
        )
        if self.task == "segment":
            arguments += (self.mask_count, self.mask_channels, self.prototype_channels)
        return arguments


def _conv_outputs(state: Dict[str, object], name: str) -> int:
    tensor = state.get(name)
    return int(tensor.shape[0]) if tensor is not None and tensor.ndim == 4 else 0


def inspect(network, state: Dict[str, object]) -> Topology:
    head = network.model[-1]
    # Keep this adapter independent of a particular Ultralytics package layout.
    # Loading the checkpoint already establishes the concrete dependency; here
    # the exported module contract is what matters.
    head_type = head.__class__.__name__
    if head_type not in ("Detect", "Segment"):
        raise ValueError("YOLOv8 adapter supports Detect and Segment checkpoints only")

    out_channels = [0] * 23
    hidden_channels = [0] * 23
    repeats = [0] * 23
    for layer in (0, 1, 3, 5, 7, 16, 19):
        out_channels[layer] = _conv_outputs(state, f"model.{layer}.conv.weight")
    for layer in (2, 4, 6, 8, 12, 15, 18, 21):
        base = f"model.{layer}"
        out_channels[layer] = _conv_outputs(state, f"{base}.cv2.conv.weight")
        hidden_channels[layer] = _conv_outputs(state, f"{base}.cv1.conv.weight") // 2
        while f"{base}.m.{repeats[layer]}.cv1.conv.weight" in state:
            repeats[layer] += 1
    out_channels[9] = _conv_outputs(state, "model.9.cv2.conv.weight")
    topology = Topology(
        task="segment" if head_type == "Segment" else "detect",
        class_count=int(head.nc),
        reg_max=int(head.reg_max),
        box_channels=_conv_outputs(state, "model.22.cv2.0.0.conv.weight"),
        class_channels=_conv_outputs(state, "model.22.cv3.0.0.conv.weight"),
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        repeats=repeats,
        mask_count=int(head.nm) if head_type == "Segment" else 0,
        mask_channels=_conv_outputs(state, "model.22.cv4.0.0.conv.weight")
            if head_type == "Segment" else 0,
        prototype_channels=_conv_outputs(state, "model.22.proto.cv1.conv.weight")
            if head_type == "Segment" else 0,
    )
    required = (
        topology.class_count,
        topology.reg_max,
        topology.box_channels,
        topology.class_channels,
        *(out_channels[index] for index in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 18, 19, 21)),
        *(hidden_channels[index] for index in (2, 4, 6, 8, 12, 15, 18, 21)),
        *(repeats[index] for index in (2, 4, 6, 8, 12, 15, 18, 21)),
    )
    if any(value <= 0 for value in required):
        raise ValueError("checkpoint does not match the supported YOLOv8 Detect topology")
    if _conv_outputs(state, "model.22.cv2.0.2.weight") != topology.reg_max * 4 or \
       _conv_outputs(state, "model.22.cv3.0.2.weight") != topology.class_count:
        raise ValueError("YOLOv8 Detect head shape does not match its metadata")
    if topology.task == "segment":
        segment_required = (
            topology.mask_count, topology.mask_channels, topology.prototype_channels,
            _conv_outputs(state, "model.22.proto.cv3.conv.weight"),
        )
        if any(value <= 0 for value in segment_required) or \
           segment_required[-1] != topology.mask_count:
            raise ValueError("YOLOv8 Segment prototype topology is invalid")
        for scale in range(3):
            base = f"model.22.cv4.{scale}"
            if _conv_outputs(state, f"{base}.0.conv.weight") != topology.mask_channels or \
               _conv_outputs(state, f"{base}.1.conv.weight") != topology.mask_channels or \
               _conv_outputs(state, f"{base}.2.weight") != topology.mask_count:
                raise ValueError("YOLOv8 Segment mask branch topology is inconsistent")
    return topology


def canonical_tensors(state: Dict[str, object]):
    for name in sorted(state):
        if not name.startswith("model.") or name.endswith("dfl.conv.weight"):
            continue
        if ".bn." in name:
            raise ValueError("checkpoint fusion left BatchNorm parameters behind")
        if not (name.endswith(".weight") or name.endswith(".bias")):
            continue
        yield name, tensor_to_numpy(state[name], force_float32=name.endswith(".bias"))


def labels(model, topology: Topology):
    return ordered_labels(model.names, topology.class_count)
