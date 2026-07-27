import importlib
import gc
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType


ROOT = Path(__file__).resolve().parents[2]
PROVIDER = ROOT / "scripts" / "conversion" / "categories" / \
    "visual_perception" / "providers" / "yolo"
sys.path.insert(0, str(PROVIDER))
sys.path.insert(1, str(ROOT / "scripts"))
v8 = importlib.import_module("versions.v8")

from conversion.common.exporter import export_model
from conversion.common.model_schema import ModelSchema
from conversion.common.schema import ModelDefinition, Parameter


class Tensor:
    def __init__(self, *shape):
        self.shape = shape
        self.ndim = len(shape)


class Detect:
    nc = 80
    reg_max = 16


class Network:
    model = [Detect()]


class Segment(Detect):
    nm = 32


class SegmentNetwork:
    model = [Segment()]


class Pose(Detect):
    kpt_shape = (17, 3)


class PoseNetwork:
    model = [Pose()]


class OBB(Detect):
    ne = 1


class OBBNetwork:
    model = [OBB()]


class Classify:
    pass


class ClassifyNetwork:
    model = [Classify()]


def conv(outputs):
    return Tensor(outputs, 1, 1, 1)


def state_dict():
    state = {}
    channels = {
        0: 16, 1: 32, 2: 32, 3: 64, 4: 64, 5: 128, 6: 128,
        7: 256, 8: 256, 9: 256, 12: 128, 15: 64, 16: 64,
        18: 128, 19: 128, 21: 256,
    }
    for layer in (0, 1, 3, 5, 7, 16, 19):
        state[f"model.{layer}.conv.weight"] = conv(channels[layer])
    for layer in (2, 4, 6, 8, 12, 15, 18, 21):
        hidden = channels[layer] // 2
        state[f"model.{layer}.cv1.conv.weight"] = conv(hidden * 2)
        state[f"model.{layer}.cv2.conv.weight"] = conv(channels[layer])
        state[f"model.{layer}.m.0.cv1.conv.weight"] = conv(hidden)
    state["model.9.cv2.conv.weight"] = conv(channels[9])
    state["model.22.cv2.0.0.conv.weight"] = conv(64)
    state["model.22.cv3.0.0.conv.weight"] = conv(80)
    state["model.22.cv2.0.2.weight"] = conv(64)
    state["model.22.cv3.0.2.weight"] = conv(80)
    return state


def segment_state_dict():
    state = state_dict()
    state["model.22.proto.cv1.conv.weight"] = conv(64)
    state["model.22.proto.cv3.conv.weight"] = conv(32)
    for scale in range(3):
        base = f"model.22.cv4.{scale}"
        state[f"{base}.0.conv.weight"] = conv(32)
        state[f"{base}.1.conv.weight"] = conv(32)
        state[f"{base}.2.weight"] = conv(32)
    return state


def pose_state_dict():
    state = state_dict()
    for scale in range(3):
        base = f"model.22.cv4.{scale}"
        state[f"{base}.0.conv.weight"] = conv(51)
        state[f"{base}.1.conv.weight"] = conv(51)
        state[f"{base}.2.weight"] = conv(51)
    return state


def obb_state_dict():
    state = state_dict()
    for scale in range(3):
        base = f"model.22.cv4.{scale}"
        state[f"{base}.0.conv.weight"] = conv(16)
        state[f"{base}.1.conv.weight"] = conv(16)
        state[f"{base}.2.weight"] = conv(1)
    return state


def classify_state_dict():
    state = {}
    channels = {0: 16, 1: 32, 2: 32, 3: 64, 4: 64, 5: 128,
                6: 128, 7: 256, 8: 256}
    for layer in (0, 1, 3, 5, 7):
        state[f"model.{layer}.conv.weight"] = conv(channels[layer])
    for layer, repeat_count in ((2, 1), (4, 2), (6, 2), (8, 1)):
        hidden = channels[layer] // 2
        state[f"model.{layer}.cv1.conv.weight"] = conv(hidden * 2)
        state[f"model.{layer}.cv2.conv.weight"] = conv(channels[layer])
        for repeat in range(repeat_count):
            state[f"model.{layer}.m.{repeat}.cv1.conv.weight"] = conv(hidden)
    state["model.9.conv.conv.weight"] = conv(1280)
    state["model.9.linear.weight"] = Tensor(1000, 1280)
    return state


class YoloV8AdapterTest(unittest.TestCase):
    def test_topology_is_derived_from_fused_tensor_names(self):
        topology = v8.inspect(Network(), state_dict())
        self.assertEqual(topology.class_count, 80)
        self.assertEqual(topology.reg_max, 16)
        self.assertEqual(topology.box_channels, 64)
        self.assertEqual(topology.out_channels[21], 256)
        self.assertEqual(topology.hidden_channels[12], 64)
        self.assertEqual(topology.repeats[8], 1)
        self.assertEqual(len(topology.schema_arguments()), 7)

    def test_non_detect_head_is_rejected(self):
        class RTDETR:
            nc = 80
            reg_max = 16

        class RTDETRNetwork:
            model = [RTDETR()]

        with self.assertRaisesRegex(ValueError, "Detect, Segment, Pose, OBB, and Classify checkpoints only"):
            v8.inspect(RTDETRNetwork(), state_dict())

    def test_segment_topology_adds_mask_schema_arguments(self):
        topology = v8.inspect(SegmentNetwork(), segment_state_dict())
        self.assertEqual(topology.task, "segment")
        self.assertEqual(topology.architecture, "yolo_v8_seg")
        self.assertEqual(topology.mask_count, 32)
        self.assertEqual(topology.mask_channels, 32)
        self.assertEqual(topology.prototype_channels, 64)
        self.assertEqual(len(topology.schema_arguments()), 10)

    def test_pose_topology_adds_keypoint_schema_arguments(self):
        topology = v8.inspect(PoseNetwork(), pose_state_dict())
        self.assertEqual(topology.task, "pose")
        self.assertEqual(topology.architecture, "yolo_v8_pose")
        self.assertEqual(topology.keypoint_count, 17)
        self.assertEqual(topology.keypoint_dimensions, 3)
        self.assertEqual(topology.keypoint_channels, 51)
        self.assertEqual(len(topology.schema_arguments()), 10)

    def test_obb_topology_adds_angle_schema_arguments(self):
        topology = v8.inspect(OBBNetwork(), obb_state_dict())
        self.assertEqual(topology.task, "obb")
        self.assertEqual(topology.architecture, "yolo_v8_obb")
        self.assertEqual(topology.angle_count, 1)
        self.assertEqual(topology.angle_channels, 16)
        self.assertEqual(len(topology.schema_arguments()), 9)

    def test_classification_topology_uses_classify_head(self):
        topology = v8.inspect(ClassifyNetwork(), classify_state_dict())
        self.assertEqual(topology.task, "classify")
        self.assertEqual(topology.architecture, "yolo_v8_cls")
        self.assertEqual(topology.class_count, 1000)
        self.assertEqual(topology.head_channels, 1280)
        self.assertEqual(topology.repeats[4], 2)
        self.assertEqual(len(topology.schema_arguments()), 5)

    def test_labels_are_ordered_by_class_id(self):
        self.assertEqual(v8.ordered_labels({0: "person", 1: "car"}, 2),
                         ["person", "car"])

    def test_yolo_conv_uses_q4_storage_through_nn_schema(self):
        schema = ModelSchema.from_json(json.dumps({"parameters": [{
            "path": "model.0.conv.weight",
            "required": True,
            "usage": "conv2d_weight",
            "direct_storage_types": ["Q4_K", "Q4_0", "Q8_0", "F16", "F32"],
            "quantized_layout": "flexible_rows",
            "logical_shape": [1, 1, 32, 64],
        }]}))
        definition = ModelDefinition("yolo_v8", schema)
        definition.parameter(Parameter(
            "model.0.conv.weight",
            np.linspace(-1.0, 1.0, 64 * 32, dtype=np.float32).reshape(64, 32, 1, 1),
        ))
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "yolo-conv-q4.gguf")
            report = export_model(definition, output, "Q4_0")
            reader = GGUFReader(output)
            self.assertEqual(report.counts["Q4_0"], 1)
            self.assertEqual(reader.tensors[0].tensor_type, GGMLQuantizationType.Q4_0)
            del reader
            gc.collect()

    def test_single_output_conv_preserves_native_logical_rank(self):
        schema = ModelSchema.from_json(json.dumps({"parameters": [{
            "path": "model.22.cv3.0.2.weight",
            "required": True,
            "usage": "conv2d_weight",
            "direct_storage_types": ["Q4_0", "F16", "F32"],
            "quantized_layout": "flexible_rows",
            "logical_shape": [1, 1, 32, 1],
        }]}))
        definition = ModelDefinition("yolo_v8_pose", schema)
        definition.parameter(Parameter(
            "model.22.cv3.0.2.weight",
            np.ones((1, 32, 1, 1), dtype=np.float32),
        ))
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "single-output-f16.gguf")
            export_model(definition, output, "F16")
            reader = GGUFReader(output)
            self.assertEqual(
                reader.fields["nn.logical_shape.names"].contents(),
                ["model.22.cv3.0.2.weight"],
            )
            self.assertEqual(
                reader.fields["nn.logical_shape.dimensions"].contents(),
                [1, 1, 32, 1],
            )
            del reader
            gc.collect()


if __name__ == "__main__":
    unittest.main()
