import gc
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from conversion.common.quantization import (
    QUANTIZED_TARGETS, QuantizationPolicy, TensorSpec, quantize_gguf, quantize_q4_k,
)
from gguf.quants import dequantize, quantize


class QuantizationPolicyTest(unittest.TestCase):
    def test_q4_k_encoder_round_trip(self):
        source = np.linspace(-2.0, 3.0, 512, dtype=np.float32).reshape(2, 256)
        packed = quantize_q4_k(source)
        restored = dequantize(packed, GGMLQuantizationType.Q4_K)
        self.assertEqual(packed.shape, (2, 144))
        self.assertLess(float(np.mean(np.abs(source - restored))), 0.03)

    def test_block_fallback_and_sensitivity(self):
        normal = TensorSpec("weight", (128, 4))
        sensitive = TensorSpec("embedding", (256, 4), sensitivity="high")
        odd = TensorSpec("odd", (127, 4))

        self.assertEqual(QuantizationPolicy("Q4_K").candidates(normal), ("Q8_0", "F16"))
        self.assertEqual(QuantizationPolicy("Q4_K").candidates(sensitive), ("Q8_0", "F16"))
        self.assertEqual(QuantizationPolicy("Q4_0").candidates(odd), ("F16",))

    def test_every_script_target_has_an_encoder(self):
        source = np.linspace(-1.0, 1.0, 512, dtype=np.float32).reshape(2, 256)
        for target in QUANTIZED_TARGETS:
            with self.subTest(target=target):
                spec = TensorSpec("weight", (256, 2), allowed_types=(target, "Q8_0", "F16"))
                self.assertEqual(QuantizationPolicy(target).candidates(spec)[0], target)
                packed = quantize_q4_k(source) if target == "Q4_K" else \
                    quantize(source, getattr(GGMLQuantizationType, target))
                self.assertGreater(packed.size, 0)

    def test_user_rule_still_obeys_allowed_types(self):
        policy = QuantizationPolicy("Q4_0", rules=[{
            "role": ["conv1d_weight"],
            "types": ["Q8_0", "F16"],
        }])
        spec = TensorSpec("conv", (32, 8), role="conv1d_weight", allowed_types=("F16",))
        self.assertEqual(policy.candidates(spec), ("F16",))

    def test_name_regex_selects_tensor_rule(self):
        policy = QuantizationPolicy("Q4_0", rules=[
            {"name": r"^decoder\.", "types": ["Q8_0", "F16"]},
        ])
        selected = TensorSpec("decoder.weight", (32, 32))
        default = TensorSpec("encoder.weight", (32, 32))
        self.assertEqual(policy.candidates(selected), ("Q8_0", "F16"))
        self.assertEqual(policy.candidates(default), ("Q4_0", "Q8_0", "F16"))


class QuantizationExecutionTest(unittest.TestCase):
    @staticmethod
    def write_source(path: str, tensor_type=GGMLQuantizationType.F16):
        writer = GGUFWriter(path, arch="test_tts")
        data = np.zeros(32, dtype=np.float16 if tensor_type == GGMLQuantizationType.F16 else np.float32)
        writer.add_tensor("test.bias", data, raw_dtype=tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_f32_is_a_target_type_and_transform_is_lazy(self):
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / "source.gguf")
            output = str(Path(directory) / "output.gguf")
            self.write_source(source)

            def describe(_arch, name, shape):
                return TensorSpec(name, shape, role="parameter", allowed_types=("F32",))

            def unexpected_transform(_arch, _spec, _data):
                raise AssertionError("F32 tensor must not enter the quantization transform")

            quantize_gguf(source, output, "Q4_0", describe, transform_tensor=unexpected_transform)
            reader = GGUFReader(output)
            self.assertEqual(reader.tensors[0].tensor_type, GGMLQuantizationType.F32)
            del reader
            gc.collect()

    def test_failed_write_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / "source.gguf")
            output = Path(directory) / "output.gguf"
            self.write_source(source, GGMLQuantizationType.F32)
            output.write_bytes(b"existing-output")

            def describe(_arch, name, shape):
                return TensorSpec(name, shape, quant_shape=(32, 1))

            def fail_transform(_arch, _spec, _data):
                raise RuntimeError("intentional failure")

            with self.assertRaisesRegex(RuntimeError, "intentional failure"):
                quantize_gguf(source, str(output), "Q4_0", describe, transform_tensor=fail_transform)
            self.assertEqual(output.read_bytes(), b"existing-output")
            self.assertEqual(list(Path(directory).glob(".model-quant-*.gguf")), [])


if __name__ == "__main__":
    unittest.main()
