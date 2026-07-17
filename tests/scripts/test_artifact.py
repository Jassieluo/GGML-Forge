import gc
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from common.artifact import ArtifactBuilder
from common.layout import LAYOUT_AXES_KEY, LAYOUT_NAMES_KEY, LAYOUT_OFFSETS_KEY, Layout, write_layout_metadata
from common.validation import ArtifactContract, validate_artifact


class ArtifactBuilderTest(unittest.TestCase):
    def test_build_and_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "model.gguf")
            builder = ArtifactBuilder(output, "test_tts")
            builder.add_string("tts.artifact.kind", "decoder")
            builder.add_tensor("weight", np.ones((32, 2), dtype=np.float16), GGMLQuantizationType.F16)
            builder.write()

            validate_artifact(
                output,
                ArtifactContract.create(metadata=["tts.artifact.kind"], tensors=["weight"]),
            )
            reader = GGUFReader(output)
            self.assertIn("general.architecture", reader.fields)
            del reader
            gc.collect()

    def test_duplicate_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = ArtifactBuilder(str(Path(directory) / "model.gguf"), "test_tts")
            with self.assertRaisesRegex(ValueError, "Duplicate GGUF metadata"):
                builder.add_string("general.architecture", "duplicate")
            data = np.ones(1, dtype=np.float32)
            builder.add_tensor("weight", data, GGMLQuantizationType.F32)
            with self.assertRaisesRegex(ValueError, "Duplicate GGUF tensor"):
                builder.add_tensor("weight", data, GGMLQuantizationType.F32)
            builder.abort()

    def test_tensor_names_must_fit_ggml(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = ArtifactBuilder(str(Path(directory) / "model.gguf"), "test_tts")
            data = np.ones(1, dtype=np.float32)
            builder.add_tensor("x" * 63, data, GGMLQuantizationType.F32)
            with self.assertRaisesRegex(ValueError, "exceeds 63 UTF-8 bytes"):
                builder.add_tensor("x" * 64, data, GGMLQuantizationType.F32)
            builder.abort()

    def test_layout_metadata_is_generic_and_flat(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "model.gguf")
            builder = ArtifactBuilder(output, "test_model")
            builder.add_tensor("conv.weight", np.ones((32, 3, 64), dtype=np.float16), GGMLQuantizationType.F16)
            write_layout_metadata(builder, {"conv.weight": Layout.permuted((1, 0, 2))})
            builder.write()

            reader = GGUFReader(output)
            self.assertIn(LAYOUT_NAMES_KEY, reader.fields)
            self.assertIn(LAYOUT_OFFSETS_KEY, reader.fields)
            self.assertIn(LAYOUT_AXES_KEY, reader.fields)
            del reader
            gc.collect()

    def test_uncommitted_builder_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.gguf"
            output.write_bytes(b"existing")
            builder = ArtifactBuilder(str(output), "test_tts")
            builder.add_tensor("weight", np.ones(1, dtype=np.float32), GGMLQuantizationType.F32)
            builder.abort()
            self.assertEqual(output.read_bytes(), b"existing")
            self.assertEqual(list(Path(directory).glob(".model-export-*.gguf")), [])


if __name__ == "__main__":
    unittest.main()
