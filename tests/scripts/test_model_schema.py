import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from conversion.common.exporter import _tensor_spec, export_model
from conversion.common.model_schema import ModelSchema
from conversion.common.schema import ModelDefinition, Parameter


class ModelSchemaTest(unittest.TestCase):
    @staticmethod
    def schema() -> ModelSchema:
        return ModelSchema.from_json(json.dumps({"parameters": [
            {
                "path": "embedding.weight",
                "required": True,
                "usage": "embedding_weight",
                "direct_storage_types": ["Q4_0", "Q8_0", "F16", "F32"],
                "quantized_layout": "native",
            },
            {
                "path": "conv.weight",
                "required": True,
                "usage": "conv1d_weight",
                "direct_storage_types": ["Q4_K", "Q4_0", "Q8_0", "F16", "F32"],
                "quantized_layout": "channel_rows",
                "logical_shape": [3, 32, 2],
            },
        ]}))

    def test_cpp_contract_controls_quantization(self):
        schema = self.schema()
        definition = ModelDefinition("schema_test", schema)
        embedding = Parameter("embedding.weight", np.zeros((2, 32), dtype=np.float32))
        convolution = Parameter("conv.weight", np.zeros((2, 32, 3), dtype=np.float32))
        definition.parameter(embedding)
        definition.parameter(convolution)

        embedding_spec = _tensor_spec(definition.parameters[0])
        conv_spec = _tensor_spec(definition.parameters[1])
        self.assertNotIn("Q4_K", embedding_spec.allowed_types)
        self.assertEqual(conv_spec.quant_shape, (32, 3, 2))
        self.assertEqual(conv_spec.transform, "channel_rows")
        schema.validate_complete(parameter.name for parameter in definition.parameters)

    def test_provider_cannot_add_unknown_parameter(self):
        definition = ModelDefinition("schema_test", self.schema())
        with self.assertRaisesRegex(ValueError, "not declared"):
            definition.parameter(Parameter("invented.weight", np.ones((2, 2), dtype=np.float32)))

    def test_required_completion_is_strict(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            self.schema().validate_complete(["embedding.weight"])

    def test_quantized_export_requires_cpp_schema(self):
        definition = ModelDefinition("schema_test")
        definition.parameter(Parameter("weight", np.ones((32, 2), dtype=np.float32)))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requires a schema"):
                export_model(definition, f"{directory}/model.gguf", "Q4_0")


if __name__ == "__main__":
    unittest.main()
