import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PROVIDER_DIR = ROOT / "scripts" / "categories" / "tts" / "providers" / "gpt_sovits"
sys.path.insert(0, str(PROVIDER_DIR))
spec = importlib.util.spec_from_file_location("gpt_sovits_process", PROVIDER_DIR / "process.py")
process = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(process)


class ProviderCheckpointTest(unittest.TestCase):
    def test_checkpoint_names_map_to_canonical_module_paths(self):
        self.assertEqual(
            process.map_bert_key("bert.encoder.layer.3.attention.self.query.weight"),
            "encoder.layers.3.self_attn.q_proj.weight",
        )
        self.assertEqual(
            process.map_hubert_key("feature_extractor.conv_layers.2.conv.weight"),
            "feature_extractor.layers.2.weight",
        )
        self.assertEqual(
            process.map_t2s_key("h.layers.1.linear1.weight"),
            "decoder.layers.1.ffn.w1.weight",
        )
        self.assertIsNone(process.map_bert_key("bert.encoder.layer.22.output.dense.weight"))

    def test_vits_checkpoint_names_map_to_canonical_module_paths(self):
        self.assertEqual(
            process.map_vits_key("enc_p.ssl_proj.weight", "v3"),
            "semantic.ssl_projection.weight",
        )
        self.assertEqual(
            process.map_vits_key("dec.ups.2.0.weight", "v3"),
            "generator.upsample.2.weight",
        )
        self.assertEqual(
            process.map_vits_key("cfm.estimator.transformer_blocks.3.attn.to_q.weight", "v3"),
            "estimator.transformer_blocks.3.attn.q_proj.weight",
        )
        self.assertIsNone(process.map_vits_key("linear_mel.weight", "v3"))

    def test_loads_legacy_utils_hparams_without_source_repository(self):
        legacy_utils = types.ModuleType("utils")
        legacy_hparams = type("HParams", (), {"__module__": "utils"})
        legacy_utils.HParams = legacy_hparams
        sys.modules["utils"] = legacy_utils
        try:
            config = legacy_hparams()
            config.model = legacy_hparams()
            config.model.n_heads = 4
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "legacy.pth"
                torch.save({"weight": {"value": torch.ones(1)}, "config": config}, path)
                del sys.modules["utils"]

                state_dict, checkpoint = process.load_checkpoint_with_metadata(str(path))

            self.assertEqual(state_dict["value"].item(), 1.0)
            self.assertEqual(checkpoint["config"].model.n_heads, 4)
            self.assertNotIn("utils", sys.modules)
        finally:
            sys.modules.pop("utils", None)

    def test_folds_weight_norm_along_output_dimension(self):
        v = torch.arange(1, 13, dtype=torch.float32).reshape(3, 2, 2)
        g = torch.tensor([2.0, 3.0, 4.0]).reshape(3, 1, 1)
        state = {"layer.weight_g": g, "layer.weight_v": v}

        self.assertEqual(process.fold_weight_norm_parameters(state), 1)

        expected = g * v / torch.linalg.vector_norm(v, dim=(1, 2), keepdim=True)
        torch.testing.assert_close(state["layer.weight"], expected)
        self.assertNotIn("layer.weight_g", state)
        self.assertNotIn("layer.weight_v", state)

    def test_folds_hubert_weight_norm_along_kernel_dimension(self):
        v = torch.arange(1, 25, dtype=torch.float32).reshape(3, 2, 4)
        g = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 4)
        state = {"pos_conv.weight_g": g, "pos_conv.weight_v": v}

        self.assertEqual(process.fold_weight_norm_parameters(state), 1)

        expected = g * v / torch.linalg.vector_norm(v, dim=(0, 1), keepdim=True)
        torch.testing.assert_close(state["pos_conv.weight"], expected)

    def test_materializes_snake_parameters_from_log_space(self):
        state = {
            "dec.resblocks.0.activations.0.act.alpha": torch.log(torch.tensor([2.0, 3.0])),
            "dec.resblocks.0.activations.0.act.beta": torch.log(torch.tensor([4.0, 5.0])),
            "unrelated.alpha": torch.tensor([7.0]),
        }

        self.assertEqual(process.exponentiate_snake_parameters(state), 2)
        torch.testing.assert_close(
            state["dec.resblocks.0.activations.0.act.alpha"], torch.tensor([2.0, 3.0])
        )
        torch.testing.assert_close(state["unrelated.alpha"], torch.tensor([7.0]))

    def test_materializes_alias_free_filters_per_channel(self):
        state = {
            "block.act.alpha": torch.ones(3),
            "block.upsample.filter": torch.tensor([1.0, 2.0]),
            "block.downsample.lowpass.filter": torch.tensor([3.0, 4.0]),
        }

        self.assertEqual(process.materialize_alias_free_filters(state), 2)
        self.assertEqual(tuple(state["block.upsample.filter_repeated"].shape), (3, 1, 2))
        self.assertEqual(tuple(state["block.downsample.filter_repeated"].shape), (3, 1, 2))
        self.assertNotIn("block.upsample.filter", state)
        self.assertNotIn("block.downsample.lowpass.filter", state)


if __name__ == "__main__":
    unittest.main()
