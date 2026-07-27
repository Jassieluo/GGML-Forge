import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
PROCESS = ROOT / "scripts" / "conversion" / "categories" / "visual_perception" / \
    "semantic_segmentation" / "providers" / "lraspp" / "process.py"
spec = importlib.util.spec_from_file_location("lraspp_process", PROCESS)
lraspp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lraspp)


class LrasppConversionTest(unittest.TestCase):
    def test_batch_norm_is_fused_into_convolution(self):
        state = {
            "conv.weight": torch.tensor([[[[2.0]]], [[[3.0]]]]),
            "bn.weight": torch.tensor([4.0, 5.0]),
            "bn.bias": torch.tensor([6.0, 7.0]),
            "bn.running_mean": torch.tensor([8.0, 9.0]),
            "bn.running_var": torch.tensor([15.0, 24.0]),
        }
        weight, bias = lraspp.fuse_conv_bn(state, "conv", "bn", 1.0)
        np.testing.assert_allclose(weight[:, 0, 0, 0], [2.0, 3.0], rtol=1e-6)
        np.testing.assert_allclose(bias, [-2.0, -2.0], rtol=1e-6)

    def test_pascal_palette_is_stable(self):
        palette = lraspp.pascal_palette(3)
        self.assertEqual(palette, [0, 0, 0, 128, 0, 0, 0, 128, 0])


if __name__ == "__main__":
    unittest.main()
