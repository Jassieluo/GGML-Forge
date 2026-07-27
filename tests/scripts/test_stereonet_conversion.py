import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
PROCESS = ROOT / "scripts/conversion/categories/visual_perception/depth_estimation/providers/stereonet/process.py"
SPEC = importlib.util.spec_from_file_location("stereonet_conversion", PROCESS)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StereoNetConversionTest(unittest.TestCase):
  def test_fuse_conv_bn_3d(self):
    state = {
        "conv.weight": torch.ones(2, 3, 3, 3, 3),
        "conv.bias": torch.tensor([1.0, -1.0]),
        "bn.weight": torch.tensor([2.0, 3.0]),
        "bn.bias": torch.tensor([0.5, -0.5]),
        "bn.running_mean": torch.tensor([0.25, -0.25]),
        "bn.running_var": torch.tensor([4.0, 9.0]),
    }
    weight, bias = MODULE.fuse_conv_bn(state, "conv", "bn", 3, eps=0.0)
    self.assertEqual(weight.shape, (2, 3, 3, 3, 3))
    self.assertTrue(torch.allclose(torch.from_numpy(weight[:, 0, 0, 0, 0]), torch.ones(2)))
    self.assertTrue(torch.allclose(torch.from_numpy(bias), torch.tensor([1.25, -1.25])))


  def test_canonical_parameter_contract(self):
    checkpoint_path = ROOT / "scratch/stereonet.ckpt"
    if not checkpoint_path.is_file():
        self.skipTest("optional public StereoNet checkpoint is not present")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    names = {name for name, _ in MODULE.canonical_tensors(checkpoint["state_dict"])}
    self.assertEqual(len(names), 126)
    self.assertIn("features.downsample.0.weight", names)
    self.assertIn("cost.layers.0.weight", names)
    self.assertIn("refiners.2.output.bias", names)


if __name__ == "__main__":
    unittest.main()
