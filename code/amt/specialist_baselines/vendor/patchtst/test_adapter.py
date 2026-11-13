"""Small CPU contracts; run directly, or discover with this directory as start."""

import hashlib
import json
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from patchtst import PatchTSTAdapter


class PatchTSTAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def small_model(self):
        return PatchTSTAdapter(sequence_length=64, context_length=32, prediction_length=16,
                               width=8, depth=1, heads=2, patch_size=8, stride=4,
                               feedforward_dim=16, dropout=0).eval()

    def sample(self):
        torch.manual_seed(42)
        values = torch.randn(3, 4, 64)
        mask = torch.ones_like(values)
        for row, horizon in enumerate((1, 8, 16)):
            mask[row, :, -horizon:] = 0
        return values, mask

    def test_forward_backward_and_observed_preservation(self):
        model = self.small_model()
        values, mask = self.sample()
        out = model(values, mask)["prediction"]
        self.assertEqual(out.shape, values.shape)
        self.assertTrue(torch.equal(out[mask.bool()], values[mask.bool()]))
        ((out-values).square() * (1-mask)).sum().backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)

    def test_hidden_nan_values_and_dict_interface_do_not_leak(self):
        model = self.small_model()
        values, mask = self.sample()
        with torch.no_grad():
            expected = model(values, mask)["prediction"]
            poisoned = values.clone()
            poisoned[~mask.bool()] = float("nan")
            actual = model({"input": poisoned, "observed_mask": mask})["prediction"]
        self.assertTrue(torch.equal(expected, actual))

    def test_latest_context_and_first_horizon_alignment(self):
        class ForecastPositions(nn.Module):
            def forward(self, context):
                self.context = context.clone()
                return context[..., -1:] + torch.arange(16)[None, None, :] + 100
        model = self.small_model()
        probe = ForecastPositions()
        model.backbone = probe
        values = torch.arange(64).float()[None, None, :].expand(3, 4, -1).clone()
        _, mask = self.sample()
        out = model(values, mask)["prediction"]
        for row, horizon in enumerate((1, 8, 16)):
            start = 64-horizon
            expected_context = torch.arange(start-32, start).float().expand(4, -1)
            self.assertTrue(torch.equal(probe.context[row], expected_context))
            expected_future = start-1 + torch.arange(horizon).float() + 100
            self.assertTrue(torch.equal(out[row, :, start:], expected_future.expand(4, -1)))

    def test_all_seven_frozen_horizons(self):
        model = PatchTSTAdapter(width=8, depth=1, heads=2, feedforward_dim=16, dropout=0).eval()
        horizons = (10, 24, 32, 36, 64, 128, 256)
        values = torch.randn(len(horizons), 4, 2048)
        mask = torch.ones_like(values)
        for row, horizon in enumerate(horizons):
            mask[row, :, -horizon:] = 0
        with torch.no_grad():
            out = model(values, mask)["prediction"]
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(out.shape, values.shape)

    def test_invalid_masks_and_context_fail(self):
        model = self.small_model()
        values, mask = self.sample()
        variants = []
        bad = mask.clone(); bad[0, 0, -2] = 0; variants.append(bad)
        bad = mask.clone(); bad[0, :, 5] = 0; variants.append(bad)
        bad = mask.clone(); bad[0, :, 0] = 0.5; variants.append(bad)
        bad = mask.clone(); bad[0] = 1; variants.append(bad)
        bad = mask.clone(); bad[0, :, -17:] = 0; variants.append(bad)
        for bad in variants:
            with self.subTest(mask=bad[0, 0].tolist()), self.assertRaises(ValueError):
                model(values, bad)
        short_context = PatchTSTAdapter(sequence_length=64, context_length=60, prediction_length=16,
                                        width=8, depth=1, heads=2, patch_size=8, stride=4,
                                        feedforward_dim=16, dropout=0)
        with self.assertRaises(ValueError):
            short_context(values, mask)

    def test_vendored_source_hashes(self):
        root = Path(__file__).resolve().parent
        metadata = json.loads((root/"UPSTREAM.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["commit"], "204c21efe0b39603ad6e2ca640ef5896646ab1a9")
        for record in metadata["files"]:
            actual = hashlib.sha256((root/record["file"]).read_bytes()).hexdigest()
            self.assertEqual(actual, record["vendored_sha256"])


if __name__ == "__main__":
    unittest.main()
