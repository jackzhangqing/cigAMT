"""Focused vendor verification; no data download, PyPOTS or model training.

Run from the new standalone specialist benchmark root:
python -B -m unittest vendor.saits.test_saits -v

From the original development staging root, use:
python -B -m unittest specialist_baselines.vendor.saits.test_saits -v
"""
import hashlib
import json
from pathlib import Path
import unittest

import torch

from .adapter import SAITSAdapter
from . import layers
from .metrics import masked_mae_cal


ROOT = Path(__file__).resolve().parent


def upstream_reference():
    """Load preserved source, replacing imports only in an isolated namespace."""
    source = (ROOT / "upstream/modeling/saits.py").read_text(encoding="utf-8")
    source = source.replace("from modeling.layers import *", "")
    source = source.replace("from modeling.utils import masked_mae_cal", "")
    namespace = dict(vars(layers))
    namespace["masked_mae_cal"] = masked_mae_cal
    exec(compile(source, str(ROOT / "upstream/modeling/saits.py"), "exec"), namespace)
    return namespace["SAITS"]


class SAITSVendorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def tiny(self, length=17, **kwargs):
        return SAITSAdapter(sequence_length=length, d_model=8, d_inner=16,
                            n_layers=2, n_heads=2, dropout=0, **kwargs)

    def test_upstream_files_and_local_import_only_patch_are_verifiable(self):
        provenance = json.loads((ROOT / "UPSTREAM.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["commit"], SAITSAdapter.upstream_commit)
        for item in provenance["files"]:
            raw = (ROOT / item["local_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), item["sha256"])
        self.assertEqual((ROOT / "layers.py").read_bytes(),
                         (ROOT / "upstream/modeling/layers.py").read_bytes())
        expected = (ROOT / "upstream/modeling/saits.py").read_text(encoding="utf-8")
        expected = expected.replace("from modeling.layers import *", "from .layers import *")
        expected = expected.replace("from modeling.utils import masked_mae_cal", "from .metrics import masked_mae_cal")
        self.assertEqual((ROOT / "official_model.py").read_text(encoding="utf-8"), expected)
        self.assertEqual((ROOT / "LICENSE").read_bytes(), (ROOT / "upstream/LICENSE").read_bytes())

    def test_forward_and_gradients_match_preserved_official_architecture(self):
        torch.manual_seed(4)
        model = self.tiny().eval()
        reference = upstream_reference()(
            n_groups=2, n_group_inner_layers=1, d_time=17, d_feature=4,
            d_model=8, d_inner=16, n_head=2, d_k=4, d_v=4, dropout=0,
            input_with_mask=True, param_sharing_strategy="inner_group", MIT=False,
            diagonal_attention_mask=True, device="cpu").eval()
        reference.load_state_dict(model.official_model.state_dict(), strict=True)
        x = torch.randn(2, 4, 17)
        mask = (torch.rand_like(x) > .4).float()
        hidden = torch.where(mask.bool(), x, torch.zeros_like(x))
        actual = model(x, mask)["prediction"]
        expected, _ = reference.impute({"X": hidden.transpose(1, 2),
                                      "missing_mask": mask.transpose(1, 2)})
        expected = expected.transpose(1, 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        target = torch.randn_like(x)
        loss_mask = 1 - mask
        (((actual-target).square()*loss_mask).sum()/loss_mask.sum()).backward()
        (((expected-target).square()*loss_mask).sum()/loss_mask.sum()).backward()
        for (name, parameter), (ref_name, ref_parameter) in zip(
                model.official_model.named_parameters(), reference.named_parameters()):
            self.assertEqual(name, ref_name)
            torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=0, atol=0)

    def test_hidden_nan_infinity_and_oracle_fields_cannot_enter_prediction(self):
        model = self.tiny().eval()
        x = torch.randn(2, 4, 17)
        mask = torch.ones_like(x)
        mask[..., 5:13] = 0
        with torch.inference_mode():
            before = model({"input": x, "observed_mask": mask,
                            "target": torch.randn_like(x), "loss_mask": 1-mask})["prediction"]
            for filler in (float("nan"), float("inf"), -1e8):
                changed = x.clone()
                changed[mask == 0] = filler
                after = model({"input": changed, "observed_mask": mask,
                               "target": torch.full_like(x, 1e8),
                               "noise_mask": torch.ones_like(x)})["prediction"]
                torch.testing.assert_close(before, after, rtol=0, atol=0)
            torch.testing.assert_close(before[mask.bool()], x[mask.bool()], rtol=0, atol=0)

    def test_all_observed_all_missing_and_checkpoint_roundtrip(self):
        model = self.tiny().eval()
        x = torch.randn(2, 4, 17)
        with torch.inference_mode():
            torch.testing.assert_close(model(x, torch.ones_like(x))["prediction"], x, rtol=0, atol=0)
            masked = model(x, torch.zeros_like(x))["prediction"]
        self.assertTrue(torch.isfinite(masked).all())
        restored = SAITSAdapter(**model.architecture_options).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.inference_mode():
            torch.testing.assert_close(masked, restored(x, torch.zeros_like(x))["prediction"], rtol=0, atol=0)

    def test_invalid_masks_and_observations_fail_instead_of_silent_leakage(self):
        model = self.tiny()
        x = torch.randn(1, 4, 17)
        mask = torch.ones_like(x)
        invalid_inputs = [lambda: model(x), lambda: model(x, mask[..., :-1]),
                          lambda: model(x[..., :-1], mask[..., :-1]),
                          lambda: model(x[:, :3], mask[:, :3]),
                          lambda: model(x, mask * .5),
                          lambda: model(x, torch.full_like(mask, float("nan"))),
                          lambda: model(torch.full_like(x, float("nan")), mask),
                          lambda: model({"input": x, "observed_mask": mask}, mask)]
        for operation in invalid_inputs:
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                operation()
        for arguments in ({"d_model": 7, "n_heads": 2}, {"sequence_length": 1},
                          {"channels": 3}, {"n_layers": 0}, {"dropout": 1}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                SAITSAdapter(**arguments)

    def test_native_2048_length_four_channel_forward_and_backward(self):
        # Physical batch one checks full-resolution native architecture without
        # pretending that batch 256 is feasible or measuring GPU throughput.
        model = SAITSAdapter(dropout=0)
        x = torch.randn(1, 4, 2048)
        mask = torch.ones_like(x)
        mask[..., 900:1156] = 0
        output = model(x, mask)["prediction"]
        self.assertEqual(tuple(output.shape), (1, 4, 2048))
        target = torch.randn_like(output)
        ((output-target).square()*(1-mask)).sum().div((1-mask).sum()).backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                            if p.grad is not None))
        self.assertGreater(sum(p.grad is not None for p in model.parameters()), 0)

    def test_cpu_bfloat16_autocast_has_finite_outputs_and_gradients(self):
        # CPU dtype contract only; this is not a CUDA speed/memory claim.
        model = self.tiny(length=33)
        x = torch.randn(2, 4, 33)
        mask = torch.ones_like(x)
        mask[..., 10:20] = 0
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = model(x, mask)["prediction"]
            loss = (result.float().square() * (1-mask)).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                            if p.grad is not None))
        with self.assertRaisesRegex(ValueError, "bf16 or fp32"):
            model(x.half(), mask)


if __name__ == "__main__":
    unittest.main()
