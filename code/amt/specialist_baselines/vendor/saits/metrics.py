"""Single upstream utility needed by the official SAITS class.

Extracted from WenjieDu/SAITS/modeling/utils.py at commit
660b87f19c1277065f314f24134f646229e89ca9. Copyright 2022 Wenjie Du,
MIT license (see LICENSE). Kept separately to avoid importing unrelated
plotting and sklearn dependencies. The benchmark adapter calls impute(),
so its training loss is computed by the benchmark engine, not this helper.
"""
import torch


def masked_mae_cal(inputs, target, mask):
    """calculate Mean Absolute Error (upstream expression unchanged)."""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)
