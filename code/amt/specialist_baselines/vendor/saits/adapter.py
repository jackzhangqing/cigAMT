"""Official SAITS architecture adapted to benchmark [B,4,T] tensors.

This uses the pinned upstream impute() path without changing attention or
the two DMSA blocks. The external benchmark trains missing-target MSE;
upstream ORT/MIT multi-branch MAE training is deliberately not invoked.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .official_model import SAITS


class SAITSAdapter(nn.Module):
    """Two-block official SAITS model for interpolation and imputation.

    Every attention layer works at the full time resolution (no patching or
    downsampling). Batch size 256 is an effective optimization batch, not a
    safe physical batch at T=2048: use external microbatch accumulation.

    Inputs are only incomplete values and their observation mask. Ground
    truth, missing-target masks, or noise labels are never passed upstream.
    """

    upstream_commit = "660b87f19c1277065f314f24134f646229e89ca9"
    supported_tasks = ("interpolation", "imputation")
    training_protocol = "official SAITS architecture; benchmark missing-target MSE; no ORT/MIT joint MAE"
    attention_backend = "official_explicit_attention"

    def __init__(self, sequence_length=2048, channels=4, d_model=64, d_inner=128,
                 n_layers=2, n_heads=4, dropout=0.1,
                 diagonal_attention_mask=True):
        super().__init__()
        for name, value in (("sequence_length", sequence_length), ("channels", channels),
                            ("d_model", d_model), ("d_inner", d_inner),
                            ("n_layers", n_layers), ("n_heads", n_heads)):
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError("%s must be a positive integer" % name)
        if int(sequence_length) < 2:
            raise ValueError("SAITS diagonal attention requires sequence_length >= 2")
        if int(channels) != 4:
            raise ValueError("This benchmark adapter expects exactly four channels: Ex/Ey/Hx/Hy")
        if int(d_model) % int(n_heads):
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= float(dropout) < 1:
            raise ValueError("dropout must be in [0,1)")
        if not isinstance(diagonal_attention_mask, bool):
            raise ValueError("diagonal_attention_mask must be boolean")
        self.sequence_length = int(sequence_length)
        self.channels = int(channels)
        self.architecture_options = dict(sequence_length=self.sequence_length,
            channels=self.channels, d_model=int(d_model), d_inner=int(d_inner),
            n_layers=int(n_layers), n_heads=int(n_heads), dropout=float(dropout),
            diagonal_attention_mask=diagonal_attention_mask)
        self.official_model = SAITS(
            n_groups=int(n_layers), n_group_inner_layers=1,
            d_time=self.sequence_length, d_feature=self.channels,
            d_model=int(d_model), d_inner=int(d_inner), n_head=int(n_heads),
            d_k=int(d_model) // int(n_heads), d_v=int(d_model) // int(n_heads),
            dropout=float(dropout), input_with_mask=True,
            param_sharing_strategy="inner_group", MIT=False,
            diagonal_attention_mask=diagonal_attention_mask, device="cpu")

    def forward(self, values, observed_mask=None):
        if isinstance(values, Mapping):
            if observed_mask is not None:
                raise ValueError("Pass observed_mask in the input mapping or as an argument, not both")
            if "input" not in values or "observed_mask" not in values:
                raise ValueError("SAITS requires input and observed_mask")
            observed_mask = values["observed_mask"]
            values = values["input"]
        if not torch.is_tensor(values) or not torch.is_tensor(observed_mask):
            raise ValueError("SAITS requires tensor values and observed_mask")
        if values.ndim != 3 or tuple(values.shape[1:]) != (self.channels, self.sequence_length):
            raise ValueError("SAITS input must have shape [B, %d, %d]" %
                             (self.channels, self.sequence_length))
        if tuple(observed_mask.shape) != tuple(values.shape):
            raise ValueError("observed_mask shape must exactly match values")
        if not torch.is_floating_point(values):
            raise ValueError("SAITS values must be floating point")
        if values.dtype == torch.float16 or (values.device.type == "cuda" and
                torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16):
            raise ValueError("Pinned official SAITS uses a -1e9 attention mask; use bf16 or fp32, not fp16")
        if observed_mask.device != values.device:
            raise ValueError("values and observed_mask must be on the same device")
        if not bool(torch.isfinite(observed_mask).all()) or not bool(
                ((observed_mask == 0) | (observed_mask == 1)).all()):
            raise ValueError("observed_mask must contain only finite 0/1 values")
        mask_bool = observed_mask.bool()
        # Multiplication does not hide NaN or infinity. torch.where makes
        # hidden values irrelevant even when imported sources use NaN gaps.
        x = torch.where(mask_bool, values, torch.zeros_like(values))
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Observed input values must be finite")
        masks = mask_bool.to(dtype=x.dtype)
        # Upstream EncoderLayer stores a plain device attribute. Keep its
        # diagonal mask on the actual input device after .to()/checkpoint load.
        for layers in (self.official_model.layer_stack_for_first_block,
                       self.official_model.layer_stack_for_second_block):
            for layer in layers:
                layer.device = x.device
        imputed, _ = self.official_model.impute(
            {"X": x.transpose(1, 2), "missing_mask": masks.transpose(1, 2)})
        prediction = imputed.transpose(1, 2).contiguous()
        # Upstream impute already preserves observations. The where operation
        # explicitly guarantees their exact values under mixed precision.
        prediction = torch.where(mask_bool, values, prediction)
        return {"prediction": prediction}
