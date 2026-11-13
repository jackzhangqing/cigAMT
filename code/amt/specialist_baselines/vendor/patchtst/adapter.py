"""Adapt the pinned, author-provided PatchTST backbone to frozen AMT windows.

The neural architecture in PatchTST_backbone.py is the author's implementation.
This adapter is not an official AMT experiment or an official fixed-horizon
paper reproduction. It supplies each example's latest fixed-length observed
prefix context to a shared maximum-horizon head, then places its first h outputs
at the actual missing suffix. Hidden input values and clean targets are never
read as context. See UPSTREAM.json and ADAPTER_NOTES.txt for provenance/protocol.
"""

from typing import Optional

import torch
from torch import nn

from .PatchTST_backbone import PatchTST_backbone


class PatchTSTAdapter(nn.Module):
    """Official channel-independent PatchTST with a variable-suffix task wrapper.

    Inputs are float tensors [B,4,T], plus binary observed_mask of the same
    shape, or a dictionary with ``input`` and ``observed_mask``. Each example
    must have a shared four-channel, contiguous observed prefix and a nonempty
    missing suffix. Contexts may differ in endpoint across a mixed-horizon batch.

    ``prediction`` has [B,4,T] shape. Observed values are copied for complete
    signal visualization; training and primary metrics must use missing points
    only. Only the first h predictions from the maximum-horizon head are used.
    """

    def __init__(
        self,
        sequence_length: int = 2048,
        context_length: int = 1792,
        prediction_length: int = 256,
        width: int = 128,
        depth: int = 3,
        heads: int = 4,
        patch_size: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        feedforward_dim: int = 256,
        attn_dropout: float = 0.0,
        head_dropout: float = 0.0,
        fc_dropout: float = 0.0,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        individual: bool = False,
        norm: str = "BatchNorm",
        padding_patch: Optional[str] = "end",
        res_attention: bool = True,
        pre_norm: bool = False,
        position_encoding: str = "zeros",
        learn_pe: bool = True,
    ):
        super().__init__()
        positive = dict(sequence_length=sequence_length, context_length=context_length,
                        prediction_length=prediction_length, width=width, depth=depth,
                        heads=heads, patch_size=patch_size, stride=stride,
                        feedforward_dim=feedforward_dim)
        for key, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if patch_size > context_length:
            raise ValueError("patch_size cannot exceed context_length")
        if context_length >= sequence_length:
            raise ValueError("context_length must leave at least one forecast point")
        if prediction_length >= sequence_length:
            raise ValueError("prediction_length must be smaller than sequence_length")
        if padding_patch not in (None, "end"):
            raise ValueError("padding_patch must be None or 'end'")
        for key, value in dict(dropout=dropout, attn_dropout=attn_dropout,
                               head_dropout=head_dropout, fc_dropout=fc_dropout).items():
            if not 0 <= value < 1:
                raise ValueError(f"{key} must be in [0,1)")
        if norm.lower() not in ("batchnorm", "layernorm"):
            raise ValueError("norm must be 'BatchNorm' or 'LayerNorm'")
        self.sequence_length = sequence_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.backbone = PatchTST_backbone(
            c_in=4, context_window=context_length, target_window=prediction_length,
            patch_len=patch_size, stride=stride, max_seq_len=max(1024, context_length),
            n_layers=depth, d_model=width, n_heads=heads, d_ff=feedforward_dim,
            dropout=dropout, attn_dropout=attn_dropout, fc_dropout=fc_dropout,
            head_dropout=head_dropout, norm=norm, padding_patch=padding_patch,
            individual=individual, revin=revin, affine=affine,
            subtract_last=subtract_last, res_attention=res_attention,
            pre_norm=pre_norm, pe=position_encoding, learn_pe=learn_pe,
            store_attn=False,
        )

    def forward(self, values, observed_mask=None):
        if isinstance(values, dict):
            observed_mask = values.get("observed_mask", observed_mask)
            values = values["input"]
        if not isinstance(values, torch.Tensor) or not values.is_floating_point():
            raise TypeError("input must be a floating-point torch.Tensor")
        if values.ndim != 3 or values.shape[1:] != (4, self.sequence_length):
            raise ValueError("input must have shape [batch,4,sequence_length]")
        if values.shape[0] < 1:
            raise ValueError("empty batches are unsupported")
        if not isinstance(observed_mask, torch.Tensor) or observed_mask.shape != values.shape:
            raise ValueError("observed_mask must match input's [batch,4,time] shape")
        observed_mask = observed_mask.to(device=values.device)
        if not torch.all((observed_mask == 0) | (observed_mask == 1)):
            raise ValueError("observed_mask must be binary and finite")
        mask = observed_mask.to(torch.bool)
        shared = mask[:, :1, :].expand_as(mask)
        if not torch.equal(mask, shared):
            raise ValueError("forecasting requires the same missing suffix in all four channels")
        # No 0 -> 1 transitions are permitted: visible points form a prefix.
        if torch.any(mask[:, 0, 1:] & ~mask[:, 0, :-1]):
            raise ValueError("forecasting requires a contiguous observed prefix, without interior holes")
        visible = mask[:, 0, :].sum(dim=-1)
        horizon = self.sequence_length - visible
        if torch.any(horizon < 1) or torch.any(horizon > self.prediction_length):
            raise ValueError("missing horizon must be between 1 and prediction_length")
        if torch.any(visible < self.context_length):
            raise ValueError("observed prefix is shorter than context_length")
        # torch.where, rather than multiplication, also removes NaN/Inf that may
        # accidentally be present exclusively at hidden positions.
        safe = torch.where(mask, values, torch.zeros((), dtype=values.dtype, device=values.device))
        if not torch.isfinite(safe).all():
            raise ValueError("visible input values must be finite")
        context_positions = visible[:, None] - self.context_length + torch.arange(
            self.context_length, device=values.device)[None, :]
        context = safe.gather(2, context_positions[:, None, :].expand(-1, 4, -1))
        forecast = self.backbone(context)
        relative = torch.arange(self.sequence_length, device=values.device)[None, :] - visible[:, None]
        relative = relative.clamp(min=0, max=self.prediction_length-1)
        aligned = forecast.gather(2, relative[:, None, :].expand(-1, 4, -1))
        return {"prediction": torch.where(mask, safe, aligned)}
