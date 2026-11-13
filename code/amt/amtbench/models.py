"""Self-contained neural baselines for the seven AMT benchmark tasks.

These are transparent benchmark adaptations, not the authors' official PatchTST,
DLinear, FNO, or AMT MaskFormer implementations and not pretrained models.
All networks consume input waveforms only. Missing-data tasks additionally
consume observed_mask; clean targets and noise masks must never be model inputs.
"""
from typing import Dict, Optional
import math

import torch
from torch import nn
from torch.nn import functional as F


CLEAN_TASKS = ("forecasting", "interpolation", "imputation")
TASKS = ("noise_localization", "noise_classification", "denoising",
         "frequency_denoising") + CLEAN_TASKS
GENERAL_MODELS = ("linear", "tcn", "unet1d", "bigru", "patch_transformer", "spectral_fno",
                  "amt_maskformer")


def supported_models(task):
    """Model identifiers supported for this task; invalid task names fail early."""
    if task not in TASKS:
        raise ValueError("Unknown task %r; expected one of %s" % (task, TASKS))
    return GENERAL_MODELS + (("dlinear",) if task == "forecasting" else ())


def _features(x, observed_mask, task):
    if isinstance(x, dict):
        observed_mask = x.get("observed_mask", observed_mask)
        x = x["input"]
    if task in CLEAN_TASKS and x.ndim == 3 and x.shape[1] == 8 and observed_mask is None:
        x, observed_mask = x[:, :4], x[:, 4:]
    if x.ndim != 3 or x.shape[1] != 4:
        raise ValueError("input must be [batch,4,time], or [batch,8,time] signal+mask for missing-data tasks")
    if task in CLEAN_TASKS:
        if observed_mask is None or observed_mask.shape != x.shape:
            raise ValueError("Missing-data tasks require observed_mask with input's [B,4,T] shape")
        observed_mask = observed_mask.to(dtype=x.dtype, device=x.device)
        # Applying the mask here also protects against mistakenly populated gaps.
        return torch.cat((x * observed_mask, observed_mask), dim=1)
    return x


def _norm(width):
    return nn.GroupNorm(1, width)


class LinearBackbone(nn.Module):
    """A shared temporal linear map plus a pointwise channel projection."""
    def __init__(self, in_channels, width, sequence_length, **unused):
        super().__init__()
        self.sequence_length = sequence_length
        self.channel = nn.Conv1d(in_channels, width, 1)
        self.temporal = nn.Linear(sequence_length, sequence_length)

    def forward(self, x):
        if x.shape[-1] != self.sequence_length:
            raise ValueError("linear was built for sequence_length=%d" % self.sequence_length)
        # Apply the expensive shared time map to 4/8 input channels, then lift.
        return self.channel(self.temporal(x))


class TemporalResidual(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2*dilation, dilation=dilation),
            _norm(width), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            _norm(width), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.net(x)


class TCNBackbone(nn.Module):
    """Dilated residual temporal CNN; uses only observed, masked input context."""
    def __init__(self, in_channels, width, depth, dropout, **unused):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(in_channels, width, 1), *[
            TemporalResidual(width, 2 ** (i % 10), dropout) for i in range(depth)])

    def forward(self, x):
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, padding=2), _norm(out_channels), nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1), _norm(out_channels),
            nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class UNetBackbone(nn.Module):
    """One-dimensional U-Net with skip connections and exact output alignment."""
    def __init__(self, in_channels, width, depth, dropout, **unused):
        super().__init__()
        self.encoders = nn.ModuleList()
        channels = in_channels
        widths = [width * min(2**i, 8) for i in range(depth+1)]
        for w in widths:
            self.encoders.append(ConvBlock(channels, w, dropout))
            channels = w
        self.decoders = nn.ModuleList([
            ConvBlock(widths[i+1]+widths[i], widths[i], dropout)
            for i in reversed(range(depth))])

    def forward(self, x):
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i < len(self.encoders)-1:
                skips.append(x)
                x = F.max_pool1d(x, 2, ceil_mode=True)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return x


class BiGRUBackbone(nn.Module):
    def __init__(self, in_channels, width, depth, dropout, **unused):
        super().__init__()
        self.gru = nn.GRU(in_channels, width, num_layers=depth, batch_first=True,
                          bidirectional=True, dropout=dropout if depth > 1 else 0.0)
        self.project = nn.Conv1d(2*width, width, 1)

    def forward(self, x):
        # cuDNN GRU bf16 support differs across server stacks; keep recurrence
        # in float32 while the surrounding task projection can use AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            y, _ = self.gru(x.float().transpose(1, 2))
        return self.project(y.transpose(1, 2))


class PatchTransformerBackbone(nn.Module):
    """Joint-channel patch Transformer, inspired by patch-based time-series work.

    Unlike official PatchTST this adaptation mixes all channels before attention,
    reconstructs every input position, and supports pointwise task heads.
    """
    def __init__(self, in_channels, width, depth, dropout, patch_size, heads, **unused):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads for patch_transformer")
        self.patch_size = patch_size
        self.width = width
        self.embedding = nn.Conv1d(in_channels, width, patch_size, stride=patch_size)
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=4*width,
                                           dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.decode = nn.Linear(width, width*patch_size)

    def forward(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, (-t) % self.patch_size))
        tokens = self.embedding(x).transpose(1, 2)
        n = tokens.shape[1]
        # Compute positions in float32 even under AMP, then cast to token dtype.
        pos = torch.arange(n, device=x.device, dtype=torch.float32).unsqueeze(1)
        rate = torch.exp(torch.arange(0, self.width, 2, device=x.device, dtype=torch.float32)
                         * (-math.log(10000.0) / self.width))
        angle = pos * rate.unsqueeze(0)
        pe = torch.zeros(n, self.width, device=x.device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(angle)
        pe[:, 1::2] = torch.cos(angle[:, :pe[:, 1::2].shape[1]])
        tokens = self.encoder(tokens + pe.to(tokens.dtype).unsqueeze(0))
        y = self.decode(tokens).reshape(x.shape[0], n, self.width, self.patch_size)
        return y.permute(0, 2, 1, 3).reshape(x.shape[0], self.width, -1)[..., :t]


def _position_encoding(length, dimension, device, dtype):
    """The original AMT reconstruction model's float32 sinusoidal encoding."""
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
                        * (-math.log(10000.0) / dimension))
    encoding = torch.zeros(length, dimension, device=device)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor[:encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


class AMTResidualConvBlock(nn.Module):
    """Faithful local refinement block from the user's AMT reconstruction code."""
    def __init__(self, channels, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2),
            nn.GroupNorm(math.gcd(8, channels), channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 5, padding=2))

    def forward(self, values):
        return F.gelu(values + self.block(values))


class AMTMaskFormer(nn.Module):
    """User's patch-attention/local-refinement AMT model with seven-task heads.

    This is the user's AMT time-series architecture, not the official vision
    MaskFormer segmentation model. With matching dimensions, reconstruction
    state_dict keys and computations match Denoising.models.AMTMaskFormer.
    Noise tasks receive a constant all-observed mask; noise labels never enter
    the network. Missing-data tasks use the supplied observed input mask.

    local_skip=False zeroes the local input skip (retaining tensor dimensions)
    for an explicit ablation; refinement_blocks=0 bypasses residual refinement.
    """
    def __init__(self, channels=4, d_model=192, patch_size=8, layers=4, heads=6,
                 feedforward_dim=768, dropout=0.1, refinement_blocks=2,
                 task="denoising", num_classes=11, local_skip=True):
        super().__init__()
        if task not in TASKS:
            raise ValueError("Unknown task %r" % task)
        if channels != 4:
            raise ValueError("This benchmark requires four channels: Ex,Ey,Hx,Hy")
        for name, value in (("d_model", d_model), ("patch_size", patch_size),
                            ("layers", layers), ("heads", heads),
                            ("feedforward_dim", feedforward_dim), ("num_classes", num_classes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("%s must be a positive integer" % name)
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        if (isinstance(refinement_blocks, bool) or not isinstance(refinement_blocks, int)
                or refinement_blocks < 0):
            raise ValueError("refinement_blocks must be a nonnegative integer")
        if not isinstance(local_skip, bool):
            raise ValueError("local_skip must be boolean")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.task = task
        self.patch_size = patch_size
        self.local_skip = local_skip
        self.patch_encoder = nn.Conv1d(2*channels, d_model, patch_size, stride=patch_size)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, feedforward_dim, dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(d_model))
        decoded_channels = max(32, d_model//2)
        self.patch_decoder = nn.ConvTranspose1d(d_model, decoded_channels, patch_size, stride=patch_size)
        self.local_stem = nn.Conv1d(decoded_channels+2*channels, decoded_channels, 7, padding=3)
        self.refinement = nn.Sequential(*[
            AMTResidualConvBlock(decoded_channels, dropout) for _ in range(refinement_blocks)])
        if task == "noise_classification":
            self.output_head = nn.Linear(decoded_channels, num_classes)
        else:
            out_channels = 1 if task == "noise_localization" else (8 if task == "frequency_denoising" else channels)
            self.output_head = nn.Conv1d(decoded_channels, out_channels, 1)

    def forward(self, values, observed_mask=None):
        if self.task in CLEAN_TASKS:
            conditioned = _features(values, observed_mask, self.task)
        else:
            signal = _features(values, None, self.task)
            # Ignore any supplied mask, including a noise region label in a
            # dictionary. Denoising/localization inputs are fully observed.
            conditioned = torch.cat((signal, torch.ones_like(signal)), dim=1)
        length = conditioned.shape[-1]
        if length < 1:
            raise ValueError("input must have a nonempty time dimension")
        if length % self.patch_size:
            conditioned = F.pad(conditioned, (0, self.patch_size-length % self.patch_size))
        tokens = self.patch_encoder(conditioned).transpose(1, 2)
        tokens = tokens + _position_encoding(
            tokens.shape[1], tokens.shape[2], tokens.device, tokens.dtype).unsqueeze(0)
        decoded = self.patch_decoder(self.transformer(tokens).transpose(1, 2))[..., :length]
        skip = conditioned[..., :length]
        if not self.local_skip:
            skip = torch.zeros_like(skip)
        local = self.local_stem(torch.cat((decoded, skip), dim=1))
        hidden = self.refinement(local)
        if self.task == "noise_classification":
            return {"logits": self.output_head(hidden.mean(dim=-1))}
        output = self.output_head(hidden)
        if self.task == "noise_localization":
            return {"logits": output}
        if self.task == "frequency_denoising":
            low, high = output[:, :4], output[:, 4:]
            return {"low": low, "high": high, "prediction": low+high}
        return {"prediction": output}


class SpectralBlock(nn.Module):
    """Low-mode spectral convolution plus a local 1x1 skip path (FNO adaptation)."""
    def __init__(self, width, modes, dropout):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(width, width, modes, 2) / math.sqrt(width))
        self.local = nn.Conv1d(width, width, 1)
        self.norm = _norm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Disabling autocast here avoids complex-half FFT/einsum limitations for
        # arbitrary lengths and does not change the real-valued head precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = torch.fft.rfft(x.float(), dim=-1)
            modes = min(self.modes, xf.shape[-1])
            weights = torch.view_as_complex(self.weight[..., :modes, :].float().contiguous())
            yf = torch.zeros(x.shape[0], x.shape[1], xf.shape[-1], device=x.device, dtype=xf.dtype)
            yf[..., :modes] = torch.einsum("bim,iom->bom", xf[..., :modes], weights)
            spectral = torch.fft.irfft(yf, n=x.shape[-1], dim=-1)
        y = self.local(x) + spectral.to(x.dtype)
        return x + self.dropout(F.gelu(self.norm(y)))


class SpectralBackbone(nn.Module):
    def __init__(self, in_channels, width, depth, dropout, modes, **unused):
        super().__init__()
        self.embedding = nn.Conv1d(in_channels, width, 1)
        self.blocks = nn.Sequential(*[SpectralBlock(width, modes, dropout) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(self.embedding(x))


class TaskModel(nn.Module):
    def __init__(self, backbone, task, width, num_classes=11):
        super().__init__()
        self.backbone = backbone
        self.task = task
        self.head = nn.Linear(width, num_classes) if task == "noise_classification" else nn.Conv1d(
            width, 1 if task == "noise_localization" else (8 if task == "frequency_denoising" else 4), 1)

    def forward(self, x, observed_mask=None):
        h = self.backbone(_features(x, observed_mask, self.task))
        if self.task == "noise_classification":
            return {"logits": self.head(h.mean(dim=-1))}
        y = self.head(h)
        if self.task == "noise_localization":
            return {"logits": y}
        if self.task == "frequency_denoising":
            low, high = y[:, :4], y[:, 4:]
            return {"low": low, "high": high, "prediction": low+high}
        return {"prediction": y}


class DLinearMasked(nn.Module):
    """DLinear-style decomposition adapted to variable-horizon masked forecasting.

    Uses a full 2048-position masked context and a learned mask contribution;
    differs from the official fixed context-to-horizon DLinear protocol.
    """
    def __init__(self, sequence_length, moving_average=25):
        super().__init__()
        if moving_average < 1 or moving_average % 2 != 1:
            raise ValueError("moving_average must be a positive odd number")
        self.sequence_length = sequence_length
        self.moving_average = moving_average
        self.seasonal = nn.Linear(sequence_length, sequence_length)
        self.trend = nn.Linear(sequence_length, sequence_length)
        self.mask_projection = nn.Linear(sequence_length, sequence_length, bias=False)
        nn.init.constant_(self.seasonal.weight, 1.0/sequence_length)
        nn.init.constant_(self.trend.weight, 1.0/sequence_length)
        nn.init.zeros_(self.seasonal.bias)
        nn.init.zeros_(self.trend.bias)
        nn.init.zeros_(self.mask_projection.weight)

    def forward(self, x, observed_mask=None):
        features = _features(x, observed_mask, "forecasting")
        signal, mask = features[:, :4], features[:, 4:]
        if signal.shape[-1] != self.sequence_length:
            raise ValueError("dlinear input length differs from configured sequence_length")
        pad = self.moving_average // 2
        trend = F.avg_pool1d(F.pad(signal, (pad, pad), mode="replicate"), self.moving_average, stride=1)
        prediction = self.seasonal(signal-trend) + self.trend(trend) + self.mask_projection(mask)
        return {"prediction": prediction}


def build_model(name, task, **kwargs):
    """Construct a randomly initialized model with no dataset target access.

    Common knobs: sequence_length=2048, width=64, depth=4, dropout=.1,
    patch_size=32, heads=4, modes=32, num_classes=11, moving_average=25.
    amt_maskformer defaults instead to width=192, patch_size=8, heads=6;
    its extra knobs are feedforward_dim=None (4*width), refinement_blocks=2,
    and local_skip=True. width/depth map to original d_model/layers.
    Unknown knobs raise, avoiding silently ignored spelling mistakes.
    """
    if name not in supported_models(task):
        raise ValueError("Model %r is unsupported for %r; choose %s" % (name, task, supported_models(task)))
    config = dict(sequence_length=2048, width=64, depth=4, dropout=0.1,
                  patch_size=32, heads=4, modes=32, num_classes=11, moving_average=25)
    amt_options = {"feedforward_dim", "refinement_blocks", "local_skip"}
    allowed = set(config) | (amt_options if name == "amt_maskformer" else set())
    unknown = set(kwargs)-allowed
    if unknown:
        raise TypeError("Unknown model parameters: %s" % sorted(unknown))
    if name == "amt_maskformer":
        config.update(width=192, patch_size=8, heads=6,
                      feedforward_dim=None, refinement_blocks=2, local_skip=True)
    config.update(kwargs)
    for key in ("sequence_length", "width", "depth", "patch_size", "heads", "modes", "num_classes"):
        if int(config[key]) < 1:
            raise ValueError("%s must be positive" % key)
    if not 0 <= config["dropout"] < 1:
        raise ValueError("dropout must be in [0,1)")
    if name == "amt_maskformer":
        return AMTMaskFormer(
            d_model=config["width"], patch_size=config["patch_size"], layers=config["depth"],
            heads=config["heads"], dropout=config["dropout"],
            feedforward_dim=(4*config["width"] if config["feedforward_dim"] is None
                             else config["feedforward_dim"]),
            refinement_blocks=config["refinement_blocks"], local_skip=config["local_skip"],
            task=task, num_classes=config["num_classes"])
    if name == "dlinear":
        return DLinearMasked(config["sequence_length"], config["moving_average"])
    backbones = dict(linear=LinearBackbone, tcn=TCNBackbone, unet1d=UNetBackbone,
                     bigru=BiGRUBackbone, patch_transformer=PatchTransformerBackbone,
                     spectral_fno=SpectralBackbone)
    backbone = backbones[name](in_channels=8 if task in CLEAN_TASKS else 4, **config)
    return TaskModel(backbone, task, config["width"], config["num_classes"])
