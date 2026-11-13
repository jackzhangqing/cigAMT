"""Local, from-scratch binary baselines with a single shared tensor contract.

create_model(name, channels=4, length=240, config=None) consumes (B,C,T) and
returns unactivated logits (B,1) for BCEWithLogitsLoss. No checkpoint, download,
ImageNet preprocessing, input z-score, or random data augmentation is applied.
Paper-inspired implementations are explicitly adaptations, not official code.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


MODEL_NAMES = ("mlp", "cnn1d", "vgg16_1d", "resnet1d", "inceptiontime", "tcn",
               "bigru", "transformer", "patchtst", "stft_resnet18", "stft_vit")

MODEL_DEFAULTS = {
    "mlp": {"width": 32, "dropout": 0.1},
    "cnn1d": {"width": 32, "dropout": 0.1},
    "vgg16_1d": {"dropout": 0.0},
    "resnet1d": {"width": 32, "dropout": 0.1},
    "inceptiontime": {"width": 32, "depth": 6, "dropout": 0.1},
    "tcn": {"width": 32, "depth": 6, "dropout": 0.1},
    "bigru": {"hidden_size": 64, "depth": 2, "dropout": 0.1},
    "transformer": {"d_model": 64, "n_heads": 4, "depth": 2, "dropout": 0.1},
    "patchtst": {"d_model": 64, "n_heads": 4, "depth": 2, "patch_length": 16,
                 "patch_stride": 8, "dropout": 0.1},
    "stft_resnet18": {"width": 32, "dropout": 0.1},
    "stft_vit": {"d_model": 64, "n_heads": 4, "depth": 3, "patch_size": 4, "dropout": 0.1},
}

_NOTES = {
    "mlp": ("MLP", "time_domain", "Flattened local dense baseline; not a named paper reproduction.", []),
    "cnn1d": ("CNN", "time_domain", "Local three-block 1D CNN baseline; not a named paper reproduction.", []),
    "vgg16_1d": ("CNN", "time_domain", "Project VGG16 1D adaptation: 13 Conv1d layers, five pools and 1024/512 binary head. Fresh weights; not ImageNet VGG16.", ["vgg"]),
    "resnet1d": ("residual_CNN", "time_domain", "1D adaptation of the ResNet18 basic-block pattern, width 32 by default; not an official time-series reproduction.", ["resnet"]),
    "inceptiontime": ("multi_scale_CNN", "time_domain", "Single six-module InceptionTime-style network, 40/20/10 kernels and residual groups; not the paper's five-network ensemble or official implementation.", ["inceptiontime"]),
    "tcn": ("causal_dilated_CNN", "time_domain", "Residual causal dilated-convolution classification adaptation; global temporal pooling, no weight normalization. Not the official TCN implementation.", ["tcn"]),
    "bigru": ("recurrent", "time_domain", "Bidirectional GRU classifier with batch_first=True and final forward/backward states; local classification adaptation, not the original encoder-decoder task.", ["gru"]),
    "transformer": ("attention", "time_domain", "Encoder-only time-step Transformer with learned positions and mean-pooling binary head; not the original translation encoder-decoder.", ["transformer"]),
    "patchtst": ("patch_attention", "time_domain", "PatchTST-inspired channel-independent shared patch encoder, concatenated channel summaries and binary head. Classification adaptation; no forecasting head, RevIN, pretraining or official-code claim.", ["patchtst"]),
    "stft_resnet18": ("vision_residual_CNN", "stft_log_power", "ResNet18 basic-block topology on native four-channel STFT maps, 3x3 stride-1 stem and width 32. Trained from scratch; no RGB conversion, ImageNet weights or official spectrogram-model claim.", ["resnet", "torch_stft"]),
    "stft_vit": ("vision_transformer", "stft_log_power", "Compact ViT-style spectrogram patch classifier with a class token. Four-channel native maps, patch size 4 and small embedding; trained from scratch, not ViT-B/16 or ImageNet-pretrained.", ["vit", "transformer", "torch_stft"]),
}
MODEL_INFO = {
    name: {"name": name, "family": row[0], "representation": row[1],
           "reimplementation_note": row[2], "reimplementationnote": row[2],
           "source_ids": row[3], "pretrained": False, "initialization": "from_scratch",
           "input": "(batch,channels,length)", "output": "(batch,1) logits",
           "default_config": copy.deepcopy(MODEL_DEFAULTS[name])}
    for name, row in _NOTES.items()
}


def model_description(name):
    """Return an independent JSON-serializable information dictionary."""
    if name not in MODEL_INFO:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    return copy.deepcopy(MODEL_INFO[name])


def _configuration(name, config):
    result = dict(MODEL_DEFAULTS[name])
    if config is not None:
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary or None")
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(f"Unsupported config keys for {name}: {sorted(unknown)}")
        result.update(config)
    for key, value in result.items():
        if key == "dropout":
            if not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ValueError("dropout must be in [0,1)")
        elif not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if "d_model" in result and result["d_model"] % result["n_heads"]:
        raise ValueError("d_model must be divisible by n_heads")
    return result


class _Contract(nn.Module):
    def __init__(self, model, name, channels, length, config):
        super().__init__()
        self.model = model
        self.name, self.channels, self.length = name, channels, length
        self.config = dict(config)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.channels, self.length):
            raise ValueError(f"{self.name} expects (B,{self.channels},{self.length}), got {tuple(x.shape)}")
        if not torch.is_floating_point(x):
            raise TypeError("Model inputs must be floating-point tensors")
        output = self.model(x)
        if output.shape != (x.shape[0], 1):
            raise RuntimeError("Model violated the single-logit output contract")
        return output


class _MLP(nn.Module):
    def __init__(self, channels, length, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(channels * length, width * 8), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(width * 8, width * 4), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(width * 4, 1))

    def forward(self, x):
        return self.net(x)


class _CNN1D(nn.Module):
    def __init__(self, channels, width, dropout):
        super().__init__()
        layers = []
        for out_channels, kernel in ((width, 7), (width * 2, 5), (width * 4, 3)):
            layers.extend([nn.Conv1d(channels, out_channels, kernel, padding=kernel // 2, bias=False),
                           nn.BatchNorm1d(out_channels), nn.ReLU(), nn.MaxPool1d(2)])
            channels = out_channels
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, 1))

    def forward(self, x):
        return self.head(self.features(x))


class _VGG16_1D(nn.Module):
    def __init__(self, channels, length, dropout):
        super().__init__()
        if length < 32:
            raise ValueError("vgg16_1d requires length >= 32 for five pooling stages")
        cfg = (64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M")
        layers = []
        for item in cfg:
            if item == "M":
                layers.append(nn.MaxPool1d(2))
            else:
                layers.extend([nn.Conv1d(channels, item, 3, padding=1), nn.ReLU()])
                channels = item
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Linear(512 * (length // 32), 1024), nn.BatchNorm1d(1024),
                                        nn.ReLU(), nn.Dropout(dropout), nn.Linear(1024, 512),
                                        nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 1))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class _Residual1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False),
                                  nn.BatchNorm1d(out_channels), nn.ReLU(),
                                  nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False),
                                  nn.BatchNorm1d(out_channels))
        self.skip = (nn.Identity() if in_channels == out_channels and stride == 1 else
                     nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm1d(out_channels)))

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


class _ResNet1D(nn.Module):
    def __init__(self, channels, width, dropout):
        super().__init__()
        layers = [nn.Conv1d(channels, width, 7, 2, 3, bias=False), nn.BatchNorm1d(width),
                  nn.ReLU(), nn.MaxPool1d(3, 2, 1)]
        previous = width
        for stage in range(4):
            out = width * 2 ** stage
            layers.extend([_Residual1D(previous, out, 1 if stage == 0 else 2), _Residual1D(out, out)])
            previous = out
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(previous, 1))

    def forward(self, x):
        return self.head(self.features(x))


class _SameConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, bias=False)

    def forward(self, x):
        return self.conv(F.pad(x, ((self.kernel - 1) // 2, self.kernel // 2)))


class _InceptionModule(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.bottleneck = nn.Conv1d(channels, width, 1, bias=False) if channels > 1 else nn.Identity()
        branch_in = width if channels > 1 else channels
        self.branches = nn.ModuleList([_SameConv1D(branch_in, width, kernel) for kernel in (40, 20, 10)])
        self.pool_branch = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(channels, width, 1, bias=False))
        self.norm = nn.BatchNorm1d(width * 4)

    def forward(self, x):
        reduced = self.bottleneck(x)
        parts = [branch(reduced) for branch in self.branches] + [self.pool_branch(x)]
        return F.relu(self.norm(torch.cat(parts, dim=1)))


class _InceptionTime(nn.Module):
    def __init__(self, channels, width, depth, dropout):
        super().__init__()
        if depth % 3:
            raise ValueError("inceptiontime depth must be divisible by 3 (residual groups)")
        self.groups, self.shortcuts = nn.ModuleList(), nn.ModuleList()
        for _ in range(depth // 3):
            self.groups.append(nn.Sequential(_InceptionModule(channels, width),
                                             _InceptionModule(width * 4, width), _InceptionModule(width * 4, width)))
            self.shortcuts.append(nn.Sequential(nn.Conv1d(channels, width * 4, 1, bias=False), nn.BatchNorm1d(width * 4)))
            channels = width * 4
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, 1))

    def forward(self, x):
        for group, shortcut in zip(self.groups, self.shortcuts):
            x = F.relu(group(x) + shortcut(x))
        return self.head(x.mean(-1))


class _CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.left_pad = 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, 3, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_pad, 0)))


class _TemporalBlock(nn.Module):
    def __init__(self, channels, width, dilation, dropout):
        super().__init__()
        self.body = nn.Sequential(_CausalConv1D(channels, width, dilation), nn.ReLU(), nn.Dropout(dropout),
                                  _CausalConv1D(width, width, dilation), nn.ReLU(), nn.Dropout(dropout))
        self.skip = nn.Conv1d(channels, width, 1) if channels != width else nn.Identity()

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


class _TCN(nn.Module):
    def __init__(self, channels, width, depth, dropout):
        super().__init__()
        self.blocks = nn.Sequential(*[_TemporalBlock(channels if index == 0 else width, width,
                                                     2 ** index, dropout) for index in range(depth)])
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        return self.head(self.blocks(x).mean(-1))


class _BiGRU(nn.Module):
    def __init__(self, channels, hidden_size, depth, dropout):
        super().__init__()
        self.rnn = nn.GRU(channels, hidden_size, num_layers=depth, batch_first=True,
                          bidirectional=True, dropout=dropout if depth > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size * 2, 1))

    def forward(self, x):
        _, hidden = self.rnn(x.transpose(1, 2))
        return self.head(torch.cat((hidden[-2], hidden[-1]), dim=-1))


def _encoder(d_model, n_heads, depth, dropout):
    layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                       dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
    encoder = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
    # PyTorch clones the prototype layer. Reinitialize matrix weights so all
    # encoder layers do not start with exactly the same parameter values.
    for parameter in encoder.parameters():
        if parameter.ndim > 1:
            nn.init.xavier_uniform_(parameter)
    return encoder


class _Transformer(nn.Module):
    def __init__(self, channels, length, d_model, n_heads, depth, dropout):
        super().__init__()
        self.embed = nn.Linear(channels, d_model)
        self.positions = nn.Parameter(torch.zeros(1, length, d_model))
        nn.init.normal_(self.positions, std=0.02)
        self.encoder = _encoder(d_model, n_heads, depth, dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        tokens = self.dropout(self.embed(x.transpose(1, 2)) + self.positions)
        return self.head(self.encoder(tokens).mean(1))


class _PatchTST(nn.Module):
    def __init__(self, channels, length, d_model, n_heads, depth, patch_length, patch_stride, dropout):
        super().__init__()
        if not 1 <= patch_stride <= patch_length <= length:
            raise ValueError("patchtst needs 1 <= patch_stride <= patch_length <= length")
        self.patch_length, self.patch_stride = patch_length, patch_stride
        self.padding = (-(length - patch_length)) % patch_stride
        count = (length + self.padding - patch_length) // patch_stride + 1
        self.embed = nn.Linear(patch_length, d_model)
        self.positions = nn.Parameter(torch.zeros(1, count, d_model))
        nn.init.normal_(self.positions, std=0.02)
        self.encoder = _encoder(d_model, n_heads, depth, dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels * d_model, 1)

    def forward(self, x):
        batch, channels, _ = x.shape
        if self.padding:
            x = F.pad(x, (0, self.padding), mode="replicate")
        patches = x.unfold(-1, self.patch_length, self.patch_stride)
        # Shared embedding/encoder, no inter-channel attention: channels become
        # independent batch entries until the final classification head.
        patches = patches.reshape(batch * channels, patches.shape[-2], self.patch_length)
        tokens = self.dropout(self.embed(patches) + self.positions)
        summary = self.encoder(tokens).mean(1).reshape(batch, -1)
        return self.head(summary)


class STFTLogPower(nn.Module):
    """Differentiable four-channel STFT; no band selection or per-map scaling.

    240 samples -> (C,33,31), n_fft=win_length=64, hop=8, periodic Hann,
    center=True, reflect padding, one-sided unnormalized FFT, log1p(|S|^2).
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(64, periodic=True))

    def forward(self, x):
        batch, channels, length = x.shape
        spectrum = torch.stft(x.reshape(batch * channels, length).float(), n_fft=64,
                              hop_length=8, win_length=64, window=self.window.float(),
                              center=True, pad_mode="reflect", normalized=False,
                              onesided=True, return_complex=True)
        power = spectrum.real.square() + spectrum.imag.square()
        return torch.log1p(power).reshape(batch, channels, power.shape[-2], power.shape[-1])


class _Residual2D(nn.Module):
    def __init__(self, channels, out_channels, stride=1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, out_channels, 3, stride, 1, bias=False),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(),
                                  nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels))
        self.skip = (nn.Identity() if channels == out_channels and stride == 1 else
                     nn.Sequential(nn.Conv2d(channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels)))

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


class _STFTResNet18(nn.Module):
    def __init__(self, channels, width, dropout):
        super().__init__()
        self.representation = STFTLogPower()
        layers = [nn.Conv2d(channels, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU()]
        previous = width
        for stage in range(4):
            out = width * 2 ** stage
            layers.extend([_Residual2D(previous, out, 1 if stage == 0 else 2), _Residual2D(out, out)])
            previous = out
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(previous, 1))

    def forward(self, x):
        return self.head(self.features(self.representation(x)))


class _STFTViT(nn.Module):
    def __init__(self, channels, length, d_model, n_heads, depth, patch_size, dropout):
        super().__init__()
        self.representation = STFTLogPower()
        height, width = 33, length // 8 + 1
        self.pad_height, self.pad_width = (-height) % patch_size, (-width) % patch_size
        tokens = math.ceil(height / patch_size) * math.ceil(width / patch_size)
        self.embed = nn.Conv2d(channels, d_model, patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positions = nn.Parameter(torch.zeros(1, tokens + 1, d_model))
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.positions, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.encoder = _encoder(d_model, n_heads, depth, dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        image = self.representation(x)
        image = F.pad(image, (0, self.pad_width, 0, self.pad_height))
        tokens = self.embed(image).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.class_token.expand(x.shape[0], -1, -1), tokens), dim=1)
        tokens = self.dropout(tokens + self.positions)
        return self.head(self.encoder(tokens)[:, 0])


def create_model(name, channels=4, length=240, config=None):
    """Construct a fresh binary classifier; supported knobs are model-specific.

    See MODEL_DEFAULTS or model_description(name)["default_config"]. Unknown
    knobs fail explicitly so a benchmark cannot silently ignore a setting.
    vgg16_1d has BatchNorm in its dense head and needs training batches >= 2.
    All models support evaluation batches of one. No sigmoid is included.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    if not isinstance(channels, int) or channels < 1 or not isinstance(length, int) or length < 8:
        raise ValueError("channels must be positive and length must be an integer >= 8")
    if name.startswith("stft_") and length < 64:
        raise ValueError("STFT baselines require length >= 64")
    cfg = _configuration(name, config)
    if name == "mlp":
        model = _MLP(channels, length, **cfg)
    elif name == "cnn1d":
        model = _CNN1D(channels, **cfg)
    elif name == "vgg16_1d":
        model = _VGG16_1D(channels, length, **cfg)
    elif name == "resnet1d":
        model = _ResNet1D(channels, **cfg)
    elif name == "inceptiontime":
        model = _InceptionTime(channels, **cfg)
    elif name == "tcn":
        model = _TCN(channels, **cfg)
    elif name == "bigru":
        model = _BiGRU(channels, **cfg)
    elif name == "transformer":
        model = _Transformer(channels, length, **cfg)
    elif name == "patchtst":
        model = _PatchTST(channels, length, **cfg)
    elif name == "stft_resnet18":
        model = _STFTResNet18(channels, **cfg)
    else:
        model = _STFTViT(channels, length, **cfg)
    return _Contract(model, name, channels, length, cfg)
