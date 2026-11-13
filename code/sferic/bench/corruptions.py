"""Synthetic stress tests; source arrays are never modified.

Random draws depend on global row index, not batch order or batch size.
The original validation-selected threshold is retained on corrupted test data.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt


DEFAULT_CORRUPTIONS = [
    dict(name="clean"),
    *[dict(name="wgn", snr_db=v) for v in (20, 10, 5, 0, -5)],
    *[dict(name="drop_channel", channel=v) for v in range(4)],
    dict(name="gain", factor=0.5), dict(name="gain", factor=2.0),
]


def corrupt(x, row_indices, spec, seed=42, fs=24000):
    values = np.asarray(x, dtype=np.float32).copy()
    name = spec["name"]
    if name == "clean":
        return values
    if name == "wgn":
        for j, row in enumerate(row_indices):
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(row), 89321]))
            centered = values[j].astype(np.float64) - values[j].mean(-1, keepdims=True, dtype=np.float64)
            power = np.mean(centered ** 2, axis=-1, keepdims=True)
            noise = rng.standard_normal(values[j].shape) * np.sqrt(power / 10 ** (float(spec["snr_db"]) / 10))
            values[j] = values[j] + noise
    elif name == "drop_channel":
        channel = int(spec["channel"])
        if channel not in range(4):
            raise ValueError("channel must be 0,1,2,3")
        values[:, channel] = 0
    elif name == "gain":
        values *= float(spec["factor"])
    elif name == "time_shift":
        # Reflect padding avoids circularly wrapping a transient across an edge.
        shift = int(spec["samples"])
        pad = abs(shift)
        if pad >= values.shape[-1]:
            raise ValueError("time shift must be shorter than one window")
        if pad:
            padded = np.pad(values, ((0,0),(0,0),(pad,pad)), mode="reflect")
            start = pad - shift
            values = padded[..., start:start + values.shape[-1]].copy()
    elif name == "bandstop":
        lo, hi = float(spec["low_hz"]), float(spec["high_hz"])
        if not 0 < lo < hi < fs / 2:
            raise ValueError("Need 0 < low_hz < high_hz < Nyquist")
        sos = butter(4, [lo, hi], btype="bandstop", fs=fs, output="sos")
        values = sosfiltfilt(sos, values, axis=-1).astype(np.float32)
    else:
        raise ValueError(f"Unknown corruption: {name}")
    if not np.isfinite(values).all():
        raise ValueError("Corruption produced nonfinite values")
    return values
