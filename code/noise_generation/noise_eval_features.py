"""Deterministic, CPU-only features for auditing generated one-dimensional noise.

This is a handcrafted feature-space audit, not an official neural embedding or a
proof of distributional equivalence / absence of memorization. Comparisons should
be made within the same noise class and declared length. A generated sample's
length label is a condition, not a measured support: we estimate its window by
maximizing energy over all contiguous windows of that declared length.

Training samples should first receive the training run's normalization and be
zero-padded by the caller. No random placement or augmentation occurs here.
Features largely ignore sign and time reversal; padding shifts are removed by
window estimation. Ties, discretization and floating point mean these are not
promised to be mathematically complete invariances. The 128-point signature is
only a candidate-search representation: interpolation can miss high-frequency
details, so copying claims require confirmation on complete original windows.

Only NumPy and SciPy are required. Extraction groups rows by length and processes
at most 1,024 rows per FFT batch, with no per-waveform FFT loop.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import next_fast_len, rfft, irfft


NOISE_TYPE_NAMES = [
    "Pulse", "Quasi-Sinusoidal", "Square-wave", "Step-wave",
    "Triangular-wave", "Power-frequency", "White-noise", "Color-noise",
    "Harmonic", "Composite",
]

_QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
_ACF_LABELS = ("1", "2", "4", "8", "16", "32", "64", "10pct", "25pct")
FEATURE_NAMES = (
    ["log1p_window_rms", "log1p_abs_window_mean", "log1p_window_peak", "log1p_window_std"]
    + [f"shape_abs_q{int(q * 100):02d}" for q in _QUANTILES]
    + ["shape_mean_abs", "shape_std_abs", "log1p_kurtosis", "log1p_crest_factor",
       "zero_crossing_rate", "mean_abs_first_difference", "rms_first_difference"]
    + [f"autocorrelation_lag_{label}" for label in _ACF_LABELS]
    + [f"spectral_band_{i:02d}_fraction" for i in range(16)]
    + ["spectral_centroid_nyquist", "spectral_spread_nyquist", "spectral_flatness",
       "spectral_entropy", "spectral_peak_frequency_nyquist", "spectral_peak_fraction",
       "spectral_rolloff85_nyquist", "spectral_rolloff95_nyquist", "spectral_dc_fraction",
       "window_energy_fraction"]
)


def _as_batch(batch: np.ndarray) -> np.ndarray:
    values = np.asarray(batch)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("batch must have shape (N, width), with width >= 1")
    if values.dtype.kind not in "fiu":
        raise ValueError("batch must contain real numeric values")
    return values


def _validate_length(length: int, width: int) -> int:
    if isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer)):
        raise ValueError("length must be an integer")
    if not 1 <= int(length) <= width:
        raise ValueError(f"length must be in [1, {width}], received {length}")
    return int(length)


def get_windows_for_length(
    batch: np.ndarray, length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return estimated windows, offsets and fractions of full-window energy.

    All rows must share the supplied declared length. Results have shapes
    ``(N, length)``, ``(N,)``, ``(N,)`` and dtypes float32, int32, float32.
    NaN/Inf values are replaced with zero. Ties select the earliest window;
    all-zero rows have offset zero and energy fraction zero. Fractions concern
    the cleaned signal, not a measured ground-truth support. To assess invalid
    input values, use ``extract_features``' finite-fraction diagnostics.
    """
    batch = _as_batch(batch)
    length = _validate_length(length, batch.shape[1])
    clean = np.nan_to_num(batch.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    energy = clean * clean
    prefix = np.empty((len(clean), clean.shape[1] + 1), dtype=np.float64)
    prefix[:, 0] = 0.0
    np.cumsum(energy, axis=1, out=prefix[:, 1:])
    moving_energy = prefix[:, length:] - prefix[:, :-length]
    offsets = np.argmax(moving_energy, axis=1).astype(np.int32)
    row_index = np.arange(len(clean))
    selected_energy = moving_energy[row_index, offsets]
    total_energy = prefix[:, -1]
    fractions = np.divide(
        selected_energy, total_energy,
        out=np.zeros_like(selected_energy), where=total_energy > 0,
    )
    windows = np.take_along_axis(clean, offsets[:, None] + np.arange(length), axis=1)
    return windows.astype(np.float32), offsets, np.clip(fractions, 0.0, 1.0).astype(np.float32)


def _normalize_windows(windows: np.ndarray, mode: str) -> np.ndarray:
    shape = windows.astype(np.float64)
    if mode == "rms":
        shape -= np.mean(shape, axis=1, keepdims=True)
        scale = np.sqrt(np.mean(shape * shape, axis=1, keepdims=True))
    elif mode == "maxabs":
        scale = np.max(np.abs(shape), axis=1, keepdims=True)
    else:  # mode == "none", validated in the public function
        return shape
    return np.divide(shape, scale, out=np.zeros_like(shape), where=scale > 1e-8)


def _features_for_windows(
    windows: np.ndarray, energy_fraction: np.ndarray, normalize_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Operate on one vectorized group with identical declared length."""
    windows64 = windows.astype(np.float64)
    n_rows, length = windows64.shape
    raw_mean = np.mean(windows64, axis=1)
    raw_rms = np.sqrt(np.mean(windows64 * windows64, axis=1))
    raw_peak = np.max(np.abs(windows64), axis=1)
    raw_std = np.std(windows64, axis=1)
    shape = _normalize_windows(windows64, normalize_mode)
    abs_shape = np.abs(shape)
    shape_rms = np.sqrt(np.mean(shape * shape, axis=1))
    centered = shape - np.mean(shape, axis=1, keepdims=True)
    centered_rms = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
    unit_centered = np.divide(
        centered, centered_rms, out=np.zeros_like(centered), where=centered_rms > 1e-12,
    )
    kurtosis = np.mean(unit_centered ** 4, axis=1)
    crest = np.divide(
        np.max(abs_shape, axis=1), shape_rms,
        out=np.zeros(n_rows, dtype=np.float64), where=shape_rms > 1e-12,
    )
    if length > 1:
        # The symmetric sign-change test is unchanged by sign/time reversal.
        zero_crossing = np.mean(shape[:, :-1] * shape[:, 1:] < 0, axis=1)
        difference = np.diff(shape, axis=1)
        mean_abs_difference = np.mean(np.abs(difference), axis=1)
        rms_difference = np.sqrt(np.mean(difference * difference, axis=1))
    else:
        zero_crossing = mean_abs_difference = rms_difference = np.zeros(n_rows)
    features = [
        np.log1p(raw_rms), np.log1p(np.abs(raw_mean)), np.log1p(raw_peak), np.log1p(raw_std),
    ]
    features.extend(np.quantile(abs_shape, _QUANTILES, axis=1))
    features.extend([
        np.mean(abs_shape, axis=1), np.std(abs_shape, axis=1), np.log1p(kurtosis),
        np.log1p(crest), zero_crossing, mean_abs_difference, rms_difference,
    ])

    # Zero-padding to >= 2L-1 gives linear, rather than circular, autocorrelation.
    n_fft = next_fast_len(max(1, 2 * length - 1))
    acf_fft = rfft(unit_centered, n=n_fft, axis=1)
    acf = irfft(acf_fft.real ** 2 + acf_fft.imag ** 2, n=n_fft, axis=1)[:, :length]
    denominator = acf[:, :1]
    acf = np.divide(acf, denominator, out=np.zeros_like(acf), where=denominator > 1e-12)
    lags = (1, 2, 4, 8, 16, 32, 64, max(1, round(length * 0.1)), max(1, round(length * 0.25)))
    features.extend(acf[:, min(lag, length - 1)] if length > 1 else np.zeros(n_rows) for lag in lags)

    # Normalize before FFT to avoid overflow for unnormalized high-amplitude inputs.
    unit_shape = np.divide(
        shape, shape_rms[:, None], out=np.zeros_like(shape), where=shape_rms[:, None] > 1e-12,
    )
    spectrum = rfft(unit_shape, axis=1)
    power = spectrum.real ** 2 + spectrum.imag ** 2
    # One-sided periodogram: interior bins represent positive and negative frequency.
    if length % 2 == 0:
        power[:, 1:-1] *= 2.0
    else:
        power[:, 1:] *= 2.0
    total_power = np.sum(power, axis=1, keepdims=True)
    spectral_fraction = np.divide(power, total_power, out=np.zeros_like(power), where=total_power > 0)
    frequencies = np.fft.rfftfreq(length) * 2.0  # unit = Nyquist frequency
    band_edges = np.searchsorted(frequencies, np.linspace(0.0, 1.0, 17), side="left")
    band_edges[-1] = len(frequencies)
    spectral_prefix = np.pad(np.cumsum(spectral_fraction, axis=1), ((0, 0), (1, 0)))
    band_fraction = spectral_prefix[:, band_edges[1:]] - spectral_prefix[:, band_edges[:-1]]
    features.extend(band_fraction.T)

    centroid = np.sum(spectral_fraction * frequencies[None, :], axis=1)
    spread = np.sqrt(np.sum(spectral_fraction * (frequencies[None, :] - centroid[:, None]) ** 2, axis=1))
    n_frequencies = len(frequencies)
    active = total_power[:, 0] > 0
    flatness = np.exp(np.mean(np.log(np.maximum(spectral_fraction, 1e-30)), axis=1)) * n_frequencies
    flatness[~active] = 0.0
    entropy = -np.sum(spectral_fraction * np.log(np.maximum(spectral_fraction, 1e-30)), axis=1)
    if n_frequencies > 1:
        entropy /= np.log(n_frequencies)
    else:
        entropy[:] = 0.0
    peak_index = np.argmax(spectral_fraction, axis=1)
    cumulative_fraction = np.cumsum(spectral_fraction, axis=1)
    rolloff85 = frequencies[np.argmax(cumulative_fraction >= 0.85, axis=1)]
    rolloff95 = frequencies[np.argmax(cumulative_fraction >= 0.95, axis=1)]
    features.extend([
        centroid, spread, flatness, entropy, frequencies[peak_index],
        np.max(spectral_fraction, axis=1), rolloff85, rolloff95,
        spectral_fraction[:, 0], energy_fraction,
    ])

    positions = np.linspace(0.0, length - 1, 128)
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, length - 1)
    weight = positions - left
    signature = shape[:, left] * (1.0 - weight) + shape[:, right] * weight
    feature_array = np.column_stack(features).astype(np.float32)
    diagnostics = {
        "window_rms": raw_rms.astype(np.float32),
        "window_mean": raw_mean.astype(np.float32),
        "window_std": raw_std.astype(np.float32),
        "peak_abs": raw_peak.astype(np.float32),
    }
    return feature_array, signature.astype(np.float32), diagnostics


def extract_features(
    batch: np.ndarray, lengths: np.ndarray, normalize_mode: str = "rms",
) -> dict:
    """Extract 54 interpretable features and 128-point candidate signatures.

    ``batch`` has shape ``(N, width)`` (normally width 2048), and ``lengths``
    supplies one integer declared support length per row. Lengths must be in
    ``[1, width]``. The result contains ``features`` (float32, N x 54),
    ``feature_names``, ``signature`` (float32, N x 128), and ``diagnostics``.

    ``normalize_mode`` controls only window morphology/signatures: ``rms``
    subtracts each window's mean then divides by RMS; ``maxabs`` divides by its
    largest absolute value; ``none`` keeps amplitude. Raw window RMS, absolute
    mean, peak and standard deviation remain represented by log1p features in
    every mode, and their untransformed values remain in diagnostics. Thus an
    amplitude mismatch is not hidden by morphology normalization. The caller
    must separately apply the checkpoint's data normalization to references.

    Diagnostic energy fractions and offsets estimate the highest-energy window
    of the *conditioned* length; they do not establish true generated support.
    Nonfinite input values become zero; full and selected-window finite fractions
    retain evidence of them. Signatures are not sign/reversal canonicalized.
    """
    batch = _as_batch(batch)
    if normalize_mode not in {"rms", "maxabs", "none"}:
        raise ValueError("normalize_mode must be 'rms', 'maxabs', or 'none'")
    lengths_array = np.asarray(lengths)
    if lengths_array.shape != (len(batch),) or lengths_array.dtype.kind not in "iu":
        raise ValueError("lengths must be a one-dimensional integer array with N entries")
    if np.any(lengths_array < 1) or np.any(lengths_array > batch.shape[1]):
        raise ValueError(f"every length must be in [1, {batch.shape[1]}]")
    output = {
        "features": np.empty((len(batch), len(FEATURE_NAMES)), dtype=np.float32),
        "feature_names": list(FEATURE_NAMES),
        "signature": np.empty((len(batch), 128), dtype=np.float32),
        "diagnostics": {
            name: np.empty(len(batch), dtype=np.int32 if name == "offset" else np.float32)
            for name in ("offset", "window_energy_fraction", "window_rms", "window_mean",
                         "window_std", "full_rms", "peak_abs", "full_peak_abs",
                         "finite_fraction", "window_finite_fraction")
        },
    }
    for length in np.unique(lengths_array):
        group = np.flatnonzero(lengths_array == length)
        for start in range(0, len(group), 1024):
            indices = group[start:start + 1024]
            raw = batch[indices]
            finite = np.isfinite(raw)
            windows, offsets, fractions = get_windows_for_length(raw, int(length))
            features, signatures, window_diagnostics = _features_for_windows(
                windows, fractions, normalize_mode,
            )
            clean = np.where(finite, raw, 0.0).astype(np.float64)
            window_finite = np.take_along_axis(finite, offsets[:, None] + np.arange(length), axis=1)
            diagnostics = {
                **window_diagnostics,
                "offset": offsets,
                "window_energy_fraction": fractions,
                "full_rms": np.sqrt(np.mean(clean * clean, axis=1)).astype(np.float32),
                "full_peak_abs": np.max(np.abs(clean), axis=1).astype(np.float32),
                "finite_fraction": np.mean(finite, axis=1).astype(np.float32),
                "window_finite_fraction": np.mean(window_finite, axis=1).astype(np.float32),
            }
            output["features"][indices] = features
            output["signature"][indices] = signatures
            for name, values in diagnostics.items():
                output["diagnostics"][name][indices] = values
    return output
