"""Independent NumPy protocols for seven paired AMT learning tasks.

All returned waveform arrays retain a batch dimension, including single-clip
inputs. Channel order is Ex, Ey, Hx, Hy. SNR energy is measured over the entire
window, including uncorrupted positions, in the supplied clean source's
coordinates. Noise events share the same temporal interval across channels.

Scheme A sets each channel's SNR exactly. Scheme C sets the E pair, H pair,
and total SNR exactly, with equal noise RMS within each pair, while individual
channel SNRs can differ. C shares a noise
template and follows the signal E/H group ratio; it is an engineering model,
not a measured four-channel noise covariance model. Five-percent pointwise
perturbations are applied *before* energy is remeasured and rescaled.

Noise tasks divide input and clean target by the same noisy-input channel RMS.
The frequency labels use complementary FFT masks at 5 Hz by default. This is
an engineering frequency split, not the uploaded paper's exact autoencoder.

The three clean reconstruction tasks retain source-normalized coordinates.
Legacy source preprocessing used full-record mean/std, including future
samples. This is not an observed-only forecasting normalization protocol.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

CHANNEL_NAMES = ("Ex", "Ey", "Hx", "Hy")
FORECAST_HORIZONS = (10, 24, 32, 36, 64, 128, 256)
INTERPOLATION_GAPS = (10, 32, 64, 128, 256)
IMPUTATION_RATIOS = tuple(index / 10.0 for index in range(1, 10))
IMPUTATION_MEAN_GAPS = (16, 32, 64, 128)
TEMPLATE_RMS_MIN = 1e-15


def _clips(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 2:
        result = result[None, ...]
    if result.ndim != 3 or result.shape[1] != 4 or result.shape[2] <= 1:
        raise ValueError("Expected clean shape [N,4,T] or [4,T] with T > 1")
    if not len(result) or not np.isfinite(result).all():
        raise ValueError("Clean clips must be nonempty and contain only finite values")
    energy = np.mean(result * result, axis=-1)
    if np.any(energy <= 0) or not np.isfinite(energy).all():
        raise ValueError("Every clean channel must have finite, positive RMS energy")
    return result


def _vector(value: Any, count: int, name: str, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim == 0:
        result = np.repeat(result, count)
    if result.shape != (count,):
        raise ValueError(f"{name} must contain one value per clip ({count})")
    return result


def _seeds(value: Any, count: int) -> np.ndarray:
    # Check before casting: negative seeds or fractional floats must not wrap
    # or silently truncate into a different replayable experiment.
    for seed in np.asarray(value).reshape(-1):
        if not np.isfinite(seed) or int(seed) != seed or not 0 <= int(seed) <= np.iinfo(np.uint64).max:
            raise ValueError("Seeds must be nonnegative uint64-range integers")
    return _vector(value, count, "seeds", np.uint64)


def _db(signal_energy: np.ndarray, noise_energy: np.ndarray) -> np.ndarray:
    if np.any(signal_energy <= 0) or np.any(noise_energy <= 0):
        raise ValueError("SNR requires positive signal and noise energy")
    return 10.0 * np.log10(signal_energy / noise_energy)


def split_frequency_target(target: np.ndarray, sampling_rate_hz: float = 150.0,
                           cutoff_hz: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Split a batch using complementary rFFT masks; DC is in the low band.

    A bin at the exact cutoff belongs to the low band. High labels are computed
    as target - low in float32 to preserve the target-complement identity.
    """
    values = _clips(target)
    fs, cutoff = float(sampling_rate_hz), float(cutoff_hz)
    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(cutoff) or not 0 < cutoff < fs / 2:
        raise ValueError("Require 0 < cutoff_hz < sampling_rate_hz / 2")
    spectra = np.fft.rfft(values, axis=-1)
    low_bins = np.fft.rfftfreq(values.shape[-1], d=1.0 / fs) <= cutoff
    low = np.fft.irfft(spectra * low_bins, n=values.shape[-1], axis=-1).astype(np.float32)
    full = values.astype(np.float32)
    high = (full - low).astype(np.float32)
    return low, high


def mix_and_labels(clean: np.ndarray, templates: Sequence[np.ndarray], starts: Any,
                   snrs: Any, scheme: str, seeds: Any, *,
                   sampling_rate_hz: float = 150.0, cutoff_hz: float = 5.0,
                   perturbation_std: float = 0.05) -> dict[str, np.ndarray]:
    """Create the common input/labels for the four noise-related tasks.

    ``templates`` has one finite nonzero one-dimensional event per clip;
    lengths may differ. Templates are not resized. ``starts`` must place each
    complete event inside its window. Raw template RMS <= 1e-15 is rejected.
    Classification source/type labels are provided by the materializer from
    its template catalog, so they are not guessed from waveform amplitudes.
    """
    signals = _clips(clean)
    count, _, length = signals.shape
    if len(templates) != count:
        raise ValueError("templates must contain one event per clip")
    event_starts = _vector(starts, count, "starts", np.int64)
    requested = _vector(snrs, count, "snrs", np.float64)
    sample_seeds = _seeds(seeds, count)
    if not np.isfinite(requested).all():
        raise ValueError("SNR targets must be finite")
    scheme = str(scheme).upper()
    if scheme not in ("A", "C"):
        raise ValueError("scheme must be A or C")
    perturbation_std = float(perturbation_std)
    if not np.isfinite(perturbation_std) or perturbation_std < 0:
        raise ValueError("perturbation_std must be finite and nonnegative")
    noise = np.zeros_like(signals)
    noise_mask = np.zeros(signals.shape, dtype=np.uint8)
    for row, raw in enumerate(templates):
        template = np.asarray(raw, dtype=np.float64)
        if template.ndim != 1 or not len(template) or not np.isfinite(template).all():
            raise ValueError(f"Invalid one-dimensional finite noise template at row {row}")
        template_rms = float(np.sqrt(np.mean(template * template)))
        if not np.isfinite(template_rms) or template_rms <= TEMPLATE_RMS_MIN:
            raise ValueError(f"Zero/insufficient-energy noise template at row {row}: RMS={template_rms}")
        begin = int(event_starts[row])
        end = begin + len(template)
        if begin < 0 or end > length:
            raise ValueError(f"Noise event does not fit at row {row}: [{begin},{end}) in T={length}")
        rng = np.random.default_rng(int(sample_seeds[row]))
        event = template[None, :] * (1.0 + rng.normal(0.0, perturbation_std, (4, len(template))))
        noise[row, :, begin:end] = event
        # The mask denotes the injected event interval, including instantaneous
        # zeros within the selected waveform; it is not amplitude thresholding.
        noise_mask[row, :, begin:end] = 1
    signal_energy = np.mean(signals * signals, axis=-1)
    pre_energy = np.mean(noise * noise, axis=-1)
    if np.any(pre_energy <= 0) or not np.isfinite(pre_energy).all():
        raise ValueError("Perturbed noise must retain positive finite energy in every channel")
    power_ratio = np.power(10.0, requested / 10.0)
    if not np.isfinite(power_ratio).all() or np.any(power_ratio <= 0):
        raise ValueError("SNR targets exceed the supported finite floating-point range")
    if scheme == "A":
        gains = np.sqrt(signal_energy / (power_ratio[:, None] * pre_energy))
    else:
        # Match the legacy C protocol: each perturbed event is normalized
        # independently to the same target RMS within its E or H pair.
        # Using one shared pair gain would retain unwanted channel noise-RMS
        # differences from the pointwise perturbations.
        wanted_energy = np.repeat(np.stack((signal_energy[:, :2].mean(axis=1),
                                           signal_energy[:, 2:].mean(axis=1)), axis=1), 2, axis=1)
        gains = np.sqrt(wanted_energy / (power_ratio[:, None] * pre_energy))
    noise *= gains[:, :, None]
    noisy = signals + noise
    scale64 = np.sqrt(np.mean(noisy * noisy, axis=-1, keepdims=True))
    if np.any(scale64 <= 0) or not np.isfinite(scale64).all():
        raise ValueError("Noisy-input RMS must remain finite and positive")
    target = (signals / scale64).astype(np.float32)
    model_input = (noisy / scale64).astype(np.float32)
    low, high = split_frequency_target(target, sampling_rate_hz, cutoff_hz)
    # Report SNR of the exported float32 pair, so recorded diagnostics describe
    # the materialized arrays rather than an unexported float64 intermediate.
    exported_noise = model_input.astype(np.float64) - target.astype(np.float64)
    exported_signal_energy = np.mean(target.astype(np.float64) ** 2, axis=-1)
    exported_noise_energy = np.mean(exported_noise ** 2, axis=-1)
    # Undo the shared normalization before aggregate E/H and total energies.
    original_signal_energy = exported_signal_energy * scale64[..., 0] ** 2
    original_noise_energy = exported_noise_energy * scale64[..., 0] ** 2
    result = {
        "input": model_input, "target": target, "target_low": low,
        "target_high": high, "noise_mask": noise_mask,
        "scale": scale64.astype(np.float32),
        "snr_target_db": requested.astype(np.float32),
        "snr_per_channel_db": _db(exported_signal_energy, exported_noise_energy).astype(np.float32),
        "snr_total_db": _db(original_signal_energy.mean(axis=1), original_noise_energy.mean(axis=1)).astype(np.float32),
        "snr_e_db": _db(original_signal_energy[:, :2].mean(axis=1), original_noise_energy[:, :2].mean(axis=1)).astype(np.float32),
        "snr_h_db": _db(original_signal_energy[:, 2:].mean(axis=1), original_noise_energy[:, 2:].mean(axis=1)).astype(np.float32),
    }
    if not all(np.isfinite(array).all() for array in result.values()):
        raise FloatingPointError("Protocol produced non-finite export arrays")
    return result


def _geometric(length: int, ratio: float, mean_gap: int,
               rng: np.random.Generator, tolerance: float = 0.03,
               attempts: int = 100) -> np.ndarray:
    if not 0 < ratio < 1 or mean_gap < 1 or tolerance <= 0 or attempts < 1:
        raise ValueError("Invalid geometric mask ratio, gap, tolerance, or attempts")
    leave_missing = 1.0 / mean_gap
    leave_observed = leave_missing * ratio / (1.0 - ratio)
    if leave_observed > 1:
        raise ValueError("Requested missing ratio and mean gap are incompatible")
    best, best_error = None, np.inf
    for _ in range(attempts):
        observed = np.empty(length, dtype=np.bool_)
        state = bool(rng.random() > ratio)
        draws = rng.random(length)
        for index, draw in enumerate(draws):
            observed[index] = state
            if draw < (leave_observed if state else leave_missing):
                state = not state
        if observed.all() or not observed.any():
            continue
        error = abs(float((~observed).mean()) - ratio)
        if error < best_error:
            best, best_error = observed, error
        if error <= tolerance:
            return observed
    if best is None:
        raise RuntimeError("Could not draw a nondegenerate geometric mask")
    return best


def clean_task_labels(target: np.ndarray, task: str, seeds: Any,
                      settings: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None
                      ) -> dict[str, Any]:
    """Generate one offline condition per clean clip for a reconstruction task.

    Global settings may contain ``sample_indices`` (global row ids) or
    ``sample_offset`` for balanced conditions across separately built batches.
    Defaults cycle through 7 horizons, 5 gaps, or 72 combinations of nominal
    missing ratio (9), mean gap (4), and synchronous/independent mode (2).
    Per-row dictionaries can instead specify explicit condition values.
    Positions and geometric realizations are randomized only by each seed.

    Returned ``observed_mask`` is True at visible positions; ``loss_mask`` is
    its complement. The input contains exact zeros at all missing positions.
    The full clean target is returned only as the supervised label.
    """
    values = _clips(target).astype(np.float32)
    count, channels, length = values.shape
    sample_seeds = _seeds(seeds, count)
    task = str(task).lower()
    if task not in ("forecasting", "interpolation", "imputation"):
        raise ValueError("Unknown clean-signal reconstruction task")
    per_row = not (settings is None or isinstance(settings, Mapping))
    if per_row:
        if len(settings) != count:
            raise ValueError("Per-row settings must match the clip count")
        configurations = [dict(item) for item in settings]
        sample_indices = np.arange(count, dtype=np.int64)
    else:
        common = dict(settings or {})
        sample_indices = _vector(common.pop("sample_indices", np.arange(count) + int(common.pop("sample_offset", 0))),
                                 count, "sample_indices", np.int64)
        if np.any(sample_indices < 0):
            raise ValueError("sample_indices must be nonnegative")
        configurations = [common] * count
    observed = np.ones(values.shape, dtype=np.bool_)
    horizons = np.zeros(count, dtype=np.int32)
    gaps = np.zeros(count, dtype=np.int32)
    gap_starts = np.full((count, channels), -1, dtype=np.int32)
    # A -1 sentinel means this metadata field is not applicable to the task.
    ratios = np.full(count, -1.0, dtype=np.float32)
    mean_gaps = np.zeros(count, dtype=np.int32)
    modes = np.empty(count, dtype="U16")
    conditions = np.zeros(count, dtype=np.int16)
    for row, cfg in enumerate(configurations):
        rng = np.random.default_rng(int(sample_seeds[row]))
        ordinal = int(sample_indices[row])
        if task == "forecasting":
            choices = tuple(int(item) for item in cfg.get("horizon_choices", FORECAST_HORIZONS))
            if not choices:
                raise ValueError("horizon_choices cannot be empty")
            condition = ordinal % len(choices)
            horizon = int(cfg.get("horizon", choices[condition]))
            if not 0 < horizon < length:
                raise ValueError("Forecast horizon must lie in [1,T-1]")
            observed[row, :, -horizon:] = False
            horizons[row] = horizon
            gap_starts[row] = length - horizon
            modes[row] = "concurrent"
        elif task == "interpolation":
            choices = tuple(int(item) for item in cfg.get("gap_choices", INTERPOLATION_GAPS))
            if not choices:
                raise ValueError("gap_choices cannot be empty")
            condition = ordinal % len(choices)
            gap = int(cfg.get("gap_length", choices[condition]))
            margin = int(cfg.get("edge_margin", 16))
            if not 0 < gap < length or margin < 1 or length - gap < 2 * margin:
                raise ValueError("Interior gap needs visible context on both sides")
            first, last = margin, length - margin - gap
            mode = str(cfg.get("channel_mode", "concurrent"))
            if mode == "concurrent":
                begins = np.repeat(rng.integers(first, last + 1), channels)
                active = range(channels)
            elif mode == "separate":
                begins = rng.integers(first, last + 1, size=channels)
                active = range(channels)
            elif mode == "single_channel":
                begins = np.repeat(rng.integers(first, last + 1), channels)
                active = (int(rng.integers(0, channels)),)
            else:
                raise ValueError("Unsupported interpolation channel mode")
            for channel in active:
                begin = int(begins[channel])
                observed[row, channel, begin:begin + gap] = False
                gap_starts[row, channel] = begin
            gaps[row], modes[row] = gap, mode
        else:
            ratio_choices = tuple(float(item) for item in cfg.get("missing_ratio_choices", IMPUTATION_RATIOS))
            gap_choices = tuple(int(item) for item in cfg.get("mean_gap_choices", IMPUTATION_MEAN_GAPS))
            if not ratio_choices or not gap_choices:
                raise ValueError("Imputation condition choices cannot be empty")
            condition_count = len(ratio_choices) * len(gap_choices) * 2
            condition = ordinal % condition_count
            ratio_index = condition % len(ratio_choices)
            gap_index = (condition // len(ratio_choices)) % len(gap_choices)
            ratio = float(cfg.get("missing_ratio", ratio_choices[ratio_index]))
            mean_gap = int(cfg.get("mean_gap", gap_choices[gap_index]))
            mode = str(cfg.get("channel_mode", "mixed"))
            if mode == "mixed":
                if per_row or not bool(cfg.get("balanced", True)):
                    chance = float(cfg.get("concurrent_probability", 0.5))
                    if not 0 <= chance <= 1:
                        raise ValueError("concurrent_probability must lie in [0,1]")
                    mode = "concurrent" if rng.random() < chance else "separate"
                else:
                    mode = "concurrent" if condition < condition_count // 2 else "separate"
            tolerance = float(cfg.get("ratio_tolerance", 0.03))
            attempts = int(cfg.get("ratio_max_attempts", 100))
            if mode == "concurrent":
                mask = _geometric(length, ratio, mean_gap, rng, tolerance, attempts)
                observed[row] = mask[None, :]
            elif mode == "separate":
                for channel in range(channels):
                    observed[row, channel] = _geometric(length, ratio, mean_gap, rng, tolerance, attempts)
            else:
                raise ValueError("Unsupported imputation channel mode")
            ratios[row], mean_gaps[row], modes[row] = ratio, mean_gap, mode
        conditions[row] = condition
    if np.any(observed.all(axis=(1, 2))) or np.any((~observed).all(axis=(1, 2))):
        raise RuntimeError("Every clip needs both visible and missing samples")
    return {
        "input": np.where(observed, values, np.float32(0.0)),
        "target": values, "observed_mask": observed, "loss_mask": ~observed,
        "metadata": {"task": task, "condition_index": conditions,
                     "horizon": horizons, "gap_length": gaps, "gap_start": gap_starts,
                     "missing_ratio_nominal": ratios, "missing_ratio_actual": (~observed).mean(axis=(1, 2)).astype(np.float32),
                     "mean_gap": mean_gaps, "channel_mode": modes},
    }


def visible_standardization(model_input: np.ndarray, target: np.ndarray,
                            observed_mask: np.ndarray, *, min_std: float = 1e-12
                            ) -> dict[str, np.ndarray]:
    """Standardize each channel using only its visible input values.

    Population mean/std (ddof=0) are measured from ``model_input`` at mask=True
    positions. Neither the target nor missing input positions contribute to
    these statistics. The target undergoes the same affine transform solely
    to provide labels. Missing input positions remain exact zeros.

    ``visible_scale_fallback`` marks channels with std <= ``min_std``; these
    channels use scale=1 in source coordinates, without consulting hidden
    samples. A constant visible context cannot identify an amplitude scale.
    Channels with no visible points are rejected rather than using labels.

    Positive per-channel affine source preprocessing is canceled by this
    transform whenever the observed std is above the threshold. Thus source
    full-record z-score coordinates can be converted into visible-context
    coordinates without using full-record statistics in the model input.
    Predictions return to source coordinates as prediction*scale + center.

    Outputs always include a batch dimension; waveform exports are float32,
    while the four-channel centers/scales are float64 to retain inversion
    precision at negligible storage cost.
    """
    values = np.asarray(model_input, dtype=np.float64)
    labels = np.asarray(target, dtype=np.float64)
    mask = np.asarray(observed_mask)
    if values.ndim == 2:
        values = values[None, ...]
    if labels.ndim == 2:
        labels = labels[None, ...]
    if mask.ndim == 2:
        mask = mask[None, ...]
    if values.ndim != 3 or values.shape[1] != 4 or values.shape[-1] <= 1 or not len(values):
        raise ValueError("Expected input shape [N,4,T] or [4,T] with T > 1")
    if labels.shape != values.shape or mask.shape != values.shape:
        raise ValueError("Input, target, and observed_mask must have the same shape")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("observed_mask must contain only boolean/0/1 values")
    mask = mask.astype(np.bool_)
    threshold = float(min_std)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("min_std must be finite and nonnegative")
    if not np.isfinite(values[mask]).all():
        raise ValueError("Visible input values must be finite")
    if not np.isfinite(labels).all():
        raise ValueError("Supervised target values must be finite")
    counts = mask.sum(axis=-1, keepdims=True, dtype=np.int64)
    if np.any(counts == 0):
        raise ValueError("Every channel needs at least one visible input value")
    sanitized = np.where(mask, values, 0.0)
    centers = sanitized.sum(axis=-1, keepdims=True) / counts
    centered_visible = np.where(mask, values - centers, 0.0)
    variances = np.sum(centered_visible * centered_visible, axis=-1, keepdims=True) / counts
    deviations = np.sqrt(variances)
    if not np.isfinite(centers).all() or not np.isfinite(deviations).all():
        raise FloatingPointError("Visible statistics exceeded finite floating-point range")
    fallback = deviations <= threshold
    scales = np.where(fallback, 1.0, deviations)
    inputs = np.where(mask, centered_visible / scales, 0.0).astype(np.float32)
    targets = ((labels - centers) / scales).astype(np.float32)
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise FloatingPointError("Visible-standardized arrays exceed finite float32 range")
    return {"input_visible": inputs, "target_visible": targets,
            "visible_center": centers, "visible_scale": scales,
            "visible_scale_fallback": fallback.astype(np.uint8),
            "visible_observed_count": counts}
