"""Deterministic, target-free classical controls for the AMT benchmark.

All methods estimate from batch['input']; missing-data methods use observed_mask
only. No method reads batch['target'], noise_mask or class_id. The robust detector's
threshold is a fixed preset; any future tuning must use validation, never test.
"""
import numpy as np

from .models import CLEAN_TASKS, TASKS


CLASSICAL_BY_TASK = {
    "noise_localization": ("robust_threshold",),
    "noise_classification": (),
    "denoising": ("identity", "zero", "wavelet"),
    "frequency_denoising": ("identity", "zero", "wavelet"),
    "forecasting": ("persistence", "mean_fill", "ridge_ar"),
    "interpolation": ("linear_interp", "mean_fill"),
    "imputation": ("linear_interp", "mean_fill"),
}


def supported_classical(task):
    if task not in TASKS:
        raise ValueError("Unknown task %r" % task)
    return CLASSICAL_BY_TASK[task]


def split_frequency(prediction, sample_rate=150.0, cutoff_hz=5.0):
    """Complementary real FFT split matching the stored <=5Hz target convention."""
    if sample_rate <= 0 or not 0 <= cutoff_hz <= sample_rate/2:
        raise ValueError("Invalid frequency split settings")
    prediction = np.asarray(prediction)
    frequencies = np.fft.rfftfreq(prediction.shape[-1], 1.0/sample_rate)
    spectrum = np.fft.rfft(prediction.astype(np.float64), axis=-1)
    spectrum[..., frequencies > cutoff_hz] = 0
    low = np.fft.irfft(spectrum, n=prediction.shape[-1], axis=-1).astype(np.float32)
    return dict(low=low, high=(prediction-low).astype(np.float32), prediction=prediction.astype(np.float32))


def _wavelet(x, wavelet="db4", level=None, threshold_scale=1.0):
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("wavelet baseline requires PyWavelets; install the benchmark requirements") from exc
    result = np.empty_like(x, dtype=np.float32)
    for b in range(len(x)):
        for c in range(4):
            series = x[b, c].astype(np.float64)
            w = pywt.Wavelet(wavelet)
            maximum = pywt.dwt_max_level(len(series), w.dec_len)
            selected = maximum if level is None else min(int(level), maximum)
            if selected < 1:
                result[b, c] = series
                continue
            coefficients = pywt.wavedec(series, w, mode="symmetric", level=selected)
            detail = coefficients[-1]
            sigma = np.median(np.abs(detail-np.median(detail))) / 0.6744897501960817
            threshold = threshold_scale*sigma*np.sqrt(2*np.log(max(len(series), 2)))
            filtered = [coefficients[0]] + [pywt.threshold(v, threshold, mode="soft") for v in coefficients[1:]]
            result[b, c] = pywt.waverec(filtered, w, mode="symmetric")[:len(series)]
    return result


def _fill(x, mask, name):
    result = (x*mask).copy()
    positions = np.arange(x.shape[-1])
    for b in range(len(x)):
        for c in range(4):
            known = np.flatnonzero(mask[b, c])
            missing = ~mask[b, c]
            if not len(known):
                result[b, c, missing] = 0
            elif name == "mean_fill":
                result[b, c, missing] = np.mean(x[b, c, known], dtype=np.float64)
            elif name == "persistence":
                # Forecasting's missing suffix follows the final visible value.
                result[b, c, missing] = x[b, c, known[-1]]
            else:
                result[b, c, missing] = np.interp(positions[missing], known, x[b, c, known])
    return result.astype(np.float32)


def _ridge_ar(x, mask, order=16, ridge=1e-2):
    if order < 1 or ridge <= 0:
        raise ValueError("AR order and ridge penalty must be positive")
    result = _fill(x, mask, "persistence")
    for b in range(len(x)):
        for c in range(4):
            known = np.flatnonzero(mask[b, c])
            if not len(known):
                continue
            n = int(known[-1]+1)
            if len(known) != n:
                raise ValueError("ridge_ar supports a contiguous visible prefix followed by a missing suffix")
            history = x[b, c, :n].astype(np.float64)
            p = min(order, max((n-1)//3, 0))
            if p < 1:
                continue
            center, scale = history.mean(), history.std()
            if scale <= 1e-12:
                continue
            normalized = (history-center)/scale
            design = np.stack([normalized[j-p:j][::-1] for j in range(p, n)])
            values = normalized[p:]
            gram = design.T.dot(design) + ridge*np.eye(p)
            coefficient = np.linalg.solve(gram, design.T.dot(values))
            series = list(normalized)
            for j in range(n, x.shape[-1]):
                # A bounded recursive extrapolation prevents unstable AR poles
                # from producing NaNs; bounds use the observed prefix only.
                value = float(np.clip(np.dot(coefficient, series[-p:][::-1]), -15, 15))
                series.append(value)
                result[b, c, j] = value*scale+center
    return result.astype(np.float32)


def classical_predict(name, task, batch, **kwargs):
    """Return NumPy outputs matching the neural output keys; never use labels.

    Settings: robust_threshold threshold=4, window=17;
    wavelet wavelet='db4', level=None, threshold_scale=1;
    ridge_ar order=16, ridge=.01; frequency sample_rate=150, cutoff_hz=5.
    """
    if name not in supported_classical(task):
        raise ValueError("Classical baseline %r is unsupported for %s; choose %s" %
                         (name, task, supported_classical(task)))
    allowed = dict(robust_threshold={"threshold", "window"},
                   wavelet={"wavelet", "level", "threshold_scale"},
                   ridge_ar={"order", "ridge"}).get(name, set()).copy()
    if task == "frequency_denoising":
        allowed |= {"sample_rate", "cutoff_hz"}
    unknown = set(kwargs)-allowed
    if unknown:
        raise TypeError("Unknown classical parameters: %s" % sorted(unknown))
    x = np.asarray(batch["input"], dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 4 or not np.isfinite(x).all():
        raise ValueError("input must be a finite NumPy [B,4,T] array")
    if task in CLEAN_TASKS:
        mask = np.asarray(batch["observed_mask"], dtype=bool)
        if mask.shape != x.shape:
            raise ValueError("observed_mask shape must match input")
        if name == "ridge_ar":
            prediction = _ridge_ar(x, mask, **kwargs)
        else:
            prediction = _fill(x, mask, name)
        return {"prediction": prediction}
    if name == "robust_threshold":
        threshold = float(kwargs.get("threshold", 4.0))
        window = int(kwargs.get("window", 17))
        if window < 1 or window % 2 != 1 or window > x.shape[-1]:
            raise ValueError("window must be positive, odd, and no longer than input")
        center = np.median(x, axis=-1, keepdims=True)
        mad = 1.4826*np.median(np.abs(x-center), axis=-1, keepdims=True)
        # Avoid amplifying floating-point jitter on practically constant channels.
        floor = np.maximum(np.sqrt(np.mean(x.astype(np.float64)**2, axis=-1, keepdims=True))*1e-6, 1e-8)
        normalized = (x-center)/np.maximum(mad, floor)
        energy = np.mean(normalized.astype(np.float64)**2, axis=1)
        kernel = np.ones(window)/window
        smooth = np.stack([np.convolve(np.pad(v, window//2, mode="edge"), kernel, mode="valid") for v in energy])
        score = np.sqrt(np.maximum(smooth, 0))
        return {"logits": np.clip(score[:, None]-threshold, -40, 40).astype(np.float32)}
    frequency_kwargs = {k: kwargs[k] for k in ("sample_rate", "cutoff_hz") if k in kwargs}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in frequency_kwargs}
    prediction = x.copy() if name == "identity" else (np.zeros_like(x) if name == "zero" else _wavelet(x, **filtered_kwargs))
    return split_frequency(prediction, **frequency_kwargs) if task == "frequency_denoising" else {"prediction": prediction}
