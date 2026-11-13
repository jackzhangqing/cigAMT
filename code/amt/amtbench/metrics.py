"""Case-wise, streaming metrics for the seven AMT benchmark tasks.

No Torch import is required. Inputs are NumPy arrays with batch axis first.
Noise-task waveforms are restored with the *input* RMS scale before reporting
source-coordinate errors. SNR, NRMSE and Pearson are computed per channel and
then averaged, so magnetic-channel amplitude cannot dominate electric channels.
Source coordinates are not certified physical SI units.

Clean-task primary losses use the visible-input normalization and missing points
only. The model must receive observed_mask, never the hidden target or loss_mask
as an extra feature. A completion preserves observed input points explicitly.

Undefined statistics are None, never NaN/Infinity, for strict JSON/CSV export.
SNR uses an explicit relative error-energy floor of 1e-12 (ceiling 120 dB).
Clean controls do not receive SNR/improvement scores. Localization uses the
channel-mean event probability at a fixed threshold, without point adjustment.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


NOISE_TASKS = frozenset(("noise_localization", "noise_classification", "denoising", "frequency_denoising"))
CLEAN_TASKS = frozenset(("forecasting", "interpolation", "imputation"))
SNR_ERROR_ENERGY_FLOOR = 1e-12
BAND_RELATIVE_ENERGY_FLOOR = 1e-12
METRIC_INFO = {
    "snr_input_db": {"direction": "higher", "unit": "dB", "coordinate": "source; channel mean"},
    "snr_output_db": {"direction": "higher", "unit": "dB", "coordinate": "source; channel mean"},
    "snr_improvement_db": {"direction": "higher", "unit": "dB", "coordinate": "source; channel mean"},
    "nrmse": {"direction": "lower", "unit": "1", "coordinate": "source; channel mean"},
    "rmse": {"direction": "lower", "unit": "source units", "coordinate": "source; channel mean"},
    "mae": {"direction": "lower", "unit": "source units", "coordinate": "source; channel mean"},
    "pearson": {"direction": "higher", "unit": "1", "coordinate": "channel mean"},
    "log_spectral_rmse": {"direction": "lower", "unit": "log10 amplitude", "coordinate": "channel mean; DC excluded"},
    "mse": {"direction": "lower", "unit": "normalized squared units", "coordinate": "model input"},
    "masked_mse": {"direction": "lower", "unit": "normalized squared units", "coordinate": "visible normalization; missing only"},
    "masked_mae": {"direction": "lower", "unit": "normalized units", "coordinate": "visible normalization; missing only"},
    "masked_nrmse": {"direction": "lower", "unit": "1", "coordinate": "source; missing only; channel mean"},
    "f1": {"direction": "higher", "unit": "1", "coordinate": "event mask; case macro; positive cases"},
    "iou": {"direction": "higher", "unit": "1", "coordinate": "event mask; case macro; positive cases"},
    "event_f1": {"direction": "higher", "unit": "1", "coordinate": "event IoU >= 0.5; positive cases"},
    "onset_mae_seconds": {"direction": "lower", "unit": "s", "coordinate": "matched events only"},
    "offset_mae_seconds": {"direction": "lower", "unit": "s", "coordinate": "matched events only"},
    "clean_false_positive_rate": {"direction": "lower", "unit": "1", "coordinate": "clean-control points only"},
    "clean_any_event_false_alarm": {"direction": "lower", "unit": "1", "coordinate": "clean-control cases only"},
    "accuracy": {"direction": "higher", "unit": "1", "coordinate": "class prediction"},
    "macro_f1_present_true": {"direction": "higher", "unit": "1", "coordinate": "pooled confusion; classes present in truth"},
    "macro_f1_all_classes": {"direction": "higher", "unit": "1", "coordinate": "pooled confusion; all configured classes; absent=0"},
    "low_nrmse": {"direction": "lower", "unit": "1", "coordinate": "source; low head; channel mean"},
    "high_nrmse": {"direction": "lower", "unit": "1", "coordinate": "source; high head; channel mean"},
}


def primary_metric(task: str) -> str:
    """Test-summary ranking key; checkpoint selection may use validation loss."""
    if task == "noise_localization":
        return "f1"
    if task == "noise_classification":
        return "macro_f1_present_true"
    if task in ("denoising", "frequency_denoising"):
        return "nrmse"
    if task in CLEAN_TASKS:
        return "masked_mse"
    raise ValueError(f"Unknown task: {task}")


def _mean(values: Any) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if valid.size else None


def _safe(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _wave(value: Any, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 2:
        value = value[None]
    if value.ndim != 3:
        raise ValueError(f"{name} must be [B,C,T] (or single [C,T]), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _field(batch: Mapping[str, Any], key: str, shape: tuple[int, ...], default: float) -> np.ndarray:
    value = np.asarray(batch.get(key, default), dtype=np.float64)
    if value.ndim == 2 and len(shape) == 3 and value.shape == shape[1:]:
        value = value[None]
    try:
        return np.broadcast_to(value, shape)
    except ValueError as exc:
        raise ValueError(f"{key} with shape {value.shape} cannot broadcast to {shape}") from exc


def _flag(batch: Mapping[str, Any], key: str, index: int, default: bool = False) -> bool:
    if key not in batch:
        return default
    v = np.asarray(batch[key]).reshape(-1)
    return bool(v[0 if len(v) == 1 else index])


def _prediction(outputs: Mapping[str, Any] | np.ndarray, key: str = "prediction") -> np.ndarray:
    if not isinstance(outputs, Mapping):
        if key != "prediction":
            raise ValueError(f"Output mapping must contain {key}")
        return _wave(outputs, key)
    if key not in outputs:
        raise ValueError(f"Output mapping must contain {key}; found {sorted(outputs)}")
    return _wave(outputs[key], key)


def _channel_error(target: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None,
                   energy_floor: np.ndarray | None = None) -> dict[str, Any]:
    mask = np.ones_like(target, dtype=bool) if mask is None else mask.astype(bool)
    rmse, mae, nrmse, corr, relative_mse = [], [], [], [], []
    for ch in range(target.shape[0]):
        t, p = target[ch, mask[ch]], pred[ch, mask[ch]]
        if t.size == 0:
            continue
        err = p - t
        mse = float(np.mean(err * err))
        energy = float(np.mean(t * t))
        rmse.append(math.sqrt(max(0.0, mse)))
        mae.append(float(np.mean(np.abs(err))))
        floor = np.finfo(np.float64).tiny if energy_floor is None else max(np.finfo(np.float64).tiny, float(energy_floor[ch]))
        if energy > floor:
            nrmse.append(math.sqrt(mse / energy))
            relative_mse.append(mse / energy)
        tc, pc = t - t.mean(), p - p.mean()
        denominator = float(np.linalg.norm(tc) * np.linalg.norm(pc))
        if denominator > np.finfo(np.float64).tiny:
            corr.append(float(np.clip(np.dot(tc, pc) / denominator, -1.0, 1.0)))
    return {"rmse": _mean(rmse), "mae": _mean(mae), "nrmse": _mean(nrmse),
            "pearson": _mean(corr), "valid_nrmse_channels": len(nrmse),
            "valid_pearson_channels": len(corr), "_relative_mse": relative_mse}


def _snr(target: np.ndarray, pred: np.ndarray) -> tuple[float | None, int, int]:
    values, capped = [], 0
    for t, p in zip(target, pred):
        energy = float(np.mean(t * t))
        if energy <= np.finfo(np.float64).tiny:
            continue
        ratio = float(np.mean((p - t) ** 2)) / energy
        capped += int(ratio < SNR_ERROR_ENERGY_FLOOR)
        values.append(-10.0 * math.log10(max(ratio, SNR_ERROR_ENERGY_FLOOR)))
    return _mean(values), len(values), capped


def _spectral(target: np.ndarray, pred: np.ndarray) -> float | None:
    values = []
    for t, p in zip(target, pred):
        reference = np.abs(np.fft.rfft(t))[1:]
        estimate = np.abs(np.fft.rfft(p))[1:]
        if not reference.size or float(np.max(reference)) <= np.finfo(np.float64).tiny:
            continue
        floor = float(np.max(reference)) * 1e-8
        d = np.log10(np.maximum(estimate, floor)) - np.log10(np.maximum(reference, floor))
        values.append(math.sqrt(float(np.mean(d * d))))
    return _mean(values)


def _denoising(batch: Mapping[str, Any], outputs: Mapping[str, Any] | np.ndarray) -> list[dict[str, Any]]:
    target, inp, pred = _wave(batch["target"], "target"), _wave(batch["input"], "input"), _prediction(outputs)
    if not target.shape == inp.shape == pred.shape:
        raise ValueError("input, target and prediction must have identical shapes")
    scale = _field(batch, "scale", (len(target), target.shape[1], 1), 1.0)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("scale must be finite and positive")
    source_t, source_i, source_p = target * scale, inp * scale, pred * scale
    rows = []
    for i in range(len(target)):
        result = _channel_error(source_t[i], source_p[i])
        result.pop("_relative_mse")
        clean = _flag(batch, "is_clean", i, bool(np.array_equal(inp[i], target[i])))
        sin, count, _ = _snr(source_t[i], source_i[i])
        sout, _, capped = _snr(source_t[i], source_p[i])
        result.update(mse=float(np.mean((pred[i] - target[i]) ** 2)),
                      log_spectral_rmse=_spectral(source_t[i], source_p[i]), is_clean=int(clean),
                      snr_input_db=None if clean else sin, snr_output_db=None if clean else sout,
                      snr_improvement_db=None if clean or sin is None or sout is None else sout - sin,
                      valid_snr_channels=0 if clean else count,
                      snr_output_capped_channels=0 if clean else capped,
                      clean_preservation_nrmse=result["nrmse"] if clean else None)
        rows.append(result)
    return rows


def _clean_task(batch: Mapping[str, Any], outputs: Mapping[str, Any] | np.ndarray) -> list[dict[str, Any]]:
    target, inp, pred = _wave(batch["target"], "target"), _wave(batch["input"], "input"), _prediction(outputs)
    if not target.shape == inp.shape == pred.shape:
        raise ValueError("input, target and prediction must have identical shapes")
    shape = target.shape
    if "loss_mask" not in batch and "observed_mask" not in batch:
        raise ValueError("Clean tasks require loss_mask or observed_mask")
    mask = (_field(batch, "loss_mask", shape, 0) > 0.5) if "loss_mask" in batch else ~(_field(batch, "observed_mask", shape, 1) > 0.5)
    observed = (_field(batch, "observed_mask", shape, 1) > 0.5) if "observed_mask" in batch else ~mask
    if np.any(mask & observed) or np.any(~(mask | observed)):
        raise ValueError("loss_mask and observed_mask must be complementary")
    center = _field(batch, "visible_center", (shape[0], shape[1], 1), 0.0)
    scale = _field(batch, "visible_scale", (shape[0], shape[1], 1), 1.0)
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)) or not np.all(np.isfinite(center)):
        raise ValueError("Visible normalization parameters must be finite with positive scale")
    source_t, source_p = target * scale + center, pred * scale + center
    rows = []
    for i in range(shape[0]):
        missing, known = mask[i], observed[i]
        err = pred[i] - target[i]
        source = _channel_error(source_t[i], source_p[i], missing)
        n = int(missing.sum())
        rows.append({"masked_mse": _mean(err[missing] ** 2), "masked_mae": _mean(np.abs(err[missing])),
                     "masked_nrmse": source["nrmse"], "source_masked_rmse": source["rmse"],
                     "source_masked_mae": source["mae"], "masked_pearson": source["pearson"],
                     "valid_nrmse_channels": source["valid_nrmse_channels"],
                     "missing_points": n, "missing_fraction": n / float(missing.size),
                     "observed_prediction_rmse": math.sqrt(float(np.mean(err[known] ** 2))) if known.any() else None,
                     # The exported completion is observed*input + missing*prediction.
                     "observed_preservation_rmse": 0.0 if known.any() else None})
    return rows


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask.astype(bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _event_matches(truth: list[tuple[int, int]], pred: list[tuple[int, int]], threshold: float = 0.5) -> list[tuple[int, int]]:
    if not truth or not pred:
        return []
    iou = np.zeros((len(truth), len(pred)), dtype=np.float64)
    for i, (a, b) in enumerate(truth):
        for j, (c, d) in enumerate(pred):
            intersection = max(0, min(b, d) - max(a, c))
            iou[i, j] = intersection / float(max(b, d) - min(a, c))
    score = (iou >= threshold) * (min(iou.shape) + 1) + iou
    try:
        from scipy.optimize import linear_sum_assignment
        ii, jj = linear_sum_assignment(-score)
        pairs = zip(ii.tolist(), jj.tolist())
    except ImportError:
        # At IoU>=0.5 for disjoint runs, candidate ambiguity is restricted to ties.
        candidates = sorted(zip(*np.where(iou >= threshold)), key=lambda p: (-iou[p], p))
        used_i, used_j, selected = set(), set(), []
        for i, j in candidates:
            if i not in used_i and j not in used_j:
                selected.append((i, j)); used_i.add(i); used_j.add(j)
        pairs = selected
    return [(i, j) for i, j in pairs if iou[i, j] >= threshold]


def _localization(batch: Mapping[str, Any], outputs: Mapping[str, Any] | np.ndarray,
                  threshold: float, sampling_rate: float) -> list[dict[str, Any]]:
    gt = _wave(batch["noise_mask"], "noise_mask")
    if isinstance(outputs, Mapping):
        if "probabilities" in outputs:
            score = np.asarray(outputs["probabilities"], dtype=np.float64)
            is_logits = False
        else:
            score = np.asarray(outputs.get("logits", outputs.get("prediction")), dtype=np.float64)
            is_logits = True
    else:
        score = np.asarray(outputs, dtype=np.float64); is_logits = True
    if score.ndim == 2:
        score = score[:, None, :]
    if score.ndim != 3 or score.shape[0] != gt.shape[0] or score.shape[-1] != gt.shape[-1]:
        raise ValueError("Localization output must be [B,1,T], [B,C,T] or [B,T]")
    if not np.all(np.isfinite(score)):
        raise ValueError("Localization output contains non-finite values")
    if is_logits:
        score = np.exp(-np.logaddexp(0.0, -score))
    elif np.any((score < 0) | (score > 1)):
        raise ValueError("probabilities must lie within [0,1]")
    probabilities = score.mean(axis=1)
    if not np.all((gt > 0.5) == (gt[:, :1] > 0.5)):
        raise ValueError("Expected a shared four-channel event mask, found differing channel labels")
    rows = []
    for k in range(len(gt)):
        truth, pred = gt[k, 0] > 0.5, probabilities[k] >= threshold
        tp, fp, fn, tn = (int(np.sum(a)) for a in (truth & pred, ~truth & pred, truth & ~pred, ~truth & ~pred))
        has_positive = bool(truth.any())
        precision = tp / float(tp + fp) if tp + fp else (0.0 if has_positive else None)
        recall = tp / float(tp + fn) if has_positive else None
        f1 = 2 * tp / float(2 * tp + fp + fn) if has_positive else None
        iou = tp / float(tp + fp + fn) if has_positive else None
        events_t, events_p = _segments(truth), _segments(pred)
        matches = _event_matches(events_t, events_p)
        nm = len(matches)
        onset = [abs(events_t[i][0] - events_p[j][0]) / sampling_rate for i, j in matches]
        offset = [abs(events_t[i][1] - events_p[j][1]) / sampling_rate for i, j in matches]
        rows.append({"precision": precision, "recall": recall, "f1": f1, "iou": iou,
                     "pointwise_accuracy": (tp + tn) / float(truth.size), "true_positive": tp,
                     "false_positive": fp, "false_negative": fn, "true_negative": tn,
                     "false_positive_rate": fp / float(fp + tn) if fp + tn else None,
                     "is_clean": int(not has_positive),
                     "clean_false_positive_rate": fp / float(truth.size) if not has_positive else None,
                     "clean_any_event_false_alarm": int(pred.any()) if not has_positive else None,
                     "event_precision": nm / float(len(events_p)) if events_p else (0.0 if events_t else None),
                     "event_recall": nm / float(len(events_t)) if events_t else None,
                     "event_f1": 2 * nm / float(len(events_t) + len(events_p)) if events_t else None,
                     "true_events": len(events_t), "predicted_events": len(events_p), "matched_events": nm,
                     "onset_mae_seconds": _mean(onset), "offset_mae_seconds": _mean(offset)})
    return rows


def _classification(batch: Mapping[str, Any], outputs: Mapping[str, Any] | np.ndarray) -> list[dict[str, Any]]:
    labels = np.asarray(batch["class_id"]).reshape(-1)
    if isinstance(outputs, Mapping):
        if "probabilities" in outputs:
            probs = np.asarray(outputs["probabilities"], dtype=np.float64)
            if probs.ndim != 2 or not np.all(np.isfinite(probs)) or np.any(probs < 0) or not np.allclose(probs.sum(axis=1), 1.0, rtol=1e-5, atol=1e-7):
                raise ValueError("Class probabilities must be finite, nonnegative [B,K] rows summing to 1")
            log_probs = np.log(np.maximum(probs, np.finfo(np.float64).tiny))
        else:
            logits = np.asarray(outputs.get("logits", outputs.get("prediction")), dtype=np.float64)
            probs = None
    else:
        logits = np.asarray(outputs, dtype=np.float64); probs = None
    if probs is None:
        if logits.ndim != 2 or not np.all(np.isfinite(logits)):
            raise ValueError("Class logits must be a finite [B,K] array")
        logits = logits - logits.max(axis=1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        probs = np.exp(log_probs)
    if len(labels) != len(probs) or np.any(labels != labels.astype(np.int64)) or np.any((labels < 0) | (labels >= probs.shape[1])):
        raise ValueError("class_id must contain one valid integer class per output")
    predictions = probs.argmax(axis=1)
    return [{"true_class": int(t), "pred_class": int(p), "accuracy": int(t == p),
             "cross_entropy": float(-log_probs[i, int(t)]), "confidence": float(probs[i, p]),
             "brier_score": float(np.sum((probs[i] - np.eye(1, probs.shape[1], int(t))[0]) ** 2))}
            for i, (t, p) in enumerate(zip(labels, predictions))]


def compute_metrics(task: str, batch: Mapping[str, Any], outputs: Mapping[str, Any] | np.ndarray,
                    threshold: float = 0.5, sampling_rate: float = 150) -> list[dict[str, Any]]:
    """Return one small, strict-JSON-safe numeric record per input case.

    ``logits`` are expected for classification/localization; use the explicit
    ``probabilities`` key when supplying probabilities. Frequency outputs need
    ``low`` and ``high``; optional ``prediction`` is an independent reconstruction.
    For clean tasks, input/target must already use the same visible normalization.
    """
    if not (0.0 <= threshold <= 1.0) or sampling_rate <= 0:
        raise ValueError("threshold must be [0,1] and sampling_rate must be positive")
    if task == "noise_localization":
        return _localization(batch, outputs, threshold, sampling_rate)
    if task == "noise_classification":
        return _classification(batch, outputs)
    if task in CLEAN_TASKS:
        return _clean_task(batch, outputs)
    if task == "denoising":
        return _denoising(batch, outputs)
    if task == "frequency_denoising":
        low, high = _prediction(outputs, "low"), _prediction(outputs, "high")
        target = _wave(batch["target"], "target")
        if low.shape != target.shape or high.shape != target.shape:
            raise ValueError("Frequency heads must have target waveform shape")
        prediction = _prediction(outputs) if isinstance(outputs, Mapping) and "prediction" in outputs else low + high
        rows = _denoising(batch, {"prediction": prediction})
        scale = _field(batch, "scale", (len(target), target.shape[1], 1), 1.0)
        references = {name: _wave(batch[f"target_{name}"], f"target_{name}") for name in ("low", "high")}
        if any(ref.shape != target.shape for ref in references.values()):
            raise ValueError("Frequency targets must have reconstruction target shape")
        for i, result in enumerate(rows):
            band_floor = np.mean((target[i] * scale[i]) ** 2, axis=-1) * BAND_RELATIVE_ENERGY_FLOOR
            for name, value in (("low", low), ("high", high)):
                ref = references[name]
                part = _channel_error(ref[i] * scale[i], value[i] * scale[i], energy_floor=band_floor)
                result[f"{name}_nrmse"] = part["nrmse"]
                result[f"{name}_valid_nrmse_channels"] = part["valid_nrmse_channels"]
                result[f"{name}_mse"] = float(np.mean((ref[i] - value[i]) ** 2))
            result["sum_consistency_rmse"] = math.sqrt(float(np.mean((prediction[i] - low[i] - high[i]) ** 2)))
        return rows
    raise ValueError(f"Unknown task: {task}")


class MetricAccumulator:
    """O(number of metrics + classes squared) streaming per-case summaries.

    Mean/std are over cases with a defined metric, not over waveform points.
    Counts disclose how many cases contributed. Classification macro scores are
    computed from pooled confusion, never averaged from per-case F1. Metadata
    identifiers are ignored. Create a separate accumulator for each stratum.
    """

    _SKIP = frozenset(("true_class", "pred_class", "is_clean", "sample_index", "source_id", "source_row",
                       "class_id", "domain_id", "noise_index", "noise_bank_row", "noise_start", "noise_length",
                       "clean_index", "draw_seed", "condition_index", "snr_condition_index", "horizon",
                       "gap_length", "gap_start", "channel_mode_id", "seed", "index"))

    def __init__(self, num_classes: int = 11):
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = int(num_classes)
        self.count = 0
        self.clean_count = 0
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self._stats: dict[str, list[float]] = {}
        self._totals: dict[str, int] = defaultdict(int)

    def update(self, rows: Iterable[Mapping[str, Any]] | Mapping[str, Any]) -> "MetricAccumulator":
        if isinstance(rows, Mapping):
            rows = [rows]
        for row in rows:
            self.count += 1
            self.clean_count += int(bool(row.get("is_clean", False)))
            if "true_class" in row and "pred_class" in row:
                t, p = int(row["true_class"]), int(row["pred_class"])
                if not (0 <= t < self.num_classes and 0 <= p < self.num_classes):
                    raise ValueError(f"Class {t}/{p} exceeds configured num_classes={self.num_classes}")
                self.confusion[t, p] += 1
            for key, value in row.items():
                if key in self._SKIP:
                    continue
                # Register undefined metrics too, so count=0 remains visible.
                if value is None:
                    self._stats.setdefault(key, [0, 0.0, 0.0])
                    continue
                if not isinstance(value, (int, float, bool, np.number)):
                    continue
                v = float(value)
                stats = self._stats.setdefault(key, [0, 0.0, 0.0])
                if not math.isfinite(v):
                    continue
                stats[0] += 1
                delta = v - stats[1]
                stats[1] += delta / stats[0]
                stats[2] += delta * (v - stats[1])
                if key in ("true_positive", "false_positive", "false_negative", "true_negative", "true_events", "predicted_events", "matched_events"):
                    self._totals[key] += int(value)
        return self

    def summary(self) -> dict[str, Any]:
        metrics = {key: {"mean": _safe(v[1]) if v[0] else None,
                         "std": math.sqrt(max(0.0, v[2] / v[0])) if v[0] else None,
                         "count": int(v[0])} for key, v in sorted(self._stats.items())}
        # Keep scalar means at top level for convenient CSV rows/ranking.
        result: dict[str, Any] = {key: value["mean"] for key, value in metrics.items()}
        result.update(count=self.count, clean_count=self.clean_count, metrics=metrics)
        if self.confusion.sum():
            cm = self.confusion.astype(np.float64)
            support, predicted, tp = cm.sum(axis=1), cm.sum(axis=0), cm.diagonal()
            precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
            recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
            f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp), where=(support + predicted) > 0)
            present, union = support > 0, (support + predicted) > 0
            result.update(confusion_matrix=self.confusion.tolist(), class_labels=list(range(self.num_classes)),
                          class_support=support.astype(np.int64).tolist(), class_precision=precision.tolist(),
                          class_recall=recall.tolist(), class_f1=f1.tolist(),
                          classification_count=int(cm.sum()), pooled_accuracy=float(tp.sum() / cm.sum()),
                          macro_f1_present_true=float(f1[present].mean()),
                          macro_f1_observed_union=float(f1[union].mean()),
                          macro_f1_all_classes=float(f1.mean()), balanced_accuracy=float(recall[present].mean()),
                          weighted_f1=float(np.dot(f1, support) / support.sum()))
            present_noise = present & (np.arange(self.num_classes) != 0)
            result["macro_f1_noise_only_present_true"] = float(f1[present_noise].mean()) if present_noise.any() else None
        if "true_positive" in self._totals:
            tp, fp, fn = (self._totals[k] for k in ("true_positive", "false_positive", "false_negative"))
            result["pooled_pointwise_f1"] = 2 * tp / float(2 * tp + fp + fn) if 2 * tp + fp + fn else None
            result["pooled_pointwise_iou"] = tp / float(tp + fp + fn) if tp + fp + fn else None
            result["localization_totals"] = dict(self._totals)
        return result


def aggregate_records(rows: Iterable[Mapping[str, Any]], num_classes: int = 11) -> dict[str, Any]:
    """Convenience wrapper; iterable may be a CSV/JSONL generator."""
    return MetricAccumulator(num_classes=num_classes).update(rows).summary()


summarize_rows = aggregate_records
