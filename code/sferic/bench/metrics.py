"""Binary ranking, fixed-threshold evaluation, and record-cluster uncertainty."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score)


METRIC_NAMES = ("pr_auc", "roc_auc", "f1", "precision", "recall", "balanced_accuracy", "mcc", "brier", "ece")


def _inputs(y_true, scores, probabilities=None):
    original_y = np.asarray(y_true).reshape(-1)
    if not np.isin(original_y, [0, 1]).all():
        raise ValueError("Labels must be actual binary 0/1 values")
    y = original_y.astype(np.int64)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not len(y) or y.shape != score.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError("Expected nonempty, aligned binary labels and finite continuous scores")
    probability = None if probabilities is None else np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probability is not None and (probability.shape != y.shape or not np.isfinite(probability).all()
                                    or np.any((probability < 0) | (probability > 1))):
        raise ValueError("Probabilities must be finite aligned values in [0,1]; decision scores are not probabilities")
    return y, score, probability


def expected_calibration_error(y_true, probabilities, n_bins=10):
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    y, _, p = _inputs(y_true, probabilities, probabilities)
    bins = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    return float(sum(np.mean(bins == i) * abs(float(y[bins == i].mean()) - float(p[bins == i].mean()))
                     for i in range(n_bins) if np.any(bins == i)))


def compute_metrics(y_true, scores, threshold=0.5, probabilities=None, n_bins=10):
    y, score, p = _inputs(y_true, scores, probabilities)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    predicted = score >= threshold
    both_classes = len(np.unique(y)) == 2
    warnings = []
    if not both_classes:
        warnings.append("Only one ground-truth class: ranking AUC, balanced accuracy, and MCC are undefined and reported as null")
    if p is None:
        warnings.append("No calibrated/probability output supplied: Brier score and ECE are unavailable")
    result = dict(pr_auc=float(average_precision_score(y, score)) if both_classes else None,
        roc_auc=float(roc_auc_score(y, score)) if both_classes else None,
        f1=float(f1_score(y, predicted, zero_division=0)),
        precision=float(precision_score(y, predicted, zero_division=0)),
        recall=float(recall_score(y, predicted, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y, predicted)) if both_classes else None,
        mcc=float(matthews_corrcoef(y, predicted)) if both_classes else None,
        confusion_matrix=confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        brier=float(brier_score_loss(y, p)) if p is not None else None,
        ece=expected_calibration_error(y, p, n_bins) if p is not None else None,
        n=int(len(y)), positives=int(y.sum()), negatives=int(len(y) - y.sum()),
        threshold=float(threshold), warnings=warnings,
        pr_auc_definition="average_precision (not trapezoidal PR integration)")
    return result


def choose_threshold(y_val, scores_val, method="best_f1", target_recall=0.9, probabilities=None):
    """Select using VALIDATION labels only; the returned threshold is frozen.

    Best-F1 ties use the highest threshold. Target-recall uses the highest
    validation threshold meeting the target, without consulting test labels.
    """
    y, score, probability = _inputs(y_val, scores_val, probabilities)
    if method not in {"best_f1", "target_recall"} or not 0 < target_recall <= 1:
        raise ValueError("Unknown threshold method or invalid target_recall")
    warnings = []
    if len(np.unique(y)) < 2:
        threshold = float(np.median(score))
        warnings.append("Validation has only one class; threshold optimization is unavailable; a median-score fallback is explicit")
    else:
        precision, recall, thresholds = precision_recall_curve(y, score)
        if method == "best_f1":
            numerator = 2 * precision[:-1] * recall[:-1]
            denominator = precision[:-1] + recall[:-1]
            f1 = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
            candidates = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
        else:
            candidates = np.flatnonzero(recall[:-1] >= target_recall)
        threshold = float(thresholds[candidates[-1]])
    return dict(threshold=threshold, method=method, target_recall=float(target_recall) if method == "target_recall" else None,
        selection_split="validation_only", validation_metrics=compute_metrics(y, score, threshold, probability), warnings=warnings)


def clustered_bootstrap(y_true, scores, record_ids, threshold, probabilities=None,
                        n_bootstrap=1000, seed=0, confidence=0.95):
    """Resample entire records with replacement, retaining within-record rows."""
    y, score, probability = _inputs(y_true, scores, probabilities)
    groups = np.asarray(record_ids, dtype=object).reshape(-1)
    if groups.shape != y.shape or any(value is None or str(value) == "" for value in groups):
        raise ValueError("Every row needs a real record ID for clustered bootstrap")
    if n_bootstrap < 0 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap count/confidence")
    base = compute_metrics(y, score, threshold, probability)
    unique = list(dict.fromkeys(groups.tolist()))
    warnings = []
    values = {name: [] for name in METRIC_NAMES}
    if len(unique) < 2:
        warnings.append("Fewer than two independent records; a record-cluster confidence interval cannot be estimated")
    elif n_bootstrap == 0:
        warnings.append("Bootstrap disabled by n_bootstrap=0")
    else:
        indices = [np.flatnonzero(groups == group) for group in unique]
        rng = np.random.default_rng(seed)
        for _ in range(n_bootstrap):
            selected = rng.integers(0, len(unique), size=len(unique))
            rows = np.concatenate([indices[int(i)] for i in selected])
            metrics = compute_metrics(y[rows], score[rows], threshold,
                                      None if probability is None else probability[rows])
            for name in METRIC_NAMES:
                if metrics[name] is not None and np.isfinite(metrics[name]):
                    values[name].append(metrics[name])
    alpha = (1 - confidence) / 2
    intervals = {}
    for name in METRIC_NAMES:
        samples = values[name]
        intervals[name] = dict(estimate=base[name], low=float(np.quantile(samples, alpha)) if samples else None,
                               high=float(np.quantile(samples, 1 - alpha)) if samples else None,
                               valid_bootstraps=len(samples))
        if samples and len(samples) < n_bootstrap:
            warnings.append(f"{name}: {n_bootstrap - len(samples)} single-class/undefined bootstrap draws excluded")
    return dict(metrics=intervals, confidence=float(confidence), n_bootstrap=int(n_bootstrap),
                n_records=len(unique), resampling_unit="record", seed=int(seed), warnings=warnings)


def grouped_metrics(y_true, scores, groups, threshold, probabilities=None):
    y, score, probability = _inputs(y_true, scores, probabilities)
    groups = np.asarray(groups, dtype=object).reshape(-1)
    if groups.shape != y.shape:
        raise ValueError("Metric group labels must align with samples")
    result = {}
    for group in sorted(set(groups.tolist()), key=str):
        selected = groups == group
        name = str(group) if group is not None and str(group) else "unresolved"
        result[name] = compute_metrics(y[selected], score[selected], threshold,
                                       None if probability is None else probability[selected])
    return result
