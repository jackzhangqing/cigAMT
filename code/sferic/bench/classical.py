"""Window-local features and training-only fitted classical baselines."""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASSICAL_NAMES = ("dummy_prior", "feature_logreg", "feature_svm", "feature_rf", "feature_extratrees")


def feature_names(n_channels=4, fs=24000, n_bands=6):
    channel_names = ("Ex", "Ey", "Hx", "Hy") if n_channels == 4 else tuple(f"ch{i}" for i in range(n_channels))
    edges = np.linspace(0, fs / 2, n_bands + 1)
    base = ["mean", "std", "rms", "ptp", "skew", "excess_kurtosis", "crest_factor", "zero_crossing_rate"]
    base += [f"relative_power_{edges[i]:g}_{edges[i + 1]:g}Hz" for i in range(n_bands)]
    return [f"{channel}/{name}" for channel in channel_names for name in base]


def _feature_batch(inputs, fs=24000, n_bands=6):
    x = np.asarray(inputs, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] < 2 or not np.isfinite(x).all() or fs <= 0:
        raise ValueError("Features require finite N,C,L inputs, L>=2, and positive sampling rate")
    mean = x.mean(axis=-1)
    centered = x - mean[..., None]
    variance = np.mean(centered * centered, axis=-1)
    std = np.sqrt(variance)
    rms = np.sqrt(np.mean(x * x, axis=-1))
    safe_std = np.where(std > 0, std, 1)
    normalized = centered / safe_std[..., None]
    skew = np.where(std > 0, np.mean(normalized ** 3, axis=-1), 0)
    kurtosis = np.where(std > 0, np.mean(normalized ** 4, axis=-1) - 3, 0)
    crest = np.divide(np.max(np.abs(x), axis=-1), rms, out=np.zeros_like(rms), where=rms > 0)
    zcr = np.mean(np.signbit(x[..., 1:]) != np.signbit(x[..., :-1]), axis=-1)
    columns = [mean, std, rms, np.ptp(x, axis=-1), skew, kurtosis, crest, zcr]
    power = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    # One-sided powers include both signs of interior frequencies. DC and an
    # even-length Nyquist bin count once. No filter or frequency cut is applied.
    if x.shape[-1] % 2 == 0:
        power[..., 1:-1] *= 2
    else:
        power[..., 1:] *= 2
    frequencies = np.fft.rfftfreq(x.shape[-1], d=1 / fs)
    edges = np.linspace(0, fs / 2, n_bands + 1)
    total = power.sum(axis=-1)
    for i in range(n_bands):
        selected = (frequencies >= edges[i]) & ((frequencies < edges[i + 1]) if i + 1 < n_bands else (frequencies <= edges[i + 1]))
        band = power[..., selected].sum(axis=-1)
        columns.append(np.divide(band, total, out=np.zeros_like(band), where=total > 0))
    return np.stack(columns, axis=-1).reshape(len(x), -1).astype(np.float32)


def iter_feature_batches(X, batch_size=1024, fs=24000):
    """Yield (start, features); accepts an ndarray or WindowDataset."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for lo in range(0, len(X), batch_size):
        hi = min(lo + batch_size, len(X))
        batch = X.get_batch(np.arange(lo, hi)) if hasattr(X, "get_batch") else X[lo:hi]
        yield lo, _feature_batch(batch, fs=fs)


def extract_features(X, batch_size=1024, fs=24000, return_names=False):
    if len(X) == 0:
        raise ValueError("Cannot extract features from an empty input")
    chunks = [features for _, features in iter_feature_batches(X, batch_size=batch_size, fs=fs)]
    result = np.concatenate(chunks, axis=0)
    n_channels = result.shape[1] // 14
    return (result, feature_names(n_channels, fs)) if return_names else result


def build_estimator(name, seed=0, n_jobs=-1):
    aliases = {"dummy": "dummy_prior", "logreg": "feature_logreg", "logisticregression": "feature_logreg",
               "svm": "feature_svm", "rbf_svm": "feature_svm", "rf": "feature_rf",
               "randomforest": "feature_rf", "extratrees": "feature_extratrees"}
    name = aliases.get(name.lower(), name.lower())
    if name == "dummy_prior":
        return Pipeline([("estimator", DummyClassifier(strategy="prior", random_state=seed))])
    if name == "feature_logreg":
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    elif name == "feature_svm":
        # Raw margins preserve ranking. No internal random-window calibration
        # CV is introduced; the engine must leave Brier/ECE unavailable.
        model = SVC(C=1, kernel="rbf", gamma="scale", probability=False,
                    class_weight="balanced", cache_size=512, random_state=seed)
    elif name == "feature_rf":
        return Pipeline([("estimator", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=n_jobs))])
    elif name == "feature_extratrees":
        return Pipeline([("estimator", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=n_jobs))])
    else:
        raise ValueError(f"Unknown classical estimator {name}; choose {CLASSICAL_NAMES}")
    return Pipeline([("scaler", StandardScaler()), ("estimator", model)])


def prediction_scores(estimator, features):
    """Return positive-class continuous scores and only real probability output."""
    classes = np.asarray(estimator.classes_)
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(features), dtype=np.float64)
        positive = np.flatnonzero(classes == 1)
        probability = proba[:, positive[0]] if len(positive) else np.zeros(len(features), dtype=np.float64)
        return dict(scores=probability, probabilities=probability, score_kind="probability")
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(features), dtype=np.float64).reshape(-1)
        if len(classes) == 2 and classes[1] != 1:
            scores = -scores
        return dict(scores=scores, probabilities=None, score_kind="decision_function_uncalibrated")
    raise ValueError("Estimator provides neither probabilities nor continuous decision scores")
