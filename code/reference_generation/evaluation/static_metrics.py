"""Non-neural metrics for multichannel generated time series.

The functions in this module are deterministic and intentionally independent of
PyTorch so that distributional and AMT-domain checks can still run on machines
without a working CUDA/PyTorch installation.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
from scipy import linalg, signal, stats
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import StandardScaler


EPS = 1e-12


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    distances = (
        np.sum(left**2, axis=1, keepdims=True)
        + np.sum(right**2, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(distances, 0.0)


def _nearest(
    queries: np.ndarray,
    reference: np.ndarray,
    neighbors: int = 1,
    exclude_self: bool = False,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Chunked NumPy k-NN that avoids sklearn/OpenMP runtime coupling."""
    if neighbors < 1 or neighbors > len(reference) - int(exclude_self):
        raise ValueError("Invalid nearest-neighbour count")
    all_distances: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    same_object = queries is reference
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        squared = _squared_distances(queries[start:stop], reference)
        if exclude_self:
            if not same_object or len(queries) != len(reference):
                raise ValueError("exclude_self requires queries and reference to be the same array")
            rows = np.arange(stop - start)
            squared[rows, start + rows] = np.inf
        candidate = np.argpartition(squared, kth=neighbors - 1, axis=1)[:, :neighbors]
        candidate_dist = np.take_along_axis(squared, candidate, axis=1)
        order = np.argsort(candidate_dist, axis=1)
        candidate = np.take_along_axis(candidate, order, axis=1)
        candidate_dist = np.take_along_axis(candidate_dist, order, axis=1)
        all_distances.append(np.sqrt(candidate_dist))
        all_indices.append(candidate)
    return np.concatenate(all_distances, axis=0), np.concatenate(all_indices, axis=0)


def channel_statistics(data_nct: np.ndarray, channel_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for channel, name in enumerate(channel_names):
        values = np.asarray(data_nct[:, channel, :], dtype=np.float64).reshape(-1)
        rows.append(
            {
                "channel": name,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "rms": float(np.sqrt(np.mean(values**2))),
                "min": float(np.min(values)),
                "p01": float(np.quantile(values, 0.01)),
                "p05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "p99": float(np.quantile(values, 0.99)),
                "max": float(np.max(values)),
                "skew": float(stats.skew(values, bias=False)),
                "kurtosis": float(stats.kurtosis(values, fisher=True, bias=False)),
            }
        )
    return rows


def channel_correlation(data_nct: np.ndarray) -> np.ndarray:
    flattened = np.transpose(data_nct, (1, 0, 2)).reshape(data_nct.shape[1], -1)
    corr = np.corrcoef(flattened)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def correlational_score(real_nct: np.ndarray, fake_nct: np.ndarray) -> dict[str, Any]:
    """Diffusion-TS-style error between channel correlation matrices."""
    real_corr = channel_correlation(real_nct)
    fake_corr = channel_correlation(fake_nct)
    difference = np.abs(real_corr - fake_corr)
    upper = np.triu_indices(real_corr.shape[0], k=1)
    return {
        "score_all_entries": float(np.mean(difference)),
        "score_off_diagonal": float(np.mean(difference[upper])) if len(upper[0]) else 0.0,
        "real": real_corr,
        "generated": fake_corr,
        "absolute_difference": difference,
    }


def _shared_histograms(real: np.ndarray, fake: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.concatenate([real.reshape(-1), fake.reshape(-1)]).astype(np.float64)
    low, high = np.quantile(pooled, [0.001, 0.999])
    if not np.isfinite(low + high) or math.isclose(float(low), float(high)):
        low = float(np.min(pooled))
        high = float(np.max(pooled))
    if math.isclose(float(low), float(high)):
        high = low + 1.0
    real_hist, _ = np.histogram(real, bins=bins, range=(low, high), density=False)
    fake_hist, _ = np.histogram(fake, bins=bins, range=(low, high), density=False)
    real_prob = (real_hist.astype(np.float64) + EPS)
    fake_prob = (fake_hist.astype(np.float64) + EPS)
    real_prob /= real_prob.sum()
    fake_prob /= fake_prob.sum()
    return real_prob, fake_prob


def marginal_jsd(
    real_nct: np.ndarray,
    fake_nct: np.ndarray,
    channel_names: list[str],
    bins: int = 200,
) -> dict[str, Any]:
    per_channel: dict[str, float] = {}
    for channel, name in enumerate(channel_names):
        p, q = _shared_histograms(real_nct[:, channel, :], fake_nct[:, channel, :], bins)
        # scipy returns Jensen-Shannon distance; square it for divergence.
        per_channel[name] = float(jensenshannon(p, q, base=2.0) ** 2)
    return {"mean": float(np.mean(list(per_channel.values()))), "per_channel": per_channel}


def mean_psd(
    data_nct: np.ndarray,
    sampling_rate: float,
    nperseg: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    segment = min(int(nperseg), data_nct.shape[-1])
    frequencies, psd = signal.welch(
        np.asarray(data_nct, dtype=np.float64),
        fs=float(sampling_rate),
        nperseg=segment,
        axis=-1,
        detrend="constant",
        scaling="density",
    )
    return frequencies, np.mean(psd, axis=0)


def spectral_metrics(
    real_nct: np.ndarray,
    fake_nct: np.ndarray,
    channel_names: list[str],
    sampling_rate: float,
    nperseg: int = 256,
) -> dict[str, Any]:
    frequencies, real_psd = mean_psd(real_nct, sampling_rate, nperseg)
    fake_frequencies, fake_psd = mean_psd(fake_nct, sampling_rate, nperseg)
    if not np.allclose(frequencies, fake_frequencies):
        raise RuntimeError("Real and generated PSD frequency grids differ")
    jsd: dict[str, float] = {}
    log_rmse: dict[str, float] = {}
    for channel, name in enumerate(channel_names):
        p = np.maximum(real_psd[channel], EPS)
        q = np.maximum(fake_psd[channel], EPS)
        p = p / p.sum()
        q = q / q.sum()
        jsd[name] = float(jensenshannon(p, q, base=2.0) ** 2)
        log_rmse[name] = float(
            np.sqrt(np.mean((10.0 * np.log10(p + EPS) - 10.0 * np.log10(q + EPS)) ** 2))
        )
    return {
        "psd_jsd_mean": float(np.mean(list(jsd.values()))),
        "psd_jsd_per_channel": jsd,
        "psd_log_rmse_db_mean": float(np.mean(list(log_rmse.values()))),
        "psd_log_rmse_db_per_channel": log_rmse,
        "frequencies": frequencies,
        "real_psd": real_psd,
        "generated_psd": fake_psd,
    }


def mean_coherence(
    data_nct: np.ndarray,
    sampling_rate: float,
    nperseg: int = 256,
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    segment = min(int(nperseg), data_nct.shape[-1])
    curves: dict[tuple[int, int], np.ndarray] = {}
    frequencies: np.ndarray | None = None
    for left, right in combinations(range(data_nct.shape[1]), 2):
        frequencies, coherence = signal.coherence(
            np.asarray(data_nct[:, left, :], dtype=np.float64),
            np.asarray(data_nct[:, right, :], dtype=np.float64),
            fs=float(sampling_rate),
            nperseg=segment,
            axis=-1,
        )
        curves[(left, right)] = np.mean(coherence, axis=0)
    if frequencies is None:
        frequencies = np.empty(0, dtype=np.float64)
    return frequencies, curves


def coherence_metrics(
    real_nct: np.ndarray,
    fake_nct: np.ndarray,
    channel_names: list[str],
    sampling_rate: float,
    nperseg: int = 256,
) -> dict[str, Any]:
    frequencies, real_curves = mean_coherence(real_nct, sampling_rate, nperseg)
    fake_frequencies, fake_curves = mean_coherence(fake_nct, sampling_rate, nperseg)
    if not np.allclose(frequencies, fake_frequencies):
        raise RuntimeError("Real and generated coherence frequency grids differ")
    scores: dict[str, float] = {}
    named_real: dict[str, np.ndarray] = {}
    named_fake: dict[str, np.ndarray] = {}
    for pair in real_curves:
        name = f"{channel_names[pair[0]]}-{channel_names[pair[1]]}"
        scores[name] = float(np.mean(np.abs(real_curves[pair] - fake_curves[pair])))
        named_real[name] = real_curves[pair]
        named_fake[name] = fake_curves[pair]
    return {
        "mean_absolute_error": float(np.mean(list(scores.values()))) if scores else 0.0,
        "per_pair": scores,
        "frequencies": frequencies,
        "real": named_real,
        "generated": named_fake,
    }


def _safe_autocorrelation(data_nt: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0 or lag >= data_nt.shape[1]:
        return np.zeros(data_nt.shape[0], dtype=np.float64)
    left = data_nt[:, :-lag]
    right = data_nt[:, lag:]
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(left**2, axis=1) * np.sum(right**2, axis=1))
    return numerator / np.maximum(denominator, EPS)


def extract_amt_features(
    data_nct: np.ndarray,
    sampling_rate: float,
    spectral_bands: int = 16,
) -> np.ndarray:
    """Deterministic sample-level time, spectral, and cross-channel features."""
    data = np.asarray(data_nct, dtype=np.float64)
    n_samples, n_channels, length = data.shape
    feature_blocks: list[np.ndarray] = []
    lags = sorted({1, max(1, length // 64), max(1, length // 16)})
    for channel in range(n_channels):
        values = data[:, channel, :]
        mean = values.mean(axis=1)
        centered = values - mean[:, None]
        std = values.std(axis=1)
        rms = np.sqrt(np.mean(values**2, axis=1))
        q01, q05, q25, q50, q75, q95, q99 = np.quantile(
            values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], axis=1
        )
        skew = stats.skew(values, axis=1, bias=False)
        kurt = stats.kurtosis(values, axis=1, fisher=True, bias=False)
        zero_cross = np.mean(np.signbit(centered[:, 1:]) != np.signbit(centered[:, :-1]), axis=1)
        temporal = np.column_stack(
            [mean, std, rms, q01, q05, q25, q50, q75, q95, q99, skew, kurt, zero_cross]
        )
        acf = np.column_stack([_safe_autocorrelation(values, lag) for lag in lags])

        spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
        frequencies = np.fft.rfftfreq(length, d=1.0 / float(sampling_rate))
        spectrum = np.maximum(spectrum, EPS)
        probability = spectrum / spectrum.sum(axis=1, keepdims=True)
        centroid = np.sum(probability * frequencies[None, :], axis=1)
        entropy = -np.sum(probability * np.log(probability + EPS), axis=1) / np.log(probability.shape[1])
        peak = frequencies[np.argmax(spectrum, axis=1)]
        edges = np.linspace(0, spectrum.shape[1], spectral_bands + 1, dtype=int)
        band_power = np.column_stack(
            [np.log10(np.sum(spectrum[:, edges[i] : edges[i + 1]], axis=1) + EPS)
             for i in range(spectral_bands)]
        )
        spectral = np.column_stack([centroid, entropy, peak, band_power])
        feature_blocks.extend([temporal, acf, spectral])

    cross_features: list[np.ndarray] = []
    for left, right in combinations(range(n_channels), 2):
        x = data[:, left, :]
        y = data[:, right, :]
        x = x - x.mean(axis=1, keepdims=True)
        y = y - y.mean(axis=1, keepdims=True)
        corr = np.sum(x * y, axis=1) / np.maximum(
            np.sqrt(np.sum(x**2, axis=1) * np.sum(y**2, axis=1)), EPS
        )
        cross_features.append(corr[:, None])
    feature_blocks.extend(cross_features)
    features = np.concatenate(feature_blocks, axis=1)
    return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def standardize_feature_pair(
    real_features: np.ndarray,
    fake_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler().fit(real_features)
    return scaler.transform(real_features), scaler.transform(fake_features), scaler


def frechet_distance(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    real = np.asarray(real_features, dtype=np.float64)
    fake = np.asarray(fake_features, dtype=np.float64)
    mu_real, mu_fake = real.mean(axis=0), fake.mean(axis=0)
    cov_real = np.atleast_2d(np.cov(real, rowvar=False))
    cov_fake = np.atleast_2d(np.cov(fake, rowvar=False))
    product_sqrt, _ = linalg.sqrtm(cov_real @ cov_fake, disp=False)
    if not np.isfinite(product_sqrt).all():
        offset = np.eye(cov_real.shape[0]) * 1e-6
        product_sqrt = linalg.sqrtm((cov_real + offset) @ (cov_fake + offset))
    if np.iscomplexobj(product_sqrt):
        product_sqrt = product_sqrt.real
    value = (
        np.sum((mu_real - mu_fake) ** 2)
        + np.trace(cov_real)
        + np.trace(cov_fake)
        - 2.0 * np.trace(product_sqrt)
    )
    return float(max(value, 0.0))


def alpha_precision_beta_recall(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    n_steps: int = 30,
) -> dict[str, Any]:
    """Implement the public Alaa et al. alpha/beta metric construction.

    Inputs must contain equal numbers of real and generated samples because the
    public authenticity estimator indexes the real nearest-neighbour radii with
    generated nearest-neighbour indices.
    """
    size = min(len(real_features), len(fake_features))
    if size < 3:
        raise ValueError("Alpha/beta metrics require at least 3 samples per set")
    real = np.asarray(real_features[:size], dtype=np.float64)
    fake = np.asarray(fake_features[:size], dtype=np.float64)
    center = np.mean(real, axis=0)
    alphas = np.linspace(0.0, 1.0, n_steps)
    real_center_distance = np.linalg.norm(real - center, axis=1)
    radii = np.quantile(real_center_distance, alphas)
    fake_center = np.mean(fake, axis=0)
    fake_to_center = np.linalg.norm(fake - center, axis=1)

    real_to_real, _ = _nearest(real, real, neighbors=1, exclude_self=True)
    real_to_real = real_to_real[:, 0]
    real_to_fake, nearest_fake_index = _nearest(real, fake, neighbors=1)
    real_to_fake = real_to_fake[:, 0]
    nearest_fake_index = nearest_fake_index[:, 0]
    nearest_fake = fake[nearest_fake_index]
    nearest_fake_center_distance = np.linalg.norm(nearest_fake - fake_center, axis=1)
    closest_fake_radii = np.quantile(nearest_fake_center_distance, alphas)

    precision_curve: list[float] = []
    recall_curve: list[float] = []
    for radius, fake_radius in zip(radii, closest_fake_radii):
        precision_curve.append(float(np.mean(fake_to_center <= radius)))
        covered = (real_to_fake <= real_to_real) & (
            nearest_fake_center_distance <= fake_radius
        )
        recall_curve.append(float(np.mean(covered)))

    # The public reference uses a rectangular sum.  Including both endpoints
    # can make the theoretical worst-case score slightly negative at a finite
    # grid resolution (for example -1/(n_steps - 1)).  Trapezoidal integration
    # preserves the same area definition and the advertised [0, 1] range.
    alpha_precision = 1.0 - 2.0 * np.trapz(
        np.abs(alphas - precision_curve), x=alphas
    )
    beta_recall = 1.0 - 2.0 * np.trapz(
        np.abs(alphas - recall_curve), x=alphas
    )
    alpha_precision = float(np.clip(alpha_precision, 0.0, 1.0))
    beta_recall = float(np.clip(beta_recall, 0.0, 1.0))
    if abs(alpha_precision) < 1e-12:
        alpha_precision = 0.0
    if abs(beta_recall) < 1e-12:
        beta_recall = 0.0
    authenticity = float(np.mean(real_to_real[nearest_fake_index] < real_to_fake))
    return {
        "alpha_precision": alpha_precision,
        "beta_recall": beta_recall,
        "authenticity": authenticity,
        "alphas": alphas,
        "alpha_precision_curve": np.asarray(precision_curve),
        "beta_recall_curve": np.asarray(recall_curve),
    }


def _manifold_membership(
    queries: np.ndarray,
    reference: np.ndarray,
    reference_radii: np.ndarray,
    chunk_size: int = 256,
) -> np.ndarray:
    membership = np.zeros(len(queries), dtype=bool)
    for start in range(0, len(queries), chunk_size):
        distances = np.sqrt(_squared_distances(queries[start : start + chunk_size], reference))
        membership[start : start + chunk_size] = np.any(
            distances <= reference_radii[None, :], axis=1
        )
    return membership


def knn_precision_recall_coverage(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    k: int = 5,
) -> dict[str, float]:
    """Compute manifold precision/recall and PRDC-style coverage."""
    real = np.asarray(real_features, dtype=np.float64)
    fake = np.asarray(fake_features, dtype=np.float64)
    if min(len(real), len(fake)) <= k:
        raise ValueError(f"Need more than k={k} samples")
    real_radii = _nearest(real, real, neighbors=k, exclude_self=True)[0][:, -1]
    fake_radii = _nearest(fake, fake, neighbors=k, exclude_self=True)[0][:, -1]
    precision = float(np.mean(_manifold_membership(fake, real, real_radii)))
    recall = float(np.mean(_manifold_membership(real, fake, fake_radii)))
    nearest_fake_distance = _nearest(real, fake, neighbors=1)[0][:, 0]
    coverage = float(np.mean(nearest_fake_distance <= real_radii))
    return {"knn_precision": precision, "knn_recall": recall, "coverage": coverage}


def nearest_neighbor_audit(
    train_features: np.ndarray,
    fake_features: np.ndarray,
    top_k: int = 20,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    distances, indices = _nearest(fake_features, train_features, neighbors=1)
    distances = distances[:, 0]
    indices = indices[:, 0]
    order = np.argsort(distances)[: min(top_k, len(distances))]
    summary = {
        "minimum": float(np.min(distances)),
        "p01": float(np.quantile(distances, 0.01)),
        "p05": float(np.quantile(distances, 0.05)),
        "median": float(np.median(distances)),
        "mean": float(np.mean(distances)),
    }
    rows = [
        {
            "generated_index": int(index),
            "training_index": int(indices[index]),
            "feature_distance": float(distances[index]),
        }
        for index in order
    ]
    return summary, rows
