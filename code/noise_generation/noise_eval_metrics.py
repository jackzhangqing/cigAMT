"""Exact Euclidean diagnostics for generated-noise feature vectors.

The caller chooses/standardizes features and samples equal-size reference and
generated sets for distribution comparisons.  These are descriptor-space
adaptations, not a reproduction of the learned one-class embedding in Alaa et al.
``How Faithful is your Synthetic Data?`` (ICML 2022), equations (6), (9), and
Section 4's authenticity classifier:
https://proceedings.mlr.press/v162/alaa22a/alaa22a.pdf

For each beta, generated points inside their beta-quantile distance from their
mean are retained.  Beta recall counts reference points whose k-neighbour ball
contains ANY retained generated point.  This follows the nearest-within-subset
definition in equation (9), rather than testing only an unrestricted nearest
neighbour.  We use strict distance < radius consistently with PRDC coverage:
https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
At beta=1 this estimate equals coverage; beta=0 is explicitly zero.  The
integrated score is 1 - 2 integral(abs(beta_recall - beta), beta).

Authenticity marks a generated point only when its distance to the closest
reference point is STRICTLY GREATER than that reference point's distance to its
closest OTHER reference point.  It is a feature-space proximity diagnostic,
not proof that the waveform is or is not copied.  For a copying audit use the
entire training reference library, rather than a subsampled reference library.
No holdout is needed to compute these diagnostics; claims about generalization
require an independently held-out evaluation.  PCA/t-SNE coordinates should not
be used as a replacement for the declared metric feature space.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def _features(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n_samples, n_features), with n_features > 0")
    if result.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def _paired_features(reference: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = _features(reference, "reference")
    queries = _features(queries, "queries")
    if reference.shape[1] != queries.shape[1]:
        raise ValueError("reference and queries must have the same feature dimension")
    return reference, queries


def _reference_radii(reference: np.ndarray, k: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return k-th and 1st other-row distances, with bounded working memory."""
    n_reference = len(reference)
    if n_reference < 2:
        invalid = np.full(n_reference, np.nan, dtype=np.float64)
        return invalid.copy(), invalid
    if not 1 <= k < n_reference:
        raise ValueError("k must be between 1 and n_reference - 1")
    kth = np.empty(n_reference, dtype=np.float64)
    first = np.empty(n_reference, dtype=np.float64)
    for start in range(0, n_reference, chunk_size):
        stop = min(start + chunk_size, n_reference)
        best = np.full((stop - start, k), np.inf, dtype=np.float64)
        for other_start in range(0, n_reference, chunk_size):
            other_stop = min(other_start + chunk_size, n_reference)
            distances = cdist(reference[start:stop], reference[other_start:other_stop])
            overlap_start = max(start, other_start)
            overlap_stop = min(stop, other_stop)
            if overlap_start < overlap_stop:
                diagonal = np.arange(overlap_start, overlap_stop)
                distances[diagonal - start, diagonal - other_start] = np.inf
            candidates = np.concatenate((best, distances), axis=1)
            best = np.partition(candidates, kth=k - 1, axis=1)[:, :k]
        kth[start:stop] = best.max(axis=1)
        first[start:stop] = best.min(axis=1)
    return kth, first


def nearest_reference(reference: np.ndarray, queries: np.ndarray, chunk_size: int = 2048) -> dict:
    """Query exact nearest reference rows and their local authenticity thresholds.

    ``nearest_index`` and ``nearest_distance`` have one entry per query;
    ``reference_nn_distance`` has one entry per reference row and excludes that
    row itself.  ``authentic`` is boolean, with one entry per query.  If fewer
    than two reference rows exist, its values are placeholders (False),
    ``authenticity_valid`` is False, and ``authenticity`` and reference thresholds
    are NaN.  Such placeholders MUST NOT be interpreted as a measured rate.
    Exact ties between reference candidates select the first reference index.
    Both axes are chunked, so the largest distance matrix is chunk_size squared.
    """
    reference, queries = _paired_features(reference, queries)
    if int(chunk_size) != chunk_size or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    chunk_size = int(chunk_size)
    nearest_distance = np.full(len(queries), np.inf, dtype=np.float64)
    nearest_index = np.full(len(queries), -1, dtype=np.int64)
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        for ref_start in range(0, len(reference), chunk_size):
            ref_stop = min(ref_start + chunk_size, len(reference))
            distances = cdist(queries[start:stop], reference[ref_start:ref_stop])
            local_index = distances.argmin(axis=1)
            local_distance = distances[np.arange(stop - start), local_index]
            update = local_distance < nearest_distance[start:stop]
            nearest_distance[start:stop][update] = local_distance[update]
            nearest_index[start:stop][update] = ref_start + local_index[update]
    _, reference_nn_distance = _reference_radii(reference, 1, chunk_size)
    valid = len(reference) >= 2
    authentic = np.zeros(len(queries), dtype=bool)
    if valid:
        authentic = nearest_distance > reference_nn_distance[nearest_index]
    return {
        "nearest_index": nearest_index,
        "nearest_distance": nearest_distance,
        "reference_nn_distance": reference_nn_distance,
        "authentic": authentic,
        "authenticity_valid": valid,
        "authenticity": float(authentic.mean()) if valid else float("nan"),
    }


def distribution_metrics(reference: np.ndarray, generated: np.ndarray, k: int = 5,
                         betas: np.ndarray | None = None) -> dict:
    """Compute beta recall, coverage and authenticity without resampling/scaling.

    ``k`` is reduced to n_reference - 1 for small sets; the effective value is
    returned.  A beta grid must be strictly increasing and span [0, 1].  Its
    default is 101 equally spaced points.  Fewer than 20 reference observations
    are flagged as small_sample.  With one reference observation all metrics
    requiring a reference neighbourhood are NaN and validity flags are False.
    Equal reference/generated counts are recommended for comparable coverage.
    """
    reference, generated = _paired_features(reference, generated)
    if int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    if betas is None:
        betas = np.linspace(0.0, 1.0, 101)
    else:
        betas = np.asarray(betas, dtype=np.float64)
    if (betas.ndim != 1 or len(betas) < 2 or not np.isfinite(betas).all()
            or betas[0] != 0.0 or betas[-1] != 1.0 or np.any(np.diff(betas) <= 0)):
        raise ValueError("betas must be a finite strictly increasing vector spanning exactly [0, 1]")

    n_reference, n_generated = len(reference), len(generated)
    effective_k = min(int(k), n_reference - 1)
    result = {
        "n_reference": n_reference,
        "n_generated": n_generated,
        "k": effective_k,
        "requested_k": int(k),
        "beta_grid": betas.copy(),
        "small_sample": n_reference < 20,
        "balanced_counts": n_reference == n_generated,
        "metrics_valid": n_reference >= 2,
        "authenticity_valid": n_reference >= 2,
    }
    if n_reference < 2:
        beta_recall = np.full_like(betas, np.nan)
        beta_recall[0] = 0.0
        result.update({
            "beta_recall": beta_recall,
            "integrated_beta_recall": float("nan"),
            "coverage": float("nan"),
            "authenticity": float("nan"),
            "median_nn_distance": float(np.median(cdist(generated, reference)[:, 0])),
            "zero_reference_nn_fraction": float("nan"),
        })
        return result

    chunk_size = 2048
    reference_radii, reference_nn = _reference_radii(reference, effective_k, chunk_size)
    generated_radius = np.linalg.norm(generated - generated.mean(axis=0), axis=1)
    beta_radii = np.quantile(generated_radius, betas)
    # A reference is covered at beta iff at least one generated neighbour in
    # its open kNN ball has radius <= beta_radii[beta].  This yields exactly the
    # nearest-within-beta-subset result without rerunning 101 nearest queries.
    first_cover_radius = np.full(n_reference, np.inf, dtype=np.float64)
    generated_to_reference = np.full(n_generated, np.inf, dtype=np.float64)
    generated_nearest_index = np.full(n_generated, -1, dtype=np.int64)
    for start in range(0, n_reference, chunk_size):
        stop = min(start + chunk_size, n_reference)
        for gen_start in range(0, n_generated, chunk_size):
            gen_stop = min(gen_start + chunk_size, n_generated)
            distances = cdist(reference[start:stop], generated[gen_start:gen_stop])
            cover = distances < reference_radii[start:stop, None]
            local_cover_radius = np.where(cover, generated_radius[None, gen_start:gen_stop], np.inf).min(axis=1)
            first_cover_radius[start:stop] = np.minimum(first_cover_radius[start:stop], local_cover_radius)
            local_index = distances.argmin(axis=0)
            local_distance = distances[local_index, np.arange(gen_stop - gen_start)]
            update = local_distance < generated_to_reference[gen_start:gen_stop]
            generated_to_reference[gen_start:gen_stop][update] = local_distance[update]
            generated_nearest_index[gen_start:gen_stop][update] = start + local_index[update]

    beta_recall = np.mean(first_cover_radius[:, None] <= beta_radii[None, :], axis=0)
    beta_recall[0] = 0.0
    coverage = float(np.isfinite(first_cover_radius).mean())
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    integrated = float(1.0 - 2.0 * trapezoid(np.abs(beta_recall - betas), betas))
    authentic = generated_to_reference > reference_nn[generated_nearest_index]
    result.update({
        "beta_recall": beta_recall,
        "integrated_beta_recall": integrated,
        "coverage": coverage,
        "authenticity": float(authentic.mean()),
        "median_nn_distance": float(np.median(generated_to_reference)),
        "zero_reference_nn_fraction": float(np.mean(reference_nn == 0.0)),
    })
    return result


def self_check() -> None:
    """Exercise copying, separation, exact chunking and invalid-size cases."""
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(64, 4))
    copied = distribution_metrics(reference, reference.copy())
    assert copied["coverage"] == 1.0
    assert copied["authenticity"] == 0.0
    assert copied["beta_recall"][0] == 0.0
    assert copied["beta_recall"][-1] == copied["coverage"]
    assert np.all(np.diff(copied["beta_recall"]) >= 0.0)
    assert copied["beta_grid"].shape == (101,)

    distant = distribution_metrics(reference, reference + 1000.0)
    assert distant["coverage"] == 0.0
    assert distant["authenticity"] == 1.0
    assert distant["beta_recall"][-1] == distant["coverage"]

    queries = np.concatenate((reference[:3], reference[:5] + 0.01), axis=0)
    nearest = nearest_reference(reference, queries, chunk_size=7)
    dense = cdist(queries, reference)
    assert np.array_equal(nearest["nearest_index"], dense.argmin(axis=1))
    assert np.allclose(nearest["nearest_distance"], dense.min(axis=1))
    assert nearest["nearest_index"].dtype == np.int64
    assert nearest["authentic"].dtype == np.bool_
    assert nearest["reference_nn_distance"].shape == (64,)
    assert nearest["authentic"].shape == (8,)
    assert not nearest["authentic"][:3].any()
    reference_distances = cdist(reference, reference)
    np.fill_diagonal(reference_distances, np.inf)
    assert np.allclose(nearest["reference_nn_distance"], reference_distances.min(axis=1))

    # Independently calculate every beta subset's nearest point, validating the
    # fast first-cover-radius construction against the literal definition.
    generated = rng.normal(size=(47, 4))
    grid = np.linspace(0.0, 1.0, 11)
    measured = distribution_metrics(reference, generated, k=3, betas=grid)
    local_radius = np.partition(reference_distances, 2, axis=1)[:, 2]
    radial_distance = np.linalg.norm(generated - generated.mean(axis=0), axis=1)
    for index, beta in enumerate(grid[1:], start=1):
        subset = generated[radial_distance <= np.quantile(radial_distance, beta)]
        expected = np.mean(cdist(reference, subset).min(axis=1) < local_radius)
        assert measured["beta_recall"][index] == expected

    singleton = distribution_metrics(reference[:1], reference[:3])
    assert singleton["k"] == 0 and not singleton["metrics_valid"]
    assert np.isnan(singleton["authenticity"]) and np.isnan(singleton["coverage"])
    singleton_nearest = nearest_reference(reference[:1], reference[:3], chunk_size=2)
    assert not singleton_nearest["authenticity_valid"]
    assert np.isnan(singleton_nearest["reference_nn_distance"]).all()
    assert np.isnan(singleton_nearest["authenticity"])
    assert distribution_metrics(reference[:3], reference[:3], k=5)["k"] == 2


if __name__ == "__main__":
    self_check()
    print("noise_eval_metrics self-check passed")
