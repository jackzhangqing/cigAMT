"""Publication-oriented plots for the evaluation pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "dataset"


def _save(fig: plt.Figure, path: str | Path, dpi: int = 300) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_waveforms(
    real_nct: np.ndarray,
    fake_nct: np.ndarray,
    channel_names: list[str],
    output: Path,
    seed: int,
    examples: int = 3,
    reference_label: str = "Real reference",
    generated_label: str = "Generated",
    dpi: int = 300,
) -> None:
    rng = np.random.default_rng(seed)
    count = min(examples, len(real_nct), len(fake_nct))
    real_indices = rng.choice(len(real_nct), size=count, replace=False)
    fake_indices = rng.choice(len(fake_nct), size=count, replace=False)
    fig, axes = plt.subplots(
        count * 2,
        len(channel_names),
        figsize=(18, 2.2 * count * 2),
        sharex=True,
        layout="constrained",
    )
    axes = np.asarray(axes).reshape(count * 2, len(channel_names))
    for row in range(count):
        for channel, name in enumerate(channel_names):
            axes[2 * row, channel].plot(real_nct[real_indices[row], channel], lw=0.7, color="#1f77b4")
            axes[2 * row + 1, channel].plot(fake_nct[fake_indices[row], channel], lw=0.7, color="#d95f02")
            
            low = min(
                float(np.min(real_nct[real_indices[row], channel])),
                float(np.min(fake_nct[fake_indices[row], channel])),
            )
            high = max(
                float(np.max(real_nct[real_indices[row], channel])),
                float(np.max(fake_nct[fake_indices[row], channel])),
            )
            margin = max((high - low) * 0.05, 1e-6)
            axes[2 * row, channel].set_ylim(low - margin, high + margin)
            axes[2 * row + 1, channel].set_ylim(low - margin, high + margin)

            if row == 0:
                axes[2 * row, channel].set_title(name)
            if channel == 0:
                axes[2 * row, channel].set_ylabel(reference_label)
                axes[2 * row + 1, channel].set_ylabel(generated_label)
    for axis in axes[-1, :]:
        axis.set_xlabel("Sample index")
    fig.suptitle("Independent waveform examples")
    _save(fig, output, dpi)


def plot_channel_distributions(
    real_nct: np.ndarray,
    fake_nct: np.ndarray,
    channel_names: list[str],
    output: Path,
    reference_label: str = "Real reference",
    generated_label: str = "Generated",
    dpi: int = 300,
) -> None:
    fig, axes = plt.subplots(
        1, len(channel_names), figsize=(4.2 * len(channel_names), 3.7), layout="constrained"
    )
    axes = np.atleast_1d(axes)
    for channel, (axis, name) in enumerate(zip(axes, channel_names)):
        real = real_nct[:, channel, :].reshape(-1)
        fake = fake_nct[:, channel, :].reshape(-1)
        pooled = np.concatenate([real, fake])
        low, high = np.quantile(pooled, [0.001, 0.999])
        axis.hist(
            real,
            bins=120,
            range=(low, high),
            density=True,
            histtype="step",
            lw=1.5,
            label=reference_label,
        )
        axis.hist(
            fake,
            bins=120,
            range=(low, high),
            density=True,
            histtype="step",
            lw=1.5,
            label=generated_label,
        )
        axis.set_title(name)
        axis.set_xlabel("Normalized amplitude")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Density")
    axes[-1].legend(frameon=False)
    _save(fig, output, dpi)


def plot_correlation_matrices(
    result: dict[str, Any],
    channel_names: list[str],
    output: Path,
    reference_label: str = "Real reference",
    generated_label: str = "Generated",
    dpi: int = 300,
) -> None:
    matrices = [result["real"], result["generated"], result["absolute_difference"]]
    titles = [reference_label, generated_label, "Absolute difference"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), layout="constrained")
    for axis, matrix, title in zip(axes, matrices, titles):
        if title == "Absolute difference":
            image = axis.imshow(matrix, cmap="magma", vmin=0)
        else:
            image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_xticks(range(len(channel_names)), channel_names)
        axis.set_yticks(range(len(channel_names)), channel_names)
        axis.set_title(title)
        
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                rgba = image.cmap(image.norm(matrix[row, column]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if luminance < 0.45 else "black",
                )
        fig.colorbar(image, ax=axis, fraction=0.045, pad=0.06, shrink=0.9)
    _save(fig, output, dpi)


def plot_psd(
    result: dict[str, Any],
    channel_names: list[str],
    output: Path,
    reference_label: str = "Real reference",
    generated_label: str = "Generated",
    dpi: int = 300,
) -> None:
    frequencies = result["frequencies"]
    fig, axes = plt.subplots(
        1,
        len(channel_names),
        figsize=(4.3 * len(channel_names), 3.8),
        sharey=True,
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    for channel, (axis, name) in enumerate(zip(axes, channel_names)):
        axis.semilogy(
            frequencies,
            np.maximum(result["real_psd"][channel], 1e-16),
            label=reference_label,
        )
        axis.semilogy(
            frequencies,
            np.maximum(result["generated_psd"][channel], 1e-16),
            label=generated_label,
        )
        axis.set_title(name)
        axis.set_xlabel("Frequency (Hz)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("PSD")
    axes[-1].legend(frameon=False)
    _save(fig, output, dpi)


def plot_coherence(
    result: dict[str, Any],
    output: Path,
    reference_label: str = "Real reference",
    generated_label: str = "Generated",
    dpi: int = 300,
) -> None:
    pairs = list(result["real"].keys())
    columns = 3
    rows = int(np.ceil(len(pairs) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 3.4 * rows),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axes = np.asarray(axes).reshape(-1)
    for axis, pair in zip(axes, pairs):
        axis.plot(result["frequencies"], result["real"][pair], label=reference_label)
        axis.plot(result["frequencies"], result["generated"][pair], label=generated_label)
        axis.set_title(pair)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
    for axis in axes[len(pairs) :]:
        axis.axis("off")
    axes[0].set_ylabel("Magnitude-squared coherence")
    for axis in axes[-columns:]:
        axis.set_xlabel("Frequency (Hz)")
    if pairs:
        axes[min(len(pairs), len(axes)) - 1].legend(frameon=False)
    _save(fig, output, dpi)


def _balanced_plot_features(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    maximum: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    count = min(len(real_features), len(fake_features), maximum)
    real_idx = rng.choice(len(real_features), size=count, replace=False)
    fake_idx = rng.choice(len(fake_features), size=count, replace=False)
    combined = np.concatenate([real_features[real_idx], fake_features[fake_idx]], axis=0)
    labels = np.concatenate([np.zeros(count, dtype=int), np.ones(count, dtype=int)])
    return combined, labels


def plot_pca_tsne(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    output: Path,
    seed: int,
    maximum: int = 1000,
    skip_tsne: bool = False,
    reference_label: str = "Real reference",
    generated_label: str = "Generated",
    feature_label: str = "sample-level features",
    dpi: int = 300,
) -> None:
    features, labels = _balanced_plot_features(real_features, fake_features, maximum, seed)
    pca = PCA(n_components=2, random_state=seed).fit_transform(features)
    panels = 1 if skip_tsne else 2
    fig, axes = plt.subplots(1, panels, figsize=(6.2 * panels, 5.2), layout="constrained")
    axes = np.atleast_1d(axes)
    plot_groups = [
        (0, "#1f77b4", reference_label),
        (1, "#d95f02", generated_label),
    ]
    for value, color, label in plot_groups:
        mask = labels == value
        axes[0].scatter(pca[mask, 0], pca[mask, 1], s=9, alpha=0.45, color=color, label=label)
    axes[0].set_title(f"PCA: {feature_label}")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    if not skip_tsne:
        perplexity = min(30.0, max(5.0, (len(features) - 1) / 10.0))
        embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            # ``exact`` avoids the optional OpenMP/threadpool nearest-neighbour
            # backend and is tractable because plotting is capped by
            # ``maximum`` (1000 samples per class by default).
            method="exact",
            random_state=seed,
        ).fit_transform(features)
        for value, color, label in plot_groups:
            mask = labels == value
            axes[1].scatter(
                embedding[mask, 0], embedding[mask, 1], s=9, alpha=0.45, color=color, label=label
            )
        axes[1].set_title(f"t-SNE: {feature_label}")
        axes[1].legend(frameon=False)
        axes[1].grid(alpha=0.2)
    _save(fig, output, dpi)


def plot_alpha_beta(result: dict[str, Any], output: Path, dpi: int = 300) -> None:
    fig, axis = plt.subplots(figsize=(6, 5), layout="constrained")
    axis.plot(result["alphas"], result["alpha_precision_curve"], label="Alpha-precision curve")
    axis.plot(result["alphas"], result["beta_recall_curve"], label="Beta-recall curve")
    axis.plot([0, 1], [0, 1], "--", color="black", lw=1, label="Ideal")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Alpha / beta")
    axis.set_ylabel("Empirical fraction")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    _save(fig, output, dpi)
