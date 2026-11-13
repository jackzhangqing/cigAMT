"""Build the human-readable report for the noise diversity/copy audit.

The evaluator owns the measurements; this module only reads its artifacts.  It
does not load the original large NPZ, import PyTorch, or modify the input data.
Usage: python noise_eval_report.py path/to/evaluation
"""

from __future__ import annotations

import argparse
import csv
import functools
import html
import inspect
import json
import math
import warnings
from contextvars import ContextVar
from pathlib import Path

import numpy as np


SOURCE_NAMES = {"real": "实测参照数据", "synthetic": "合成参照数据", "pooled": "合并参照数据"}
SOURCE_LABELS = {0: "Real reference", 1: "Synthetic reference", 2: "Generated"}
SOURCE_COLORS = {0: "#0072b2", 1: "#e69f00", 2: "#009e73"}
TYPE_NAMES = [
    "Pulse", "Quasi-Sinusoidal", "Square-wave", "Step-wave", "Triangular-wave",
    "Power-frequency", "White-noise", "Color-noise", "Harmonic", "Composite",
]
_PDF_PAGES = ContextVar("noise_evaluation_pdf_pages", default=None)


def prepare_threadpool_compat():
    """Handle a threadpoolctl 2.x Windows/OpenBLAS version-detection bug.

    Some OpenBLAS DLLs omit legacy metadata symbols.  The old detector calls
    ``None.split()`` or an absent ``openblas_get_parallel`` and aborts even
    though computations work.  The guard marks missing metadata as unknown;
    it neither edits installed packages nor changes numerical operations.
    Returns True when the guard is installed (or was already installed).
    """
    import threadpoolctl

    module_class = getattr(threadpoolctl, "_OpenBLASModule", None)
    if module_class is None:
        return False
    original = module_class.get_version
    if getattr(original, "_noise_eval_null_version_guard", False):
        return True

    @functools.wraps(original)
    def safe_version(self):
        try:
            return original(self)
        except AttributeError as exc:
            if "'NoneType' object has no attribute 'split'" in str(exc):
                return None
            raise

    safe_version._noise_eval_null_version_guard = True
    module_class.get_version = safe_version
    original_layer = module_class.get_threading_layer

    @functools.wraps(original_layer)
    def safe_layer(self):
        if not hasattr(self._dynlib, "openblas_get_parallel"):
            return None
        return original_layer(self)

    module_class.get_threading_layer = safe_layer
    return True


def _read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _integer(value, default=0):
    number = _float(value)
    return int(number) if np.isfinite(number) else default


def _truth(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def _percent(value):
    number = _float(value)
    return f"{number:.1%}" if np.isfinite(number) else "—"


def _count(value):
    number = _float(value)
    return f"{int(number):,}" if np.isfinite(number) else "—"


def _escaped(value):
    return html.escape(str(value), quote=True)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "savefig.dpi": 300,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def _save_figure(figure, path, plt):
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    pdf_pages = _PDF_PAGES.get()
    if pdf_pages is not None:
        pdf_pages.savefig(figure, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _panel_labels(axes):
    for index, axis in enumerate(np.asarray(axes, dtype=object).reshape(-1)):
        axis.text(0.0, 1.045, chr(ord("a") + index), transform=axis.transAxes, fontsize=15, fontweight="bold", ha="left", va="bottom")


def _heatmap(rows, source, output_dir, plt):
    relevant = [row for row in rows if row.get("source") == source]
    lengths = sorted({_integer(row.get("length")) for row in rows if _integer(row.get("length")) > 0})
    type_ids = sorted({_integer(row.get("type_id")) for row in rows})
    if not lengths or not type_ids:
        raise ValueError("No valid evaluated length/type groups for heatmap")
    single_length = len(lengths) == 1
    names = {index: name for index, name in enumerate(TYPE_NAMES)}
    names.update({_integer(row.get("type_id")): row.get("type_name", "") for row in relevant})
    index_by_type = {type_id: index for index, type_id in enumerate(type_ids)}
    index_by_length = {length: index for index, length in enumerate(lengths)}
    metrics = [
        ("coverage", "Coverage (balanced samples)"),
        ("integrated_beta_recall", "Integrated beta-recall (descriptor space)"),
        ("authenticity_full", "Authenticity (full reference set)"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(11.8, 7.4) if single_length else (19.8, 6.8), sharey=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dfe3e8")
    for axis, (metric, title) in zip(axes, metrics):
        values = np.full((len(type_ids), len(lengths)), np.nan)
        small = np.zeros(values.shape, dtype=bool)
        for row in relevant:
            pos = (index_by_type[_integer(row.get("type_id"))], index_by_length[_integer(row.get("length"))])
            values[pos] = _float(row.get(metric))
            small[pos] = _truth(row.get("small_sample")) or _integer(row.get("n_reference")) < 10
        plot = axis.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        axis.set_xticks(range(len(lengths)), [str(length) for length in lengths], rotation=0 if single_length else 55, ha="center" if single_length else "right")
        axis.set_yticks(range(len(type_ids)), [names.get(type_id, str(type_id)) for type_id in type_ids])
        axis.set_title(title.replace(" (", "\n(") if single_length else title, pad=20 if single_length else 12, fontsize=12 if single_length else 13)
        axis.set_xlabel("Sample length")
        for r, c in zip(*np.where(np.isfinite(values))):
            value = values[r, c]
            color = "#111827" if value > 0.66 else "white"
            axis.text(c, r, f"{value:.2f}{'*' if small[r, c] else ''}", ha="center", va="center", fontsize=11 if single_length else 8, color=color)
        axis.set_xticks(np.arange(len(lengths) + 1) - 0.5, minor=True)
        axis.set_yticks(np.arange(len(type_ids) + 1) - 0.5, minor=True)
        axis.grid(which="minor", color="white", linewidth=0.5)
        axis.tick_params(which="minor", bottom=False, left=False)
    axes[0].set_ylabel("Noise type")
    _panel_labels(axes)
    figure.suptitle(f"Generated noise vs {source} reference" + (f" | L={lengths[0]}" if single_length else ""), fontsize=16, y=0.99)
    figure.text(0.13, 0.015, "Gray: unavailable / insufficient reference.  * Small reference group.\nScores are diagnostics, not proof of originality." if single_length else "Gray: unavailable / insufficient reference.  * Small reference group.  Scores are diagnostics, not a proof of originality.", fontsize=9)
    figure.subplots_adjust(left=0.22 if single_length else 0.13, right=0.88, top=0.84 if single_length else 0.88, bottom=0.14 if single_length else 0.16, wspace=0.25 if single_length else 0.12)
    colorbar_axis = figure.add_axes([0.90, 0.20, 0.012, 0.62])
    figure.colorbar(plot, cax=colorbar_axis, label="Score (0-1)")
    filename = f"metric_heatmap_{source}.png"
    _save_figure(figure, output_dir / filename, plt)
    return filename


def _beta_curves(rows, output_dir, plt):
    figure, axis = plt.subplots(figsize=(8.4, 5.4))
    source_styles = {"real": ("#0072b2", "Real reference"), "synthetic": ("#e69f00", "Synthetic reference"), "pooled": ("#009e73", "Pooled reference")}
    for source, (color, label) in source_styles.items():
        by_beta = {}
        for row in rows:
            if row.get("source") != source:
                continue
            beta, recall = _float(row.get("beta")), _float(row.get("recall"))
            if np.isfinite(beta) and np.isfinite(recall):
                by_beta.setdefault(beta, []).append(recall)
        if by_beta:
            xs = sorted(by_beta)
            ys = [float(np.mean(by_beta[x])) for x in xs]
            counts = [len(by_beta[x]) for x in xs]
            label += f" (groups={min(counts)}" + (f"-{max(counts)})" if min(counts) != max(counts) else ")")
            axis.plot(xs, ys, linewidth=2, color=color, label=label)
    axis.plot([0, 1], [0, 1], color="#6b7280", linewidth=1.3, linestyle="--", label="Ideal distribution match: recall = beta")
    axis.set(xlim=(0, 1), ylim=(0, 1.025), xlabel="Beta: generated-distribution mass", ylabel="Recall: reference mass covered", title="Beta-recall: unweighted mean across valid type-length groups")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=9)
    figure.tight_layout()
    filename = "beta_recall.png"
    _save_figure(figure, output_dir / filename, plt)
    return filename


def _summary_figure(aggregate, output_dir, plt, external_reference=False):
    """Render the three-panel summary with explicit metric directions."""
    from matplotlib.ticker import PercentFormatter

    lookup = {row.get("source"): row for row in aggregate}
    sources = [source for source in ["real", "synthetic", "pooled"] if source in lookup]
    if not sources:
        raise ValueError("No source aggregate rows available")
    rows = [lookup[source] for source in sources]
    labels = [dict(real="Real", synthetic="Synthetic", pooled="Pooled")[source] for source in sources]
    x = np.arange(len(rows))
    colors = ["#0072b2", "#e69f00", "#009e73"]
    figure, axes = plt.subplots(
        1, 3, figsize=(9.4, 3.7),
        gridspec_kw={"width_ratios": [1.4, 1.0, 1.1]},
    )
    figure.subplots_adjust(left=.065, right=.99, bottom=.285, top=.70, wspace=.43)

    def values(key):
        return np.asarray([_float(row.get(key)) for row in rows])

    def annotate(axis, bars, digits=1, fontsize=9.5):
        for bar in bars:
            height = bar.get_height()
            if np.isfinite(height):
                axis.annotate(f"{height:.{digits}%}", (bar.get_x() + bar.get_width() / 2, height), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=fontsize)

    axis = axes[0]
    annotate(axis, axis.bar(x - .24, values("coverage_macro"), .40, color=colors[0], label="Coverage ↑"), fontsize=8.5)
    annotate(axis, axis.bar(x + .24, values("integrated_beta_recall_macro"), .40, color=colors[1], label="Integrated β-recall ↑"), fontsize=8.5)
    axis.set_title("a  Diversity ↑", loc="left", pad=45, fontsize=12.5, fontweight="bold")
    axis.text(0, 1.20, "Mean across type-length groups", transform=axis.transAxes, fontsize=8.5, color="#52616b")
    axis.legend(loc="upper left", bbox_to_anchor=(.025, .97), ncol=1, fontsize=8.5, frameon=False,
                borderaxespad=0, labelspacing=.35, handlelength=1.6)
    axis.set_ylim(0, 1.02)
    axis = axes[1]
    annotate(axis, axis.bar(x, values("authenticity_full"), .55, color=colors[2]))
    axis.set_title("b  Authenticity ↑", loc="left", pad=45, fontsize=12.5, fontweight="bold")
    axis.text(0, 1.20, "Full reference set", transform=axis.transAxes, fontsize=8.5, color="#52616b")
    axis.set_ylim(0, 1.02)
    axis = axes[2]
    near_copy = values("near_copy_rate")
    annotate(axis, axis.bar(x, near_copy, .55, color="#cc79a7"), digits=2)
    finite_near = near_copy[np.isfinite(near_copy)]
    near_limit = math.ceil(max(.01, float(finite_near.max()) * 1.3 if len(finite_near) else .01) / .0025) * .0025
    axis.set_ylim(0, near_limit)
    axis.set_title("c  Near-copy rate ↓", loc="left", pad=45, fontsize=12.5, fontweight="bold")
    axis.text(0, 1.20, "Normalized RMSE ≤ 0.05", transform=axis.transAxes, fontsize=8.5, color="#52616b")
    axis.set_yticks(np.linspace(0, near_limit, 6))
    axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=2))
    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Reference source", fontsize=10, labelpad=6)
        axis.set_ylabel("Fraction", fontsize=10, labelpad=5)
        axis.tick_params(labelsize=9.5)
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    for axis in axes[:2]:
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    figure.suptitle("Generated noise: diversity and " + ("reference-set" if external_reference else "training-set") + " proximity", fontsize=14, y=.985)
    footer = ("Selected reference library; original training set unverified.\n"
              "Descriptor-space diagnostics; not proof of originality. Candidate screening is not exhaustive.") if external_reference else (
              "↑ Higher is preferred; ↓ lower is preferred. Descriptor-space diagnostics, not proof of originality.\n"
              "No independent holdout; waveform candidate screening is not exhaustive.")
    figure.text(.5, .048, footer, fontsize=8.2, ha="center", va="bottom", linespacing=1.4, color="#52616b")
    filename = "evaluation_summary.png"
    with plt.rc_context({"savefig.dpi": 600}):
        _save_figure(figure, output_dir / filename, plt)
    return filename


def _stratified_indices(source, type_id, limit=3000, seed=42):
    """Round-robin allocation prevents a large source/type dominating the plot."""
    rng = np.random.default_rng(seed)
    buckets = []
    for source_value, type_value in sorted(set(zip(source.tolist(), type_id.tolist()))):
        indices = np.flatnonzero((source == source_value) & (type_id == type_value))
        buckets.append(rng.permutation(indices).tolist())
    selected = []
    while len(selected) < min(limit, len(source)):
        for bucket in buckets:
            if bucket:
                selected.append(bucket.pop())
                if len(selected) == min(limit, len(source)):
                    break
    return np.asarray(selected, dtype=np.int64)


def _fit_reference_scaler(reference_features):
    """Use the evaluator's near-constant guard, without clipping any data."""
    from sklearn.preprocessing import StandardScaler

    reference_features = np.asarray(reference_features, dtype=np.float64)
    scaler = StandardScaler().fit(reference_features)
    reference_std = reference_features.std(axis=0, dtype=np.float64)
    guarded = reference_std <= 1e-6
    scaler.mean_ = reference_features.mean(axis=0, dtype=np.float64)
    scaler.var_ = reference_std ** 2
    scaler.scale_ = np.where(guarded, 1.0, reference_std)
    return scaler, guarded


def _pca_tsne(path, output_dir, plt):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits

    prepare_threadpool_compat()
    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float64)
        source = np.asarray(data["source"] if "source" in data else data["source_id"]).reshape(-1)
        type_id = np.asarray(data["type_id"]).reshape(-1)
        length = np.asarray(data["length"]).reshape(-1)
        feature_names = np.asarray(data["feature_names"]).astype(str).tolist() if "feature_names" in data else [f"feature_{index}" for index in range(features.shape[1])]
    if features.ndim != 2 or not (len(features) == len(source) == len(type_id) == len(length)):
        raise ValueError("visualization.npz features/source/type_id/length shape mismatch")
    available_lengths = sorted(int(value) for value in np.unique(length))
    eligible_lengths = [value for value in available_lengths if int(np.sum((length == value) & ((source == 0) | (source == 1)))) >= 3 and int(np.sum(length == value)) >= 4]
    if not eligible_lengths:
        raise ValueError("No length has at least 4 plotted rows and 3 reference rows")
    selected_length = 2048 if 2048 in eligible_lengths else eligible_lengths[-1]
    mask = length == selected_length
    features, source, type_id = features[mask], source[mask], type_id[mask]
    reference_mask = (source == 0) | (source == 1)
    if len(features) < 4 or int(reference_mask.sum()) < 3:
        raise ValueError(f"At least 4 plotted rows and 3 reference rows at length {selected_length} are required")
    finite = np.isfinite(features).all(axis=1)
    removed_nonfinite = int((~finite).sum())
    features, source, type_id = features[finite], source[finite], type_id[finite]
    reference_mask = (source == 0) | (source == 1)
    if len(features) < 4 or int(reference_mask.sum()) < 3:
        raise ValueError(f"Too few finite rows for PCA/t-SNE at length {selected_length}")
    selected = _stratified_indices(source, type_id)
    components = min(20, features.shape[1], int(reference_mask.sum()) - 1)
    if components < 2:
        raise ValueError("At least two feature/PCA dimensions are required")
    with threadpool_limits(limits=1):
        scaler, guarded_features = _fit_reference_scaler(features[reference_mask])
        standardized_reference = scaler.transform(features[reference_mask])
        pca = PCA(n_components=components, svd_solver="full", random_state=42).fit(standardized_reference)
        projected = pca.transform(scaler.transform(features[selected]))
        parameters = dict(n_components=2, perplexity=min(30.0, (len(selected) - 1) / 3), init="pca", learning_rate="auto", random_state=42, method="barnes_hut")
        parameters["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"] = 750
        embedding = TSNE(**parameters).fit_transform(projected)
    sample_source, sample_type = source[selected], type_id[selected]
    figure, axes = plt.subplots(2, 2, figsize=(13.7, 11.0))
    source_counts = {SOURCE_LABELS.get(int(value), str(value)): int(np.sum(sample_source == value)) for value in np.unique(sample_source)}
    for column, (points, title) in enumerate([(projected[:, :2], "PCA"), (embedding, "t-SNE")]):
        axis = axes[0, column]
        for value in [0, 1, 2]:
            matches = sample_source == value
            if matches.any():
                axis.scatter(points[matches, 0], points[matches, 1], s=8, alpha=0.5, color=SOURCE_COLORS[value], label=f"{SOURCE_LABELS[value]} (n={int(matches.sum())})", rasterized=True, linewidths=0)
        axis.set_title(f"{title}: colored by source")
        axis.legend(fontsize=8, markerscale=2)
        axis = axes[1, column]
        for value in sorted(np.unique(sample_type)):
            matches = sample_type == value
            label = TYPE_NAMES[int(value)] if 0 <= int(value) < len(TYPE_NAMES) else str(value)
            axis.scatter(points[matches, 0], points[matches, 1], s=8, alpha=0.5, color=plt.get_cmap("tab10")(int(value) % 10), label=label, rasterized=True, linewidths=0)
        axis.set_title(f"{title}: colored by noise type")
        axis.legend(fontsize=8, markerscale=2, ncol=2)
        for row in [0, 1]:
            axes[row, column].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})" if column == 0 else "t-SNE dimension 1")
            axes[row, column].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})" if column == 0 else "t-SNE dimension 2")
            axes[row, column].grid(alpha=0.12)
            if column == 0:
                axes[row, column].ticklabel_format(axis="x", style="sci", scilimits=(-3, 3), useMathText=True)
    figure.suptitle(f"Length {selected_length} only | stratified visualization sample n={len(selected):,} | seed=42", fontsize=15)
    _panel_labels(axes)
    figure.text(0.5, 0.012, "Reference-only scaling/PCA; std <= 1e-6 uses scale = 1. No data clipping. t-SNE is exploratory, not an authenticity test.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    filename = "pca_tsne.png"
    _save_figure(figure, output_dir / filename, plt)
    return filename, {"n_plotted": len(selected), "reference_fit_rows": int(reference_mask.sum()), "source_counts": source_counts, "removed_nonfinite": removed_nonfinite, "pca_dimensions_before_tsne": components, "pca_2d_explained_variance": float(np.sum(pca.explained_variance_ratio_[:2])), "near_constant_std_threshold": 1e-6, "guarded_feature_names": [feature_names[index] for index in np.flatnonzero(guarded_features)], "tsne_perplexity": parameters["perplexity"], "tsne_iterations": 750, "seed": 42, "length": selected_length}


def _pca_all_lengths(path, output_dir, plt):
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits

    prepare_threadpool_compat()
    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float64)
        source = np.asarray(data["source"] if "source" in data else data["source_id"]).reshape(-1)
        type_id = np.asarray(data["type_id"]).reshape(-1)
        length = np.asarray(data["length"]).reshape(-1)
    if features.ndim != 2 or not (len(features) == len(source) == len(type_id) == len(length)):
        raise ValueError("visualization.npz shape mismatch")
    finite = np.isfinite(features).all(axis=1)
    features, source, type_id, length = features[finite], source[finite], type_id[finite], length[finite]
    reference_mask = (source == 0) | (source == 1)
    if int(reference_mask.sum()) < 3 or features.shape[1] < 2:
        raise ValueError("At least 3 reference rows and 2 features are required")
    # Treat each source/type/length combination as one allocation stratum.
    unique_pairs = sorted(set(zip(type_id.tolist(), length.tolist())))
    pair_ids = {pair: index for index, pair in enumerate(unique_pairs)}
    strata = np.asarray([pair_ids[pair] for pair in zip(type_id.tolist(), length.tolist())])
    selected = _stratified_indices(source, strata, limit=6000)
    with threadpool_limits(limits=1):
        scaler, _ = _fit_reference_scaler(features[reference_mask])
        pca = PCA(n_components=2, svd_solver="full", random_state=42).fit(scaler.transform(features[reference_mask]))
        points = pca.transform(scaler.transform(features[selected]))
    sample_source, sample_length = source[selected], length[selected]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for value in [0, 1, 2]:
        mask = sample_source == value
        if mask.any():
            axes[0].scatter(points[mask, 0], points[mask, 1], s=6, alpha=0.4, color=SOURCE_COLORS[value], label=f"{SOURCE_LABELS[value]} (n={int(mask.sum())})", linewidths=0, rasterized=True)
    lengths = sorted(np.unique(sample_length))
    for index, value in enumerate(lengths):
        mask = sample_length == value
        axes[1].scatter(points[mask, 0], points[mask, 1], s=6, alpha=0.45, color=plt.get_cmap("viridis")(index / max(1, len(lengths) - 1)), label=str(int(value)), linewidths=0, rasterized=True)
    axes[0].set_title("Colored by source")
    axes[1].set_title(f"Single sample length: {int(lengths[0])}" if len(lengths) == 1 else "Colored by sample length")
    for axis in axes:
        axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        axis.grid(alpha=0.12)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3), useMathText=True)
        axis.legend(fontsize=8, markerscale=2, ncol=2 if axis is axes[1] else 1)
    length_title = f"Length {int(lengths[0])}" if len(lengths) == 1 else f"{len(lengths)} evaluated lengths ({int(lengths[0])}-{int(lengths[-1])})"
    figure.suptitle(f"{length_title}: stratified PCA display (n={len(selected):,})", fontsize=15)
    _panel_labels(axes)
    figure.text(0.5, 0.012, "Reference-only scaling/PCA; std <= 1e-6 uses scale = 1. No data clipping. Source overlap is a display diagnostic.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    filename = "pca_all_lengths.png"
    _save_figure(figure, output_dir / filename, plt)
    return filename


def _nearest_waveforms(path, output_dir, plt):
    with np.load(path, allow_pickle=False) as data:
        generated = np.asarray(data["generated"])
        training = np.asarray(data["training"])
        length = np.asarray(data["length"]).reshape(-1)
        names = np.asarray(data["type_name"]).reshape(-1)
        sources = np.asarray(data["source"]).reshape(-1)
        generated_indices = np.asarray(data["generated_index"]).reshape(-1)
        rmse = np.asarray(data["rmse"]).reshape(-1)
    n_examples = min(12, len(generated))
    if not n_examples:
        raise ValueError("No waveform examples provided")
    order = np.argsort(np.nan_to_num(rmse, nan=np.inf), kind="stable")[:n_examples]
    rows = math.ceil(n_examples / 3)
    figure, axes = plt.subplots(rows, 3, figsize=(17.5, rows * 2.5), squeeze=False)
    for axis, index in zip(axes.flat, order):
        n = min(_integer(length[index]), generated.shape[-1], training.shape[-1])
        time = np.arange(n)
        axis.plot(time, training[index].reshape(-1)[:n], color="#0072b2", linewidth=1.05, alpha=0.9, label="Reference waveform")
        axis.plot(time, generated[index].reshape(-1)[:n], color="#d55e00", linewidth=0.8, alpha=0.8, label="Generated (aligned)")
        axis.set_title(f"{str(names[index])} | L={n} | {str(sources[index])}\nGenerated #{int(generated_indices[index])}, RMSE={float(rmse[index]):.4g}", fontsize=9)
        axis.set_xlabel("Effective-segment sample")
        axis.set_ylabel("Normalized amplitude")
        axis.grid(alpha=0.18)
    for axis in list(axes.flat)[n_examples:]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    _panel_labels(list(axes.flat)[:n_examples])
    figure.suptitle("Closest saved waveform candidates (ordered by normalized RMSE)", fontsize=15)
    figure.text(0.5, 0.012, "Candidate screening, not exhaustive waveform search. Inspect training_path and candidate metrics in the CSV.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    filename = "nearest_waveforms.png"
    _save_figure(figure, output_dir / filename, plt)
    return filename


def _aggregate_table(aggregate):
    headers = ["参照来源", "评估生成数", "分组数", "Coverage 宏平均", "综合 β-recall 宏平均", "Authenticity", "近似复制候选率", "精确匹配数"]
    parts = ["<table><thead><tr>" + "".join(f"<th>{header}</th>" for header in headers) + "</tr></thead><tbody>"]
    for row in aggregate:
        source = row.get("source", "")
        fields = [SOURCE_NAMES.get(source, source), _count(row.get("total_generated_evaluated")), _count(row.get("groups")), _percent(row.get("coverage_macro")), _percent(row.get("integrated_beta_recall_macro")), _percent(row.get("authenticity_full")), _percent(row.get("near_copy_rate")), _count(row.get("exact_matches"))]
        parts.append("<tr>" + "".join(f"<td>{_escaped(field)}</td>" for field in fields) + "</tr>")
    if not aggregate:
        parts.append('<tr><td colspan="8">本次结果没有提供汇总行，请查看分组 CSV。</td></tr>')
    parts.append("</tbody></table>")
    return "".join(parts)


def _calibration_table(aggregate):
    available = [row for row in aggregate if _integer(row.get("calibration_groups")) > 0]
    if not available:
        return "<p>本次结果未提供可用的参照拆分对照；不能据指标绝对值推断达到参照自身的覆盖水平。</p>"
    headers = ["参照来源", "有效对照组", "参照拆分 Coverage", "等规模生成 Coverage", "Coverage 差值（生成−参照）", "参照拆分综合 β-recall", "等规模生成综合 β-recall"]
    parts = ['<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{header}</th>" for header in headers) + "</tr></thead><tbody>"]
    for row in available:
        reference = _float(row.get("reference_split_coverage_macro"))
        generated = _float(row.get("matched_generated_coverage_macro"))
        gap = f"{(generated - reference) * 100:+.1f} 个百分点" if np.isfinite(reference) and np.isfinite(generated) else "—"
        fields = [SOURCE_NAMES.get(row.get("source"), row.get("source", "")), _count(row.get("calibration_groups")), _percent(reference), _percent(generated), gap, _percent(row.get("reference_split_irbeta_macro")), _percent(row.get("matched_generated_irbeta_macro"))]
        parts.append("<tr>" + "".join(f"<td>{_escaped(field)}</td>" for field in fields) + "</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _data_driven_text(aggregate, rows):
    paragraphs = []
    for item in aggregate:
        source = SOURCE_NAMES.get(item.get("source", ""), item.get("source", "未知来源"))
        paragraphs.append(f"与{source}比较：Coverage 宏平均 {_percent(item.get('coverage_macro'))}，综合 β-recall 宏平均 {_percent(item.get('integrated_beta_recall_macro'))}；Authenticity 为 {_percent(item.get('authenticity_full'))}，近似复制候选率为 {_percent(item.get('near_copy_rate'))}，精确匹配数 {_count(item.get('exact_matches'))}。")
    small = [row for row in rows if _truth(row.get("small_sample")) or _integer(row.get("n_reference")) < 10]
    invalid = [row for row in rows if not np.isfinite(_float(row.get("coverage")))]
    if small:
        paragraphs.append(f"本次 {len(rows):,} 个参照来源/类别/长度分组中，有 {len(small):,} 组被标为小样本（或参照少于 10 条）；这些组的估计不稳定，不应与大样本组等量解读。")
    if invalid:
        paragraphs.append(f"其中 {len(invalid):,} 个结果行没有可用 Coverage 数值；缺失值不等于 0，详见 CSV 的 status 与 n_reference。")
    paragraphs.append("以上数值应分别理解为覆盖程度、描述符空间中的分布重叠诊断和参照邻近度诊断。未设置经过验证的通过阈值，因此本报告不直接宣布“多样性合格”或“没有复制训练集”。")
    return "".join(f"<p>{_escaped(paragraph)}</p>" for paragraph in paragraphs)


def _build_report(output_dir):
    """Read evaluator artifacts, render PNG figures, and write report.html.

    Returns a dictionary with the report path, figures, and optional-plot errors.
    Required artifact: summary.json. Missing optional artifacts are reported in
    the HTML so that an incomplete visualization cannot look like a full audit.
    """
    prepare_threadpool_compat()
    output_dir = Path(output_dir).resolve()
    with (output_dir / "summary.json").open("r", encoding="utf-8-sig") as handle:
        summary = json.load(handle)
    provenance_data = summary.get("provenance", {})
    provenance_data = provenance_data if isinstance(provenance_data, dict) else {}
    scope_warning = provenance_data.get("reference_scope_warning")
    external_reference = bool(scope_warning) or provenance_data.get("reference_selection") == "explicit_directories"
    rows = _read_csv(output_dir / "group_metrics.csv")
    curves = _read_csv(output_dir / "beta_curves.csv")
    candidates = _read_csv(output_dir / "copy_candidates.csv")
    aggregate = summary.get("aggregate", [])
    if isinstance(aggregate, dict):
        aggregate = [dict(value, source=key) if isinstance(value, dict) else {"source": key} for key, value in aggregate.items()]
    aggregate = aggregate if isinstance(aggregate, list) else []
    plt = _pyplot()
    figures = {}
    errors = []
    projection_info = None

    def render(name, function):
        try:
            figures[name] = function()
        except Exception as exc:
            message = f"{name}: {type(exc).__name__}: {exc}"
            errors.append(message)
            warnings.warn(message, RuntimeWarning)
            plt.close("all")

    if aggregate:
        render("evaluation_summary", lambda: _summary_figure(aggregate, output_dir, plt, external_reference=external_reference))
    if rows:
        for source in ["synthetic", "real"]:
            render(f"heatmap_{source}", lambda source=source: _heatmap(rows, source, output_dir, plt))
    else:
        errors.append("group_metrics.csv 缺失或为空，未生成分组热图。")
    if curves:
        render("beta_recall", lambda: _beta_curves(curves, output_dir, plt))
    else:
        errors.append("beta_curves.csv 缺失或为空，未生成 β-recall 曲线。")
    if (output_dir / "visualization.npz").exists():
        render("pca_all_lengths", lambda: _pca_all_lengths(output_dir / "visualization.npz", output_dir, plt))
        try:
            figures["pca_tsne"], projection_info = _pca_tsne(output_dir / "visualization.npz", output_dir, plt)
        except Exception as exc:
            message = f"pca_tsne: {type(exc).__name__}: {exc}"
            errors.append(message)
            warnings.warn(message, RuntimeWarning)
            plt.close("all")
    else:
        errors.append("visualization.npz 缺失，未生成 PCA/t-SNE。")
    if (output_dir / "waveform_examples.npz").exists():
        render("nearest_waveforms", lambda: _nearest_waveforms(output_dir / "waveform_examples.npz", output_dir, plt))
    else:
        errors.append("waveform_examples.npz 缺失，未生成波形候选对照。")

    def figure_block(name, caption):
        if name not in figures:
            return ""
        filename = _escaped(figures[name])
        svgname = _escaped(str(Path(figures[name]).with_suffix(".svg")))
        png_dpi = 600 if name == "evaluation_summary" else 300
        return f'<figure><a href="{filename}"><img src="{filename}" alt="{_escaped(caption)}" loading="lazy"></a><figcaption>{_escaped(caption)} <a href="{filename}">PNG {png_dpi} dpi</a> · <a href="{svgname}">SVG 矢量图</a></figcaption></figure>'

    artifacts = ["summary.json", "group_metrics.csv", "reference_calibration.csv", "beta_curves.csv", "copy_candidates.csv", "visualization.npz", "waveform_examples.npz"]
    if _PDF_PAGES.get() is not None:
        artifacts.insert(0, "noise_evaluation_figures.pdf")
    artifacts.extend(path.name for path in sorted(output_dir.glob("*.log")))
    download_links = "".join(f'<li><a href="{_escaped(filename)}">{_escaped(filename)}</a></li>' for filename in artifacts if (output_dir / filename).exists())
    issue_block = "" if not errors else '<aside class="warning"><strong>报告完整性提示</strong><ul>' + "".join(f"<li>{_escaped(error)}</li>" for error in errors) + "</ul></aside>"
    provenance = json.dumps(provenance_data, ensure_ascii=False, indent=2)
    methodology = json.dumps(summary.get("methodology", {}), ensure_ascii=False, indent=2)
    evaluated_lengths = sorted({_integer(row.get("length")) for row in rows if _integer(row.get("length")) > 0})
    length_text = "、".join(str(length) for length in evaluated_lengths) or "未提供"
    generated_count = provenance_data.get("generated_count")
    scope_sentence = f"本次生成数据共 {_count(generated_count)} 条；评估长度集合为 {length_text}。" if generated_count is not None else f"本次评估长度集合为 {length_text}。"
    reference_term = "用户指定参照" if external_reference else "训练参照"
    report_title = "生成噪声：多样性与参考库相似度诊断" if external_reference else "生成噪声：多样性与训练集复制诊断"
    report_subtitle = "与用户指定参考库的相似度" if external_reference else "与是否复制训练集"
    scope_block = ""
    if external_reference:
        scope_parts = ["本次使用用户明确指定的参考库；参照身份不等同于已核实的完整模型训练集。Authenticity、近似复制及精确匹配只描述生成数据与这些参考库的关系。"]
        if provenance_data.get("recorded_include_real") is False:
            scope_parts.append("训练配置明确排除了实测数据，因此 Real 是未参与模型训练的实测对照；与 Real 相似不能称为复制了实测训练集。")
        if provenance_data.get("training_distribution_matches_record") is False:
            scope_parts.append("当前参考库的样本分布与原始训练记录不一致；合成参照是否属于原训练集、是否覆盖完整训练集尚未核实。")
        recorded = provenance_data.get("recorded_training", {})
        selected_reference = provenance_data.get("reference", {})
        if isinstance(recorded, dict) and isinstance(selected_reference, dict):
            original_sources = recorded.get("source_distribution", {})
            reference_sources = selected_reference.get("source_distribution", {})
            if isinstance(original_sources, dict) and isinstance(reference_sources, dict) and "synthetic" in original_sources and "synthetic" in reference_sources:
                scope_parts.append(f"原训练记录包含 {_count(original_sources['synthetic'])} 条合成样本；本次合成参考库为 {_count(reference_sources['synthetic'])} 条。即使 Authenticity 较高或匹配数为零，也不能据此断言未复制原始 {_count(recorded.get('n_samples'))} 条训练样本。")
        if scope_warning:
            scope_parts.append(str(scope_warning))
        directories = provenance_data.get("reference_directories", {})
        if isinstance(directories, dict):
            scope_parts.extend(f"{source} 参照目录：{path}" for source, path in directories.items())
        scope_block = '<aside class="warning"><strong>参照范围与结论边界</strong>' + "".join(f"<p>{_escaped(part)}</p>" for part in scope_parts) + "</aside>"
    calibration_scope = "A、B 都来自当前指定参考库，其拆分不构成新的模型训练/测试划分。" if external_reference else "B 也来自参与模型训练的数据，这不是独立 holdout 测试。"
    holdout_scope = "实测对照按训练配置未参与模型训练；合成参考库与原训练集的身份及完整性未核实。本次属于指定参考库的比较，不能据此证明模型没有记忆完整训练集。" if external_reference and provenance_data.get("recorded_include_real") is False else ("当前指定参照的训练身份与独立性未核实，不能将其自动视为独立 holdout 或完整训练集。" if external_reference else "没有独立的未参与训练 holdout 参照。当前结果描述生成数据与本地训练数据的关系，不能单独回答对新实测数据的泛化能力。")
    sample_shape = provenance_data.get("generated_shape", [])
    full_window = bool(len(evaluated_lengths) == 1 and sample_shape and evaluated_lengths[0] == sample_shape[-1])
    window_scope = f"本次只评估长度 {evaluated_lengths[0]}，与保存波形窗口等长，无需从更长窗口定位短片段；归一化和符号/时间反转等价处理仍会影响距离。" if full_window else "当片段短于保存窗口时，评估需估计有效片段并对齐；归一化、符号/时间反转等价处理和窗口定位会影响距离，应结合方法配置及能量指标检查。"
    projection = ""
    projection_length = projection_info["length"] if projection_info else (evaluated_lengths[-1] if evaluated_lengths else None)
    if projection_info:
        projection = f"<p>PCA/t-SNE 对照图仅使用长度 {projection_length}：绘制 " + _count(projection_info["n_plotted"]) + " 条，StandardScaler/PCA 在 " + _count(projection_info["reference_fit_rows"]) + " 条可用参照上拟合；PCA 前两维解释方差 " + _percent(projection_info["pca_2d_explained_variance"]) + "。t-SNE 先降至最多 20 维，随机种子 42，750 次迭代。按来源/类别分层选择至多 3,000 条，图中的数量不是总体比例。</p>"
        if projection_info["removed_nonfinite"]:
            projection += f'<p>可视化排除了 {projection_info["removed_nonfinite"]:,} 条特征含非有限值的记录。</p>'
    candidate_preview = ""
    if candidates:
        candidate_preview = '<div class="table-wrap"><table><thead><tr><th>参照来源</th><th>类型</th><th>长度</th><th>生成索引</th><th>归一化 RMSE</th><th>参照文件</th></tr></thead><tbody>'
        ordered = sorted(candidates, key=lambda row: _float(row.get("rmse")) if np.isfinite(_float(row.get("rmse"))) else float("inf"))[:12]
        for row in ordered:
            fields = [row.get("source", ""), row.get("type_name", ""), row.get("length", ""), row.get("generated_index", ""), row.get("rmse", ""), row.get("training_path", "")]
            candidate_preview += "<tr>" + "".join(f"<td>{_escaped(value)}</td>" for value in fields) + "</tr>"
        candidate_preview += "</tbody></table></div>"
    report = f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{report_title}</title>
<style>
:root{{--ink:#173047;--muted:#526474;--line:#dce4eb;--accent:#076c89}}
*{{box-sizing:border-box}}body{{margin:0;background:#f2f5f8;color:var(--ink);font:16px/1.75 "Microsoft YaHei","Noto Sans CJK SC",sans-serif}}
main{{max-width:1400px;margin:auto;padding:32px 38px 60px;background:white}}header{{border-bottom:5px solid var(--accent);padding-bottom:24px;margin-bottom:30px}}
h1{{font-size:32px;line-height:1.35;margin:0 0 12px}}h2{{font-size:23px;margin-top:38px;border-left:5px solid var(--accent);padding-left:12px}}h3{{font-size:18px}}
p,li{{max-width:1150px}}.subtitle,figcaption{{color:var(--muted)}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:18px 0}}
th,td{{border:1px solid var(--line);padding:9px 11px;text-align:left;vertical-align:top}}th{{background:#eaf1f6;white-space:nowrap}}td:last-child{{overflow-wrap:anywhere}}
figure{{margin:24px 0}}img{{display:block;width:100%;height:auto;border:1px solid var(--line)}}figcaption{{font-size:14px;margin-top:8px}}
aside{{background:#edf5f9;border-left:4px solid #3886a2;padding:16px 20px}}aside.warning{{background:#fff6e9;border-color:#ba7c22}}a{{color:var(--accent)}}
details{{border:1px solid var(--line);padding:10px 16px;margin:16px 0}}summary{{cursor:pointer;font-weight:bold}}pre{{overflow:auto;white-space:pre-wrap;font:12px/1.65 Consolas,monospace}}
.artifacts{{columns:2}}footer{{margin-top:36px;border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:13px}}
@media(max-width:720px){{main{{padding:20px 15px}}h1{{font-size:25px}}.artifacts{{columns:1}}}}@media print{{body{{background:white}}main{{max-width:none;padding:0}}figure,aside{{break-inside:avoid}}details{{display:block}}}}
</style></head><body><main>
<header><h1>生成噪声数据的多样性<br>{report_subtitle}</h1><div class="subtitle">{'用户指定的实测 / 合成参照' if external_reference else '实测 / 合成训练参照'} · 类别与长度分组 · 可复核的离线诊断报告</div></header>
{issue_block}
{scope_block}
<h2>1. 本次结果</h2>
<p>{_escaped(scope_sentence)}</p>
{_data_driven_text(aggregate, rows)}
<div class="table-wrap">{_aggregate_table(aggregate)}</div>
<p>Coverage 与综合 β-recall 的宏平均先按类别/长度分别计算，再对有效组等权汇总；Authenticity 和复制候选率按各行定义的生成样本统计。各来源参照使用同一批生成数据，不能把 real、synthetic、pooled 的评估条数相加当作独立样本数。“—”表示不可用，而不是零。</p>
{figure_block('evaluation_summary', '英文三联汇总图：多样性 ↑、Authenticity ↑、近似复制候选率 ↓。箭头表示指标的偏好方向；各来源有效组集合不同，未显示置信区间，不能据柱高差异作显著性判断。')}
<h2>2. 多样性：覆盖与 β-recall</h2>
<p>Coverage 检查生成样本是否进入{reference_term}的局部近邻范围，综合 β-recall 检查描述符分布覆盖。两者在每个“噪声类别 × 样本长度”组中使用平衡的参照/生成抽样，减少样本数差异带来的影响；实际平衡数量见 n_balanced。高覆盖本身不能排除训练集复制。</p>
<h3>同规模的参照拆分对照</h3>
<p>对每组参照随机拆出互不重叠的 A、B 两部分，每部分 m = min(512, ⌊n_reference / 2⌋) 条；再取同样 m 条生成数据 G，使用同一个 A 分别评价 B 与 G。该对照帮助识别有限样本规模造成的指标偏差，避免只看综合 β-recall 绝对高低判断好坏。<strong>{calibration_scope}</strong></p>
{_calibration_table(aggregate)}
<p>对照表仅汇总可拆分且可计算的分组，因样本规模与有效组集合可能不同，不能将该表的数值与上方主指标直接作差。逐组对照见 reference_calibration.csv。</p>
{figure_block('heatmap_synthetic', '合成参照：三项分组指标。灰格为缺组或不可计算，星号为小样本组；数值均显示至两位小数。')}
{figure_block('heatmap_real', '实测参照：仅在有对应类别和长度参照的组中评估。灰格不能解释为生成失败。')}
{figure_block('beta_recall', 'β-recall 按有效类别/长度组做等权平均；虚线表示理想分布匹配。曲线重叠诊断依赖所用特征空间与有限样本。')}
<h2>3. PCA / t-SNE 分布视图</h2>
{projection}
<p>两张图的标准化与主评估保持相同的近常数保护：参照特征标准差 ≤ 10⁻⁶ 时，缩放因子设为 1；其他特征按参照标准差缩放，均使用参照均值中心化。这避免 RMS 归一化后近常数特征的浮点误差被放大。未裁剪生成特征或投影坐标；t-SNE 使用同一保护后的 PCA 表示。</p>
<p>第一张图覆盖本次长度集合（{_escaped(length_text)}），分别按来源和样本长度着色；只有一种长度时，长度颜色仅说明统一的采样长度，不代表跨长度覆盖。第二张图固定长度 {_escaped(projection_length if projection_length is not None else '未提供')}，上排按来源着色、下排按类别着色。两张图都只用于展示结构和异常；点云重叠、分离或覆盖面积都不能单独证明分布质量，也不能证明没有记忆训练数据。</p>
{figure_block('pca_all_lengths', f'本次评估长度集合（{length_text}）的分层抽样 PCA：最多 6,000 条，仅参照数据拟合 StandardScaler/PCA。')}
{figure_block('pca_tsne', f'长度 {projection_length if projection_length is not None else "未提供"} 的分层抽样可视化。PCA 与 t-SNE 使用相同记录；图形是探索性证据，不能替代定量指标。')}
<h2>4. 邻近度与复制诊断：Authenticity 和近邻波形</h2>
<p>Authenticity 对比生成样本到最近参照的距离与该参照到其他参照样本的最近邻距离。本报告使用本地信号描述符空间的适配版；该分数是过度靠近{reference_term}的诊断量，<strong>不构成“没有复制”的证明</strong>。参照少于两条时不能建立参照间最近邻基线；参照不等于完整训练集时，也不能用该分数判断模型是否记忆全部训练数据。</p>
<p>波形复核在同类同长度条件下，从各参照来源的重采样波形签名近邻中筛选 5 个参照候选，再对完整有效片段进行形状匹配，并考虑符号翻转和时间反转。pooled 汇合实测与合成各 5 个候选后取最小值，因此最多复核 10 个候选。候选中的最小归一化 RMSE 不是在所有参照波形中穷尽搜索得到的全局最小值；本次近似复制阈值为单位 RMS 归一化后 RMSE ≤ 0.05。</p>
<p>精确匹配比较的是<strong>未经再次归一化的生成有效窗口</strong>与<strong>按训练配置预处理的参照有效片段</strong>的 float32 字节，考虑符号/时间反转及窗口偏移对齐。计算哈希前，两侧都去除片段首尾的精确零值，内部数值与内部零值保持不变，以避免最大能量并列窗口造成偏移漏检。因此它区别于原始文件字节相同，也区别于容差内近似相同。即使精确匹配数为零，也不能排除加了微小扰动的复制；近似匹配还受候选搜索范围限制。</p>
{figure_block('nearest_waveforms', '保存候选中最接近的至多 12 对波形。已对齐到有效片段，右侧补零不进入绘图区；请结合 CSV 中 training_path（参照文件路径）逐条复核。')}
{candidate_preview}
<h2>5. 方法边界与可复现信息</h2>
<aside><ul>
<li>本报告是信号描述符空间中的适配评估，并非原论文学习特征表示的原样复现；它主要检查给定描述符能表达的统计与形状差异。</li>
<li>{holdout_scope}</li>
<li>{window_scope}</li>
<li>小样本组、参照自身重复或近重复，会影响 kNN 半径和 Authenticity 基线。来源之间组数及样本数不同，不宜只比较汇总百分比排名。</li>
<li>多样性与真实性/不复制是不同问题：一个样本可以偏离训练数据很多而失真，也可以具有很好的覆盖同时包含复制样本。</li>
</ul></aside>
<details><summary>数据来源与运行溯源（provenance）</summary><pre>{_escaped(provenance)}</pre></details>
<details><summary>本次方法与参数（methodology）</summary><pre>{_escaped(methodology)}</pre></details>
<details><summary>完整汇总 JSON</summary><pre>{_escaped(json.dumps(summary, ensure_ascii=False, indent=2))}</pre></details>
<h2>6. 下载与参考</h2><ul class="artifacts">{download_links}</ul>
<p>指标依据：<a href="https://proceedings.mlr.press/v162/alaa22a.html">Alaa et al., 2022 — How Faithful is Your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models</a>；<a href="https://proceedings.mlr.press/v119/naeem20a.html">Naeem et al., 2020 — Reliable Fidelity and Diversity Metrics for Generative Models</a>。</p>
<footer>报告由 noise_eval_report.py 从同目录评估产物生成。该 HTML 不依赖外部脚本；分享时请连同 PNG 图与相关 CSV 一起保留。</footer>
</main></body></html>'''
    report_path = output_dir / "report.html"
    report_path.write_text(report, encoding="utf-8")
    return {"report": str(report_path), "figures": figures, "plot_errors": errors, "projection": projection_info}


def build_report(output_dir, export_pdf=False):
    """Render the report; optionally export all figures into one vector PDF.

    PDF export is opt-in.  The normal evaluation path writes only HTML, PNG,
    and SVG.  Scatter points are rasterized inside the PDF/SVG; text, axes,
    bars, and waveform curves retain vector geometry.
    """
    if not export_pdf:
        return _build_report(output_dir)
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = Path(output_dir).resolve() / "noise_evaluation_figures.pdf"
    with (Path(output_dir) / "summary.json").open("r", encoding="utf-8-sig") as handle:
        provenance = json.load(handle).get("provenance", {})
    external = isinstance(provenance, dict) and (bool(provenance.get("reference_scope_warning")) or provenance.get("reference_selection") == "explicit_directories")
    pdf_title = "Generated noise: diversity and " + ("reference-set" if external else "training-set") + " proximity"
    with PdfPages(pdf_path, metadata={"Title": pdf_title, "Subject": "Descriptor-space diagnostics and nearest-reference waveform review", "Creator": "noise_eval_report.py"}) as pdf_pages:
        token = _PDF_PAGES.set(pdf_pages)
        try:
            result = _build_report(output_dir)
        finally:
            _PDF_PAGES.reset(token)
    result["figures_pdf"] = str(pdf_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="Render an existing noise evaluation as HTML and PNG figures")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--export_pdf", action="store_true", help="Also write one multi-page PDF containing all figures")
    args = parser.parse_args()
    print(json.dumps(build_report(args.output_dir, export_pdf=args.export_pdf), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
