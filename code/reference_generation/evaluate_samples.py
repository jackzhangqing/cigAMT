"""Evaluate generated four-channel GeoAMT time series.

This pipeline implements the principal unconditional-generation metrics used by
Diffusion-TS and TransFusion, plus frequency-domain checks suited to AMT data.
Input arrays may be HDF5 (dataset ``data`` by default) or NumPy ``.npy`` files
and must have shape (N, C, T) or (N, T, C).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from evaluation.data_io import (
    ChannelStandardizer,
    file_sha256,
    hash_inventory,
    load_array,
    parse_named_path,
)
from evaluation.plots import (
    plot_alpha_beta,
    plot_channel_distributions,
    plot_coherence,
    plot_correlation_matrices,
    plot_pca_tsne,
    plot_psd,
    plot_waveforms,
    safe_name,
)
from evaluation.static_metrics import (
    alpha_precision_beta_recall,
    channel_statistics,
    coherence_metrics,
    correlational_score,
    extract_amt_features,
    frechet_distance,
    knn_precision_recall_coverage,
    marginal_jsd,
    nearest_neighbor_audit,
    spectral_metrics,
    standardize_feature_pair,
)


PAPER_METRIC_NOTES = {
    "lds": "TransFusion Long-Sequence Discriminative Score: abs(accuracy - 0.5).",
    "lps": "TransFusion TSTR next-step/five-step prediction MAE.",
    "context_fid": "Diffusion-TS Context-FID using official TS2Vec full-series embeddings.",
    "correlational_score": "Diffusion-TS error between real/generated channel correlation matrices.",
    "marginal_jsd": "TransFusion Jensen-Shannon divergence of marginal amplitudes.",
    "alpha_precision_beta_recall": (
        "Alaa et al. public alpha-precision/beta-recall construction with "
        "range-preserving trapezoidal integration."
    ),
    "coverage": "Nearest-neighbour manifold coverage of real samples.",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(v) for v in value]
    return value


def _save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return result


def _sample_limit(data: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or len(data) <= maximum:
        return data
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(data), size=maximum, replace=False))
    return data[indices]


def _metric_view(result: dict[str, Any]) -> dict[str, Any]:
    """Drop arrays that belong in plots rather than the JSON results."""
    excluded = {
        "real",
        "generated",
        "absolute_difference",
        "frequencies",
        "real_psd",
        "generated_psd",
        "alphas",
        "alpha_precision_curve",
        "beta_recall_curve",
    }
    return {key: value for key, value in result.items() if key not in excluded}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diffusion-TS/TransFusion plus AMT-domain evaluation for generated waveforms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--real", required=True, help="Independent real reference HDF5/NPY file")
    parser.add_argument(
        "--generated",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Generated file; repeat for DDIM, DDPM, or baselines",
    )
    parser.add_argument(
        "--train",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Optional training files for exact/nearest-neighbour memorization audits",
    )
    parser.add_argument("--key", default="data", help="HDF5 dataset key")
    parser.add_argument("--output", default="evaluation_results", help="Output directory")
    parser.add_argument("--channels", default="Ex,Ey,Hx,Hy", help="Comma-separated channel names")
    parser.add_argument("--sampling-rate", type=float, default=150.0, help="Sampling rate in Hz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum samples loaded per dataset for distributional metrics",
    )
    parser.add_argument("--max-neural", type=int, default=3000, help="Maximum samples for LDS/LPS")
    parser.add_argument("--max-tsne", type=int, default=1000)
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=300,
        help="PNG resolution; 300 is journal-ready for color figures",
    )
    parser.add_argument("--hist-bins", type=int, default=200)
    parser.add_argument("--welch-nperseg", type=int, default=256)
    parser.add_argument("--feature-pca-dims", type=int, default=50)
    parser.add_argument("--manifold-k", type=int, default=5)
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Stop instead of recording optional metric errors")
    parser.add_argument("--repeats", type=int, default=5, help="Use 10 for final paper results")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--horizons", type=_parse_int_list, default=[1, 5])
    parser.add_argument(
        "--ts2vec-repo",
        default=None,
        help="Path to official zhihanyue/ts2vec repo; enables paper-exact Context-FID features",
    )
    parser.add_argument("--ts2vec-checkpoint", default=None)
    parser.add_argument("--ts2vec-epochs", type=int, default=20)
    parser.add_argument("--full-dedup", action="store_true", help="SHA-256 every sample, not just loaded subsets")
    parser.add_argument("--hash-files", action="store_true", help="SHA-256 each full input file")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    output = Path(args.output)
    figures = output / "figures"
    tables = output / "tables"
    output.mkdir(parents=True, exist_ok=True)
    channel_names = [item.strip() for item in args.channels.split(",") if item.strip()]
    if not channel_names:
        raise ValueError("No channel names provided")

    generated_specs = [parse_named_path(spec) for spec in args.generated]
    train_specs = [parse_named_path(spec) for spec in args.train]
    paths = [Path(args.real)] + [path for _, path in generated_specs] + [path for _, path in train_specs]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    print(f"[1/6] Loading real reference: {args.real}")
    real, real_info = load_array(
        args.real,
        args.key,
        args.max_samples,
        args.seed,
        expected_channels=len(channel_names),
    )
    reference_source = str(real_info.attrs.get("source", "")).strip().lower()
    reference_label = (
        "Mixed reference" if reference_source == "mixed" else "Real reference"
    )
    generated: dict[str, np.ndarray] = {}
    inventory: dict[str, Any] = {"real": real_info.to_dict(), "generated": {}, "train": {}}
    for offset, (name, path) in enumerate(generated_specs):
        print(f"      Loading generated set {name}: {path}")
        data, info = load_array(
            path,
            args.key,
            args.max_samples,
            args.seed + 10 + offset,
            expected_channels=len(channel_names),
        )
        if data.shape[1:] != real.shape[1:]:
            raise ValueError(f"Shape mismatch: real={real.shape}, {name}={data.shape}")
        generated[name] = data
        inventory["generated"][name] = info.to_dict()

    standardizer = ChannelStandardizer.fit(real)
    real_scaled = standardizer.transform(real)
    generated_scaled = {name: standardizer.transform(data) for name, data in generated.items()}

    print("[2/6] Extracting deterministic AMT features")
    real_features_raw = extract_amt_features(real_scaled, args.sampling_rate)
    fake_features_raw = {
        name: extract_amt_features(data, args.sampling_rate) for name, data in generated_scaled.items()
    }
    feature_scaler = None
    standardized_features: dict[str, np.ndarray] = {}
    for name, fake in fake_features_raw.items():
        real_standard, fake_standard, fitted = standardize_feature_pair(real_features_raw, fake)
        if feature_scaler is None:
            feature_scaler = fitted
            real_features = real_standard
        standardized_features[name] = fake_standard

    pca_dims = min(args.feature_pca_dims, real_features.shape[1], len(real_features) - 1)
    feature_pca = PCA(n_components=pca_dims, random_state=args.seed).fit(real_features)
    real_manifold_features = feature_pca.transform(real_features)
    fake_manifold_features = {
        name: feature_pca.transform(features) for name, features in standardized_features.items()
    }

    ts2vec_real = None
    ts2vec_generated: dict[str, np.ndarray] = {}
    ts2vec_status: dict[str, Any] = {"enabled": bool(args.ts2vec_repo)}
    if args.ts2vec_repo:
        print("[3/6] Training/loading official TS2Vec for Context-FID")
        try:
            from evaluation.neural_metrics import resolve_device, ts2vec_embeddings

            device = resolve_device(args.device)
            ts2vec_real, ts2vec_generated, checkpoint = ts2vec_embeddings(
                real_scaled,
                generated_scaled,
                args.ts2vec_repo,
                device,
                output,
                args.ts2vec_epochs,
                args.batch_size,
                args.ts2vec_checkpoint,
            )
            ts2vec_status.update({"status": "computed", "checkpoint": checkpoint, "device": device})
        except Exception as exc:
            ts2vec_status.update({"status": "error", "error": str(exc)})
            if args.strict:
                raise
            print(f"      Context-FID skipped after TS2Vec error: {exc}")
    else:
        print("[3/6] Context-FID skipped (provide --ts2vec-repo for the official implementation)")
        ts2vec_status.update(
            {
                "status": "not_computed",
                "reason": "No --ts2vec-repo supplied. AMT feature FID is not labeled Context-FID.",
            }
        )

    results: dict[str, Any] = {
        "protocol": {
            "paper_metric_notes": PAPER_METRIC_NOTES,
            "channel_standardizer": standardizer.to_dict(),
            "feature_representation": {
                "fallback": "standardized handcrafted AMT time/spectral/cross-channel features",
                "pca_dimensions_for_manifold_metrics": pca_dims,
                "ts2vec": ts2vec_status,
            },
            "warning": (
                "--real must point to records never used to train the diffusion model. "
                "This script cannot prove that provenance automatically."
            ),
        },
        "datasets": inventory,
        "methods": {},
        "errors": {},
    }

    print("[4/6] Computing distributional, manifold, and AMT-domain metrics")
    stats_rows: list[dict[str, Any]] = []
    for row in channel_statistics(real, channel_names):
        stats_rows.append({"dataset": "real", **row})
    summary_rows: list[dict[str, Any]] = []
    for index, (name, fake) in enumerate(generated.items()):
        print(f"      Static metrics: {name}")
        method_dir = figures / safe_name(name)
        corr = correlational_score(real, fake)
        jsd = marginal_jsd(real, fake, channel_names, args.hist_bins)
        spectral = spectral_metrics(real, fake, channel_names, args.sampling_rate, args.welch_nperseg)
        coherence = coherence_metrics(
            _sample_limit(real, 1000, args.seed),
            _sample_limit(fake, 1000, args.seed + index + 1),
            channel_names,
            args.sampling_rate,
            args.welch_nperseg,
        )
        # Use the real-fitted PCA space to keep covariance estimation stable and
        # identical across every compared method.
        amt_fid = frechet_distance(real_manifold_features, fake_manifold_features[name])

        if ts2vec_real is not None:
            representation_real = ts2vec_real
            representation_fake = ts2vec_generated[name]
            context_fid = frechet_distance(representation_real, representation_fake)
            manifold_source = "TS2Vec full-series embeddings"
            rep_real_standard, rep_fake_standard, _ = standardize_feature_pair(
                representation_real, representation_fake
            )
            rep_dims = min(args.feature_pca_dims, rep_real_standard.shape[1], len(rep_real_standard) - 1)
            rep_pca = PCA(n_components=rep_dims, random_state=args.seed).fit(rep_real_standard)
            manifold_real = rep_pca.transform(rep_real_standard)
            manifold_fake = rep_pca.transform(rep_fake_standard)
            ts2vec_plot_features = (rep_real_standard, rep_fake_standard)
        else:
            context_fid = None
            manifold_source = "handcrafted AMT features (fallback; not TS2Vec)"
            manifold_real = real_manifold_features
            manifold_fake = fake_manifold_features[name]
            ts2vec_plot_features = None

        balanced = min(len(manifold_real), len(manifold_fake))
        rng = np.random.default_rng(args.seed + index)
        real_idx = rng.choice(len(manifold_real), size=balanced, replace=False)
        fake_idx = rng.choice(len(manifold_fake), size=balanced, replace=False)
        alpha_beta = alpha_precision_beta_recall(
            manifold_real[real_idx], manifold_fake[fake_idx]
        )
        knn = knn_precision_recall_coverage(
            manifold_real[real_idx], manifold_fake[fake_idx], args.manifold_k
        )

        method_result = {
            "correlational": _metric_view(corr),
            "marginal_jsd": jsd,
            "amt_domain": {
                "psd": _metric_view(spectral),
                "coherence": _metric_view(coherence),
            },
            "amt_feature_fid": amt_fid,
            "context_fid": context_fid,
            "manifold_representation": manifold_source,
            "alpha_precision_beta_recall": _metric_view(alpha_beta),
            "knn_manifold": knn,
        }
        results["methods"][name] = method_result
        summary_rows.append(
            {
                "method": name,
                "LDS_down": None,
                "LPS_1_down": None,
                "LPS_5_down": None,
                "Context_FID_down": context_fid,
                "AMT_feature_FID_down": amt_fid,
                "Correlational_down": corr["score_off_diagonal"],
                "JSD_down": jsd["mean"],
                "Alpha_precision_up": alpha_beta["alpha_precision"],
                "Beta_recall_up": alpha_beta["beta_recall"],
                "Coverage_up": knn["coverage"],
                "Authenticity_up": alpha_beta["authenticity"],
                "PSD_JSD_down": spectral["psd_jsd_mean"],
                "Coherence_MAE_down": coherence["mean_absolute_error"],
            }
        )
        for row in channel_statistics(fake, channel_names):
            stats_rows.append({"dataset": name, **row})

        plot_waveforms(
            real,
            fake,
            channel_names,
            method_dir / "waveforms.png",
            args.seed + index,
            reference_label=reference_label,
            generated_label=name,
            dpi=args.figure_dpi,
        )
        plot_channel_distributions(
            real,
            fake,
            channel_names,
            method_dir / "amplitude_distributions.png",
            reference_label=reference_label,
            generated_label=name,
            dpi=args.figure_dpi,
        )
        plot_correlation_matrices(
            corr,
            channel_names,
            method_dir / "correlation_matrices.png",
            reference_label=reference_label,
            generated_label=name,
            dpi=args.figure_dpi,
        )
        plot_psd(
            spectral,
            channel_names,
            method_dir / "power_spectral_density.png",
            reference_label=reference_label,
            generated_label=name,
            dpi=args.figure_dpi,
        )
        plot_coherence(
            coherence,
            method_dir / "channel_coherence.png",
            reference_label=reference_label,
            generated_label=name,
            dpi=args.figure_dpi,
        )
        plot_pca_tsne(
            real_features,
            standardized_features[name],
            method_dir / "pca_tsne_amt_features.png",
            args.seed + index,
            args.max_tsne,
            args.skip_tsne,
            reference_label=reference_label,
            generated_label=name,
            feature_label="handcrafted AMT features",
            dpi=args.figure_dpi,
        )
        if ts2vec_plot_features is not None:
            plot_pca_tsne(
                ts2vec_plot_features[0],
                ts2vec_plot_features[1],
                method_dir / "pca_tsne_ts2vec.png",
                args.seed + index,
                args.max_tsne,
                args.skip_tsne,
                reference_label=reference_label,
                generated_label=name,
                feature_label="TS2Vec full-series embeddings",
                dpi=args.figure_dpi,
            )
        plot_alpha_beta(
            alpha_beta, method_dir / "alpha_beta_curves.png", dpi=args.figure_dpi
        )

    _save_rows(tables / "channel_statistics.csv", stats_rows)

    print("[5/6] Computing LDS/LPS neural metrics")
    neural_rows_lds: list[dict[str, Any]] = []
    neural_rows_lps: list[dict[str, Any]] = []
    if not args.skip_neural:
        try:
            from evaluation.neural_metrics import (
                long_sequence_discriminative_score,
                long_sequence_predictive_score,
                resolve_device,
            )

            device = resolve_device(args.device)
            neural_real = _sample_limit(real_scaled, args.max_neural, args.seed)
            for index, name in enumerate(generated):
                print(f"      LDS/LPS: {name} on {device}")
                neural_fake = _sample_limit(
                    generated_scaled[name], args.max_neural, args.seed + 100 + index
                )
                lds_summary, lds_rows = long_sequence_discriminative_score(
                    neural_real,
                    neural_fake,
                    args.repeats,
                    args.seed,
                    device,
                    args.epochs,
                    args.batch_size,
                    args.learning_rate,
                    args.patch_size,
                    args.d_model,
                    args.heads,
                    args.layers,
                    args.dropout,
                    args.patience,
                )
                lps_summary, lps_rows = long_sequence_predictive_score(
                    neural_real,
                    neural_fake,
                    args.horizons,
                    standardizer.std,
                    args.repeats,
                    args.seed,
                    device,
                    args.epochs,
                    args.batch_size,
                    args.learning_rate,
                    args.patch_size,
                    args.d_model,
                    args.heads,
                    args.layers,
                    args.dropout,
                    args.patience,
                )
                results["methods"][name]["lds"] = lds_summary
                results["methods"][name]["lps"] = lps_summary
                for row in lds_rows:
                    neural_rows_lds.append({"method": name, **row})
                for row in lps_rows:
                    neural_rows_lps.append({"method": name, **row})
                summary = next(row for row in summary_rows if row["method"] == name)
                summary["LDS_down"] = lds_summary.get("lds_mean")
                for horizon in args.horizons:
                    column = f"LPS_{horizon}_down"
                    if column not in summary:
                        summary[column] = None
                    summary[column] = lps_summary[str(horizon)].get("synthetic_mae_mean")
        except Exception as exc:
            results["errors"]["neural_metrics"] = str(exc)
            if args.strict:
                raise
            print(f"      LDS/LPS not computed: {exc}")
    else:
        results["errors"]["neural_metrics"] = "Skipped by --skip-neural"
    if neural_rows_lds:
        _save_rows(tables / "lds_runs.csv", neural_rows_lds)
    if neural_rows_lps:
        _save_rows(tables / "lps_runs.csv", neural_rows_lps)

    if train_specs:
        print("      Training-set nearest-neighbour audit")
        train_arrays: list[np.ndarray] = []
        per_file_limit = max(1, args.max_samples // len(train_specs))
        for offset, (train_name, train_path) in enumerate(train_specs):
            data, info = load_array(
                train_path,
                args.key,
                per_file_limit,
                args.seed + 500 + offset,
                expected_channels=len(channel_names),
            )
            inventory["train"][train_name] = info.to_dict()
            train_arrays.append(data)
        train = np.concatenate(train_arrays, axis=0)
        train_features = extract_amt_features(standardizer.transform(train), args.sampling_rate)
        train_features = feature_scaler.transform(train_features)
        train_features = feature_pca.transform(train_features)
        nearest_rows: list[dict[str, Any]] = []
        for name, fake_features in fake_manifold_features.items():
            audit_summary, audit_rows = nearest_neighbor_audit(train_features, fake_features)
            results["methods"][name]["training_nearest_neighbor"] = audit_summary
            nearest_rows.extend({"method": name, **row} for row in audit_rows)
        _save_rows(tables / "nearest_training_samples.csv", nearest_rows)

    if args.full_dedup:
        print("      Full exact-duplicate SHA-256 audit")
        dedup: dict[str, Any] = {"train": {}, "generated": {}, "cross_generated": {}}
        train_hashes: set[str] = set()
        for name, path in train_specs:
            hash_counts, total, internal = hash_inventory(
                path, args.key, expected_channels=len(channel_names)
            )
            train_hashes.update(hash_counts)
            dedup["train"][name] = {
                "samples": total,
                "unique": len(hash_counts),
                "internal_duplicates": internal,
            }
        generated_hashes: dict[str, set[str]] = {}
        for name, path in generated_specs:
            hash_counts, total, internal = hash_inventory(
                path, args.key, expected_channels=len(channel_names)
            )
            hashes = set(hash_counts)
            generated_hashes[name] = hashes
            matching_unique = len(hashes & train_hashes) if train_hashes else None
            matching_samples = (
                sum(count for digest, count in hash_counts.items() if digest in train_hashes)
                if train_hashes
                else None
            )
            dedup["generated"][name] = {
                "samples": total,
                "unique": len(hashes),
                "internal_duplicates": internal,
                "exact_unique_matches_to_training": matching_unique,
                "exact_matching_samples_to_training": matching_samples,
                "exact_training_match_rate": (
                    matching_samples / total if matching_samples is not None and total else None
                ),
            }
        names = list(generated_hashes)
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                key = f"{names[left]}__vs__{names[right]}"
                dedup["cross_generated"][key] = len(
                    generated_hashes[names[left]] & generated_hashes[names[right]]
                )
        results["exact_duplicate_audit"] = dedup

    print("[6/6] Writing report artifacts")
    if args.hash_files:
        results["file_sha256"] = {str(path.resolve()): file_sha256(path) for path in paths}
    results["run"] = {
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": time.time() - started,
        "seed": args.seed,
        "repeats": args.repeats,
    }
    _save_rows(tables / "paper_summary.csv", summary_rows)
    _write_json(output / "data_inventory.json", inventory)
    _write_json(output / "results.json", _finite_or_none(results))
    _write_json(output / "run_config.json", vars(args))
    print(f"Done. Summary: {tables / 'paper_summary.csv'}")
    print(f"Full results: {output / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
