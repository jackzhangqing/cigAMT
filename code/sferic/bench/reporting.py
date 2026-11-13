"""Summarize observed completed runs without mixing seeds, folds or protocols."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np

from .runtime import digest, read_json, scientific_config, write_json


METRICS = ("pr_auc", "roc_auc", "f1", "precision", "recall", "balanced_accuracy", "mcc", "brier", "ece")
GROUP_KEYS = ("model", "version", "augmentation", "channels", "train_fraction", "protocol",
              "group_by", "n_splits", "fold", "condition_id", "implementation_id", "dataset_id", "smoke_only")


def _csv(path, rows, fallback):
    keys = list(rows[0]) if rows else list(fallback)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _metrics(value):
    if not isinstance(value, dict):
        return {}
    candidate = value.get("metrics", value)
    return candidate if isinstance(candidate, dict) else {}


def _condition(config):
    data = scientific_config(config)
    for key in ("model", "seed", "fold"):
        data.pop(key, None)
    return digest(data)[:12]


def _row(path, result):
    config = result.get("config", {})
    training = result.get("training", {})
    provenance_path = path.parent / "provenance.json"
    provenance = read_json(provenance_path) if provenance_path.exists() else {}
    channels = config.get("channels", "all")
    if isinstance(channels, (list, tuple)):
        channels = ",".join(str(value) for value in channels)
    row = dict(run_dir=str(path.parent), model=config.get("model", result.get("model", "unknown")),
        seed=int(config.get("seed", result.get("seed", 0))), version=config.get("version", "standardized"),
        augmentation=config.get("augmentation", "all"), channels=str(channels),
        train_fraction=float(config.get("train_fraction", 1.0)), protocol=config.get("protocol", "fixed"),
        group_by=config.get("group_by", "record"), n_splits=int(config.get("n_splits", 1)),
        fold=int(config.get("fold", 0)), condition_id=_condition(config),
        implementation_id=provenance.get("code", {}).get("fingerprint", "not_recorded"),
        dataset_id=provenance.get("data", {}).get("fingerprint", "not_recorded"),
        smoke_only=bool(result.get("smoke_only", config.get("smoke_only", False))),
        threshold=_number(result.get("threshold")), best_epoch=training.get("best_epoch", result.get("best_epoch")),
        parameters=training.get("parameters", result.get("params", result.get("parameters", result.get("n_parameters")))),
        train_seconds=_number(training.get("total_recorded_epoch_seconds", training.get("current_session_training_seconds", result.get("train_seconds", result.get("training_seconds"))))),
        total_seconds=_number(result.get("total_seconds", result.get("elapsed_seconds"))),
        test_inference_seconds=_number(result.get("test_inference_seconds")),
        test_samples_per_second=_number(result.get("end_to_end_test_samples_per_second")))
    for split in ("val", "test", "record_macro"):
        metrics = _metrics(result.get(split, {}))
        for key in METRICS:
            row[f"{split}_{key}"] = _number(metrics.get(key))
    return row


def _plot(groups, output):
    if not groups:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = []
    versions = sorted({row["version"] for row in groups})
    for version in versions:
        candidates = defaultdict(list)
        for row in groups:
            if row["version"] == version and row.get("test_pr_auc_mean") is not None:
                key = tuple(row[name] for name in GROUP_KEYS if name != "model")
                candidates[key].append(row)
        if not candidates:
            continue
        def rank(item):
            rows = item[1]
            row = rows[0]
            baseline = row["protocol"] == "fixed" and row["augmentation"] == "all" and row["channels"] == "all" and row["train_fraction"] == 1.0
            return (baseline, len(rows), -row["fold"])
        _, chosen = max(candidates.items(), key=rank)
        chosen.sort(key=lambda row: row["test_pr_auc_mean"], reverse=True)
        figure, axes = plt.subplots(1, 2, figsize=(max(9, len(chosen) * .8), 4.9))
        x = np.arange(len(chosen))
        for axis, metric, title, color in zip(axes, ("pr_auc", "f1"), ("Test average precision", "Test F1"), ("#355b78", "#587365")):
            means = [row.get(f"test_{metric}_mean") for row in chosen]
            deviations = [row.get(f"test_{metric}_std") for row in chosen]
            axis.bar(x, [np.nan if value is None else value for value in means], color=color, width=.72)
            for position, (mean, deviation) in enumerate(zip(means, deviations)):
                if mean is not None and deviation is not None:
                    axis.errorbar(position, mean, yerr=deviation, color="#202830", capsize=3, fmt="none", linewidth=1)
            axis.set_xticks(x)
            axis.set_xticklabels([row["model"] for row in chosen], rotation=40, ha="right", fontsize=8)
            axis.set_ylim(0, 1.04)
            axis.set_title(title, fontsize=11)
            axis.spines[["top", "right"]].set_visible(False)
            axis.yaxis.grid(True, alpha=.16)
            axis.set_axisbelow(True)
        condition = {key: chosen[0][key] for key in GROUP_KEYS if key != "model"}
        title = f"{version}; augmentation={condition['augmentation']}; channels={condition['channels']}; {condition['protocol']}; fold={condition['fold']}"
        if condition["smoke_only"]:
            title = "SMOKE ONLY — " + title
        figure.suptitle(title, fontsize=10)
        figure.text(.02, .01, "Bars: mean across seeds within this exact condition/fold. Error bars: sample SD (ddof=1); omitted when n<2.", fontsize=8)
        figure.tight_layout(rect=(0, .04, 1, .94))
        filename = f"AP_F1_{version}.png"
        figure.savefig(output / filename, dpi=220)
        plt.close(figure)
        write_json(output / f"AP_F1_{version}_condition.json", dict(condition=condition,
            models=[row["model"] for row in chosen], seed_counts={row["model"]: row["n_seeds"] for row in chosen},
            selection="One baseline condition per version; complete tables contain all conditions. Folds are never treated as seeds."))
        plots.append(filename)
    return plots


def summarize(root, output_dir=None, include_smoke=False):
    root = Path(root).resolve()
    output = Path(output_dir).resolve() if output_dir else root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    rows, ignored, malformed = [], [], []
    for path in sorted(root.rglob("run_metrics.json")):
        try:
            result = read_json(path)
            if result.get("status") != "complete":
                ignored.append(dict(path=str(path), reason="not_complete", status=result.get("status")))
                continue
            row = _row(path, result)
            if row["smoke_only"] and not include_smoke:
                ignored.append(dict(path=str(path), reason="smoke_excluded"))
                continue
            row["_mtime"] = path.stat().st_mtime_ns
            rows.append(row)
        except (OSError, ValueError, TypeError, KeyError) as error:
            malformed.append(dict(path=str(path), error=str(error)))
    # A repeated directory copy or rerun is not another random seed. Retain the
    # newest complete artifact per exact condition+seed and record duplicates.
    unique = {}
    duplicates = []
    for row in sorted(rows, key=lambda item: item["_mtime"]):
        key = tuple(row[name] for name in GROUP_KEYS) + (row["seed"],)
        if key in unique:
            duplicates.append(dict(ignored_run=unique[key]["run_dir"], retained_run=row["run_dir"], reason="same_condition_and_seed_latest_complete_kept"))
        unique[key] = row
    selected = list(unique.values())
    for row in rows:
        row.pop("_mtime", None)
    grouped = defaultdict(list)
    for row in selected:
        grouped[tuple(row[name] for name in GROUP_KEYS)].append(row)
    summaries = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        result = dict(zip(GROUP_KEYS, key))
        result.update(n_seeds=len({row["seed"] for row in values}),
                      seeds=";".join(str(value) for value in sorted({row["seed"] for row in values})), runs=len(values))
        for split in ("val", "test", "record_macro"):
            for metric in METRICS:
                data = [_number(row[f"{split}_{metric}"]) for row in values]
                data = [value for value in data if value is not None]
                result[f"{split}_{metric}_mean"] = float(np.mean(data)) if data else None
                result[f"{split}_{metric}_std"] = float(np.std(data, ddof=1)) if len(data) >= 2 else None
                result[f"{split}_{metric}_n"] = len(data)
        summaries.append(result)
    _csv(output / "runs.csv", rows, ("run_dir", "model", "seed", "version", "smoke_only"))
    _csv(output / "mean_std.csv", summaries, GROUP_KEYS + ("n_seeds",))
    _csv(output / "duplicate_runs.csv", duplicates, ("ignored_run", "retained_run", "reason"))
    plots = _plot(summaries, output)
    report = dict(status="complete" if rows else "no_complete_runs", root=str(root), output_dir=str(output),
        observed_complete_runs=len(rows), aggregated_unique_runs=len(selected), summary_groups=len(summaries),
        include_smoke=bool(include_smoke), ignored=ignored, malformed=malformed, duplicates=len(duplicates), plots=plots,
        aggregation="Each row aggregates distinct seeds within exact model/version/augmentation/channels/fraction/protocol/group/fold/config condition. Sample SD uses ddof=1; n=1 SD remains missing.",
        warning="Synthetic/model smoke checks are excluded by default. Existing observations only; no results are fabricated. CV folds are not independent random seeds.")
    write_json(output / "summary.json", report)
    return report
