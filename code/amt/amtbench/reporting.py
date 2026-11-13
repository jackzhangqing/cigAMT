"""Bounded-memory aggregation and publication exports for benchmark runs.

Only compact summaries, strata and explicitly exported NPZ examples are read;
the potentially multi-million-row evaluation_rows.csv is never scanned here.
Seed uncertainty is the sample standard deviation of independent seed means,
not the spread between examples. Single-seed standard deviations stay undefined.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


CLEAN_TASKS = {"forecasting", "interpolation", "imputation"}
CLASS_NAMES = ("Clean", "Pulse", "Quasi-sinusoidal", "Square wave", "Step wave",
               "Triangular wave", "Power-frequency", "White noise", "Colored noise",
               "Harmonic", "Composite")
LABELS = {
    "f1": "Localization F1", "iou": "Localization IoU", "event_f1": "Event F1",
    "macro_f1_present_true": "Classification macro F1 (present classes)",
    "macro_f1_all_classes": "Classification macro F1 (all 11 classes)",
    "accuracy": "Classification accuracy", "nrmse": "NRMSE",
    "snr_improvement_db": "SNR improvement (dB)", "snr_output_db": "Output SNR (dB)",
    "masked_mse": "MSE on missing points (visible coordinates)",
    "masked_nrmse": "NRMSE on missing points", "low_nrmse": "Low-frequency NRMSE",
    "high_nrmse": "High-frequency NRMSE", "mae": "MAE (source coordinates)",
    "clean_false_positive_rate": "Clean-control false positive rate",
    "log_spectral_rmse": "Log-spectral RMSE"}
PRIMARY = {"noise_localization": "f1", "noise_classification": "macro_f1_present_true",
           "denoising": "nrmse", "frequency_denoising": "nrmse",
           "forecasting": "masked_mse", "interpolation": "masked_mse", "imputation": "masked_mse"}


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     default=str, allow_nan=False).encode()).hexdigest()


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        else:
            stream.write("status\nno_complete_evaluations\n")


def _read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _means(metrics):
    """Accept engine Moments and richer MetricAccumulator summary layouts."""
    if "mean" in metrics:
        result = {k: _number(v) for k, v in metrics["mean"].items()}
    else:
        result = {k: _number(v) for k, v in metrics.items()
                  if isinstance(v, (int, float)) and k not in ("n", "count", "clean_count")}
        result.update({k: _number(v.get("mean")) for k, v in metrics.get("metrics", {}).items()})
    aliases = {"macro_f1_present": "macro_f1_present_true", "macro_f1_all11": "macro_f1_all_classes"}
    for key in ("macro_f1_present", "macro_f1_all11", "macro_f1_present_true",
                "macro_f1_all_classes", "pooled_accuracy", "accuracy", "balanced_accuracy"):
        if key in metrics:
            result[aliases.get(key, key)] = _number(metrics[key])
    identifiers = {"true_class", "pred_class", "is_clean", "sample_index", "source_id",
                   "source_row", "class_id", "domain_id", "noise_index", "noise_bank_row",
                   "noise_start", "noise_length", "clean_index", "draw_seed", "seed"}
    return {aliases.get(k, k): v for k, v in result.items() if k not in identifiers}


def _selection(summary):
    fingerprint = summary.get("data_fingerprints") or summary.get("evaluation_identity", {}).get("dataset", {})
    if fingerprint.get("selection_sha256"):
        return fingerprint["selection_sha256"], "exact selected-row fingerprint"
    # Older summaries lack selected-row hashes. Never pool their seeds: one
    # evaluation signature defines a conservative singleton comparison group.
    return "legacy-singleton-" + summary.get("evaluation_signature", _digest(summary.get("data", {}))), \
        "legacy summary lacks exact selected-row fingerprint; seeds not pooled"


def _run_protocol(run, summary):
    evaluation = summary.get('evaluation_identity', {})
    numerical = {k:evaluation.get(k) for k in ('inference_device_type','inference_precision','batch_size')}
    path = run/"run_config.json"
    if path.exists():
        info = _read_json(path)
        identity = info.get("identity", {})
        config = dict(identity.get("config", info.get("config", {})))
        for key in ("num_workers", "prefetch_factor", "cpu_threads", "log_interval"):
            config.pop(key, None)
        return _digest({"config": config, "training": identity.get("training"), "evaluation_numerics": numerical,
                        "validation": identity.get("validation"), "code": identity.get("code")})
    # Classical settings must also match; unrecorded settings remain isolated.
    identity = summary.get("evaluation_identity", {})
    if summary.get("classical") and identity:
        return _digest({"classical": identity.get("classical"), "code": identity.get("code"), "evaluation_numerics": numerical,
                        "settings": identity.get("classical_parameters", {}), "threshold": identity.get("threshold")})
    return "singleton-" + summary.get("evaluation_signature", str(run))


def _grouper(row):
    keys = ("dataset_id", "task", "model", "train_scheme", "eval_scheme", "split",
            "selection_sha256", "training_protocol_sha256")
    return tuple(row.get(k, "") for k in keys)


def _group_fields(row):
    return {k: row[k] for k in ("dataset_id", "task", "model", "train_scheme", "eval_scheme", "split",
                                "selection_sha256", "training_protocol_sha256")}


def _seed_statistics(rows, value_key):
    values = [_number(r.get(value_key)) for r in rows]
    finite = [x for x in values if x is not None]
    return dict(mean=float(np.mean(finite)) if finite else None,
                std=float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
                metric_seed_count=len(finite), seed_count=len(rows),
                seeds=json.dumps(sorted(int(r["seed"]) for r in rows)),
                uncertainty="sample standard deviation of seed means; undefined for one seed")


def aggregate_runs(root, output):
    """Write compact per-run, seed and condition CSVs without row-file scans.

    Runs with different exact evaluation selections, optimization settings,
    training selections, model code, or A/C protocols are never seed-averaged.
    Duplicate runs of the same seed are excluded and listed in audit JSON.
    """
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run_rows, condition_runs, ignored, duplicates = [], [], [], []
    seen = {}
    for path in sorted(root.rglob("summary.json")):
        if path.parent.parent.name != "evaluation":
            continue
        summary = _read_json(path)
        if summary.get("state") != "evaluation_complete":
            ignored.append(str(path)); continue
        run = path.parent.parent.parent
        selection, scope = _selection(summary)
        row = {k: summary.get(k, "") for k in ("dataset_id", "task", "model", "train_scheme", "eval_scheme", "split", "seed")}
        row.update(seed=int(summary.get("seed", 0)), rows=int(summary.get("rows", 0)),
                   run=str(run.resolve()), summary_path=str(path.resolve()),
                   evaluation_signature=summary.get("evaluation_signature", ""),
                   selection_sha256=selection, selection_scope=scope,
                   training_protocol_sha256=_run_protocol(run, summary),
                   threshold=summary.get("threshold"), seconds=summary.get("seconds"),
                   valid_counts_json=json.dumps(summary.get("metrics", {}).get("valid_counts", {}), sort_keys=True))
        row.update(_means(summary.get("metrics", {})))
        complete = run/"training_complete.json"
        if complete.exists():
            training = _read_json(complete)
            for key in ("parameters", "best_epoch", "best_validation_loss", "epochs_completed", "max_gpu_memory_bytes"):
                row[key] = training.get(key)
        duplicate_key = (_grouper(row), row["seed"])
        if duplicate_key in seen:
            row["included_in_seed_summary"] = False
            duplicates.append({"excluded": str(path), "kept": seen[duplicate_key], "seed": row["seed"]})
        else:
            row["included_in_seed_summary"] = True
            seen[duplicate_key] = str(path)
        run_rows.append(row)
        if not row["included_in_seed_summary"]:
            continue
        for condition in _read_csv(path.parent/"strata.csv"):
            condition_runs.append({**_group_fields(row), "seed": row["seed"],
                                   "stratum": condition["stratum"], "value": condition["value"],
                                   "rows": int(float(condition.get("n", condition.get("count", 0)))),
                                   "metrics": {({"macro_f1_present": "macro_f1_present_true", "macro_f1_all11": "macro_f1_all_classes"}.get(k, k)): _number(v)
                                               for k, v in condition.items() if k not in ("stratum", "value", "n", "count")}})
    grouped = defaultdict(list)
    for row in run_rows:
        if row["included_in_seed_summary"]:
            grouped[_grouper(row)].append(row)
    seed_rows = []
    for rows in grouped.values():
        metric_keys = set().union(*(set(_means(_read_json(r["summary_path"]).get("metrics", {}))) for r in rows))
        for metric in sorted(metric_keys):
            seed_rows.append({**_group_fields(rows[0]), "metric": metric,
                              **_seed_statistics(rows, metric),
                              "rows_per_seed": json.dumps([r["rows"] for r in rows]),
                              "summary_paths_json": json.dumps([r["summary_path"] for r in rows]),
                              "selection_scope": rows[0]["selection_scope"]})
    condition_groups = defaultdict(list)
    for row in condition_runs:
        condition_groups[(_grouper(row), row["stratum"], row["value"])].append(row)
    condition_rows = []
    for rows in condition_groups.values():
        keys = set().union(*(r["metrics"] for r in rows))
        for metric in sorted(keys):
            values = [{**r, "value_for_stats": r["metrics"].get(metric)} for r in rows]
            condition_rows.append({**_group_fields(rows[0]), "stratum": rows[0]["stratum"],
                                   "value": rows[0]["value"], "metric": metric,
                                   **_seed_statistics(values, "value_for_stats"),
                                   "rows_per_seed": json.dumps([r["rows"] for r in rows])})
    _write_csv(output/"per_run_summary.csv", run_rows)
    _write_csv(output/"seed_summary.csv", seed_rows)
    _write_csv(output/"condition_summary.csv", condition_rows)
    audit = {"complete_evaluations": len(run_rows), "seed_summary_groups": len(grouped),
             "condition_groups": len(condition_groups), "duplicate_seed_runs_excluded": duplicates,
             "ignored_incomplete": ignored, "row_files_scanned": 0,
             "note": "No cross-seed pooling without exact selected-row fingerprints; no test-based selection."}
    (output/"aggregation_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold",
                         "savefig.bbox": "tight", "svg.fonttype": "none", "pdf.fonttype": 42})
    return plt


def _save(fig, stem):
    stem = Path(stem); stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(str(stem)+"."+extension, dpi=300, facecolor="white")


def _slug(value):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))[:160]


def _spans(ax, time, mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    dt = time[1]-time[0] if len(time) > 1 else 1
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        ax.axvspan(time[a], time[min(b, len(time)-1)]+(dt if b == len(time) else 0),
                   color="#edb66f", alpha=0.24, linewidth=0)


def _training_plot(run, output):
    rows = _read_csv(run/"history.csv")
    if not rows:
        return []
    plt = _plt(); fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), constrained_layout=True)
    epoch = [_number(r.get("epoch")) for r in rows]
    for key, label, color in (("train_loss", "Training loss", "#3767a3"),
                               ("validation_loss", "Validation loss", "#db7b30")):
        axes[0].plot(epoch, [_number(r.get(key)) for r in rows], label=label, color=color, lw=1.8)
    axes[0].set(xlabel="Epoch", ylabel="Task loss", title="Optimization history")
    axes[0].legend(frameon=False)
    axes[1].plot(epoch, [_number(r.get("rows_per_second")) for r in rows], color="#33866e", lw=1.8)
    axes[1].set(xlabel="Epoch", ylabel="Training samples / second", title="Observed training throughput")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(run.name+" | "+str(run.parent.name))
    _save(fig, output/"training_curves"); plt.close(fig)
    return [output/"training_curves.png"]


def _confusion(summary, output):
    matrix = summary.get("metrics", {}).get("confusion_matrix")
    if matrix is None:
        return []
    cm = np.asarray(matrix, dtype=float)
    denom = cm.sum(1, keepdims=True)
    normalized = np.divide(cm, denom, out=np.zeros_like(cm), where=denom > 0)
    plt = _plt(); fig, ax = plt.subplots(figsize=(8.8, 7.8), constrained_layout=True)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    labels = CLASS_NAMES if len(cm) == 11 else [str(i) for i in range(len(cm))]
    ax.set(xticks=np.arange(len(cm)), yticks=np.arange(len(cm)), xticklabels=labels,
           yticklabels=labels, xlabel="Predicted class", ylabel="True class",
           title="Row-normalized confusion matrix (counts annotated)")
    ax.tick_params(axis="x", rotation=55)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i, j]:
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                        color="white" if normalized[i, j] > .5 else "#203144", fontsize=7)
    fig.colorbar(image, ax=ax, label="Fraction of true-class samples", shrink=.8)
    _save(fig, output/"classification_confusion"); plt.close(fig)
    return [output/"classification_confusion.png"]


def _example(path, summary, output):
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    task = summary["task"]
    metadata = json.loads(str(data.get("metadata_json", np.asarray("{}"))))
    inp = np.asarray(data["input"], dtype=float)
    target = np.asarray(data["target"], dtype=float)
    fs = float(summary.get("data", {}).get("sampling_rate_hz", 150))
    time = np.arange(inp.shape[-1])/fs
    clean = task in CLEAN_TASKS
    scale = np.asarray(data.get("visible_scale" if clean else "scale", np.ones((4,1))), dtype=float)
    center = np.asarray(data.get("visible_center", np.zeros((4,1))), dtype=float) if clean else np.zeros((4,1))
    source_i, source_t = inp*scale+center, target*scale+center
    mask = np.asarray(data.get("loss_mask" if clean else "noise_mask", np.zeros_like(inp))) > .5
    title = "%s | %s | train %s / evaluate %s | case %s" % (
        task, summary["model"], summary.get("train_scheme", ""), summary.get("eval_scheme", ""),
        metadata.get("evaluation_index", path.stem))
    if metadata.get("snr_target_db") is not None and _number(metadata["snr_target_db"]) is not None:
        title += " | input SNR %g dB" % metadata["snr_target_db"]
    plt = _plt(); outputs = []
    if task in ("noise_localization", "noise_classification"):
        fig, axes = plt.subplots(5, 1, figsize=(12, 10), constrained_layout=True)
        for ch, name in enumerate(("Ex", "Ey", "Hx", "Hy")):
            axes[ch].plot(time, source_i[ch], color="#7c8da5", lw=.8, label="Noisy input")
            axes[ch].plot(time, source_t[ch], color="#202934", lw=.85, label="Clean reference")
            _spans(axes[ch], time, mask[ch])
            axes[ch].set(ylabel=name+"\nSource coordinates")
            axes[ch].grid(alpha=.15)
        axes[0].legend(loc="upper right", ncol=2, frameon=False)
        logits = np.asarray(data.get("output_logits"))
        if task == "noise_localization":
            probability = np.exp(-np.logaddexp(0, -logits)).reshape(-1, inp.shape[-1]).mean(0)
            truth = mask[0].astype(int)
            axes[4].plot(time, probability, color="#3269a8", lw=1.2, label="Predicted probability")
            axes[4].step(time, truth, where="post", color="#a7504a", lw=1.1, label="True event mask")
            axes[4].axhline(summary.get("threshold", .5), color="#555555", ls="--", lw=.9, label="Validation-selected threshold")
            axes[4].set(xlabel="Time (s)", ylabel="Event probability", ylim=(-.05, 1.05))
            axes[4].legend(ncol=3, frameon=False, fontsize=8)
        else:
            logits = logits.reshape(-1); probabilities = np.exp(logits-logits.max()); probabilities /= probabilities.sum()
            colors = ["#3d7cab"]*len(logits)
            truth = int(metadata.get("true_class", metadata.get("class_id", -1)))
            if 0 <= truth < len(colors):
                colors[truth] = "#d48a32"
            axes[4].bar(np.arange(len(logits)), probabilities, color=colors)
            axes[4].set(xticks=np.arange(len(logits)), xticklabels=CLASS_NAMES[:len(logits)],
                        ylabel="Class probability", title="Orange: true class; blue: other classes", ylim=(0, 1))
            axes[4].tick_params(axis="x", rotation=30)
        for ax in axes[:4]:
            ax.set_xlabel("Time (s)")
        fig.suptitle(title, fontsize=11)
        _save(fig, output/path.stem); plt.close(fig); outputs.append(output/(path.stem+".png"))
        return outputs
    if "output_prediction" not in data:
        return []
    predicted = np.asarray(data["output_prediction"], dtype=float)*scale+center
    observed = ~mask if clean else None
    completion = np.where(observed, source_i, predicted) if clean else predicted
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True, constrained_layout=True)
    for ch, name in enumerate(("Ex", "Ey", "Hx", "Hy")):
        visible_input = np.where(observed[ch], source_i[ch], np.nan) if clean else source_i[ch]
        axes[ch, 0].plot(time, visible_input, color="#9aa8b7", lw=.75,
                         label="Observed input" if clean else "Noisy input")
        axes[ch, 0].plot(time, source_t[ch], color="#202934", lw=1, label="Clean reference")
        axes[ch, 0].plot(time, completion[ch], color="#3269a8", lw=1, alpha=.85,
                         label="Completed signal" if clean else "Denoised prediction")
        residual = completion[ch]-source_t[ch]
        if clean:
            residual = np.where(mask[ch], residual, np.nan)
        axes[ch, 1].plot(time, residual, color="#b66b2e", lw=.9)
        axes[ch, 1].axhline(0, color="#555555", lw=.7)
        for ax in axes[ch]:
            _spans(ax, time, mask[ch]); ax.grid(alpha=.15)
        axes[ch, 0].set_ylabel(name+"\nSource coordinates")
        axes[ch, 1].set_ylabel(name+"\nPrediction - reference")
    axes[0, 0].set_title("Signal reconstruction"); axes[0, 1].set_title("Missing-point residual" if clean else "Prediction residual")
    axes[0, 0].legend(ncol=3, frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(title+"\nShading: missing interval" if clean else title+"\nShading: true injected-noise interval", fontsize=11)
    _save(fig, output/path.stem); plt.close(fig); outputs.append(output/(path.stem+".png"))
    if task == "frequency_denoising" and all("output_"+x in data and "target_"+x in data for x in ("low", "high")):
        fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True, constrained_layout=True)
        for ch, name in enumerate(("Ex", "Ey", "Hx", "Hy")):
            for col, part in enumerate(("low", "high")):
                axes[ch, col].plot(time, data["target_"+part][ch]*scale[ch], color="#202934", lw=1, label="Reference")
                axes[ch, col].plot(time, data["output_"+part][ch]*scale[ch], color="#3269a8", lw=1, label="Prediction")
                axes[ch, col].set_ylabel(name+"\nSource coordinates"); axes[ch, col].grid(alpha=.15)
        axes[0,0].set_title("Low-frequency component (<= 5 Hz)")
        axes[0,1].set_title("High-frequency component (> 5 Hz)")
        axes[0,0].legend(frameon=False)
        for ax in axes[-1]:
            ax.set_xlabel("Time (s)")
        fig.suptitle(title, fontsize=11)
        _save(fig, output/(path.stem+"_frequency_heads")); plt.close(fig)
        outputs.append(output/(path.stem+"_frequency_heads.png"))
    return outputs


def _html_index(output, title, images, links=()):
    content = ["<!doctype html><meta charset='utf-8'><title>"+html.escape(title)+"</title>",
               "<style>body{font:16px Arial;margin:28px;max-width:1450px}img{max-width:100%;border:1px solid #ddd}figure{margin:24px 0}code{background:#eee;padding:3px}</style>",
               "<h1>"+html.escape(title)+"</h1>",
               "<p>All plots use recorded evaluations. Error bands denote sample standard deviations across seeds when available. Source coordinates are not certified physical SI units.</p>"]
    for path in links:
        path = Path(path)
        content.append("<p><a href='"+html.escape(path.name, quote=True)+"'>"+html.escape(path.name)+"</a></p>")
    for path in images:
        rel = Path(path).relative_to(output).as_posix()
        content.append("<figure><a href='"+html.escape(rel, quote=True)+"'><img loading='lazy' src='"+html.escape(rel, quote=True)+"'></a><figcaption>"+html.escape(rel)+"</figcaption></figure>")
    (output/"index.html").write_text("\n".join(content), encoding="utf-8")


def plot_run(run, output=None):
    """Export training, confusion and saved-example figures in PNG/PDF/SVG."""
    run = Path(run); output = Path(output) if output is not None else run/"figures"
    output.mkdir(parents=True, exist_ok=True)
    images = _training_plot(run, output)
    for path in sorted((run/"evaluation").glob("*/summary.json")):
        summary = _read_json(path)
        if summary.get("state") != "evaluation_complete":
            continue
        folder = output/path.parent.name; folder.mkdir(parents=True, exist_ok=True)
        images.extend(_confusion(summary, folder))
        for example in sorted((path.parent/"examples").glob("case_*.npz")):
            images.extend(_example(example, summary, folder))
    _html_index(output, "AMT benchmark run review", images)
    return {"output": str(output), "figures": len(images), "index": str(output/"index.html")}


def _condition_curves(rows, output):
    plt = _plt(); images = []; panels = defaultdict(list)
    for row in rows:
        if "task" not in row:
            continue
        task = row["task"]
        stratum = ("snr_target_db" if task not in CLEAN_TASKS else
                   {"forecasting": "horizon", "interpolation": "gap_length", "imputation": "missing_ratio_nominal"}[task])
        metrics = {PRIMARY.get(task)}
        if task in ("denoising", "frequency_denoising"):
            metrics |= {"snr_improvement_db"}
        if task == "frequency_denoising":
            metrics |= {"low_nrmse", "high_nrmse"}
        if task in CLEAN_TASKS:
            metrics |= {"masked_nrmse"}
        if row["stratum"] != stratum or row["metric"] not in metrics:
            continue
        x, y = _number(row["value"]), _number(row.get("mean"))
        if x is None or y is None or (task in CLEAN_TASKS and x <= 0):
            continue
        # Evaluation selections are an explicit figure boundary, not just a
        # seed pooling boundary: incomparable test subsets never share a panel.
        key = (row["dataset_id"], task, row["eval_scheme"], row["split"], row["selection_sha256"], row["metric"])
        panels[key].append((row, x, y))
    for key, panel in sorted(panels.items()):
        dataset, task, scheme, split, selection, metric = key
        groups = defaultdict(list)
        for row, x, y in panel:
            groups[(row["model"], row["train_scheme"], row["training_protocol_sha256"])].append((x, y, _number(row.get("std"))))
        fig, ax = plt.subplots(figsize=(8.8, 5), constrained_layout=True)
        for (model, training, protocol), points in sorted(groups.items()):
            points.sort(); x = np.asarray([p[0] for p in points]); y = np.asarray([p[1] for p in points])
            label = model+" / train "+training+" ["+protocol[:6]+"]"
            line, = ax.plot(x, y, "o-", ms=4, lw=1.7, label=label)
            std = np.asarray([p[2] if p[2] is not None else np.nan for p in points])
            valid = np.isfinite(std)
            if valid.any():
                ax.fill_between(x, y-std, y+std, where=valid, color=line.get_color(), alpha=.13)
        xlabel = "Target input SNR (dB)" if task not in CLEAN_TASKS else {
            "forecasting": "Forecast horizon (samples)", "interpolation": "Internal gap length (samples)",
            "imputation": "Nominal missing fraction"}[task]
        ax.set(xlabel=xlabel, ylabel=LABELS.get(metric, metric),
               title="%s | %s | evaluation %s" % (task, split, scheme))
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8, loc="best")
        stem = output/("condition_"+_slug(task+"_"+split+"_"+scheme+"_"+metric+"_"+selection[:8]))
        _save(fig, stem); plt.close(fig); images.append(Path(str(stem)+".png"))
    return images


def _overview_training(root, output):
    plt = _plt(); groups = defaultdict(list); images = []
    for path in sorted(Path(root).rglob("training_complete.json")):
        complete = _read_json(path); rows = _read_csv(path.parent/"history.csv")
        if not rows:
            continue
        key = (complete["dataset_id"], complete["task"], complete["model"], complete["train_scheme"],
               _run_protocol(path.parent, complete))
        groups[key].append((int(complete["seed"]), rows))
    for key, histories in sorted(groups.items()):
        dataset, task, model, scheme, protocol = key
        # Reject duplicate seeds in the visualization just as in aggregation.
        unique = {}
        for seed, rows in histories:
            unique.setdefault(seed, rows)
        epochs = defaultdict(lambda: defaultdict(list))
        for rows in unique.values():
            for row in rows:
                epoch = int(float(row["epoch"]))
                for metric in ("train_loss", "validation_loss"):
                    value = _number(row.get(metric))
                    if value is not None:
                        epochs[epoch][metric].append(value)
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for metric, label in (("train_loss", "Training"), ("validation_loss", "Validation")):
            xs, means, stds = [], [], []
            for epoch in sorted(epochs):
                values = epochs[epoch][metric]
                if values:
                    xs.append(epoch); means.append(np.mean(values)); stds.append(np.std(values, ddof=1) if len(values)>1 else np.nan)
            line, = ax.plot(xs, means, lw=1.8, label=label)
            means, stds = np.asarray(means), np.asarray(stds)
            if np.isfinite(stds).any():
                ax.fill_between(xs, means-stds, means+stds, where=np.isfinite(stds), color=line.get_color(), alpha=.15)
        ax.set(xlabel="Epoch", ylabel="Task loss", title="%s | %s | train %s | %d seeds" % (task, model, scheme, len(unique)))
        ax.grid(alpha=.2); ax.legend(frameon=False)
        ax.text(.99, .02, "Per-epoch available seeds; no extrapolation after early stopping", transform=ax.transAxes,
                ha="right", fontsize=7, color="#666666")
        stem = output/("training_"+_slug(task+"_"+model+"_"+scheme+"_"+protocol[:8]))
        _save(fig, stem); plt.close(fig); images.append(Path(str(stem)+".png"))
    return images


def plot_overview(root, output):
    """Aggregate compact summaries and export protocol-separated English plots."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    audit = aggregate_runs(root, output)
    images = _condition_curves(_read_csv(output/"condition_summary.csv"), output)
    images.extend(_overview_training(root, output))
    _html_index(output, "AMT benchmark comparison", images,
                [output/x for x in ("per_run_summary.csv", "seed_summary.csv", "condition_summary.csv", "aggregation_audit.json")])
    return {**audit, "output": str(output), "figures": len(images), "index": str(output/"index.html")}
