"""Training and evaluation with validation-only model/threshold selection."""
from __future__ import annotations

import csv
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
from time import perf_counter

import joblib
import numpy as np

from .classical import CLASSICAL_NAMES, build_estimator, extract_features, prediction_scores
from .corruptions import DEFAULT_CORRUPTIONS, corrupt
from .data import CHANNELS, PARTITION_DIRS, _channels, load_bundle
from .metrics import (METRIC_NAMES, choose_threshold, clustered_bootstrap,
                      compute_metrics, grouped_metrics)
from .runtime import (BASE, digest, environment, read_json, scientific_config,
                      seed_everything, seed_worker, sha256, write_json)


def _torch():
    try:
        import torch
    except (ImportError, OSError) as error:
        raise RuntimeError("Use a working PyTorch environment; see README_运行指南.txt. " + str(error)) from error
    return torch


def _make_grad_scaler(torch_module, enabled):
    """Use the current AMP API, falling back to the older CUDA namespace."""
    modern = getattr(getattr(torch_module, "amp", None), "GradScaler", None)
    if callable(modern):
        return modern("cuda", enabled=enabled)
    cuda_amp = getattr(getattr(torch_module, "cuda", None), "amp", None)
    legacy = getattr(cuda_amp, "GradScaler", None)
    if callable(legacy):
        return legacy(enabled=enabled)
    raise RuntimeError(
        "This PyTorch build provides neither torch.amp.GradScaler nor "
        "torch.cuda.amp.GradScaler. Use a working PyTorch environment; "
        "see README_运行指南.txt."
    )


def _optimizer_step(torch_module, model, optimizer, scaler, max_norm=5.0):
    """Clip finite unscaled gradients; let AMP recover from an overflow."""
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("Backward produced no parameter gradients")
    finite = bool(torch_module.stack(
        [torch_module.isfinite(gradient).all() for gradient in gradients]
    ).all().item())
    if not finite:
        if not scaler.is_enabled():
            raise RuntimeError("Nonfinite gradients in full-precision training; "
                               "inspect the model, inputs, and learning rate")
        # unscale_ has already recorded the overflow. GradScaler skips the
        # optimizer update and lowers its scale; do not clip Inf/NaN gradients.
        scaler.step(optimizer)
        scaler.update()
        return False
    torch_module.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_norm, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return True


def _device(value):
    torch = _torch()
    value = ("cuda:0" if torch.cuda.is_available() else "cpu") if value == "auto" else value
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
            raise ValueError(f"Requested unavailable GPU: {value}")
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Supported devices: cpu, auto, cuda:N")
    return device


def _snapshot(config, full=False):
    root = Path(config["data_root"])
    if (root / "train/windows.h5").exists() and config['version'] == 'standardized':
        # Publication format: hash precisely the files consumed by the HDF5 reader.
        files = {}
        for relative in PARTITION_DIRS.values():
            for name in ("windows.h5", "windows.csv", "records.csv"):
                path = root / relative / name
                files[path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size, sha256=sha256(path))
        if float(config["fs_hz"]) != 24000:
            raise ValueError("Published sferic data have sampling rate 24 kHz")
        return dict(fingerprint=digest(files), selected_input="windows.h5:x", files=files,
                    hash_scope="Published labeled HDF5 tensors and split-local manifests")
    summary = read_json(root/"metadata/summary.json")
    if float(config["fs_hz"]) != float(summary["fs_hz"]):
        raise ValueError("Configured sampling rate differs from the dataset's declared rate")
    selected = "X.npy" if config["version"] == "standardized" else "X_raw.npy"
    files = {}
    for relative in PARTITION_DIRS.values():
        for name in (selected, "y.npy", "windows.csv"):
            path = root / relative / name
            if not path.is_file():
                raise FileNotFoundError(f"Required benchmark input is missing: {path}")
            files[path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    for name in ("metadata/records.csv", "metadata/summary.json"):
        path = root / name
        files[name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    if config.get("augmentation", "").startswith("matched_"):
        path = root/"train/windows.h5"
        files["train/windows.h5"] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    if full:
        for name in ("X.npy", "X_raw.npy", "y.npy"):
            path = root / "test/unlabeled" / name
            if path.exists():
                files[path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    return dict(fingerprint=digest(files), selected_input=selected, files=files,
                hash_scope="All selected labeled arrays, labels and source manifests" +
                    ("; existing unlabeled arrays also hashed" if full else ""))


def _code_snapshot():
    paths = [BASE/"benchmark.py", *sorted((BASE/"bench").glob("*.py"))]
    files = {path.relative_to(BASE).as_posix(): sha256(path) for path in paths}
    return dict(fingerprint=digest(files), files=files)


def _archive_code(run_dir):
    for path in [BASE/"benchmark.py", *sorted((BASE/"bench").glob("*.py"))]:
        target = run_dir/"source"/path.relative_to(BASE)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)
        elif sha256(target) != sha256(path):
            raise ValueError("Existing source archive differs; refuse to overwrite provenance")


def audit_dataset(config, full=False):
    bundle = load_bundle(config["data_root"], config["version"], config.get("channels", "all"))
    try:
        split = bundle.fixed_split()
        reports = {}
        groups = []
        for name, ds in (("train", split.train), ("val", split.val), ("test", split.test)):
            for x, _, _ in ds.iter_batches(1024):
                if not np.isfinite(x).all():
                    raise ValueError(f"Nonfinite input in {name}")
            groups.append(set(ds.record_ids))
            reports[name] = dict(windows=len(ds), positive=int(ds.y.sum()),
                negative=int(len(ds) - ds.y.sum()), records=len(set(ds.record_ids)),
                record_ids=sorted(set(ds.record_ids)), regions=sorted(set(ds.regions)),
                sites=sorted(set(ds.sites)))
        if any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
            raise ValueError("A source record appears in multiple supervised partitions")
        all_clean = [row for row in bundle.rows if row["transform"] == 0]
        by_region = {}
        for row in all_clean:
            by_region.setdefault(row["region"] or "unresolved", set()).add(row["record_id"])
        report = dict(status="passed", dataset_root=str(Path(config["data_root"]).resolve()),
            selected_version=config["version"], partitions=reports,
            unique_labeled_records=len(set(row["record_id"] for row in all_clean)),
            clean_windows=len(all_clean), frozen_augmented_windows=len(bundle.rows)-len(all_clean),
            records_by_region={k: len(v) for k,v in by_region.items()},
            source_grouping_provenance=sorted(set(row["group_provenance"] for row in all_clean)),
            record_disjoint=True, fingerprint=_snapshot(config, full=full), environment=environment(),
            warnings=["Fixed partitions are record-disjoint, not site/region-disjoint.",
                      "Augmented/overlapping windows are correlated; independent record count is the relevant acquisition scale.",
                      "Only four records in the fixed labeled test: record bootstrap intervals are fragile.",
                      "Unlabeled field y=-1 is unknown and cannot be scored as negative ground truth."])
        write_json(Path(config["output_root"])/"data_audit.json", report)
        return report
    finally:
        bundle.close()


def _limit(ds, limit, seed):
    if limit is None or int(limit) >= len(ds):
        return ds
    limit = int(limit)
    if limit < 2:
        raise ValueError("A debug sample limit must be at least two")
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(ds.y == 1), np.flatnonzero(ds.y == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Both classes are required for this benchmark")
    n_positive = min(len(positive), max(1, min(limit-1, round(limit * len(positive) / len(ds)))))
    n_negative = min(len(negative), limit-n_positive)
    n_positive = min(len(positive), limit-n_negative)
    ids = np.sort(np.r_[rng.permutation(positive)[:n_positive], rng.permutation(negative)[:n_negative]])
    return (ds.subset_local(ids) if hasattr(ds, "subset_local") else ds.bundle.subset(ds.row_indices[ids]))


def _split(config):
    bundle = load_bundle(config["data_root"], config["version"], config["channels"])
    split_seed = int(config.get("split_seed", 20261014))
    if config["protocol"] == "fixed":
        matched = config["augmentation"].startswith("matched_")
        split = bundle.fixed_split("all" if matched else config["augmentation"], float(config["train_fraction"]), split_seed)
        if matched:
            from .matched import MatchedWindowDataset
            split.train = MatchedWindowDataset(split.train, config["augmentation"])
            split.protocol.update(augmentation=config["augmentation"],
                matched_control="Same frozen rows/labels/noise draws; clean control duplicates parents to match samples and SGD steps",
                warnings=["Matched clean duplicates do not increase independent acquisition count",
                          "Noise is extracted using each augmented row's own clean-crop statistics; noise-only is restored with its parent's statistics"])
    else:
        if config["augmentation"] != "clean":
            raise ValueError("group_cv is clean-only; set augmentation=clean, not frozen augmentation from old splits")
        folds = bundle.group_folds(group_by=config["group_by"], n_splits=int(config["n_splits"]),
            seed=split_seed, val_fraction=float(config["val_fraction"]), val_group_by="record")
        if not 0 <= int(config["fold"]) < len(folds):
            raise ValueError("fold is outside the available cross-validation range")
        split = folds[int(config["fold"])]
    split.train = _limit(split.train, config.get("max_train_samples"), split_seed+10)
    split.val = _limit(split.val, config.get("max_eval_samples"), split_seed+20)
    split.test = _limit(split.test, config.get("max_eval_samples"), split_seed+30)
    for name, ds in (("train", split.train), ("val", split.val), ("test", split.test)):
        if not len(ds) or len(np.unique(ds.y)) != 2:
            raise ValueError(f"{name} must contain both classes")
    return bundle, split


def _split_manifest(split):
    out = {"protocol": split.protocol}
    for name, ds in (("train", split.train), ("val", split.val), ("test", split.test)):
        out[name] = dict(rows=len(ds), positives=int(ds.y.sum()), records=sorted(set(ds.record_ids)),
            global_row_indices=ds.row_indices.tolist(),
            source_windows=[dict(row_index=int(row["row_index"]), partition=row["partition"],
                local_index=int(row["local_index"]), record_id=row["record_id"],
                **{key: row[key] for key in ("matched_mode", "effective_start", "matched_parent_index", "matched_parent_row_index", "noise_source_index") if key in row}) for row in ds.metadata])
    return out


def _predict(model, ds, config, device, spec=None):
    began = perf_counter()
    if config["model"] in CLASSICAL_NAMES:
        features = []
        for x, _, rows in ds.iter_batches(int(config["batch_size"])):
            if spec:
                x = corrupt(x, rows, spec, int(config["seed"]), config["fs_hz"])
            features.append(extract_features(x, fs=config["fs_hz"]))
        result = prediction_scores(model, np.concatenate(features))
    else:
        torch = _torch()
        model.to(device).eval()
        chunks = []
        with torch.no_grad():
            for x, _, rows in ds.iter_batches(int(config["batch_size"])):
                if spec:
                    x = corrupt(x, rows, spec, int(config["seed"]), config["fs_hz"])
                tensor = torch.from_numpy(x).to(device)
                with torch.autocast(device_type=device.type, enabled=bool(config["amp"] and device.type == "cuda")):
                    logits = model(tensor)
                if tuple(logits.shape) != (len(x), 1) or not torch.isfinite(logits).all():
                    raise ValueError("Model must return finite (batch,1) logits")
                chunks.append(torch.sigmoid(logits.float()).squeeze(1).cpu().numpy())
        score = np.concatenate(chunks)
        result = dict(scores=score, probabilities=score, score_kind="sigmoid_probability")
    result["elapsed_seconds"] = perf_counter()-began
    result["end_to_end_samples_per_second"] = len(ds)/max(result["elapsed_seconds"], 1e-12)
    return result


def _save_predictions(path, ds, prediction, threshold):
    temporary = Path(str(path) + ".writing.npz")
    np.savez_compressed(temporary, y=ds.y, scores=prediction["scores"],
        probabilities=(np.empty(0) if prediction["probabilities"] is None else prediction["probabilities"]),
        predictions=(np.asarray(prediction["scores"],dtype=np.float64) >= threshold).astype(np.int64),
        record_ids=np.asarray(ds.record_ids, dtype=str), regions=np.asarray(ds.regions, dtype=str),
        sites=np.asarray(ds.sites, dtype=str), indices=ds.row_indices)
    temporary.replace(path)


def _metrics(ds, prediction, threshold):
    return compute_metrics(ds.y, prediction["scores"], threshold,
                           probabilities=prediction["probabilities"])


def _record_macro(groups):
    result = {}
    for metric in METRIC_NAMES:
        valid = [row[metric] for row in groups.values() if row.get(metric) is not None]
        result[metric] = float(np.mean(valid)) if valid else None
    return result


def _torch_save(path, checkpoint):
    torch = _torch()
    temporary = Path(str(path) + ".writing")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _train_deep(config, split, run_dir, run_id, resume):
    torch = _torch()
    from .models import create_model, model_description
    device = _device(config["device"])
    torch.set_num_threads(int(config["cpu_threads"]))
    seed_everything(int(config["seed"]), config["deterministic"])
    model = create_model(config["model"], channels=4, length=split.train.shape[-1], config=config.get("model_config")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]),
        eta_min=float(config["learning_rate"])*0.01)
    use_amp = bool(config["amp"] and device.type == "cuda")
    scaler = _make_grad_scaler(torch, enabled=use_amp)
    print(f"Training {config['model']} seed={config['seed']} device={device} "
          f"torch={torch.__version__} amp={use_amp} "
          f"scaler={type(scaler).__module__}.{type(scaler).__name__}", flush=True)
    pos_weight = (len(split.train.y)-split.train.y.sum())/split.train.y.sum() if config["class_weight"] == "auto" else 1.0
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], device=device))
    generator = torch.Generator().manual_seed(int(config["seed"]))
    start, best, best_epoch, stale, history = 0, -float("inf"), -1, 0, []
    last = run_dir/"last.pt"
    if resume and last.exists():
        checkpoint = torch.load(last, map_location=device, weights_only=False)
        if checkpoint["run_id"] != run_id:
            raise ValueError("Resume checkpoint provenance/config differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best, best_epoch, stale = checkpoint["epoch"]+1, checkpoint["best_score"], checkpoint["best_epoch"], checkpoint["stale"]
        history = checkpoint["history"]
        random.setstate(checkpoint["rng_python"])
        np.random.set_state(checkpoint["rng_numpy"])
        torch.set_rng_state(checkpoint["rng_torch"].cpu())
        generator.set_state(checkpoint["loader_rng"].cpu())
        if device.type == "cuda" and checkpoint.get("rng_cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["rng_cuda"]])
    loader = torch.utils.data.DataLoader(split.train, batch_size=int(config["batch_size"]), shuffle=True,
        num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda",
        drop_last=len(split.train) % int(config["batch_size"]) == 1,
        generator=generator, worker_init_fn=seed_worker)
    if int(config["batch_size"]) < 2:
        raise ValueError("Use batch_size>=2 for BatchNorm-based baseline training")
    began = perf_counter()
    for epoch in range(start, int(config["epochs"])):
        if stale >= int(config["patience"]) and epoch >= int(config["min_epochs"]):
            break
        epoch_began = perf_counter()
        model.train()
        total_loss, samples = 0.0, 0
        optimizer_steps, amp_skipped_steps = 0, 0
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True).reshape(-1,1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                if logits.shape != y.shape:
                    raise ValueError("Expected model/label shape (batch,1)")
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; inspect source amplitudes and model hyperparameters")
            scaler.scale(loss).backward()
            if _optimizer_step(torch, model, optimizer, scaler):
                optimizer_steps += 1
            else:
                amp_skipped_steps += 1
                print(f"AMP overflow: skipped optimizer step at epoch={epoch+1}; "
                      f"new_scale={scaler.get_scale():g}", flush=True)
            total_loss += float(loss.detach().cpu())*len(x)
            samples += len(x)
        if not samples:
            raise ValueError("No training batches were produced")
        if not optimizer_steps:
            raise RuntimeError("No optimizer updates completed in this epoch; all AMP steps "
                               "overflowed. Retry with --set amp=false in a fresh output directory.")
        val_prediction = _predict(model, split.val, config, device)
        val_metrics = _metrics(split.val, val_prediction, 0.5)
        score = val_metrics["pr_auc"]
        if score is None:
            raise ValueError("Validation AP is undefined; both classes are needed")
        improved = score > best + 1e-8
        if improved:
            best, best_epoch, stale = score, epoch, 0
            _torch_save(run_dir/"best.pt", dict(run_id=run_id, model=model.state_dict(), config=config, epoch=epoch,
                validation_pr_auc=score, model_description=model_description(config["model"])))
        else:
            stale += 1
        history.append(dict(epoch=epoch+1, train_loss=total_loss/samples, samples=samples,
            validation_pr_auc=score, validation_roc_auc=val_metrics["roc_auc"],
            learning_rate=optimizer.param_groups[0]["lr"], seconds=perf_counter()-epoch_began,
            optimizer_steps=optimizer_steps, amp_skipped_steps=amp_skipped_steps,
            grad_scale=float(scaler.get_scale())))
        scheduler.step()
        _torch_save(last, dict(run_id=run_id, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=epoch, best_score=best,
            best_epoch=best_epoch, stale=stale, history=history, rng_python=random.getstate(),
            rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(), loader_rng=generator.get_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None))
        write_json(run_dir/"history.json", history)
        print(f"{config['model']} seed={config['seed']} epoch={epoch+1} loss={total_loss/samples:.5f} "
              f"val_AP={score:.5f} updates={optimizer_steps} amp_skipped={amp_skipped_steps}", flush=True)
    if not (run_dir/"best.pt").exists():
        raise ValueError("Training produced no valid best checkpoint")
    checkpoint = torch.load(run_dir/"best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, device, dict(parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        best_epoch=best_epoch+1, best_validation_pr_auc=best, completed_epochs=len(history),
        current_session_training_seconds=perf_counter()-began,
        total_recorded_epoch_seconds=sum(h["seconds"] for h in history),
        optimizer_steps=sum(h["optimizer_steps"] for h in history),
        amp_skipped_steps=sum(h["amp_skipped_steps"] for h in history),
        amp_effective=use_amp, pos_weight=float(pos_weight), model_description=model_description(config["model"]))


def _train_classical(config, split, run_dir):
    began = perf_counter()
    features, names = extract_features(split.train, fs=config["fs_hz"], return_names=True)
    model = build_estimator(config["model"], seed=int(config["seed"]), n_jobs=int(config["cpu_threads"]))
    candidates = [k for k in model.get_params(deep=True) if k.split("__")[-1] == "class_weight"]
    if candidates:
        model.set_params(**{k: ("balanced" if config["class_weight"] == "auto" else None) for k in candidates})
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=int(config["cpu_threads"])):
        model.fit(features, split.train.y)
    temporary = run_dir/"estimator.joblib.writing"
    joblib.dump(model, temporary)
    temporary.replace(run_dir/"estimator.joblib")
    write_json(run_dir/"features.json", dict(names=names, count=len(names),
        scaler="Pipeline scaler, when present, is fitted only on the training partition",
        svm="SVM uses continuous decision scores without pretending they are calibrated probabilities"))
    return model, None, dict(parameters=None, best_epoch=None, completed_epochs=1,
        current_session_training_seconds=perf_counter()-began, feature_count=len(names),
        estimator_parameters={key: (value if isinstance(value,(str,int,float,bool,type(None))) else repr(value))
                              for key,value in model.get_params(deep=True).items()})


def train_run(config, resume=False):
    config = dict(config)
    debug = config.get("max_train_samples") is not None or config.get("max_eval_samples") is not None
    config["smoke_only"] = bool(config.get("smoke_only") or debug)
    snapshot = _snapshot(config)
    code = _code_snapshot()
    run_id = digest(dict(config=scientific_config(config), data=snapshot["fingerprint"], implementation=code["fingerprint"]))
    run_dir = Path(config["output_root"])/config["model"]/f"seed_{config['seed']}_{run_id[:12]}"
    completed = run_dir/"run_metrics.json"
    if completed.exists():
        if resume and read_json(completed).get("run_id") == run_id:
            print(f"Already complete: {run_dir}", flush=True)
            return run_dir
        raise FileExistsError(f"Completed run exists. Use --resume to keep it: {run_dir}")
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(f"Partial run exists. Use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir/"config.json", config)
    write_json(run_dir/"provenance.json", dict(run_id=run_id, data=snapshot, code=code, environment=environment()))
    _archive_code(run_dir)
    bundle, split = _split(config)
    try:
        write_json(run_dir/"split.json", _split_manifest(split))
        if config["model"] in CLASSICAL_NAMES:
            model, device, training = _train_classical(config, split, run_dir)
        else:
            model, device, training = _train_deep(config, split, run_dir, run_id, resume)
        val = _predict(model, split.val, config, device)
        selection = choose_threshold(split.val.y, val["scores"], method=config["threshold_method"],
            target_recall=float(config["target_recall"]))
        threshold = float(selection["threshold"])
        # Test is read only after model selection and validation threshold are fixed.
        test = _predict(model, split.test, config, device)
        records = grouped_metrics(split.test.y, test["scores"], split.test.record_ids, threshold, test["probabilities"])
        report = dict(status="complete", run_id=run_id, config=config, protocol=split.protocol,
            smoke_only=config["smoke_only"], threshold=threshold, threshold_selection=selection,
            model_selection="Best validation average precision; early stopping never reads test labels",
            val=_metrics(split.val, val, threshold), test=_metrics(split.test, test, threshold),
            per_record=records, record_macro=_record_macro(records),
            record_macro_support={metric: sum(row.get(metric) is not None for row in records.values()) for metric in METRIC_NAMES},
            per_region=grouped_metrics(split.test.y, test["scores"], split.test.regions, threshold, test["probabilities"]),
            per_site=grouped_metrics(split.test.y, test["scores"], split.test.sites, threshold, test["probabilities"]),
            record_bootstrap=clustered_bootstrap(split.test.y, test["scores"], split.test.record_ids, threshold,
                test["probabilities"], n_bootstrap=int(config["bootstrap_iterations"]), seed=int(config["seed"])),
            training=training, test_inference_seconds=test["elapsed_seconds"],
            end_to_end_test_samples_per_second=test["end_to_end_samples_per_second"],
            score_kind=test["score_kind"], warnings=list(split.protocol.get("warnings", [])), environment=environment())
        if len(set(split.test.record_ids)) < 10:
            report["warnings"].append("Test has fewer than ten independent records; confidence intervals and domain generalization claims are fragile")
        _save_predictions(run_dir/"val_predictions.npz", split.val, val, threshold)
        _save_predictions(run_dir/"test_predictions.npz", split.test, test, threshold)
        write_json(completed, report)
        print(f"Complete: {run_dir}; test_AP={report['test']['pr_auc']:.5f} F1={report['test']['f1']:.5f}", flush=True)
        del model
        gc.collect()
        return run_dir
    finally:
        if hasattr(split.train, "mode"):
            split.train.close()
        bundle.close()


def _load_run(run_dir, device="auto", data_root=None):
    run_dir = Path(run_dir).resolve()
    report = read_json(run_dir/"run_metrics.json")
    trained_code = read_json(run_dir/"provenance.json").get("code")
    if trained_code is None or trained_code["fingerprint"] != _code_snapshot()["fingerprint"]:
        raise ValueError("Implementation differs from this checkpoint's training code. Use its archived source/benchmark.py and original run config for reproducible evaluation.")
    config = dict(report["config"])
    config["device"] = device
    if data_root is not None:
        config["data_root"] = str(Path(data_root).expanduser().resolve())
    if config["model"] in CLASSICAL_NAMES:
        # Only load locally generated trusted artifacts, never third-party pickles.
        model = joblib.load(run_dir/"estimator.joblib")
        actual_device = None
    else:
        from .models import create_model
        torch = _torch()
        actual_device = _device(device)
        torch.set_num_threads(int(config["cpu_threads"]))
        model = create_model(config["model"], channels=4, length=240, config=config.get("model_config")).to(actual_device)
        checkpoint = torch.load(run_dir/"best.pt", map_location=actual_device, weights_only=True)
        if checkpoint["run_id"] != report["run_id"]:
            raise ValueError("Checkpoint and run provenance differ")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
    return run_dir, report, config, model, actual_device


def evaluate_run(run_dir, device="auto", bootstrap_iterations=None, data_root=None):
    run_dir, report, config, model, actual_device = _load_run(run_dir, device, data_root)
    expected = read_json(run_dir/"provenance.json")["data"]["fingerprint"]
    if _snapshot(config)["fingerprint"] != expected:
        raise ValueError("Dataset changed since training; refuse untracked evaluation")
    bundle, split = _split(config)
    try:
        result = _predict(model, split.test, config, actual_device)
        threshold = report["threshold"]
        scored = dict(status="passed", threshold_source="Stored validation-only threshold; not reselected",
            run_id=report["run_id"], test=_metrics(split.test, result, threshold),
            record_bootstrap=clustered_bootstrap(split.test.y, result["scores"], split.test.record_ids,
                threshold, result["probabilities"], n_bootstrap=int(config["bootstrap_iterations"] if bootstrap_iterations is None else bootstrap_iterations),
                seed=int(config["seed"])))
        write_json(run_dir/"reevaluation.json", scored)
        return scored
    finally:
        bundle.close()


def robustness_run(run_dir, specs=None, device="auto", limit=None, data_root=None):
    run_dir, report, config, model, actual_device = _load_run(run_dir, device, data_root)
    expected = read_json(run_dir/"provenance.json")["data"]["fingerprint"]
    if _snapshot(config)["fingerprint"] != expected:
        raise ValueError("Dataset changed since training; refuse untracked robustness evaluation")
    bundle, split = _split(config)
    ds = _limit(split.test, limit, int(config.get("split_seed", 20261014))+30)
    try:
        conditions = []
        for spec in DEFAULT_CORRUPTIONS if specs is None else specs:
            if spec.get("name") == "time_shift":
                raise ValueError("time_shift can move an annotated event outside a window; original y cannot be reused without center-level label validation")
            prediction = _predict(model, ds, config, actual_device, spec=spec)
            conditions.append(dict(corruption=spec, metrics=_metrics(ds, prediction, report["threshold"])))
        result = dict(status="complete", run_id=report["run_id"], threshold=report["threshold"],
            threshold_source="Original clean validation; never adjusted on corrupted test",
            rows=len(ds), smoke_only=bool(limit or config.get("smoke_only")), conditions=conditions,
            frequency_processing="No source filtering. Bandstop is an optional explicit stress test, not a dataset definition.",
            caveat="Stress is applied to prepared model inputs, not an acquisition/preprocessing simulator. Source event labels are retained; synthetic stress does not replace real noisy-domain annotation.")
        write_json(run_dir/("robustness_smoke.json" if result["smoke_only"] else "robustness.json"), result)
        return result
    finally:
        bundle.close()


def predict_field(run_dir, device="auto", batch_size=None, limit=None, output_dir=None, data_root=None):
    run_dir, report, config, model, actual_device = _load_run(run_dir, device, data_root)
    root = Path(config["data_root"])
    name = "X.npy" if config["version"] == "standardized" else "X_raw.npy"
    source = root/"test/unlabeled"/name
    values = np.load(source, mmap_mode="r", allow_pickle=False)
    if values.dtype != np.float32 or values.shape[1:] != (4,240):
        values._mmap.close()
        raise ValueError("Unexpected prepared field-input shape/dtype")
    count = len(values) if limit is None else min(int(limit), len(values))
    if count <= 0:
        values._mmap.close()
        raise ValueError("limit must be positive")
    out = Path(output_dir).resolve() if output_dir else run_dir/"field_predictions"/("all" if limit is None else f"first_{count}")
    out.mkdir(parents=True, exist_ok=True)
    if (out/"prediction_summary.json").exists():
        values._mmap.close()
        raise FileExistsError("Prediction output already exists; choose another --output-dir")
    batch_size = int(batch_size or config["batch_size"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    selected = _channels(config["channels"])
    maps = {key: np.lib.format.open_memmap(out/f"{key}.writing.npy", mode="w+", dtype=dtype, shape=(count,))
            for key,dtype in (("scores",np.float64),("predictions",np.uint8))}
    probability_available = None
    input_digest = hashlib.sha256()
    began = perf_counter()
    try:
        for lo in range(0,count,batch_size):
            hi = min(lo+batch_size,count)
            x = np.array(values[lo:hi], dtype=np.float32, copy=True)
            input_digest.update(x.tobytes(order="C"))
            for c, channel in enumerate(CHANNELS):
                if channel not in selected:
                    x[:,c] = 0
            if config["model"] in CLASSICAL_NAMES:
                prediction = prediction_scores(model, extract_features(x,fs=config["fs_hz"]))
            else:
                torch = _torch()
                with torch.no_grad(), torch.autocast(device_type=actual_device.type,
                     enabled=bool(config["amp"] and actual_device.type=="cuda")):
                    logits = model(torch.from_numpy(x).to(actual_device))
                score = torch.sigmoid(logits.float()).reshape(-1).cpu().numpy()
                prediction = dict(scores=score, probabilities=score, score_kind="sigmoid_probability")
            scores = prediction["scores"]
            if len(scores) != hi-lo or not np.isfinite(scores).all():
                raise ValueError("Invalid field predictions")
            has_probability = prediction["probabilities"] is not None
            if probability_available is None:
                probability_available = has_probability
                if has_probability:
                    maps["probabilities"] = np.lib.format.open_memmap(out/"probabilities.writing.npy",mode="w+",dtype=np.float32,shape=(count,))
            if has_probability != probability_available:
                raise ValueError("Probability availability changed between batches")
            maps["scores"][lo:hi] = scores
            maps["predictions"][lo:hi] = np.asarray(scores,dtype=np.float64) >= report["threshold"]
            if has_probability:
                maps["probabilities"][lo:hi] = prediction["probabilities"]
            if hi==count or hi//100000 != lo//100000:
                print(f"Field predictions {hi:,}/{count:,}",flush=True)
    finally:
        values._mmap.close()
        for array in maps.values():
            array.flush()
            array._mmap.close()
    for key in maps:
        (out/f"{key}.writing.npy").replace(out/f"{key}.npy")
    scores = np.load(out/"scores.npy",mmap_mode="r")
    predictions = np.load(out/"predictions.npy",mmap_mode="r")
    try:
        with (root/"test/unlabeled/windows.csv").open(encoding="utf-8-sig",newline="") as stream:
            records=list(csv.DictReader(stream))
        with (out/"record_candidates.csv").open("w",encoding="utf-8-sig",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=["record_id","row_start","row_stop","windows","positive_windows","candidate_rate","mean_score"])
            writer.writeheader()
            for row in records:
                lo=int(row["virtual_window_start"]); hi=min(int(row["virtual_window_stop_exclusive"]),count)
                if hi<=lo:continue
                positives=int(predictions[lo:hi].sum())
                writer.writerow(dict(record_id=row["record_id"],row_start=lo,row_stop=hi,windows=hi-lo,
                    positive_windows=positives,candidate_rate=positives/(hi-lo),mean_score=float(scores[lo:hi].mean(dtype=np.float64))))
        result=dict(status="complete",run_id=report["run_id"],rows=count,source=str(source),version=config["version"],
            channels=list(selected),threshold=report["threshold"],score_kind=prediction["score_kind"],
            probabilities_available=probability_available,positive_windows=int(predictions.sum()),
            elapsed_seconds=perf_counter()-began,labels="Unknown. No accuracy, precision, recall, or F1 is reported.",
            row_mapping="test/unlabeled/windows.csv virtual_window_start/stop intervals, not one CSV row per window",
            input_values_sha256=input_digest.hexdigest(), input_hash_scope="Processed original float32 rows in contiguous C order, before channel masking; no .npy header",
            windows_manifest_sha256=sha256(root/"test/unlabeled/windows.csv"),
            training_code=read_json(run_dir/"provenance.json")["code"],
            partial=limit is not None,output_dir=str(out),files={key:dict(path=f"{key}.npy",sha256=sha256(out/f"{key}.npy")) for key in maps})
    finally:
        scores._mmap.close();predictions._mmap.close()
    write_json(out/"prediction_summary.json",result)
    return result
