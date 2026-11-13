"""Configuration, provenance, and atomic artifact helpers."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
from datetime import datetime, timezone

import numpy as np


BASE = Path(__file__).resolve().parents[1]
DEFAULTS = dict(data_root="..", output_root="results/main", version="standardized",
    augmentation="all", channels="all", train_fraction=1.0, protocol="fixed",
    group_by="record", n_splits=5, val_fraction=0.2, fold=0, split_seed=20261014,
    epochs=100, batch_size=128, learning_rate=0.001, weight_decay=0.0001,
    patience=10, min_epochs=5, num_workers=0, device="auto", amp=False,
    deterministic=True, cpu_threads=4, class_weight="none",
    threshold_method="best_f1", target_recall=0.95, bootstrap_iterations=1000,
    seed=42, model="cnn1d", model_config={}, max_train_samples=None,
    max_eval_samples=None, smoke_only=False, fs_hz=24000)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(clean_json(value), sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def sha256(path):
    out = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            out.update(block)
    return out.hexdigest()


def resolve_path(value):
    value = Path(value).expanduser()
    return value.resolve() if value.is_absolute() else (BASE / value).resolve()


def load_config(path=None, overrides=None):
    config = DEFAULTS.copy()
    config["model_config"] = {}
    if path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = BASE / candidate
        config.update(read_json(candidate))
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    for key in ("data_root", "output_root"):
        config[key] = str(resolve_path(config[key]))
    if config["version"] not in {"raw", "standardized"}:
        raise ValueError("version must be raw or standardized")
    if config["protocol"] not in {"fixed", "group_cv"}:
        raise ValueError("protocol must be fixed or group_cv")
    for key in ("epochs", "batch_size", "patience", "cpu_threads"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 < float(config["train_fraction"]) <= 1:
        raise ValueError("train_fraction must lie in (0,1]")
    if int(config["num_workers"]) < 0 or int(config["bootstrap_iterations"]) < 0:
        raise ValueError("num_workers/bootstrap_iterations must be nonnegative")
    if config["class_weight"] not in {"none", "auto"}:
        raise ValueError("class_weight must be none or auto")
    if config["threshold_method"] not in {"best_f1", "target_recall"}:
        raise ValueError("threshold_method must be best_f1 or target_recall")
    if config["protocol"] != "fixed" and config["train_fraction"] != 1.0:
        raise ValueError("Learning-curve fractions currently use the fixed protocol only")
    return config


def environment():
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        packages=versions, utc=datetime.now(timezone.utc).isoformat(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))


def scientific_config(config):
    ignored = {"data_root", "output_root", "device", "num_workers", "cpu_threads",
               "models", "seeds", "experiments", "description"}
    return {k: v for k, v in config.items() if k not in ignored}


def seed_everything(seed, deterministic=True):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def seed_worker(worker_id):
    import torch
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
