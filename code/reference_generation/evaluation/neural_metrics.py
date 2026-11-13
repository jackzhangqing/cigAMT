"""Transformer LDS/LPS metrics and optional official TS2Vec embeddings."""

from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def import_torch():
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "PyTorch could not be imported. LDS/LPS/TS2Vec require a working PyTorch "
            f"environment. Original error: {exc}"
        ) from exc
    return torch


def resolve_device(requested: str) -> str:
    torch = import_torch()
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    torch = import_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_models(
    channels: int,
    length: int,
    patch_size: int,
    d_model: int,
    heads: int,
    layers: int,
    dropout: float,
):
    torch = import_torch()
    nn = torch.nn

    class PatchTransformerEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_size = patch_size
            self.patch = nn.Conv1d(
                channels, d_model, kernel_size=patch_size, stride=patch_size
            )
            max_tokens = int(np.ceil(length / patch_size))
            self.position = nn.Parameter(torch.zeros(1, max_tokens, d_model))
            nn.init.trunc_normal_(self.position, std=0.02)
            block = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(block, num_layers=layers)
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            remainder = x.shape[-1] % self.patch_size
            if remainder:
                x = nn.functional.pad(x, (0, self.patch_size - remainder))
            tokens = self.patch(x).transpose(1, 2)
            tokens = tokens + self.position[:, : tokens.shape[1]]
            tokens = self.encoder(tokens)
            return self.norm(tokens.mean(dim=1))

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = PatchTransformerEncoder()
            self.head = nn.Linear(d_model, 1)

        def forward(self, x):
            return self.head(self.backbone(x)).squeeze(-1)

    class Forecaster(nn.Module):
        def __init__(self, horizon: int):
            super().__init__()
            self.backbone = PatchTransformerEncoder()
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, channels * horizon)
            )

        def forward(self, x):
            return self.head(self.backbone(x))

    return Classifier, Forecaster


def _loaders(x_train, y_train, x_val, y_val, batch_size: int, seed: int):
    torch = import_torch()
    generator = torch.Generator().manual_seed(seed)
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(np.asarray(x_train, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_train, dtype=np.float32)),
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(np.asarray(x_val, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_val, dtype=np.float32)),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader


def _train(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    task: str,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
) -> int:
    torch = import_torch()
    model = model.to(device)
    train_loader, val_loader = _loaders(x_train, y_train, x_val, y_val, batch_size, seed)
    criterion = torch.nn.BCEWithLogitsLoss() if task == "classification" else torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    stale = 0
    epochs_run = 0
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                loss = criterion(model(batch_x), batch_y)
                total += float(loss.item()) * len(batch_x)
                count += len(batch_x)
        val_loss = total / max(count, 1)
        epochs_run = epoch + 1
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return epochs_run


def _predict(model, array: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    torch = import_torch()
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.asarray(array, dtype=np.float32)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            outputs.append(model(batch.to(device)).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _aggregate_runs(rows: list[dict[str, float | int]]) -> dict[str, float]:
    result: dict[str, float] = {}
    if not rows:
        return result
    for key in rows[0]:
        if key in {"repeat", "seed", "epochs"}:
            continue
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return result


def long_sequence_discriminative_score(
    real_nct: np.ndarray,
    fake_nct: np.ndarray,
    repeats: int,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patch_size: int,
    d_model: int,
    heads: int,
    layers: int,
    dropout: float,
    patience: int,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    """Transformer LDS with held-out accuracy, AUROC, and class-wise F1."""
    count = min(len(real_nct), len(fake_nct))
    Classifier, _ = _build_models(
        real_nct.shape[1], real_nct.shape[2], patch_size, d_model, heads, layers, dropout
    )
    rows: list[dict[str, float | int]] = []
    for repeat in range(repeats):
        run_seed = seed + repeat
        set_seed(run_seed)
        rng = np.random.default_rng(run_seed)
        real_idx = rng.choice(len(real_nct), size=count, replace=False)
        fake_idx = rng.choice(len(fake_nct), size=count, replace=False)
        x = np.concatenate([real_nct[real_idx], fake_nct[fake_idx]], axis=0)
        y = np.concatenate([np.ones(count), np.zeros(count)]).astype(np.float32)
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y, random_state=run_seed
        )
        fit_idx, val_idx = train_test_split(
            train_idx, test_size=0.2, stratify=y[train_idx], random_state=run_seed
        )
        model = Classifier()
        epochs_run = _train(
            model,
            x[fit_idx],
            y[fit_idx],
            x[val_idx],
            y[val_idx],
            "classification",
            device,
            epochs,
            batch_size,
            learning_rate,
            patience,
            run_seed,
        )
        logits = _predict(model, x[test_idx], device, batch_size)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        predictions = (probabilities >= 0.5).astype(int)
        truth = y[test_idx].astype(int)
        accuracy = accuracy_score(truth, predictions)
        rows.append(
            {
                "repeat": repeat,
                "seed": run_seed,
                "epochs": epochs_run,
                "lds": float(abs(accuracy - 0.5)),
                "accuracy": float(accuracy),
                "auroc": float(roc_auc_score(truth, probabilities)),
                "f1_generated": float(f1_score(truth, predictions, pos_label=0, zero_division=0)),
                "f1_real": float(f1_score(truth, predictions, pos_label=1, zero_division=0)),
            }
        )
    return _aggregate_runs(rows), rows


def _forecast_arrays(data_nct: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    if horizon <= 0 or horizon >= data_nct.shape[-1]:
        raise ValueError(f"Invalid horizon {horizon} for length {data_nct.shape[-1]}")
    x = np.ascontiguousarray(data_nct[:, :, :-horizon], dtype=np.float32)
    y = np.ascontiguousarray(data_nct[:, :, -horizon:].reshape(len(data_nct), -1), dtype=np.float32)
    return x, y


def _fit_forecaster(
    train_nct: np.ndarray,
    val_nct: np.ndarray,
    horizon: int,
    Forecaster,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
):
    x_train, y_train = _forecast_arrays(train_nct, horizon)
    x_val, y_val = _forecast_arrays(val_nct, horizon)
    model = Forecaster(horizon)
    epochs_run = _train(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        "regression",
        device,
        epochs,
        batch_size,
        learning_rate,
        patience,
        seed,
    )
    return model, epochs_run


def long_sequence_predictive_score(
    real_nct: np.ndarray,
    fake_nct: np.ndarray,
    horizons: list[int],
    channel_std: np.ndarray,
    repeats: int,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patch_size: int,
    d_model: int,
    heads: int,
    layers: int,
    dropout: float,
    patience: int,
) -> tuple[dict[str, Any], list[dict[str, float | int]]]:
    """Transformer TSTR LPS, plus a train-real/test-real reference."""
    _, Forecaster = _build_models(
        real_nct.shape[1], real_nct.shape[2], patch_size, d_model, heads, layers, dropout
    )
    all_rows: list[dict[str, float | int]] = []
    summaries: dict[str, Any] = {}
    for horizon in horizons:
        rows: list[dict[str, float | int]] = []
        for repeat in range(repeats):
            run_seed = seed + 1000 + repeat
            set_seed(run_seed)
            rng = np.random.default_rng(run_seed)
            real_train_idx, real_test_idx = train_test_split(
                np.arange(len(real_nct)), test_size=0.25, random_state=run_seed
            )
            real_fit_idx, real_val_idx = train_test_split(
                real_train_idx, test_size=0.2, random_state=run_seed
            )
            fake_count = min(len(fake_nct), len(real_train_idx))
            fake_idx = rng.choice(len(fake_nct), size=fake_count, replace=False)
            fake_fit_idx, fake_val_idx = train_test_split(
                fake_idx, test_size=0.2, random_state=run_seed
            )

            synthetic_model, synthetic_epochs = _fit_forecaster(
                fake_nct[fake_fit_idx],
                fake_nct[fake_val_idx],
                horizon,
                Forecaster,
                device,
                epochs,
                batch_size,
                learning_rate,
                patience,
                run_seed,
            )
            real_model, real_epochs = _fit_forecaster(
                real_nct[real_fit_idx],
                real_nct[real_val_idx],
                horizon,
                Forecaster,
                device,
                epochs,
                batch_size,
                learning_rate,
                patience,
                run_seed + 100,
            )
            x_test, y_test = _forecast_arrays(real_nct[real_test_idx], horizon)
            synthetic_prediction = _predict(synthetic_model, x_test, device, batch_size)
            real_prediction = _predict(real_model, x_test, device, batch_size)
            synthetic_mae = float(np.mean(np.abs(synthetic_prediction - y_test)))
            real_mae = float(np.mean(np.abs(real_prediction - y_test)))
            scale = np.repeat(channel_std.astype(np.float64), horizon)
            synthetic_raw_mae = float(np.mean(np.abs(synthetic_prediction - y_test) * scale[None, :]))
            real_raw_mae = float(np.mean(np.abs(real_prediction - y_test) * scale[None, :]))
            row: dict[str, float | int] = {
                "horizon": horizon,
                "repeat": repeat,
                "seed": run_seed,
                "synthetic_epochs": synthetic_epochs,
                "real_epochs": real_epochs,
                "synthetic_mae": synthetic_mae,
                "real_reference_mae": real_mae,
                "mae_gap": synthetic_mae - real_mae,
                "mae_ratio": synthetic_mae / max(real_mae, 1e-12),
                "synthetic_raw_mae": synthetic_raw_mae,
                "real_reference_raw_mae": real_raw_mae,
            }
            rows.append(row)
            all_rows.append(row)
        # Aggregate without horizon and the two epoch counters.
        aggregate_input = [
            {k: v for k, v in row.items() if k not in {"horizon", "synthetic_epochs", "real_epochs"}}
            for row in rows
        ]
        summaries[str(horizon)] = _aggregate_runs(aggregate_input)
    return summaries, all_rows


def ts2vec_embeddings(
    real_nct: np.ndarray,
    generated: dict[str, np.ndarray],
    repo_path: str | Path,
    device: str,
    output_dir: str | Path,
    epochs: int,
    batch_size: int,
    checkpoint: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    """Fit/load the official TS2Vec implementation and encode full sequences."""
    import_torch()
    repo = Path(repo_path).resolve()
    if not (repo / "ts2vec.py").exists():
        raise FileNotFoundError(f"Official TS2Vec repo not found at {repo}")
    sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module("ts2vec")
        TS2Vec = module.TS2Vec
        model = TS2Vec(
            input_dims=real_nct.shape[1],
            device=device,
            output_dims=320,
            batch_size=batch_size,
        )
        if checkpoint:
            model.load(str(checkpoint))
            checkpoint_used = str(Path(checkpoint).resolve())
        else:
            model.fit(np.transpose(real_nct, (0, 2, 1)), n_epochs=epochs, verbose=True)
            checkpoint_path = Path(output_dir) / "ts2vec_encoder.pkl"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(checkpoint_path))
            checkpoint_used = str(checkpoint_path.resolve())
        real_features = model.encode(
            np.transpose(real_nct, (0, 2, 1)), encoding_window="full_series"
        )
        fake_features = {
            name: model.encode(np.transpose(data, (0, 2, 1)), encoding_window="full_series")
            for name, data in generated.items()
        }
    finally:
        if sys.path and sys.path[0] == str(repo):
            sys.path.pop(0)
    return np.asarray(real_features), {k: np.asarray(v) for k, v in fake_features.items()}, checkpoint_used

