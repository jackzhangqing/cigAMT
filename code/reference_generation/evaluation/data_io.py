"""Input, sampling, scaling, and hashing helpers for evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


@dataclass
class ArrayInfo:
    path: str
    key: str
    shape: tuple[int, ...]
    dtype: str
    attrs: dict[str, Any]
    loaded_samples: int = 0
    dropped_nonfinite: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "key": self.key,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "attrs": self.attrs,
            "loaded_samples": self.loaded_samples,
            "dropped_nonfinite": self.dropped_nonfinite,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def parse_named_path(spec: str) -> tuple[str, Path]:
    """Parse NAME=PATH, falling back to the file stem as NAME."""
    if "=" in spec:
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip())
    else:
        path = Path(spec)
        name = path.stem
    if not name:
        raise ValueError(f"Empty dataset name in {spec!r}")
    return name, path


def _orient_nct(array: np.ndarray, expected_channels: int | None) -> np.ndarray:
    """Return an array in (sample, channel, time) order."""
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape {array.shape}")
    if expected_channels is not None:
        if array.shape[1] == expected_channels:
            return array
        if array.shape[2] == expected_channels:
            return np.transpose(array, (0, 2, 1))
        raise ValueError(
            f"Neither axis 1 nor axis 2 has expected channel count "
            f"{expected_channels}: shape={array.shape}"
        )
    # Time is normally much longer than channel count.
    if array.shape[1] <= array.shape[2]:
        return array
    return np.transpose(array, (0, 2, 1))


def inspect_array(path: str | Path, key: str = "data") -> ArrayInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path, mmap_mode="r")
        return ArrayInfo(str(path.resolve()), key, tuple(arr.shape), str(arr.dtype), {})
    with h5py.File(path, "r") as handle:
        if key not in handle:
            raise KeyError(f"Dataset {key!r} not found in {path}; keys={list(handle.keys())}")
        ds = handle[key]
        attrs = {str(k): _jsonable(v) for k, v in handle.attrs.items()}
        return ArrayInfo(str(path.resolve()), key, tuple(ds.shape), str(ds.dtype), attrs)


def _sample_indices(total: int, maximum: int | None, seed: int) -> np.ndarray:
    if maximum is None or maximum <= 0 or maximum >= total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    # h5py requires monotonically increasing fancy indices.
    return np.sort(rng.choice(total, size=maximum, replace=False).astype(np.int64))


def load_array(
    path: str | Path,
    key: str = "data",
    max_samples: int | None = None,
    seed: int = 42,
    expected_channels: int | None = 4,
) -> tuple[np.ndarray, ArrayInfo]:
    """Load a deterministic subset and return float32 data in N,C,T order."""
    path = Path(path)
    info = inspect_array(path, key)
    indices = _sample_indices(info.shape[0], max_samples, seed)
    if path.suffix.lower() == ".npy":
        source = np.load(path, mmap_mode="r")
        array = np.asarray(source[indices], dtype=np.float32)
    else:
        with h5py.File(path, "r") as handle:
            array = np.asarray(handle[key][indices], dtype=np.float32)
    array = _orient_nct(array, expected_channels)
    finite = np.isfinite(array).all(axis=(1, 2))
    info.dropped_nonfinite = int((~finite).sum())
    array = np.ascontiguousarray(array[finite], dtype=np.float32)
    info.loaded_samples = int(array.shape[0])
    if not len(array):
        raise ValueError(f"No finite samples loaded from {path}")
    return array, info


@dataclass
class ChannelStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, reference_nct: np.ndarray) -> "ChannelStandardizer":
        mean = reference_nct.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
        std = reference_nct.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
        std = np.maximum(std, np.float32(1e-6))
        return cls(mean=mean, std=std)

    def transform(self, array_nct: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            (array_nct - self.mean[None, :, None]) / self.std[None, :, None],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def iter_samples(
    path: str | Path,
    key: str = "data",
    chunk_size: int = 256,
    expected_channels: int | None = 4,
) -> Iterable[np.ndarray]:
    """Yield samples one at a time without loading the full file."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        source = np.load(path, mmap_mode="r")
        for start in range(0, source.shape[0], chunk_size):
            chunk = _orient_nct(np.asarray(source[start : start + chunk_size]), expected_channels)
            for sample in chunk:
                yield np.asarray(sample, dtype=np.float32)
        return
    with h5py.File(path, "r") as handle:
        ds = handle[key]
        for start in range(0, ds.shape[0], chunk_size):
            chunk = _orient_nct(np.asarray(ds[start : start + chunk_size]), expected_channels)
            for sample in chunk:
                yield np.asarray(sample, dtype=np.float32)


def sample_sha256(sample_ct: np.ndarray) -> str:
    canonical = np.ascontiguousarray(sample_ct, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def hash_inventory(
    path: str | Path,
    key: str = "data",
    chunk_size: int = 256,
    expected_channels: int | None = 4,
) -> tuple[dict[str, int], int, int]:
    """Return SHA-256 counts, total samples, and internal duplicate count."""
    hash_counts: dict[str, int] = {}
    total = 0
    duplicates = 0
    for sample in iter_samples(path, key, chunk_size, expected_channels):
        digest = sample_sha256(sample)
        total += 1
        if digest in hash_counts:
            duplicates += 1
        hash_counts[digest] = hash_counts.get(digest, 0) + 1
    return hash_counts, total, duplicates


def file_sha256(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
