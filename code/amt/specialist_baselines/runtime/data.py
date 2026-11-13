"""Direct readers for the materialized AMT HDF5 shards.

No build manifest, source bank, schedule or original D: drive is consulted.
Files are discovered below ``offline/`` and their own headers are checked.
The module only needs NumPy and h5py; PyTorch can wrap it in a DataLoader.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

NOISE_TASKS = ("noise_localization", "noise_classification", "denoising", "frequency_denoising")
CLEAN_TASKS = ("forecasting", "interpolation", "imputation")
TASKS = NOISE_TASKS + CLEAN_TASKS
SOURCE_NAMES = ("real", "syn_amt_A", "syn_amt_B", "syn_mt", "ddpm", "ddim",
                "syn_continuous_A", "syn_continuous_B", "syn_3d_A", "syn_3d_B")
DOMAIN_NAMES = {-1: "clean", 0: "real", 1: "synthetic", 2: "generated"}
CLASS_NAMES = ("clean", "pulse", "quasi_sinusoidal", "square_wave", "step_wave",
               "triangular_wave", "power_frequency", "white_noise", "color_noise",
               "harmonic", "composite")

# A fixed set of scalar/vector metadata keeps mixed noise/control batches collatable.
_INT_META = {
    "clean_index": -1, "source_id": -1, "source_row": -1, "domain_id": -1,
    "noise_length": 0, "noise_start": -1, "noise_crop_start": -1,
    "noise_index": -1, "noise_bank_row": -1, "sample_index": -1,
    "class_id": -1, "is_clean": 0, "snr_condition_index": -1,
    "condition_index": -1, "horizon": 0, "gap_length": 0,
    "mean_gap": 0, "channel_mode_id": -1,
}
_FLOAT_META = {
    "snr_target_db": float("nan"), "snr_total_db": float("nan"),
    "snr_e_db": float("nan"), "snr_h_db": float("nan"),
    "missing_ratio_actual": 0.0, "missing_ratio_nominal": 0.0,
}


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _hash_values(values, seed):
    """Stable uint64 priorities, independent of Python hash randomization."""
    x = np.asarray(values, dtype=np.uint64) ^ np.uint64(int(seed) & ((1 << 64) - 1))
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@dataclass
class _Shard:
    path: Path
    relative: str
    rows: int
    kind: str
    scheme: str
    signature: str
    file_bytes: int
    raw_offset: int
    selected: np.ndarray | None = None

    @property
    def count(self):
        return self.rows if self.selected is None else len(self.selected)


class H5Dataset:
    """Read finished shard files, with deterministic optional filters/subsampling.

    ``root`` is the dataset root or its ``offline`` directory. For noise tasks,
    ``scheme='AC'`` combines A/C and adds shared controls only once. ``snr_db``
    filters noisy records; controls remain when ``include_controls=True``.
    Clean-task readers default to the context-only ``*_visible`` coordinates.

    ``max_samples`` uses a reproducible proportional subset stratified jointly
    by clean source, scheme, SNR and control status. Every nonempty stratum is
    represented when the budget is large enough. ``selection_mode='probe'`` is
    an explicitly limited smoke-test mode: it samples contiguous metadata blocks
    across every shard and stratifies within that candidate pool, without
    claiming exact whole-dataset stratum proportions. No waveform is loaded
    while selecting. Log this budget instead of calling a subset a full run.
    """

    def __init__(self, root, task, split, scheme="A", include_controls=True,
                 normalization="visible", snr_db=None, source_names=None,
                 max_samples=None, selection_seed=2025, cache_files=4,
                 selection_mode="exact"):
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}; choose from {TASKS}")
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        if scheme not in ("A", "C", "AC", "none"):
            raise ValueError("scheme must be A, C, AC, or none")
        if task in NOISE_TASKS and scheme == "none":
            raise ValueError("Noise tasks require A, C, or AC")
        if normalization not in ("visible", "legacy"):
            raise ValueError("normalization must be visible or legacy")
        if max_samples is not None and int(max_samples) < 1:
            raise ValueError("max_samples must be positive or None")
        if int(cache_files) < 1:
            raise ValueError("cache_files must be positive")
        if selection_mode not in ("exact", "probe"):
            raise ValueError("selection_mode must be exact or probe")
        if selection_mode == "probe" and max_samples is None:
            raise ValueError("probe selection is a limited smoke mode; specify max_samples")
        self.root = Path(root).expanduser().resolve()
        self.offline_root = self.root if self.root.name.lower() == "offline" else self.root / "offline"
        self.task, self.split = task, split
        self.scheme = "none" if task in CLEAN_TASKS else scheme
        self.normalization = normalization
        self.include_controls = bool(include_controls) and task in NOISE_TASKS
        self.selection_seed = int(selection_seed)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.selection_mode = selection_mode
        self.cache_files = int(cache_files)
        self.source_names = None if source_names is None else tuple(source_names)
        if self.source_names is not None:
            unknown = set(self.source_names) - set(SOURCE_NAMES)
            if unknown:
                raise ValueError(f"Unknown clean source names: {sorted(unknown)}")
            if not self.source_names:
                raise ValueError("source_names cannot be an empty list")
        if snr_db is None:
            self.snr_db = None
        else:
            values = [snr_db] if np.isscalar(snr_db) else list(snr_db)
            self.snr_db = tuple(sorted(set(float(x) for x in values)))
            if not self.snr_db or not np.all(np.isfinite(self.snr_db)):
                raise ValueError("snr_db must contain finite numbers")
            if task in CLEAN_TASKS:
                raise ValueError("Clean tasks have no SNR condition")
        self._handles = OrderedDict()
        self._pid = os.getpid()
        self._shards = []
        self.dataset_id = None
        self.sequence_length = None
        self.sampling_rate_hz = None
        self._discover()
        self.unfiltered_rows = sum(s.rows for s in self._shards)
        self.candidate_pool_rows = self.unfiltered_rows
        self._strata_summary = None
        self._select()
        self._prefix = np.cumsum([0] + [s.count for s in self._shards], dtype=np.int64)
        if not int(self._prefix[-1]):
            raise ValueError("No HDF5 rows remain after the requested filters")

    def _discover(self):
        folders = []
        if self.task in CLEAN_TASKS:
            folders.append((self.offline_root / "clean_tasks" / self.task / self.split, "clean", ""))
        else:
            for scheme in ("A", "C") if self.scheme == "AC" else (self.scheme,):
                folders.append((self.offline_root / "noise_tasks" / scheme / self.split, "noise", scheme))
            if self.include_controls:
                folders.append((self.offline_root / "clean_controls" / self.split, "control", ""))
        raw_offset = 0
        for folder, kind, scheme in folders:
            paths = sorted(folder.glob("shard_*.h5"))
            if not paths:
                raise FileNotFoundError(f"No complete shard_*.h5 files in {folder}")
            for path in paths:
                with h5py.File(path, "r") as h:
                    if not bool(h.attrs.get("complete", False)):
                        raise ValueError(f"Incomplete HDF5 shard: {path}")
                    for key, expected in (("split", self.split), ("kind", kind), ("scheme", scheme),
                                          ("task", self.task if kind == "clean" else "")):
                        if _text(h.attrs.get(key, "<missing>")) != expected:
                            raise ValueError(f"Shard {key} mismatch in {path}: expected {expected!r}")
                    dataset_id = _text(h.attrs.get("dataset_id", ""))
                    if not dataset_id:
                        raise ValueError(f"Missing dataset_id in {path}")
                    if self.dataset_id is not None and dataset_id != self.dataset_id:
                        raise ValueError(f"Mixed dataset_id values in {path}")
                    self.dataset_id = dataset_id
                    required = {"input", "target", "source_id", "source_row", "clean_index", "sample_index"}
                    if kind == "clean":
                        required.update(("observed_mask", "loss_mask"))
                        if self.normalization == "visible":
                            required.update(("input_visible", "target_visible", "visible_center", "visible_scale"))
                    else:
                        required.update(("noise_mask", "scale", "class_id", "is_clean", "snr_target_db"))
                        if self.task == "frequency_denoising":
                            required.update(("target_low", "target_high"))
                    if required - set(h):
                        raise ValueError(f"Missing fields {sorted(required - set(h))} in {path}")
                    shape = h["input"].shape
                    if len(shape) != 3 or shape[1] != 4 or shape[0] < 1:
                        raise ValueError(f"Expected [N, 4, T] input in {path}, got {shape}")
                    rows, _, length = shape
                    if h["target"].shape != shape:
                        raise ValueError(f"Input/target shape mismatch in {path}")
                    if int(h.attrs.get("sequence_length", -1)) != length:
                        raise ValueError(f"sequence_length header mismatch in {path}")
                    if int(h.attrs.get("committed_rows", -1)) != rows:
                        raise ValueError(f"committed_rows header mismatch in {path}")
                    for key in required:
                        if h[key].shape[0] != rows:
                            raise ValueError(f"Field row count mismatch: {path}:{key}")
                    waveform_fields = {"input", "target", "noise_mask", "observed_mask", "loss_mask",
                                       "input_visible", "target_visible", "target_low", "target_high"}
                    for key in required & waveform_fields:
                        if h[key].shape != shape:
                            raise ValueError(f"Waveform shape mismatch: {path}:{key}")
                    for key in required & {"scale", "visible_center", "visible_scale"}:
                        if h[key].shape != (rows, 4, 1):
                            raise ValueError(f"Normalization shape mismatch: {path}:{key}")
                    fs = float(h.attrs.get("sampling_rate_hz", 0))
                    if fs <= 0 or not math.isfinite(fs):
                        raise ValueError(f"Invalid sampling_rate_hz in {path}")
                    if self.sequence_length is not None and length != self.sequence_length:
                        raise ValueError(f"Mixed sequence lengths in {path}")
                    if self.sampling_rate_hz is not None and fs != self.sampling_rate_hz:
                        raise ValueError(f"Mixed sampling rates in {path}")
                    self.sequence_length, self.sampling_rate_hz = length, fs
                    signature = _text(h.attrs.get("signature", ""))
                self._shards.append(_Shard(path, path.relative_to(self.offline_root).as_posix(),
                                           rows, kind, scheme or ("none" if kind == "clean" else "control"),
                                           signature, path.stat().st_size, raw_offset))
                raw_offset += rows

    def _metadata_blocks(self, shard):
        """Bounded contiguous probes; all rows in exact mode or small shards."""
        if self.selection_mode == "exact":
            return [(0, shard.rows)]
        width = max(32, int(math.ceil(self.max_samples * 2 / (8 * len(self._shards)))))
        width = min(shard.rows, int(math.ceil(width / 32)) * 32)
        starts = np.linspace(0, max(0, shard.rows - width), 8, dtype=np.int64)
        starts[:-1] = starts[:-1] // 32 * 32
        merged = []
        for start in np.unique(starts):
            start, stop = int(start), min(int(start) + width, shard.rows)
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
            else:
                merged.append((start, stop))
        return merged

    def _selection_metadata(self, shard, blocks):
        with h5py.File(shard.path, "r") as h:
            source = np.concatenate([h["source_id"][a:b] for a,b in blocks]).astype(np.int64)
            snr = np.concatenate([h["snr_target_db"][a:b] for a,b in blocks]) if "snr_target_db" in h else np.full(len(source),np.nan)
            clean = np.concatenate([h["is_clean"][a:b] for a,b in blocks]) if "is_clean" in h else np.zeros(len(source),dtype=np.uint8)
        return source, snr, clean

    def _select(self):
        if self.snr_db is None and self.source_names is None and self.max_samples is None:
            self.filtered_rows = self.unfiltered_rows
            return
        source_ids = None if self.source_names is None else [SOURCE_NAMES.index(n) for n in self.source_names]
        strata = defaultdict(list)
        candidate_count = 0
        for shard in self._shards:
            blocks = self._metadata_blocks(shard)
            row_numbers = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in blocks])
            candidate_count += len(row_numbers)
            source, snr, is_clean = self._selection_metadata(shard, blocks)
            keep = np.ones(len(row_numbers), dtype=bool)
            if source_ids is not None:
                keep &= np.isin(source, source_ids)
            if self.snr_db is not None and shard.kind != "control":
                keep &= np.any(np.isclose(snr[:, None], np.asarray(self.snr_db)[None, :], atol=1e-4, rtol=0), axis=1)
            selected_positions = np.flatnonzero(keep)
            selected = row_numbers[selected_positions]
            shard.selected = None if len(selected) == shard.rows else selected
            if self.max_samples is not None:
                for sid in np.unique(source[selected_positions]):
                    source_rows = selected_positions[source[selected_positions] == sid]
                    # SNR values are few and fixed; -999 denotes no applicable SNR.
                    snr_codes = np.where(np.isfinite(snr[source_rows]), np.rint(snr[source_rows] * 1000), -999).astype(np.int64)
                    for code in np.unique(snr_codes):
                        local = source_rows[snr_codes == code]
                        for flag in np.unique(is_clean[local]):
                            local_flag = local[is_clean[local] == flag]
                            key = (int(sid), shard.scheme, int(code), int(flag))
                            strata[key].append(row_numbers[local_flag] + shard.raw_offset)
        self.candidate_pool_rows = candidate_count
        self.filtered_rows = sum(s.count for s in self._shards)
        if (self.selection_mode == "probe" and (self.source_names is not None or self.snr_db is not None)
                and self.filtered_rows < min(self.max_samples, self.unfiltered_rows)):
            raise ValueError(f"Probe candidate filters leave {self.filtered_rows} rows, below requested budget "
                             f"{self.max_samples}; use selection_mode='exact' or reduce max_samples. "
                             "Probe counts are not whole-dataset source/SNR counts.")
        if self.max_samples is None or self.max_samples >= self.filtered_rows:
            return
        keys = sorted(strata)
        counts = np.array([sum(len(a) for a in strata[k]) for k in keys], dtype=np.int64)
        budget = self.max_samples
        quotas = np.zeros(len(keys), dtype=np.int64)
        rng = np.random.default_rng(self.selection_seed)
        if budget < len(keys):
            quotas[rng.permutation(len(keys))[:budget]] = 1
        else:
            quotas[:] = 1
            remaining = budget - len(keys)
            capacity = counts - 1
            weights = remaining * capacity / max(1, int(capacity.sum()))
            extra = np.floor(weights).astype(np.int64)
            quotas += extra
            residual = budget - int(quotas.sum())
            # Stable random tie breaks avoid always favoring the first source.
            tie = rng.random(len(keys))
            order = np.lexsort((tie, -(weights - extra)))
            for i in order:
                if residual == 0:
                    break
                if quotas[i] < counts[i]:
                    quotas[i] += 1
                    residual -= 1
        selected_global = []
        for key, quota in zip(keys, quotas):
            if quota == 0:
                continue
            values = np.concatenate(strata[key])
            if quota < len(values):
                priorities = _hash_values(values, self.selection_seed)
                values = values[np.argpartition(priorities, int(quota) - 1)[:int(quota)]]
            selected_global.append(values)
        selected_global = np.sort(np.concatenate(selected_global))
        self._strata_summary = [{"source": SOURCE_NAMES[k[0]] if 0 <= k[0] < len(SOURCE_NAMES) else str(k[0]),
                                  "scheme": k[1], "snr_db": None if k[2] == -999 else k[2] / 1000,
                                  "is_clean": k[3], "available": int(c), "selected": int(q)}
                                 for k, c, q in zip(keys, counts, quotas)]
        for shard in self._shards:
            a = np.searchsorted(selected_global, shard.raw_offset)
            b = np.searchsorted(selected_global, shard.raw_offset + shard.rows)
            shard.selected = (selected_global[a:b] - shard.raw_offset).astype(np.int64)

    def __len__(self):
        return int(self._prefix[-1])

    def _locate(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        file_id = bisect_right(self._prefix, index) - 1
        shard = self._shards[file_id]
        local = index - int(self._prefix[file_id])
        return file_id, local if shard.selected is None else int(shard.selected[local])

    def _file(self, file_id):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if file_id in self._handles:
            self._handles.move_to_end(file_id)
            return self._handles[file_id]
        h = h5py.File(self._shards[file_id].path, "r", rdcc_nbytes=8 * 1024 * 1024, rdcc_nslots=1009)
        self._handles[file_id] = h
        while len(self._handles) > self.cache_files:
            self._handles.popitem(last=False)[1].close()
        return h

    def _fields(self, h, metadata_only=False):
        fields = set(_INT_META) | set(_FLOAT_META) | {"gap_start", "snr_per_channel_db"}
        if not metadata_only:
            fields.update(("input", "target", "scale", "observed_mask", "loss_mask", "noise_mask"))
            if self.task in CLEAN_TASKS and self.normalization == "visible":
                # Do not decompress the unused legacy waveforms as well.
                fields.difference_update(("input", "target"))
                fields.update(("input_visible", "target_visible", "visible_center", "visible_scale"))
            if self.task == "frequency_denoising":
                fields.update(("target_low", "target_high"))
        return sorted(fields & set(h))

    def _sample(self, file_id, row, values, metadata_only=False):
        shard = self._shards[file_id]
        result = {k: np.int64(values.get(k, v)) for k, v in _INT_META.items()}
        result.update({k: np.float32(values.get(k, v)) for k, v in _FLOAT_META.items()})
        result["gap_start"] = np.asarray(values.get("gap_start", np.full(4, -1)), dtype=np.int64)
        result["snr_per_channel_db"] = np.asarray(values.get("snr_per_channel_db", np.full(4, np.nan)), dtype=np.float32)
        result.update(scheme=shard.scheme, shard=shard.relative, row=np.int64(row),
                      sample_uid=f"{self.dataset_id}/{shard.relative}/{row}")
        if metadata_only:
            return result
        shape = (4, self.sequence_length)
        clean = self.task in CLEAN_TASKS
        visible = clean and self.normalization == "visible"
        result["input"] = np.asarray(values["input_visible" if visible else "input"], dtype=np.float32)
        result["target"] = np.asarray(values["target_visible" if visible else "target"], dtype=np.float32)
        for key in ("observed_mask", "loss_mask"):
            result[key] = np.asarray(values.get(key, np.ones(shape)), dtype=np.float32)
        result["noise_mask"] = np.asarray(values.get("noise_mask", np.zeros(shape)), dtype=np.float32)
        result["scale"] = np.asarray(values.get("scale", np.ones((4, 1))), dtype=np.float32)
        result["visible_center"] = np.asarray(values.get("visible_center", np.zeros((4, 1))) if visible else np.zeros((4, 1)), dtype=np.float32)
        result["visible_scale"] = np.asarray(values.get("visible_scale", np.ones((4, 1))) if visible else np.ones((4, 1)), dtype=np.float32)
        if self.task == "frequency_denoising":
            result["target_low"] = np.asarray(values["target_low"], dtype=np.float32)
            result["target_high"] = np.asarray(values["target_high"], dtype=np.float32)
        return result

    def __getitem__(self, index):
        file_id, row = self._locate(index)
        h = self._file(file_id)
        return self._sample(file_id, row, {k: h[k][row] for k in self._fields(h)})

    def __getitems__(self, indices):
        """Batch read each field once per shard, restoring requested order/duplicates."""
        locations = [self._locate(i) for i in indices]
        groups = defaultdict(list)
        for output, (file_id, row) in enumerate(locations):
            groups[file_id].append((output, row))
        results = [None] * len(locations)
        for file_id, rows in groups.items():
            h = self._file(file_id)
            unique, inverse = np.unique([r for _, r in rows], return_inverse=True)
            selection = slice(int(unique[0]), int(unique[-1]) + 1) if len(unique) == int(unique[-1] - unique[0] + 1) else unique
            arrays = {k: h[k][selection] for k in self._fields(h)}
            for (output, row), value_index in zip(rows, inverse):
                results[output] = self._sample(file_id, row, {k: v[value_index] for k, v in arrays.items()})
        return results

    def get_metadata(self, index):
        file_id, row = self._locate(index)
        h = self._file(file_id)
        return self._sample(file_id, row, {k: h[k][row] for k in self._fields(h, True)}, True)

    def describe(self):
        return {"reader": "direct_h5_shards", "dataset_id": self.dataset_id,
                "task": self.task, "split": self.split, "scheme": self.scheme,
                "normalization": self.normalization, "include_controls": self.include_controls,
                "rows": len(self), "unfiltered_rows": self.unfiltered_rows,
                "filtered_rows_before_budget": self.filtered_rows,
                "selection_mode": self.selection_mode, "candidate_pool_rows": self.candidate_pool_rows,
                "stratum_available_scope": "whole dataset" if self.selection_mode == "exact" else "contiguous metadata probes only; not global proportions",
                "max_samples": self.max_samples, "selection_seed": self.selection_seed,
                "snr_db": self.snr_db, "source_names": self.source_names,
                "sequence_length": self.sequence_length, "sampling_rate_hz": self.sampling_rate_hz,
                "strata": self._strata_summary,
                "shards": [{"path": s.relative, "rows": s.rows, "selected_rows": s.count,
                            "scheme": s.scheme, "kind": s.kind} for s in self._shards]}

    def fingerprints(self):
        """Protocol/header identity; this is not a full waveform-content checksum."""
        digest = hashlib.sha256()
        digest.update(json.dumps({"dataset_id": self.dataset_id, "task": self.task, "split": self.split,
                                  "scheme": self.scheme, "normalization": self.normalization,
                                  "selection_mode": self.selection_mode,
                                  "snr_db": self.snr_db, "source_names": self.source_names,
                                  "selection_seed": self.selection_seed, "max_samples": self.max_samples},
                                 sort_keys=True).encode())
        for s in self._shards:
            digest.update(json.dumps([s.relative, s.rows, s.signature, s.file_bytes]).encode())
            digest.update(b"all" if s.selected is None else s.selected.astype("<i8", copy=False).tobytes())
        return {"dataset_id": self.dataset_id, "selection_sha256": digest.hexdigest(),
                "scope": "HDF5 headers, paths, sizes, signatures and exact selected row indices"}

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["_pid"] = None
        return state

    def __del__(self):
        if hasattr(self, "_handles"):
            self.close()


def model_input(sample, task):
    """Four noisy channels, or four clean channels plus their visibility masks.

    Accepts either a single [4,T] sample or a collated [B,4,T] batch. Labels,
    clean references, true noise masks and noise types never enter this helper.
    PyTorch is only imported lazily if this function receives torch tensors.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}")
    x = sample["input"]
    if task in NOISE_TASKS:
        return x
    if isinstance(x, np.ndarray):
        return np.concatenate((x, sample["observed_mask"]), axis=-2)
    import torch
    return torch.cat((x, sample["observed_mask"].to(dtype=x.dtype)), dim=-2)


class BlockShuffleSampler:
    """Shuffle contiguous blocks, preserving efficient HDF5 reads within blocks.

    A full epoch visits each selected row exactly once. A budget smaller than
    the dataset visits a reproducible sample of whole blocks plus the needed
    prefix of the final block. Budgets cannot exceed the selected dataset.
    Keep DataLoader ``shuffle=False`` when passing this sampler.
    """
    def __init__(self, dataset, seed=2025, block_size=256, shuffle=True, samples_per_epoch=None):
        self.dataset = dataset
        self.seed = int(seed)
        self.block_size = int(block_size)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        self.samples_per_epoch = len(dataset) if samples_per_epoch is None else int(samples_per_epoch)
        if self.samples_per_epoch < 1 or self.samples_per_epoch > len(dataset):
            raise ValueError("samples_per_epoch must be between 1 and the selected dataset length")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        blocks = np.arange((len(self.dataset) + self.block_size - 1) // self.block_size)
        if self.shuffle:
            np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch])).shuffle(blocks)
        remaining = self.samples_per_epoch
        for block in blocks:
            start = int(block) * self.block_size
            stop = min(start + self.block_size, len(self.dataset), start + remaining)
            yield from range(start, stop)
            remaining -= stop - start
            if remaining == 0:
                break

# Publication adapter: the shared core can replay fixed epoch-0 rows without
# downloading the much larger historical offline materialization.
OfflineH5Dataset = H5Dataset

def H5Dataset(root, *args, **kwargs):
    root = Path(root).expanduser().resolve()
    if (root / "manifest.json").is_file() and not (root / "offline/noise_tasks").is_dir() and root.name.lower() != "offline":
        from publication_reader import PortableDataset
        return PortableDataset(root, *args, base_module=__import__(__name__, fromlist=["x"]), **kwargs)
    return OfflineH5Dataset(root, *args, **kwargs)
