"""Matched frozen-row controls for noise and crop-jitter ablations.

The same source rows, labels, record identities and optimization-step count are
kept in every mode. No random noise is drawn and no source array is modified.
Only the training partition is supported; CV must use its original clean rows.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .data import CHANNELS


MATCHED_MODES = ("matched_clean", "matched_noise", "matched_jitter", "matched_both")


def _statistics(raw):
    """Float32 source crop, float64 population statistics and normalized result."""
    raw = np.asarray(raw, dtype=np.float32)
    mean = raw.mean(axis=-1, keepdims=True, dtype=np.float64)
    std = raw.std(axis=-1, keepdims=True, dtype=np.float64, ddof=0)
    scale = np.where(std > 0, std, 1.0)
    return (raw - mean) / scale, mean, scale


def _h5_rows(array, indices):
    """h5py requires sorted distinct fancy indices; restore arbitrary requests."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    unique, inverse = np.unique(indices, return_inverse=True)
    return array[unique.tolist()][inverse]


class MatchedWindowDataset:
    """Wrap a WindowDataset without dropping or reordering any frozen rows.

    Clean rows are unchanged in all modes. On augmented rows:
      matched_clean: exact prepared parent input, duplicated as a step control;
      matched_noise: unshifted parent + the row's frozen normalized noise draw;
      matched_jitter: the row's clean shifted crop, without noise;
      matched_both: exact original prepared input.

    Normalized noise is frozen_H5_x_aug - zscore(own_H5_sx_aug), calculated in
    float64. Raw noise-only inputs restore that draw using PARENT clean std.
    Unselected channels are zeroed only after the four-channel calculation.
    """

    def __init__(self, base, mode):
        if mode not in MATCHED_MODES:
            raise ValueError(f"mode must be one of {MATCHED_MODES}")
        self.base, self.mode = base, mode
        self.matched_mode = mode
        self.bundle = base.bundle
        self.root = Path(getattr(base, "root", self.bundle.root)).resolve()
        self.channels = base.channels
        self.variant = self.bundle.variant
        if self.variant not in {"standardized", "raw"}:
            raise ValueError("Matched controls require standardized or raw inputs")
        self.row_indices = base.row_indices
        self.indices = self.row_indices
        self.y, self.record_ids = base.y, base.record_ids
        self.sites, self.regions, self.shape = base.sites, base.regions, base.shape
        self._h5, self._h5_pid = None, None
        self.metadata = []
        for original in base.metadata:
            row = dict(original)
            if row["partition"] != "train":
                raise ValueError("MatchedWindowDataset is training-only; do not wrap CV/validation/test")
            augmented = bool(row["transform"])
            parent_global = int(row["parent_row_index"]) if augmented else -1
            parent = self.bundle.rows[parent_global] if augmented else None
            if augmented and (parent["partition"] != "train" or parent["transform"] != 0 or
                              parent["record_id"] != row["record_id"] or parent["y"] != row["y"]):
                raise ValueError("An augmented row needs a clean same-record/same-label train parent")
            use_parent = augmented and mode in {"matched_clean", "matched_noise"}
            row.update(matched_mode=mode,
                effective_start=int(parent["start"] if use_parent else row["start"]),
                matched_parent_index=int(parent["local_index"]) if augmented else -1,
                matched_parent_row_index=parent_global,
                noise_source_index=int(row["local_index"]) if augmented and mode in {"matched_noise", "matched_both"} else -1,
                noise_source_partition="train" if augmented and mode in {"matched_noise", "matched_both"} else "none")
            self.metadata.append(row)

    def __len__(self):
        return len(self.base)

    def _local_indices(self, values):
        if isinstance(values, slice):
            return np.arange(len(self), dtype=np.int64)[values]
        values = np.asarray(values, dtype=np.int64).reshape(-1)
        if np.any((values < -len(self)) | (values >= len(self))):
            raise IndexError("Matched dataset index out of range")
        return np.where(values < 0, values + len(self), values)

    def _file(self):
        if self._h5 is None or self._h5_pid != os.getpid():
            self._close_h5()
            try:
                import h5py
            except ImportError as error:
                raise ImportError("Matched noise/jitter controls need h5py; see requirements-matched.txt") from error
            path = self.root / "train/windows.h5"
            if not path.is_file():
                raise FileNotFoundError(f"Matched controls need the frozen training archive: {path}")
            self._h5 = h5py.File(path, "r")
            self._h5_pid = os.getpid()
            expected = sum(row["partition"] == "train" for row in self.bundle.rows)
            shape = (expected, 4, self.bundle.window_length)
            if any(self._h5[name].shape != shape or self._h5[name].dtype != np.dtype("float32")
                   for name in ("sx", "x")):
                self._close_h5()
                raise ValueError("Training HDF5 sx/x headers disagree with the frozen prepared rows")
        return self._h5

    def get_batch(self, local_indices):
        local_indices = self._local_indices(local_indices)
        output = self.base.get_batch(local_indices)
        if self.mode == "matched_both" or not len(local_indices):
            return output
        positions = np.asarray([position for position, local in enumerate(local_indices)
                                if self.metadata[int(local)]["transform"]], dtype=np.int64)
        if not len(positions):
            return output
        rows = [self.metadata[int(local_indices[position])] for position in positions]
        parent_rows = [self.bundle.rows[int(row["parent_row_index"])] for row in rows]
        parent_indices = np.asarray([row["local_index"] for row in parent_rows], dtype=np.int64)
        if self.mode == "matched_clean":
            # Exact float32 copies, including the frozen parent's rounding.
            output[positions] = self.bundle.arrays["train"][parent_indices]
        else:
            handle = self._file()
            own_indices = np.asarray([row["local_index"] for row in rows], dtype=np.int64)
            own_raw = _h5_rows(handle["sx"], own_indices)
            own_z, own_mean, own_scale = _statistics(own_raw)
            frozen_x = _h5_rows(handle["x"], own_indices).astype(np.float64)
            expected_prepared = (frozen_x.astype(np.float32) if self.variant == "standardized" else
                                 (frozen_x * own_scale + own_mean).astype(np.float32))
            if not np.array_equal(expected_prepared, self.bundle.arrays["train"][own_indices]):
                raise ValueError("Training HDF5 frozen rows disagree with the prepared input version")
            if self.mode == "matched_jitter":
                values = own_z if self.variant == "standardized" else own_raw
            else:
                noise_unit = frozen_x - own_z
                parent_raw = _h5_rows(handle["sx"], parent_indices)
                parent_z, _, parent_scale = _statistics(parent_raw)
                values = (parent_z + noise_unit if self.variant == "standardized" else
                          parent_raw.astype(np.float64) + parent_scale * noise_unit)
            output[positions] = np.asarray(values, dtype=np.float32)
        for channel, name in enumerate(CHANNELS):
            if name not in self.channels:
                output[:, channel] = 0
        return output

    def __getitem__(self, index):
        local = int(self._local_indices([index])[0])
        return self.get_batch([local])[0], int(self.y[local]), int(self.row_indices[local])

    def iter_batches(self, batch_size=1024):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for lo in range(0, len(self), batch_size):
            hi = min(len(self), lo + batch_size)
            yield self.get_batch(np.arange(lo, hi)), self.y[lo:hi], self.row_indices[lo:hi]

    def subset_local(self, local_indices):
        """Keep the matched transformation when capping or sampling local rows."""
        local_indices = self._local_indices(local_indices)
        return MatchedWindowDataset(self.bundle.subset(self.row_indices[local_indices]), self.mode)

    def _close_h5(self):
        if self._h5 is not None:
            self._h5.close()
        self._h5, self._h5_pid = None, None

    def close(self):
        self._close_h5()
        self.base.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_h5=None, _h5_pid=None)
        return state

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self._close_h5()
        except Exception:
            pass
