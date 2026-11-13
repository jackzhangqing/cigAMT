"""Read-only, record-aware data protocols for the frozen sferic dataset."""
from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from .h5_release import open_input, close_input
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


CHANNELS = ("Ex", "Ey", "Hx", "Hy")
PARTITION_DIRS = {"train": Path("train"), "val": Path("val"), "test_labeled": Path("test/labeled")}
AUGMENTATIONS = ("all", "clean", "noise_only", "jitter_noise")


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _channels(selection):
    aliases = {"all": CHANNELS, "electric": CHANNELS[:2], "magnetic": CHANNELS[2:],
               "exey": CHANNELS[:2], "hxhy": CHANNELS[2:]}
    if isinstance(selection, str):
        values = aliases.get(selection.lower())
        if values is None:
            values = tuple(item.strip() for item in selection.split(",") if item.strip())
    else:
        values = tuple(selection)
    canonical = {name.lower(): name for name in CHANNELS}
    try:
        values = tuple(canonical[name.lower()] for name in values)
    except (KeyError, AttributeError) as error:
        raise ValueError(f"Channels must come from {CHANNELS}") from error
    if not values or len(set(values)) != len(values):
        raise ValueError("Choose at least one channel, without duplicates")
    return tuple(name for name in CHANNELS if name in values)


def _source_groups(record):
    """Only accept explicit metadata or consistent site tokens in every alias."""
    region = (record.get("region") or "").strip()
    site = (record.get("site") or record.get("site_id") or "").strip()
    if region and site:
        return region, f"{region}/{site}", "explicit metadata region/site"
    aliases = [value.strip() for value in record.get("source_names", "").split("|") if value.strip()]
    parsed = []
    for alias in aliases:
        basename = PurePosixPath(alias.replace("\\", "/")).name
        match = re.match(r"^sx_(.+?)_(site[A-Za-z0-9-]+)_", basename)
        if not match:
            return "", "", "unresolved: source aliases lack an unambiguous region/site token"
        parsed.append(match.groups())
    if not parsed or len(set(parsed)) != 1:
        return "", "", "unresolved: source aliases disagree or are absent"
    parsed_region, parsed_site = parsed[0]
    if (region and region != parsed_region) or (site and site != parsed_site):
        return "", "", "unresolved: explicit metadata conflicts with source aliases"
    return parsed_region, f"{parsed_region}/{parsed_site}", "consistent region/site tokens in every source alias"


@dataclass
class SplitData:
    train: "WindowDataset"
    val: "WindowDataset"
    test: "WindowDataset"
    protocol: dict


class WindowDataset:
    """PyTorch-compatible triples; channel ablation masks rather than drops rows.

    The row index is a stable global index across train, val, test/labeled.
    Original memory maps are never modified. Returned windows are writable
    copies, suitable for a default DataLoader without read-only-array warnings.
    """

    def __init__(self, bundle, indices):
        self.bundle = bundle
        self.channels = bundle.channels
        self.row_indices = np.asarray(indices, dtype=np.int64)
        self.indices = self.row_indices
        if len(set(self.row_indices.tolist())) != len(self.row_indices):
            raise ValueError("A dataset subset must not duplicate frozen rows")
        self.metadata = [dict(bundle.rows[int(i)], channels=list(bundle.channels)) for i in self.row_indices]
        self.y = np.asarray([row["y"] for row in self.metadata], dtype=np.int64)
        self.record_ids = np.asarray([row["record_id"] for row in self.metadata], dtype=object)
        self.regions = np.asarray([row["region"] for row in self.metadata], dtype=object)
        self.sites = np.asarray([row["site"] for row in self.metadata], dtype=object)
        self.shape = (len(self), 4, bundle.window_length)

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, index):
        index = int(index)
        x = self.get_batch([index])[0]
        return x, int(self.y[index]), int(self.row_indices[index])

    def get_batch(self, local_indices):
        if isinstance(local_indices, slice):
            local_indices = np.arange(len(self))[local_indices]
        local_indices = np.asarray(local_indices, dtype=np.int64).reshape(-1)
        rows = self.row_indices[local_indices]
        output = np.empty((len(rows), 4, self.bundle.window_length), dtype=np.float32)
        for partition, mapping in self.bundle.arrays.items():
            positions = [i for i, row in enumerate(rows) if self.bundle.rows[int(row)]["partition"] == partition]
            if positions:
                source_rows = [self.bundle.rows[int(rows[i])]["local_index"] for i in positions]
                output[positions] = mapping[source_rows]
        for channel, name in enumerate(CHANNELS):
            if name not in self.bundle.channels:
                output[:, channel] = 0
        return output

    def iter_batches(self, batch_size=1024):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for lo in range(0, len(self), batch_size):
            hi = min(lo + batch_size, len(self))
            yield self.get_batch(np.arange(lo, hi)), self.y[lo:hi], self.row_indices[lo:hi]

    def close(self):
        self.bundle.close()


class BenchmarkData:
    def __init__(self, root, variant="standardized", channels="all"):
        self.root = Path(root).resolve()
        if variant not in {"standardized", "raw"}:
            raise ValueError("variant must be standardized or raw")
        self.variant, self.channels = variant, _channels(channels)
        global_records = self.root / "metadata/records.csv"
        if not all((self.root / relative / 'records.csv').exists()
                   for relative in PARTITION_DIRS.values()) and global_records.exists():
            records = _csv(global_records)
        else:
            records = []
            for partition, relative in PARTITION_DIRS.items():
                for record in _csv(self.root / relative / "records.csv"):
                    record["partition"] = partition
                    records.append(record)
        record_metadata = {row["record_id"]: row for row in records}
        self.rows, self._arrays, self._paths = [], {}, {}
        self._pid = os.getpid()
        self.window_length = None
        for partition, relative in PARTITION_DIRS.items():
            directory = self.root / relative
            input_path = directory / ("X.npy" if variant == "standardized" else "X_raw.npy")
            # Prefer the published HDF5 representation even when local NPY
            # convenience exports are also present.
            if variant == 'standardized' and (directory / 'windows.h5').exists():
                input_path = directory / 'windows.h5'
            elif not input_path.exists():
                input_path = directory / "windows.h5"
            mapping = open_input(input_path, variant)
            if input_path.suffix == ".h5":
                import h5py
                with h5py.File(input_path, "r") as h5:
                    labels = np.asarray(h5["y"], dtype=np.int64)
            else:
                labels = np.load(directory / "y.npy", mmap_mode="r", allow_pickle=False)
            manifest = _csv(directory / "windows.csv")
            if mapping.ndim != 3 or mapping.shape[1] != 4 or mapping.dtype != np.float32:
                raise ValueError(f"Invalid input tensor in {partition}: {mapping.shape}/{mapping.dtype}")
            if labels.dtype != np.int64 or labels.shape != (len(mapping),) or len(manifest) != len(mapping):
                raise ValueError(f"Input/label/window manifest mismatch: {partition}")
            if self.window_length is not None and mapping.shape[2] != self.window_length:
                raise ValueError("Window lengths differ across partitions")
            self.window_length = mapping.shape[2]
            offset = len(self.rows)
            for i, row in enumerate(manifest):
                record = record_metadata.get(row["record_id"])
                if record is None or record.get("partition") != partition:
                    raise ValueError(f"Record belongs to the wrong partition: {row['record_id']}")
                y, transform, parent = int(row["y"]), int(row["transform"]), int(row["parent_index"])
                if int(row["index"]) != i or y != labels[i] or y not in {0, 1} or transform not in {0, 1}:
                    raise ValueError(f"Invalid window metadata: {partition}/{i}")
                if transform == 0 and parent != -1:
                    raise ValueError("Clean windows must have parent_index=-1")
                if transform and (not 0 <= parent < i or int(manifest[parent]["transform"]) != 0
                                  or manifest[parent]["record_id"] != row["record_id"]
                                  or int(manifest[parent]["y"]) != y):
                    raise ValueError("Augmentation parent must be an earlier clean row of the same record/class")
                region, site, provenance = _source_groups(record)
                self.rows.append(dict(row_index=offset + i, local_index=i, partition=partition,
                    record_id=row["record_id"], region=region, site=site, group_provenance=provenance,
                    y=y, transform=transform, parent_index=parent,
                    parent_row_index=offset + parent if parent >= 0 else -1,
                    start=int(row["start"]), crop_shift_samples=int(row["crop_shift_samples"])))
            self._arrays[partition] = mapping
            self._paths[partition] = input_path
            close_input(labels)
        self.all_indices = np.arange(len(self.rows), dtype=np.int64)

    @property
    def arrays(self):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if not self._arrays:
            self._arrays = {partition: open_input(path, self.variant)
                            for partition, path in self._paths.items()}
        return self._arrays

    def close(self):
        for mapping in self._arrays.values():
            close_input(mapping)
        self._arrays = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_arrays={}, _pid=None)
        return state

    def subset(self, indices):
        return WindowDataset(self, indices)

    def fixed_split(self, augmentation="all", train_fraction=1.0, seed=0):
        if augmentation not in AUGMENTATIONS or not 0 < train_fraction <= 1:
            raise ValueError("Invalid augmentation or train_fraction")
        train_records = sorted({row["record_id"] for row in self.rows if row["partition"] == "train"})
        keep_count = max(1, math.ceil(len(train_records) * train_fraction))
        chosen = set(np.random.default_rng(seed).permutation(train_records)[:keep_count])
        train, val, test = [], [], []
        for row in self.rows:
            if row["partition"] == "train" and row["record_id"] in chosen:
                include = row["transform"] == 0 or augmentation == "all"
                if row["transform"]:
                    include |= augmentation == "noise_only" and row["crop_shift_samples"] == 0
                    include |= augmentation == "jitter_noise" and row["crop_shift_samples"] != 0
                if include:
                    train.append(row["row_index"])
            elif row["partition"] == "val" and row["transform"] == 0:
                val.append(row["row_index"])
            elif row["partition"] == "test_labeled" and row["transform"] == 0:
                test.append(row["row_index"])
        protocol = dict(name="fixed", variant=self.variant, channels=list(self.channels),
            channel_policy="Unselected channels are zeroed; shape remains N,4,L",
            augmentation=augmentation, train_fraction=train_fraction, fraction_unit="whole training records",
            selected_train_records=sorted(chosen), seed=seed,
            grouping_provenance=sorted({row["group_provenance"] for row in self.rows}),
            augmentation_semantics={"all": "clean plus all frozen augmented rows", "clean": "clean rows only",
                "noise_only": "clean plus augmented rows with zero actual crop shift",
                "jitter_noise": "clean plus augmented rows with nonzero actual crop shift"},
            warnings=["noise_only/jitter_noise are subsets of frozen augmentation, not a fully decoupled causal ablation; class/record counts can differ"])
        return SplitData(self.subset(train), self.subset(val), self.subset(test), protocol)

    def group_folds(self, group_by="record", n_splits=5, seed=0, val_fraction=0.2, val_group_by="record"):
        fields = {"record": "record_id", "site": "site", "region": "region"}
        if group_by not in fields or val_group_by not in fields or not 0 < val_fraction < 1:
            raise ValueError("Invalid grouping or validation fraction")
        clean = np.asarray([row["row_index"] for row in self.rows if row["transform"] == 0], dtype=np.int64)
        labels = np.asarray([self.rows[i]["y"] for i in clean], dtype=np.int64)
        outer = np.asarray([self.rows[i][fields[group_by]] for i in clean], dtype=object)
        if not all(outer):
            raise ValueError(f"Reliable {group_by} grouping is unavailable; provide explicit metadata, no fallback is used")
        group_count = len(set(outer))
        if not 2 <= n_splits <= group_count:
            raise ValueError(f"Requested {n_splits} folds but only {group_count} reliable {group_by} groups exist; choose 2..{group_count}")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        result = []
        for fold, (pool, heldout) in enumerate(splitter.split(np.zeros(len(clean)), labels, outer)):
            inner = np.asarray([self.rows[i][fields[val_group_by]] for i in clean[pool]], dtype=object)
            if not all(inner) or len(set(inner)) < 2:
                raise ValueError(f"Fold {fold}: insufficient reliable {val_group_by} groups for train/validation; no window-random fallback")
            inner_splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed + fold)
            training, validation = next(inner_splitter.split(np.zeros(len(pool)), labels[pool], inner))
            train_ids, val_ids, test_ids = clean[pool[training]], clean[pool[validation]], clean[heldout]
            for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
                if len(set(self.rows[i]["y"] for i in ids)) < 2:
                    raise ValueError(f"Fold {fold}: {name} has one class; change group protocol, no window-level fallback")
            warning = ["Cross-validation uses only clean rows from all originally labeled partitions; no frozen or online augmentation"]
            if len(set(inner)) <= 2:
                warning.append("Only two training-pool groups remain: train and validation each contain one group; estimates are fragile")
            protocol = dict(name="group_cv", fold=fold, n_splits=n_splits, seed=seed,
                outer_test_group_by=group_by, validation_group_by=val_group_by, val_fraction=val_fraction,
                augmentation="clean", variant=self.variant, channels=list(self.channels),
                test_groups=sorted(set(outer[heldout])), train_groups=sorted(set(outer[pool[training]])),
                val_groups=sorted(set(outer[pool[validation]])), warnings=warning,
                grouping_provenance=sorted({self.rows[i]["group_provenance"] for i in clean}))
            result.append(SplitData(self.subset(train_ids), self.subset(val_ids), self.subset(test_ids), protocol))
        return result


def load_bundle(dataset_root, variant="standardized", channels="all"):
    return BenchmarkData(dataset_root, variant=variant, channels=channels)
