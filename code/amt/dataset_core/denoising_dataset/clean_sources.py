"""Prepare portable clean-signal banks without importing the training project.

``prepare_clean(root, config, existing_inventory, new_inventory)`` copies legacy
banks, inherits the whole legacy held-out pools, and streams new 150 Hz long
records into paired raw/normalised HDF5 banks. No source file is modified.

All output paths returned in the manifest are relative to ``root``. The data
coordinates are the original stored coordinates, which must not be assumed to
be calibrated SI fields. Generated clean signals are train-only and their
generator-training provenance remains unverified.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Any

import h5py
import numpy as np

CHANNELS = ("Ex", "Ey", "Hx", "Hy")
WINDOW = 2048
STRIDE = 1024
SPLITS = ("train", "validation", "test")
LEGACY_IDS = ("real", "syn_amt_A", "syn_amt_B", "syn_mt", "ddpm", "ddim")
NEW_BANKS = {
    "syn_continuous_A": ("syn_continuous_res", "A", "syn_continuous"),
    "syn_continuous_B": ("syn_continuous_res", "B", "syn_continuous"),
    "syn_3d_A": ("syn_3d_res", "A", "syn_3d"),
    "syn_3d_B": ("syn_3d_res", "B", "syn_3d"),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _copy_verified(source: Path, target: Path) -> dict:
    """Atomic byte copy with source-change checks and SHA-256-verified restart."""
    target.parent.mkdir(parents=True, exist_ok=True)
    marker = target.with_name(target.name + ".copy.json")
    before = _fingerprint(source)
    if marker.exists() and target.exists():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old.get("source_fingerprint") == before and _sha256(target) == old.get("sha256"):
            return old
    temporary = target.with_name(target.name + ".partial")
    digest = hashlib.sha256()
    with source.open("rb") as reader, temporary.open("wb") as writer:
        for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            writer.write(block)
            digest.update(block)
    if _fingerprint(source) != before:
        raise RuntimeError(f"Source changed while copying: {source}")
    source_sha = digest.hexdigest()
    if _sha256(temporary) != source_sha:
        raise RuntimeError(f"Copy checksum mismatch: {source} -> {target}")
    os.replace(temporary, target)
    result = {"source_fingerprint": before, "sha256": source_sha, "bytes": target.stat().st_size}
    _write_json(marker, result)
    return result


def _settings(config: dict) -> dict:
    config = {**config, **config.get("clean", {})}
    split = config.get("split", {})
    seed = int(split.get("seed", config.get("split_seed", config.get("seed", 2025))))
    validation = float(split.get("validation_fraction", config.get("validation_fraction", 0.15)))
    test = float(split.get("test_fraction", config.get("test_fraction", 0.20)))
    if not (0 <= validation < 1 and 0 <= test < 1 and validation + test < 1):
        raise ValueError("Invalid validation/test split fractions")
    return {"seed": seed, "validation_fraction": validation, "test_fraction": test,
            "compression": config.get("compression", "lzf"),
            "chunk_rows": int(config.get("chunk_rows", 128)),
            "window": WINDOW, "stride": STRIDE, "sampling_rate_hz": 150,
            "spike_filter": bool(config.get("spike_filter", False)),
            "inherit_excitation_split": bool(config.get("inherit_excitation_split", True)),
            "normalization": "per_record_per_channel_zscore",
            "module_sha256": _sha256(Path(__file__))}


def _decode(array: np.ndarray) -> list[str]:
    return [str(_jsonable(x)) for x in array.tolist()]


def _assert_splits(split_indices: dict, count: int, groups: list[str] | None = None,
                   require_all: bool = True) -> dict:
    seen: set[int] = set()
    group_owner: dict[str, str] = {}
    report = {}
    for split in SPLITS:
        indices = np.asarray(split_indices.get(split, []), dtype=np.int64)
        if len(indices) != len(set(indices.tolist())):
            raise ValueError(f"Duplicate indices in {split}")
        if np.any(indices < 0) or np.any(indices >= count):
            raise ValueError(f"Out-of-range indices in {split}")
        if seen.intersection(indices.tolist()):
            raise ValueError("Clean split index overlap")
        seen.update(indices.tolist())
        split_groups = set(groups[int(i)] for i in indices) if groups is not None else set()
        for group in split_groups:
            if group in group_owner and group_owner[group] != split:
                raise ValueError(f"Geology/record group crosses splits: {group}")
            group_owner[group] = split
        report[split] = {"count": len(indices), "group_count": len(split_groups),
                         "groups": sorted(split_groups)}
    if require_all and len(seen) != count:
        raise ValueError(f"Unassigned clean rows: {count - len(seen)}")
    report["unassigned_count"] = count - len(seen)
    report["group_overlap_count"] = 0
    return report


def _legacy_source(root: Path, source_id: str, inventory: dict, legacy: dict,
                   settings: dict) -> dict:
    probe = inventory["sources"][source_id]
    source = Path(probe["path"])
    fingerprint = _fingerprint(source)
    for field, probe_field in (("size_bytes", "file_size_bytes"), ("mtime_ns", "file_mtime_ns")):
        if probe_field in probe and int(probe[probe_field]) != fingerprint[field]:
            raise RuntimeError(f"Clean source changed after its energy audit: {source}; re-run the inventory")
    folder = root / "clean" / source_id
    folder.mkdir(parents=True, exist_ok=True)
    norm_path = folder / "clean_normalized.h5"
    copied = _copy_verified(source, norm_path)
    with h5py.File(norm_path, "r") as handle:
        key = "data" if "data" in handle else ("samples" if "samples" in handle else None)
        if key is None:
            candidates = [k for k, value in handle.items()
                          if isinstance(value, h5py.Dataset) and value.ndim == 3 and value.shape[1:] == (4, WINDOW)]
            if len(candidates) != 1:
                raise ValueError(f"Cannot identify clean dataset: {norm_path}")
            key = candidates[0]
        shape = list(handle[key].shape)
        if len(shape) != 3 or shape[1:] != [4, WINDOW]:
            raise ValueError(f"Unexpected clean shape: {norm_path}, {shape}")
        if handle[key].dtype != np.dtype("float32"):
            raise ValueError(f"Expected audited float32 source: {norm_path}")
        if "data_shape" in probe and shape != probe["data_shape"]:
            raise RuntimeError(f"Clean shape changed after energy audit: {source}")
        count = shape[0]
        attrs = {k: _jsonable(v) for k, v in handle.attrs.items()}
        generated = source_id in ("ddpm", "ddim")
        entry = legacy.get("sources", {}).get(source_id, {})
        if generated:
            split_indices = {"train": list(range(count)), "validation": [], "test": []}
            groups = [f"{source_id}:generated:{i}" for i in range(count)]
            split_unit = "generated_train_only"
        else:
            if not entry:
                raise ValueError(f"Missing legacy split for {source_id}")
            split_indices = {"train": entry.get("train_indices", []),
                             "validation": entry.get("validation_indices", []),
                             "test": entry.get("heldout_indices", [])}
            split_unit = entry.get("split_unit", "record_idx")
            if split_unit not in handle:
                raise ValueError(f"Missing provenance column {split_unit} in {norm_path}")
            groups = [f"{entry.get('group_namespace', source_id)}:{value}"
                      for value in _decode(handle[split_unit][:])]
    energy = probe.get("full_energy_scan", {})
    bad = sorted(set(int(i) for field in ("nonfinite_sample_indices",
                                         "any_zero_energy_channel_sample_indices",
                                         "any_constant_channel_sample_indices")
                     for i in energy.get(field, [])))
    if bad:
        bad_set = set(bad)
        split_indices = {name: [int(i) for i in values if int(i) not in bad_set]
                         for name, values in split_indices.items()}
    split_indices = {name: [int(i) for i in values] for name, values in split_indices.items()}
    audit = _assert_splits(split_indices, count, groups, require_all=not bad)
    raw_probe = probe.get("normalization_reconstruction_audit", {}) or {}
    raw_source = Path(raw_probe.get("raw_path", source.parent / "clean_raw.h5"))
    raw_path = None
    raw_copy = None
    if not generated and raw_source.exists():
        raw_path = folder / "clean_raw.h5"
        raw_copy = _copy_verified(raw_source, raw_path)
        with h5py.File(raw_path, "r") as handle:
            if handle["data"].shape != tuple(shape):
                raise ValueError(f"Raw/normalised shape mismatch for {source_id}")
    source_meta = source.parent / "metadata.json"
    metadata = json.loads(source_meta.read_text(encoding="utf-8")) if source_meta.exists() else {}
    previous_count = metadata.get("n_clips_total")
    metadata.update({"source": source_id, "n_clips_total": count,
                     "original_metadata_n_clips_total": previous_count,
                     "metadata_count_corrected": previous_count is not None and previous_count != count,
                     "amplitude_space": "source_stored_coordinates",
                     "calibrated_physical_units_verified": False,
                     "legacy_h5_attrs": attrs,
                     "source_h5_sha256": copied["sha256"],
                     "raw_source_h5_sha256": raw_copy["sha256"] if raw_copy else None,
                     "excluded_invalid_indices": bad,
                     "split_audit": audit})
    if source_meta.exists():
        metadata["original_metadata_sha256"] = _sha256(source_meta)
    if generated:
        metadata["generator_provenance"] = probe.get("generator_provenance", {})
        metadata["generator_training_provenance_status"] = "unverified_training_provenance"
        metadata["generated_signal_split_policy"] = "train_only"
    metadata_path = folder / "metadata.json"
    _write_json(metadata_path, metadata)
    family = "generated" if generated else ("real" if source_id == "real" else
                                             "syn_discrete" if source_id.startswith("syn_amt") else "syn_mt")
    return {"id": source_id, "family": family, "path": _relative(root, norm_path),
            "raw_path": _relative(root, raw_path) if raw_path else None, "key": key,
            "count": count, "usable_count": sum(len(x) for x in split_indices.values()),
            "shape": shape, "dtype": "float32", "sampling_rate_hz": 150,
            "channel_names": list(CHANNELS), "generated": generated,
            "normalization": "generator_output_coordinates" if generated else settings["normalization"],
            "normalization_metadata_path": _relative(root, metadata_path),
            "amplitude_space": "source_stored_coordinates",
            "calibrated_physical_units_verified": False,
            "split_unit": split_unit, "split_policy": "train_only" if generated else "inherited_whole_legacy_test_pool",
            "split_indices": split_indices, "split_audit": audit,
            "excluded_invalid_indices": bad, "file_sha256": copied["sha256"],
            "raw_file_sha256": raw_copy["sha256"] if raw_copy else None,
            "generator_provenance_status": "unverified_training_provenance" if generated else None,
            "source_fingerprint": copied["source_fingerprint"]}


def _h_signature(path: Path) -> str:
    """Exact 150 Hz Hx/Hy stored-coordinate content identity, independent of E."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as handle:
        fs = float(np.asarray(handle["fs"]).item())
        digest.update(struct.pack("<d", fs))
        for key in ("Hx", "Hy"):
            values = np.asarray(handle[key], dtype="<f4")
            digest.update(struct.pack("<Q", values.size))
            digest.update(values.tobytes())
    return digest.hexdigest()


def _excitation_plan(existing_inventory: dict, new_inventory: dict,
                     legacy: dict, settings: dict) -> tuple[dict, dict, dict]:
    """Keep duplicate excitation content on one side of the legacy holdout.

    Equal integer seeds are only candidates: exact whole-record Hx/Hy hashes
    confirm the links, separately for both A/B pipelines. Existing heldout rows
    are never moved into training. For repeated old excitation groups, test has
    priority over validation, which has priority over train.
    """
    priority = {"train": 0, "validation": 1, "test": 2}
    inherited = {}
    audit = {"enabled": settings["inherit_excitation_split"], "old_duplicate_seed_count": 0,
             "matched_old_new_pairs": [], "unmatched_seed_candidates": [],
             "legacy_clip_migrations": {}, "new_3d_cross_parent_H_duplicates": [],
             "new_3d_policy": "one_train_one_test", "status": "not_applicable"}
    if not settings["inherit_excitation_split"]:
        audit["status"] = "disabled_by_configuration"
        return legacy, inherited, audit
    # Verify the two new 3D parents do not share excitation before splitting them.
    if "syn_3d_res" in new_inventory.get("sources", {}):
        signatures = {}
        records = new_inventory["sources"]["syn_3d_res"]["records"]
        for rec in records:
            signature = _h_signature(Path(rec["path"]))
            signatures.setdefault(signature, []).append({"parent": rec["geology_group"],
                                                         "site": rec.get("site_id"), "pipeline": rec["pipeline"]})
        duplicates = [{"HxHy_sha256": sha, "records": rows} for sha, rows in signatures.items()
                      if len({row["parent"] for row in rows}) > 1]
        audit["new_3d_cross_parent_H_duplicates"] = duplicates
        audit["new_3d_H_signature_count"] = len(signatures)
        audit["new_3d_record_count"] = len(records)
        if duplicates:
            audit["new_3d_policy"] = "all_train_only_shared_excitation_across_parents"
    if "syn_amt_A" not in existing_inventory.get("sources", {}) or "syn_amt_A" not in legacy.get("sources", {}):
        return legacy, inherited, audit
    old_bank_path = Path(existing_inventory["sources"]["syn_amt_A"]["path"])
    old_bank_metadata = json.loads((old_bank_path.parent / "metadata.json").read_text(encoding="utf-8"))
    if not old_bank_metadata.get("records"):
        raise ValueError("Legacy discrete metadata lacks original record provenance")
    old_long_root = Path(old_bank_metadata["records"][0]["path"]).parent.parent
    old_model_metadata_path = old_long_root / "metadata.json"
    old_model_metadata = json.loads(old_model_metadata_path.read_text(encoding="utf-8"))
    audit["legacy_geology_metadata_path"] = str(old_model_metadata_path)
    audit["legacy_geology_metadata_sha256"] = _sha256(old_model_metadata_path)
    old_models = old_model_metadata["models"]
    old_seed_models = {}
    for model in old_models:
        old_seed_models.setdefault(int(model["seed"]), []).append(model["model_id"])
    audit["old_geology_model_count"] = len(old_models)
    audit["old_unique_seed_count"] = len(old_seed_models)
    audit["old_duplicate_seed_count"] = sum(len(models) > 1 for models in old_seed_models.values())
    group_owner_sets = {}
    for source_id in ("syn_amt_A", "syn_amt_B"):
        if source_id not in legacy["sources"]:
            continue
        entry = legacy["sources"][source_id]
        for split, key in (("train", "train_groups"), ("validation", "validation_groups"), ("test", "heldout_groups")):
            for group in entry.get(key, []):
                group_owner_sets.setdefault(str(group), set()).add(split)
    def model_group(model_id):
        match = re.match(r"M(\d+)", model_id)
        if not match:
            raise ValueError(f"Cannot map legacy physical model: {model_id}")
        return match.group(1)
    owners = {group: max(values, key=priority.get) for group, values in group_owner_sets.items()}
    new_info = new_inventory.get("sources", {}).get("syn_continuous_res", {})
    new_models = {int(item["seed"]): item for item in new_info.get("source_metadata", {}).get("models", [])}
    if len(new_models) != len(new_info.get("source_metadata", {}).get("models", [])):
        raise ValueError("New continuous source repeats excitation seeds; add a content-group audit before splitting")
    new_paths = {(rec["geology_group"], rec["pipeline"]): Path(rec["path"]) for rec in new_info.get("records", [])}
    cache = {}
    def old_signatures(model_id):
        if model_id not in cache:
            cache[model_id] = tuple(_h_signature(old_long_root / model_id / f"pipeline{pipeline}_low.npz") for pipeline in ("A", "B"))
        return cache[model_id]
    for seed, model_ids in old_seed_models.items():
        if len(model_ids) <= 1 and seed not in new_models:
            continue
        signature_groups = {}
        for model_id in model_ids:
            signature_groups.setdefault(old_signatures(model_id), []).append(model_id)
        # Promote old duplicated excitations to the most restrictive existing split.
        for signatures, matching_models in signature_groups.items():
            roles = [owners[model_group(model_id)] for model_id in matching_models]
            owner = max(roles, key=priority.get)
            for model_id in matching_models:
                owners[model_group(model_id)] = owner
        if seed not in new_models:
            continue
        new_model = new_models[seed]
        new_id = new_model["model_id"]
        if (new_id, "A") not in new_paths or (new_id, "B") not in new_paths:
            raise ValueError(f"Missing new A/B excitation candidate for {new_id}")
        new_signatures = tuple(_h_signature(new_paths[(new_id, pipeline)]) for pipeline in ("A", "B"))
        matched = signature_groups.get(new_signatures, [])
        if matched:
            owner = max((owners[model_group(model_id)] for model_id in matched), key=priority.get)
            inherited[new_id] = owner
            audit["matched_old_new_pairs"].append({"seed": seed, "old_models": matched,
                                                   "new_model": new_id, "inherited_split": owner,
                                                   "HxHy_sha256_by_pipeline": dict(zip(("A", "B"), new_signatures))})
        else:
            audit["unmatched_seed_candidates"].append({"seed": seed, "old_models": model_ids, "new_model": new_id})
    # Only update split index lists; preserve original HDF5 row coordinates.
    amended = {**legacy, "sources": {source_id: dict(entry) for source_id, entry in legacy["sources"].items()}}
    for source_id in ("syn_amt_A", "syn_amt_B"):
        if source_id not in existing_inventory["sources"] or source_id not in amended["sources"]:
            continue
        with h5py.File(existing_inventory["sources"][source_id]["path"], "r") as handle:
            ids = _decode(handle["model_id"][:])
        previous = amended["sources"][source_id]
        old_row_owner = {int(i): split for split, key in (("train", "train_indices"),
                                                         ("validation", "validation_indices"),
                                                         ("test", "heldout_indices")) for i in previous.get(key, [])}
        updated = {split: [] for split in SPLITS}
        migrations = Counter()
        for i, group in enumerate(ids):
            owner = owners[group]
            old_owner = old_row_owner[i]
            if priority[owner] < priority[old_owner]:
                raise RuntimeError("An old heldout clip would be returned to a less restrictive split")
            if owner != old_owner:
                migrations[f"{old_owner}_to_{owner}"] += 1
            updated[owner].append(i)
        for split, index_key, group_key in (("train", "train_indices", "train_groups"),
                                           ("validation", "validation_indices", "validation_groups"),
                                           ("test", "heldout_indices", "heldout_groups")):
            previous[index_key] = updated[split]
            previous[group_key] = sorted({ids[i] for i in updated[split]})
        audit["legacy_clip_migrations"][source_id] = dict(migrations)
    audit["status"] = "exact_Hx_Hy_excitation_links_checked"
    audit["new_continuous_inherited_group_count"] = len(inherited)
    audit["new_continuous_inherited_split_counts"] = dict(Counter(inherited.values()))
    # Geological profile identity is checked independently from excitation identity.
    def profile_hash(model):
        if isinstance(model["rho"], dict):
            return "structured:" + _digest({"rho": model["rho"], "thick": model["thick"]})
        rho = np.asarray(model["rho"], dtype="<f8")
        thick = np.asarray(model["thick"], dtype="<f8")
        return hashlib.sha256(struct.pack("<QQ", rho.size, thick.size) + rho.tobytes() + thick.tobytes()).hexdigest()
    old_profiles = {profile_hash(model) for model in old_models}
    audit["old_new_exact_geology_profile_overlap_count"] = sum(profile_hash(model) in old_profiles for model in new_info.get("source_metadata", {}).get("models", []))
    return amended, inherited, audit


def _assign_new_groups(new_inventory: dict, settings: dict, inherited: dict | None = None,
                       excitation_audit: dict | None = None) -> dict:
    result = {}
    for source, info in new_inventory["sources"].items():
        groups = sorted({rec["geology_group"] for rec in info["records"]})
        rng = np.random.default_rng(settings["seed"])
        ordered = [groups[int(i)] for i in rng.permutation(len(groups))]
        if source == "syn_3d_res":
            if len(groups) != 2:
                raise ValueError("Expected exactly two new 3D-parent geology groups")
            if (excitation_audit or {}).get("new_3d_cross_parent_H_duplicates"):
                result[source] = {group: "train" for group in groups}
            else:
                result[source] = {ordered[0]: "train", ordered[1]: "test"}
        else:
            locked = {group: split for group, split in (inherited or {}).items() if group in groups}
            ordered = [group for group in ordered if group not in locked]
            n_test = int(round(len(groups) * settings["test_fraction"]))
            n_validation = int(round(len(groups) * settings["validation_fraction"]))
            locked_counts = Counter(locked.values())
            n_test = max(0, n_test - locked_counts["test"])
            n_validation = max(0, n_validation - locked_counts["validation"])
            n_test = min(n_test, len(ordered))
            n_validation = min(n_validation, len(ordered) - n_test)
            result[source] = {**locked, **{group: "test" if i < n_test else "validation" if i < n_test + n_validation else "train"
                                          for i, group in enumerate(ordered)}}
    return result


def _record_stats(data: np.ndarray) -> dict:
    work = np.asarray(data, dtype=np.float64)
    center = np.mean(work, axis=1)
    original_scale = np.std(work, axis=1)
    scale = np.where(original_scale < 1e-12, 1.0, original_scale)
    return {"method": "zscore", "normalization_type": "per_record_per_channel_zscore",
            "channel_center": center.tolist(), "channel_scale": scale.tolist(),
            "channel_std_before_fallback": original_scale.tolist(),
            "E_center": center[:2].tolist(), "H_center": center[2:].tolist(),
            "E_scale": float(np.mean(scale[:2])), "H_scale": float(np.mean(scale[2:]))}


def _max_constant_run(clip: np.ndarray) -> int:
    diffs = np.abs(np.diff(clip.astype(np.float64), axis=1)).sum(axis=0)
    threshold = max(1e-12, float(np.abs(clip).max()) * 1e-8)
    mask = diffs < threshold
    changes = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).astype(np.int8)))
    return int(np.max(changes[1::2] - changes[::2], initial=0))


def _clip_quality(clip: np.ndarray, stats: dict, spike_filter: bool = False) -> tuple[bool, str]:
    work = clip.astype(np.float64)
    if not np.isfinite(work).all():
        return False, "nan_or_inf"
    if np.any(np.mean(work ** 2, axis=1) == 0):
        return False, "zero_energy_channel"
    if np.any(np.ptp(work, axis=1) == 0):
        return False, "constant_channel"
    if _max_constant_run(work) > 200:
        return False, "constant_run"
    if spike_filter:
        center = np.asarray(stats["channel_center"])[:, None]
        if np.abs(work[:2] - center[:2]).max() > 30 * stats["E_scale"]:
            return False, "E_spike"
        if np.abs(work[2:] - center[2:]).max() > 30 * stats["H_scale"]:
            return False, "H_spike"
    if work[:2].max() - work[:2].min() < 1e-3 * stats["E_scale"]:
        return False, "E_range_too_small"
    if work[2:].max() - work[2:].min() < 1e-3 * stats["H_scale"]:
        return False, "H_range_too_small"
    return True, "ok"


def _new_h5(path: Path, source_id: str, settings: dict) -> h5py.File:
    handle = h5py.File(path, "w")
    chunk_rows = max(1, settings["chunk_rows"])
    compression = settings["compression"]
    if compression in ("none", "", False):
        compression = None
    handle.create_dataset("data", shape=(0, 4, WINDOW), maxshape=(None, 4, WINDOW),
                          chunks=(chunk_rows, 4, WINDOW), dtype="float32", compression=compression)
    for key, dtype in (("record_idx", "int32"), ("start_idx", "int64")):
        handle.create_dataset(key, shape=(0,), maxshape=(None,), chunks=(max(256, chunk_rows),), dtype=dtype,
                              compression=compression)
    string = h5py.string_dtype(encoding="utf-8")
    for key in ("model_id", "group_id", "site_id", "split"):
        handle.create_dataset(key, shape=(0,), maxshape=(None,), chunks=(max(256, chunk_rows),),
                              dtype=string, compression=compression)
    handle.attrs.update({"source": source_id, "window_size": WINDOW, "stride": STRIDE,
                         "sampling_rate": 150, "channel_names": json.dumps(CHANNELS),
                         "amplitude_space": "source_stored_coordinates", "calibrated_physical_units_verified": False})
    return handle


def _append_h5(handle: h5py.File, clips: np.ndarray, columns: dict) -> tuple[int, int]:
    start = handle["data"].shape[0]
    end = start + clips.shape[0]
    handle["data"].resize(end, axis=0)
    handle["data"][start:end] = clips
    for key, value in columns.items():
        handle[key].resize(end, axis=0)
        handle[key][start:end] = value
    return start, end


def _build_new_source(root: Path, source_id: str, new_inventory: dict,
                      group_assignment: dict, settings: dict) -> dict:
    inventory_id, pipeline, family = NEW_BANKS[source_id]
    info = new_inventory["sources"][inventory_id]
    input_metadata = Path(info["root"]) / "metadata.json"
    if info.get("metadata_sha256") and _sha256(input_metadata) != info["metadata_sha256"]:
        raise RuntimeError(f"New clean metadata changed since inventory: {input_metadata}")
    selected = sorted((rec for rec in info["records"] if rec["pipeline"] == pipeline), key=lambda rec: rec["path"])
    if not selected:
        raise ValueError(f"No records found for {source_id}")
    input_fingerprints = [_fingerprint(Path(rec["path"])) for rec in selected]
    signature = _digest({"settings": settings, "source_id": source_id, "input_fingerprints": input_fingerprints,
                         "inventory_metadata_sha256": info.get("metadata_sha256"),
                         "groups": group_assignment[inventory_id]})
    folder = root / "clean" / source_id
    folder.mkdir(parents=True, exist_ok=True)
    norm_path, raw_path = folder / "clean_normalized.h5", folder / "clean_raw.h5"
    marker_path = folder / "complete.json"
    if marker_path.exists() and norm_path.exists() and raw_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("input_signature") == signature and all(
                _sha256(folder / file) == sha for file, sha in marker.get("file_sha256", {}).items()):
            print(f"CLEAN SKIP {source_id}: verified {marker['manifest_source']['count']} rows", flush=True)
            return marker["manifest_source"]
    norm_partial = norm_path.with_name(norm_path.name + ".partial")
    raw_partial = raw_path.with_name(raw_path.name + ".partial")
    counters = Counter()
    records_table = []
    split_indices = {split: [] for split in SPLITS}
    all_groups = []
    norm_handle = _new_h5(norm_partial, source_id, settings)
    raw_handle = _new_h5(raw_partial, source_id, settings)
    try:
        for record_idx, rec in enumerate(selected):
            source = Path(rec["path"])
            expected = input_fingerprints[record_idx]
            if _fingerprint(source) != expected:
                raise RuntimeError(f"Long record changed before loading: {source}")
            with np.load(source, allow_pickle=False) as npz:
                if any(channel not in npz for channel in CHANNELS):
                    raise ValueError(f"Missing channels: {source}")
                fs = float(np.asarray(npz["fs"]).item())
                if not math.isclose(fs, 150, rel_tol=0, abs_tol=1e-8):
                    raise ValueError(f"Expected 150 Hz: {source}, fs={fs}")
                arrays = [npz[channel] for channel in CHANNELS]
                if any(x.ndim != 1 for x in arrays) or len({x.shape for x in arrays}) != 1:
                    raise ValueError(f"Invalid channel shapes: {source}")
                data = np.stack(arrays).astype(np.float32)
            if rec.get("n_samples") is not None and data.shape[1] != rec["n_samples"]:
                raise RuntimeError(f"Record length changed since header inventory: {source}")
            if rec.get("file_bytes") is not None and expected["size_bytes"] != rec["file_bytes"]:
                raise RuntimeError(f"Record file size changed since header inventory: {source}")
            source_sha256 = _sha256(source)
            if _fingerprint(source) != expected:
                raise RuntimeError(f"Long record changed during loading: {source}")
            group = rec["geology_group"]
            split = group_assignment[inventory_id][group]
            starts = list(range(0, data.shape[1] - WINDOW + 1, STRIDE))
            counters["candidate_windows"] += len(starts)
            record_info = {**rec, "record_idx": record_idx, "source_sha256": source_sha256,
                           "source_fingerprint": expected, "split": split,
                           "stats": None, "n_clips_kept": 0, "candidate_windows": len(starts)}
            if not np.isfinite(data).all():
                counters["record_nan_or_inf"] += len(starts)
                record_info["rejected_reason"] = "record_nan_or_inf"
                records_table.append(record_info)
                continue
            stats = _record_stats(data)
            record_info["stats"] = stats
            if np.any(np.asarray(stats["channel_std_before_fallback"]) == 0):
                counters["record_constant_channel"] += len(starts)
                record_info["rejected_reason"] = "record_constant_channel"
                records_table.append(record_info)
                continue
            kept, kept_starts = [], []
            for start in starts:
                clip = data[:, start:start + WINDOW]
                valid, reason = _clip_quality(clip, stats, settings["spike_filter"])
                counters[reason] += 1
                if valid:
                    kept.append(clip)
                    kept_starts.append(start)
            record_info["n_clips_kept"] = len(kept)
            records_table.append(record_info)
            if kept:
                raw = np.stack(kept)
                center = np.asarray(stats["channel_center"], dtype=np.float64)[None, :, None]
                scale = np.asarray(stats["channel_scale"], dtype=np.float64)[None, :, None]
                normalized = ((raw.astype(np.float64) - center) / scale).astype(np.float32)
                if not np.isfinite(normalized).all():
                    raise ValueError(f"Nonfinite normalized data: {source}")
                n = len(kept)
                columns = {"record_idx": np.full(n, record_idx, dtype=np.int32),
                           "start_idx": np.asarray(kept_starts, dtype=np.int64),
                           "model_id": [group] * n, "group_id": [f"{inventory_id}:{group}"] * n,
                           "site_id": [rec.get("site_id") or ""] * n, "split": [split] * n}
                row_start, row_end = _append_h5(norm_handle, normalized, columns)
                if _append_h5(raw_handle, raw, columns) != (row_start, row_end):
                    raise RuntimeError("Raw/normalized row alignment lost")
                split_indices[split].extend(range(row_start, row_end))
                all_groups.extend(columns["group_id"])
            if record_idx % 25 == 0 or record_idx + 1 == len(selected):
                print(f"CLEAN {source_id} {record_idx + 1}/{len(selected)} records; {norm_handle['data'].shape[0]} rows", flush=True)
        count = norm_handle["data"].shape[0]
        norm_handle.attrs.update({"n_clips": count, "normalization": settings["normalization"]})
        raw_handle.attrs.update({"n_clips": count, "normalization": "none_original_source_stored_coordinates"})
    finally:
        norm_handle.close()
        raw_handle.close()
    audit = _assert_splits(split_indices, count, all_groups)
    metadata_path = folder / "metadata.json"
    metadata = {"source": source_id, "family": family, "n_clips_total": count,
                "n_records_total": len(records_table), "window_size": WINDOW, "stride": STRIDE,
                "sampling_rate": 150, "channel_names": list(CHANNELS), "records": records_table,
                "normalization": settings["normalization"], "filter_stats": dict(counters),
                "split_audit": audit, "amplitude_space": "source_stored_coordinates",
                "calibrated_physical_units_verified": False,
                "source_geometry": info.get("source_metadata", {}).get("geometry", "layered_continuous_1d_resistivity_profile"),
                "source_signal_scaling": info.get("source_metadata", {}).get("signal_scaling", "not_explicitly_declared_in_source_metadata"),
                "input_signature": signature}
    _write_json(metadata_path, metadata)
    os.replace(norm_partial, norm_path)
    os.replace(raw_partial, raw_path)
    hashes = {path.name: _sha256(path) for path in (norm_path, raw_path, metadata_path)}
    entry = {"id": source_id, "family": family, "path": _relative(root, norm_path),
             "raw_path": _relative(root, raw_path), "key": "data", "count": count, "usable_count": count,
             "shape": [count, 4, WINDOW], "dtype": "float32", "sampling_rate_hz": 150,
             "channel_names": list(CHANNELS), "generated": False,
             "normalization": settings["normalization"],
             "normalization_metadata_path": _relative(root, metadata_path),
             "amplitude_space": "source_stored_coordinates", "calibrated_physical_units_verified": False,
             "split_unit": "geology_parent_group", "split_policy":
             ("all_train_only_shared_parent_excitation" if set(group_assignment[inventory_id].values()) == {"train"}
              else "one_train_one_test_no_validation") if family == "syn_3d" else "seeded_geology_group_split",
             "split_indices": split_indices, "split_audit": audit, "file_sha256": hashes[norm_path.name],
             "raw_file_sha256": hashes[raw_path.name], "candidate_count": counters["candidate_windows"],
             "filter_stats": dict(counters), "source_geometry": metadata["source_geometry"]}
    _write_json(marker_path, {"input_signature": signature, "file_sha256": hashes, "manifest_source": entry})
    return entry


def prepare_clean(root: Path, config: dict, existing_inventory: dict, new_inventory: dict) -> dict:
    """Build/copy all requested clean banks and return their portable manifest.

    ``config['include_sources']`` can select source IDs for QA; production omits
    it to require all ten banks. Legacy split lists are inherited, including
    *every* held-out source row rather than only the selected 2,000-row export.
    Newly generated continuous and 3D data share one geology split assignment
    across their A/B pipelines. The two 3D parents form one train and one test
    parent, with no validation parent.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    settings = _settings(config)
    clean_config = {**config, **config.get("clean", {})}
    include = clean_config.get("include_sources", list(LEGACY_IDS) + list(NEW_BANKS))
    unknown = set(include) - set(LEGACY_IDS) - set(NEW_BANKS)
    if unknown:
        raise ValueError(f"Unknown clean source IDs: {sorted(unknown)}")
    legacy_path = Path(clean_config.get("legacy_manifest_path", existing_inventory["legacy_manifest_path"]))
    legacy = clean_config.get("legacy_manifest") or json.loads(legacy_path.read_text(encoding="utf-8"))
    if any(k in NEW_BANKS for k in include):
        legacy, inherited, excitation_audit = _excitation_plan(existing_inventory, new_inventory, legacy, settings)
        group_assignment = _assign_new_groups(new_inventory, settings, inherited, excitation_audit)
    else:
        inherited, group_assignment, excitation_audit = {}, {}, {"status": "no_new_sources_requested"}
    excitation_path = root / "clean" / "excitation_isolation_audit.json"
    _write_json(excitation_path, excitation_audit)
    sources = []
    for source_id in include:
        if source_id in LEGACY_IDS:
            if source_id not in existing_inventory["sources"]:
                raise ValueError(f"Missing requested inventory source: {source_id}")
            print(f"CLEAN COPY {source_id}", flush=True)
            sources.append(_legacy_source(root, source_id, existing_inventory, legacy, settings))
        else:
            sources.append(_build_new_source(root, source_id, new_inventory, group_assignment, settings))
    for source in sources:
        source["excitation_isolation_audit_path"] = _relative(root, excitation_path)
        if source["family"] == "syn_continuous" and inherited:
            source["split_policy"] = "seeded_geology_split_with_exact_Hx_Hy_excitation_inheritance"
    # Explicitly verify shared physical parents across the legacy A/B pipelines.
    combined_owner = {}
    for source in sources:
        if source["id"] not in ("syn_amt_A", "syn_amt_B"):
            continue
        for split in SPLITS:
            for group in source["split_audit"][split]["groups"]:
                if group in combined_owner and combined_owner[group] != split:
                    raise ValueError(f"Legacy A/B physical-parent split mismatch: {group}")
                combined_owner[group] = split
    split_folder = root / "splits"
    split_folder.mkdir(parents=True, exist_ok=True)
    split_path = split_folder / "clean_splits.npz"
    split_partial = split_path.with_name(split_path.name + ".partial")
    arrays = {f"{source['id']}__{split}": np.asarray(source["split_indices"][split], dtype=np.int64)
              for source in sources for split in SPLITS}
    with split_partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(split_partial, split_path)
    for source in sources:
        source["split_indices_path"] = _relative(root, split_path)
        source["split_indices_keys"] = {split: f"{source['id']}__{split}" for split in SPLITS}
    manifest = {"schema_version": "clean_bank_v1", "channel_names": list(CHANNELS),
                "sequence_length": WINDOW, "sampling_rate_hz": 150, "sources": sources,
                "settings": settings, "legacy_manifest_id": legacy.get("manifest_id"),
                "legacy_manifest_sha256": _sha256(legacy_path) if legacy_path.exists() else None,
                "split_indices_path": _relative(root, split_path), "split_indices_sha256": _sha256(split_path),
                "new_geology_group_assignment": group_assignment,
                "excitation_isolation_audit_path": _relative(root, excitation_path),
                "excitation_isolation_audit": excitation_audit,
                "totals": {split: sum(len(source["split_indices"][split]) for source in sources) for split in SPLITS},
                "count": sum(source["count"] for source in sources),
                "usable_count": sum(source["usable_count"] for source in sources),
                "warnings": ["Generated DDPM/DDIM signals are training-only; generator fitting provenance remains unverified.",
                             ("New 3D-parent source shares H excitation across both parents and is training-only."
                              if excitation_audit.get("new_3d_cross_parent_H_duplicates") else
                              "New 3D-parent source contains only two geology groups: one train and one test, no validation."),
                             "New 3D-parent signals were generated using independent local 1D columns without lateral coupling.",
                             "Original source coordinates are retained; calibrated SI physical units are not established.",
                             "Legacy real metadata pre-dates curation; actual HDF5 shapes and provenance indices are authoritative.",
                             "Exact shared Hx/Hy excitations inherit the old split; old holdouts are never returned to training."],
                "cross_pipeline_parent_overlap_audit": {"legacy_A_B_split_mismatch": 0}}
    manifest["manifest_id"] = _digest(manifest)
    _write_json(root / "clean" / "manifest.json", manifest)
    return manifest
