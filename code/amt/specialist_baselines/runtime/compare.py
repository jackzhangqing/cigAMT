"""Bounded-memory paired case comparison and cluster bootstrap.

Cases are joined by the full immutable sample_uid, never by CSV row order.
Temporary SQLite tables live in the requested output directory. Default cluster
is clean_index: this groups corruptions of the same clip, but does NOT establish
field-record independence because different clips may overlap or share a parent.
For publication, supply a source-parent mapping when available and repeat across
training seeds. Cluster intervals here condition on the two fitted checkpoints.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

import numpy as np

from .metrics import METRIC_INFO, NOISE_TASKS


_CASE_FIELDS = ("scheme", "sample_index", "clean_index", "source_id", "source_row", "class_id", "domain_id",
                "noise_length", "noise_start", "is_clean", "snr_target_db", "condition_index", "horizon",
                "gap_length", "missing_ratio_nominal", "missing_ratio_actual", "mean_gap", "channel_mode_id")
_IDENTITY_FIELDS = ("dataset_id", "task", "split")


def _canonical(value: Any) -> str:
    value = str(value or "").strip()
    if value.lower() in ("", "none", "null", "nan"):
        return ""
    try:
        number = Decimal(value)
        if number.is_finite():
            return format(number.normalize(), "f")
    except InvalidOperation:
        pass
    return value


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            part = stream.read(8 * 1024 * 1024)
            if not part:
                break
            digest.update(part)
    return digest.hexdigest()


def _load_csv(connection: sqlite3.Connection, table: str, path: Path, metric: str,
              cluster: str, using_map: bool) -> dict[str, Any]:
    # table names are constants controlled exclusively by this module.
    if table not in ("left_rows", "right_rows"):
        raise ValueError("Invalid internal table")
    connection.execute(f"CREATE TABLE {table} (sample_uid TEXT PRIMARY KEY, value REAL, cluster_key TEXT, metadata TEXT, source_key TEXT)")
    before = path.stat()
    identity: dict[str, str] | None = None
    schemes, models, training_schemes, seeds = set(), set(), set(), set()
    rows, finite = 0, 0
    pending = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        required = set(_IDENTITY_FIELDS) | {"sample_uid", metric}
        if not using_map:
            required.add(cluster)
        if required - columns:
            raise ValueError(f"{path}: missing columns {sorted(required - columns)}")
        for line, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}:{line}: malformed CSV row")
            current = {key: str(row.get(key, "")).strip() for key in _IDENTITY_FIELDS}
            if not all(current.values()):
                raise ValueError(f"{path}:{line}: dataset_id, task and split must be nonempty")
            if identity is None:
                identity = current
            elif identity != current:
                raise ValueError(f"{path}:{line}: mixed dataset_id/task/split in one input")
            uid = str(row["sample_uid"]).strip()
            if not uid:
                raise ValueError(f"{path}:{line}: empty sample_uid")
            if not uid.startswith(current["dataset_id"] + "/"):
                raise ValueError(f"{path}:{line}: sample_uid does not belong to dataset_id")
            scheme = str(row.get("scheme", "")).strip()
            if scheme:
                schemes.add(scheme)
            for key, destination in (("model", models), ("train_scheme", training_schemes), ("seed", seeds)):
                if str(row.get(key, "")).strip():
                    destination.add(str(row[key]).strip())
            value = _finite(row.get(metric))
            finite += int(value is not None)
            metadata = {key: _canonical(row[key]) for key in _CASE_FIELDS if key in row}
            source_key = None
            if row.get("source_id", "") != "" and row.get("source_row", "") != "":
                source_key = _canonical(row["source_id"]) + ":" + _canonical(row["source_row"])
            pending.append((uid, value, _canonical(row.get(cluster)), json.dumps(metadata, sort_keys=True), source_key))
            rows += 1
            if len(pending) >= 5000:
                try:
                    connection.executemany(f"INSERT INTO {table} VALUES (?,?,?,?,?)", pending)
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"{path}: duplicate sample_uid; pairings would be ambiguous") from exc
                pending.clear()
        if pending:
            try:
                connection.executemany(f"INSERT INTO {table} VALUES (?,?,?,?,?)", pending)
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"{path}: duplicate sample_uid; pairings would be ambiguous") from exc
    if identity is None:
        raise ValueError(f"{path}: contains no data rows")
    if any(len(values) > 1 for values in (models, training_schemes, seeds)):
        raise ValueError(f"{path}: one comparison input must contain one model/training scheme/seed")
    summary_path = path.parent / "summary.json"
    declared_scheme = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        if summary.get("rows_path", "evaluation_rows.csv") == path.name:
            for key, expected in identity.items():
                if key in summary and str(summary[key]) != expected:
                    raise ValueError(f"{path}: neighboring summary.json {key} conflicts with rows")
            declared_scheme = summary.get("eval_scheme")
    if identity["task"] in NOISE_TASKS:
        if schemes - {"A", "C", "control"}:
            raise ValueError(f"{path}: unexpected noise scheme values {sorted(schemes)}")
        noise_schemes = schemes - {"control"}
        inferred_scheme = "AC" if noise_schemes == {"A", "C"} else next(iter(noise_schemes), None)
        if declared_scheme is not None:
            declared_allowed = {"A", "C"} if str(declared_scheme) == "AC" else {str(declared_scheme)}
            if str(declared_scheme) not in ("A", "C", "AC") or noise_schemes - declared_allowed:
                raise ValueError(f"{path}: summary eval_scheme conflicts with sample schemes")
        evaluation_scheme = str(declared_scheme or inferred_scheme or "")
        if not evaluation_scheme:
            raise ValueError(f"{path}: clean-only noise evaluation needs neighboring summary.json eval_scheme")
    else:
        if schemes - {"none"}:
            raise ValueError(f"{path}: clean completion task unexpectedly contains A/C schemes")
        evaluation_scheme = "none"
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise ValueError(f"{path}: changed during comparison; wait for evaluation to finish")
    connection.commit()
    return {**identity, "eval_scheme": evaluation_scheme, "path": str(path.resolve()), "rows": rows,
            "finite_metric_rows": finite, "undefined_metric_rows": rows - finite,
            "model": next(iter(models), None), "train_scheme": next(iter(training_schemes), None),
            "seed": next(iter(seeds), None), "file_bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns, "sha256": _file_hash(path)}


def _load_cluster_map(connection: sqlite3.Connection, path: Path) -> dict[str, Any]:
    connection.execute("CREATE TABLE cluster_map (map_key TEXT PRIMARY KEY, cluster_id TEXT NOT NULL)")
    rows, pending = 0, []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        if "cluster_id" not in columns:
            raise ValueError("cluster-map CSV needs cluster_id")
        if "sample_uid" in columns:
            mode = "sample_uid"
        elif {"source_id", "source_row"} <= columns:
            mode = "source_id_source_row"
        else:
            raise ValueError("cluster-map needs sample_uid,cluster_id OR source_id,source_row,cluster_id")
        for row in reader:
            key = str(row.get("sample_uid", "")).strip() if mode == "sample_uid" else _canonical(row.get("source_id")) + ":" + _canonical(row.get("source_row"))
            cluster_id = str(row.get("cluster_id", "")).strip()
            if not key or key == ":" or not cluster_id:
                raise ValueError("cluster-map has empty key or cluster_id")
            pending.append((key, cluster_id)); rows += 1
            if len(pending) >= 5000:
                try:
                    connection.executemany("INSERT INTO cluster_map VALUES (?,?)", pending)
                except sqlite3.IntegrityError as exc:
                    raise ValueError("cluster-map has duplicate keys") from exc
                pending.clear()
        try:
            connection.executemany("INSERT INTO cluster_map VALUES (?,?)", pending)
        except sqlite3.IntegrityError as exc:
            raise ValueError("cluster-map has duplicate keys") from exc
    connection.commit()
    return {"path": str(path.resolve()), "rows": rows, "key_mode": mode, "sha256": _file_hash(path)}


def _cluster_bootstrap(connection: sqlite3.Connection, bootstrap: int, seed: int) -> dict[str, Any]:
    # One numeric mean per cluster is retained, not all individual cases.
    means = np.fromiter((row[0] for row in connection.execute("SELECT AVG(delta) FROM pairs GROUP BY cluster_key ORDER BY cluster_key")), dtype=np.float64)
    oriented = np.fromiter((row[0] for row in connection.execute("SELECT AVG(advantage) FROM pairs GROUP BY cluster_key ORDER BY cluster_key")), dtype=np.float64)
    n = len(means)
    result = {"clusters": n, "bootstrap_replicates": int(bootstrap), "seed": int(seed),
              "statistic": "equal-weight mean of within-cluster paired case mean differences (right minus left)",
              "mean_delta": float(means.mean()) if n else None,
              "mean_right_advantage": float(oriented.mean()) if n else None,
              "ci95_delta": None, "ci95_right_advantage": None,
              "inference": "conditional on the two trained checkpoints; does not include training-seed variance"}
    if n < 2 or bootstrap < 2:
        result["status"] = "insufficient_clusters_or_replicates_for_interval"
        return result
    rng = np.random.default_rng(seed)
    deltas, advantages = np.empty(bootstrap), np.empty(bootstrap)
    block = max(1, min(64, 1_000_000 // n))
    for start in range(0, bootstrap, block):
        end = min(bootstrap, start + block)
        selected = rng.integers(0, n, size=(end - start, n))
        deltas[start:end] = means[selected].mean(axis=1)
        advantages[start:end] = oriented[selected].mean(axis=1)
    result.update(status="computed_percentile_interval", ci95_delta=np.quantile(deltas, [.025, .975]).tolist(),
                  ci95_right_advantage=np.quantile(advantages, [.025, .975]).tolist())
    return result


def compare_cases(left_csv: str | Path, right_csv: str | Path, output: str | Path,
                  metric: str = "snr_improvement_db", cluster: str = "clean_index",
                  bootstrap: int = 2000, seed: int = 2025, cluster_map: str | Path | None = None,
                  direction: str | None = None, tie_tolerance: float = 1e-12) -> dict[str, Any]:
    """Pair two completed evaluation CSVs; positive advantage means right wins.

    The default clean_index cluster handles repeated corruptions of one clip.
    Use cluster_map for original-record/geological-parent uncertainty, with CSV
    columns sample_uid,cluster_id or source_id,source_row,cluster_id. Missing map
    entries, duplicate case IDs and mismatched experimental identities fail.
    Unmatched cases and undefined scores are counted and excluded explicitly.
    """
    left_csv, right_csv, output = Path(left_csv), Path(right_csv), Path(output)
    if not left_csv.is_file() or not right_csv.is_file():
        raise FileNotFoundError("Both comparison inputs must be existing evaluation CSV files")
    if bootstrap < 0 or not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("bootstrap and tie_tolerance must be nonnegative")
    if direction is None:
        direction = METRIC_INFO.get(metric, {}).get("direction")
    if direction not in ("higher", "lower"):
        raise ValueError(f"Unknown metric direction for {metric}; specify direction='higher' or 'lower'")
    output.mkdir(parents=True, exist_ok=True)
    ranked_path, report_path = output / "ranked_cases.csv", output / "paired_comparison.json"
    if ranked_path.resolve() in (left_csv.resolve(), right_csv.resolve()):
        raise ValueError("Output must not overwrite an input CSV")
    with tempfile.NamedTemporaryFile(prefix=".paired_", suffix=".sqlite", dir=output, delete=False) as temporary:
        database = Path(temporary.name)
    connection = sqlite3.connect(str(database))
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA cache_size=-32768")
        connection.execute("PRAGMA temp_store=FILE")
        mapping = _load_cluster_map(connection, Path(cluster_map)) if cluster_map is not None else None
        left = _load_csv(connection, "left_rows", left_csv, metric, cluster, mapping is not None)
        right = _load_csv(connection, "right_rows", right_csv, metric, cluster, mapping is not None)
        for field in (*_IDENTITY_FIELDS, "eval_scheme"):
            if left[field] != right[field]:
                raise ValueError(f"Cannot pair different {field}: {left[field]!r} versus {right[field]!r}")
        connection.execute("CREATE TABLE pairs (sample_uid TEXT PRIMARY KEY, left_value REAL, right_value REAL, delta REAL, advantage REAL, cluster_key TEXT, metadata TEXT)")
        map_join = ""
        map_column = "NULL"
        if mapping:
            key = "a.sample_uid" if mapping["key_mode"] == "sample_uid" else "a.source_key"
            map_join = f"LEFT JOIN cluster_map m ON m.map_key={key}"
            map_column = "m.cluster_id"
        query = ("SELECT a.sample_uid,a.value,b.value,a.cluster_key,b.cluster_key,a.metadata,b.metadata," + map_column +
                 " FROM left_rows a JOIN right_rows b ON a.sample_uid=b.sample_uid " + map_join)
        matched, undefined = 0, 0
        pending = []
        sign = 1.0 if direction == "higher" else -1.0
        for uid, lv, rv, lc, rc, lm, rm, mapped in connection.execute(query):
            matched += 1
            if lm != rm:
                raise ValueError(f"Case metadata mismatch for {uid}; cannot compare changed noise/mask conditions")
            if not mapping and lc != rc:
                raise ValueError(f"Cluster assignment differs between inputs for {uid}")
            if lv is None or rv is None:
                undefined += 1
                continue
            cluster_key = mapped if mapping else lc
            if not cluster_key:
                raise ValueError(f"Missing cluster assignment for finite matched case {uid}")
            delta = rv - lv
            pending.append((uid, lv, rv, delta, sign * delta, cluster_key, lm))
            if len(pending) >= 5000:
                connection.executemany("INSERT INTO pairs VALUES (?,?,?,?,?,?,?)", pending); pending.clear()
        if pending:
            connection.executemany("INSERT INTO pairs VALUES (?,?,?,?,?,?,?)", pending)
        connection.commit()
        paired = matched - undefined
        if not paired:
            raise ValueError("No matched cases have a defined finite metric in both inputs")
        mean_left, mean_right, delta, advantage, wins, ties, losses = connection.execute(
            "SELECT AVG(left_value),AVG(right_value),AVG(delta),AVG(advantage),"
            "SUM(advantage>?),SUM(ABS(advantage)<=?),SUM(advantage<?) FROM pairs",
            (tie_tolerance, tie_tolerance, -tie_tolerance)).fetchone()
        bootstrap_report = _cluster_bootstrap(connection, bootstrap, seed)
        connection.execute("CREATE INDEX pairs_ranking ON pairs(advantage DESC,sample_uid)")
        columns = ["rank", "sample_uid", "cluster_id", "left_value", "right_value", "right_minus_left",
                   "right_advantage", *_CASE_FIELDS]
        with ranked_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for rank, (uid, lv, rv, dv, adv, ck, meta) in enumerate(connection.execute(
                    "SELECT sample_uid,left_value,right_value,delta,advantage,cluster_key,metadata FROM pairs ORDER BY advantage DESC,sample_uid"), start=1):
                writer.writerow({"rank": rank, "sample_uid": uid, "cluster_id": ck, "left_value": lv,
                                 "right_value": rv, "right_minus_left": dv, "right_advantage": adv, **json.loads(meta)})
        independence = ("User-supplied cluster map. The statistical unit is exactly cluster_id; source-parent independence must be established by the mapping author."
                        if mapping else
                        ("clean_index groups all corruption realizations of one clean clip. Different clips may overlap or share a field record/geological parent; this is NOT a parent-independent confidence interval."
                         if cluster == "clean_index" else f"Clusters are the supplied {cluster} column; statistical independence is not established automatically."))
        report = {"state": "paired_comparison_complete", "metric": metric, "direction": direction,
                  "delta_convention": "right minus left", "advantage_convention": "positive favors right; metric direction applied",
                  "left": left, "right": right, "matched_sample_uids": matched,
                  "left_unmatched_rows": left["rows"] - matched, "right_unmatched_rows": right["rows"] - matched,
                  "matched_undefined_metric_rows": undefined, "paired_finite_rows": paired,
                  "case_weighted": {"mean_left": mean_left, "mean_right": mean_right, "mean_delta": delta,
                                    "mean_right_advantage": advantage, "right_wins": int(wins), "ties": int(ties),
                                    "left_wins": int(losses), "right_win_fraction": wins / paired,
                                    "tie_fraction": ties / paired, "left_win_fraction": losses / paired,
                                    "tie_tolerance": tie_tolerance},
                  "cluster": {"column": cluster if not mapping else None, "mapping": mapping, "independence_scope": independence},
                  "cluster_bootstrap": bootstrap_report, "ranked_cases": ranked_path.name,
                  "selection_scope": "Intersection of exact sample_uid values with finite metric on both sides; no row-order pairing or cherry-picked removal.",
                  "limitations": ["Case metrics are paired, but a single checkpoint per method does not measure training-seed variability.",
                                  "Clip-level clusters are not a substitute for original-record or geological-parent clusters.",
                                  "This estimates the chosen decomposable per-case metric. Classification macro-F1 differences require paired resampling of confusion matrices, not per-case F1.",
                                  "When comparing many models/conditions, report all planned comparisons and address multiple comparisons."]}
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        return report
    finally:
        connection.close()
        # Only this exact temporary SQLite file is removed.
        database.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="Left completed evaluation_rows.csv")
    parser.add_argument("--right", required=True, help="Right completed evaluation_rows.csv")
    parser.add_argument("--output", required=True, help="Output directory for report and full ranked case CSV")
    parser.add_argument("--metric", default="snr_improvement_db")
    parser.add_argument("--cluster", default="clean_index")
    parser.add_argument("--cluster-map", help="CSV sample_uid,cluster_id OR source_id,source_row,cluster_id")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--direction", choices=("higher", "lower"))
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    args = parser.parse_args(argv)
    result = compare_cases(args.left, args.right, args.output, args.metric, args.cluster,
                           args.bootstrap, args.seed, args.cluster_map, args.direction, args.tie_tolerance)
    print(json.dumps({"state": result["state"], "paired_finite_rows": result["paired_finite_rows"],
                      "mean_right_advantage": result["case_weighted"]["mean_right_advantage"],
                      "report": str(Path(args.output) / "paired_comparison.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
