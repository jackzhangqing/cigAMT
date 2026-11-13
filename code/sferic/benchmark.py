"""Single Windows/Linux command line for frozen-data baseline experiments."""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from bench.runtime import BASE, digest, environment, load_config, read_json, sha256, write_json


def registry_names():
    """Read literal registries without importing a potentially unavailable torch."""
    result = {}
    for filename, symbol, kind in (("models.py", "MODEL_NAMES", "deep"),
                                    ("classical.py", "CLASSICAL_NAMES", "classical")):
        tree = ast.parse((BASE / "bench" / filename).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == symbol
                                                   for target in node.targets):
                result[kind] = list(ast.literal_eval(node.value))
                break
        if kind not in result:
            raise RuntimeError(f"Could not read {symbol} registry")
    return result


def _overrides(args):
    result = {}
    for item in getattr(args, "sets", []) or []:
        if "=" not in item:
            raise ValueError("--set needs KEY=JSON, for example --set num_workers=4")
        key, value = item.split("=", 1)
        if not key or "." in key:
            raise ValueError("--set keys must be top-level configuration names")
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            result[key] = value
    for name in ("model", "seed", "device", "output_root"):
        value = getattr(args, name, None)
        if value is not None:
            result[name] = value
    return result


def _cases(config):
    all_names = sum(registry_names().values(), [])
    experiments = config.get("experiments") or [{}]
    if not isinstance(experiments, list) or not all(isinstance(case, dict) for case in experiments):
        raise ValueError("experiments must be a list of override dictionaries")
    cases = []
    seen = set()
    for experiment_index, overrides in enumerate(experiments):
        merged = dict(config)
        merged.update(overrides)
        models = overrides.get("models", [overrides["model"]] if "model" in overrides else
                               config.get("models", [config["model"]]))
        seeds = overrides.get("seeds", [overrides["seed"]] if "seed" in overrides else
                              config.get("seeds", [config["seed"]]))
        if not isinstance(models, list) or not models or not isinstance(seeds, list) or not seeds:
            raise ValueError("models and seeds must be nonempty lists")
        if merged["protocol"] == "group_cv":
            folds = range(int(merged["n_splits"]))
            if len(folds) < 2:
                raise ValueError("group_cv needs n_splits >= 2")
            merged["augmentation"] = "clean"
        else:
            folds = [int(merged.get("fold", 0))]
        for model in models:
            if model not in all_names:
                raise ValueError(f"Unknown model {model!r}")
            for seed in seeds:
                for fold in folds:
                    case = {key: value for key, value in merged.items() if key not in {"models", "seeds", "experiments"}}
                    case.update(model=model, seed=int(seed), fold=int(fold))
                    case = load_config(overrides=case)
                    signature = digest({key: value for key, value in case.items() if key != "description"})
                    if signature in seen:
                        continue
                    seen.add(signature)
                    cases.append(dict(index=len(cases), experiment_index=experiment_index,
                                      case_id=f"{model}_seed{seed}_fold{fold}_{signature[:10]}", config=case))
    return cases


def run_suite(config, resume=False, dry_run=False, limit_runs=None):
    cases = _cases(config)
    if limit_runs is not None and limit_runs < 1:
        raise ValueError("limit-runs must be positive")
    implementation = {str(path.relative_to(BASE)): sha256(path)
                      for path in [BASE / "benchmark.py"] + sorted((BASE / "bench").glob("*.py"))}
    key = digest(dict(cases=[case["config"] for case in cases], implementation=implementation))[:12]
    suite_dir = Path(config["output_root"]) / "_suites" / key
    suite_dir.mkdir(parents=True, exist_ok=True)
    plan = dict(status="planned", suite_id=key, cases=len(cases), resume_requested=bool(resume),
                smoke_only=bool(config.get("smoke_only")), implementation=implementation,
                environment=environment(), runs=[])
    for case in cases:
        path = suite_dir / "configs" / (case["case_id"] + ".json")
        write_json(path, case["config"])
        plan["runs"].append({key: value for key, value in case.items() if key != "config"})
        plan["runs"][-1].update(config_path=str(path), status="queued")
    write_json(suite_dir / "suite_plan.json", plan)
    if dry_run:
        return dict(status="planned", suite_dir=str(suite_dir), cases=len(cases), dry_run=True)
    selected = plan["runs"][:limit_runs] if limit_runs else plan["runs"]
    plan["status"] = "running"
    for row in selected:
        command = [sys.executable, str(BASE / "benchmark.py"), "train", "--config", row["config_path"]]
        if resume:
            command.append("--resume")
        log = suite_dir / "logs" / (row["case_id"] + ".log")
        log.parent.mkdir(exist_ok=True)
        row.update(status="running", command=command, log_path=str(log))
        write_json(suite_dir / "suite_status.json", plan)
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        print(f"[{row['index'] + 1}/{len(cases)}] {row['case_id']}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=str(BASE), env=child_env,
                                       stdout=stream, stderr=subprocess.STDOUT, check=False)
        row["return_code"] = completed.returncode
        row["status"] = "complete" if completed.returncode == 0 else "failed"
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                row["result"] = payload
                break
        if completed.returncode:
            row["failure_reason"] = row.get("result", {}).get("error") or "\n".join(lines[-15:])
            print(f"  FAILED: {row['failure_reason']}\n  Log: {log}", flush=True)
        else:
            destination = row.get("result", {}).get("run_dir", str(log))
            print(f"  COMPLETE: {destination}", flush=True)
        write_json(suite_dir / "suite_status.json", plan)
    failures = sum(row["status"] == "failed" for row in plan["runs"])
    successes = sum(row["status"] == "complete" for row in plan["runs"])
    plan["status"] = "failed" if failures else ("complete" if successes == len(cases) else "partial")
    plan.update(completed_runs=successes, failed_runs=failures, unstarted_runs=len(cases) - len(selected))
    write_json(suite_dir / "suite_status.json", plan)
    return dict(status=plan["status"], suite_dir=str(suite_dir), cases=len(cases),
                completed_runs=successes, failed_runs=failures, unstarted_runs=plan["unstarted_runs"])


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("list-models", "audit", "train", "suite", "smoke", "evaluate", "robustness", "predict", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--config", help="JSON configuration; relative paths resolve under Benchmark")
        command.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=JSON")
        command.add_argument("--traceback", action="store_true", help="Print a diagnostic traceback on failure")
        if name in {"train", "suite", "smoke"}:
            command.add_argument("--model")
            command.add_argument("--seed", type=int)
            command.add_argument("--output-root")
            command.add_argument("--resume", action="store_true", help="Explicitly resume compatible existing runs")
        if name in {"train", "suite", "smoke", "evaluate", "robustness", "predict"}:
            command.add_argument("--device", default="cpu" if name == "smoke" else None)
        if name in {"suite", "smoke"}:
            command.add_argument("--dry-run", action="store_true")
            command.add_argument("--limit-runs", type=int)
        if name == "audit":
            command.add_argument("--full", action="store_true")
        if name in {"evaluate", "robustness", "predict"}:
            command.add_argument("--run-dir", required=True)
            command.add_argument("--data-root", help="Relocate copied runs to this dataset root (same input checksums)")
        if name == "evaluate":
            command.add_argument("--bootstrap-iterations", type=int)
        if name in {"robustness", "predict"}:
            command.add_argument("--limit", type=int)
        if name == "robustness":
            command.add_argument("--specs", help="JSON file containing corruption specifications")
        if name == "predict":
            command.add_argument("--batch-size", type=int)
            command.add_argument("--output-dir")
        if name == "summarize":
            command.add_argument("--root", default="results/main")
            command.add_argument("--output-dir")
            command.add_argument("--include-smoke", action="store_true")
    return parser


def _path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (BASE / path).resolve()


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.sets and args.command in {"evaluate", "robustness", "predict", "summarize", "list-models"}:
            raise ValueError("--set is only for audit/train/suite/smoke; this command uses its dedicated flags")
        if args.command == "list-models":
            result = registry_names()
        elif args.command == "summarize":
            from bench.reporting import summarize
            result = summarize(_path(args.root), _path(args.output_dir) if args.output_dir else None,
                               include_smoke=args.include_smoke)
        else:
            config = load_config(args.config, _overrides(args))
            if args.command == "smoke":
                config.update(models=sum(registry_names().values(), []), seeds=[42], experiments=[],
                    epochs=1, min_epochs=1, patience=1, bootstrap_iterations=20,
                    max_train_samples=64, max_eval_samples=32, batch_size=16,
                    smoke_only=True, protocol="fixed", augmentation="all", channels="all",
                    train_fraction=1.0, output_root=str(BASE / "results/smoke"))
                result = run_suite(config, args.resume, args.dry_run, args.limit_runs)
            elif args.command == "suite":
                if args.model:
                    config["models"] = [args.model]
                if args.seed is not None:
                    config["seeds"] = [args.seed]
                result = run_suite(config, args.resume, args.dry_run, args.limit_runs)
            else:
                engine = importlib.import_module("bench.engine")
                if args.command == "audit":
                    result = engine.audit_dataset(config, full=args.full)
                elif args.command == "train":
                    path = engine.train_run(config, resume=args.resume)
                    result = dict(status="complete", run_dir=str(path))
                elif args.command == "evaluate":
                    result = engine.evaluate_run(_path(args.run_dir), device=args.device or "auto",
                                                 bootstrap_iterations=args.bootstrap_iterations,
                                                 data_root=_path(args.data_root) if args.data_root else None)
                elif args.command == "robustness":
                    specs = read_json(_path(args.specs)) if args.specs else None
                    result = engine.robustness_run(_path(args.run_dir), specs=specs,
                        device=args.device or "auto", limit=args.limit,
                        data_root=_path(args.data_root) if args.data_root else None)
                else:
                    result = engine.predict_field(_path(args.run_dir), device=args.device or "auto",
                        batch_size=args.batch_size, limit=args.limit,
                        output_dir=_path(args.output_dir) if args.output_dir else None,
                        data_root=_path(args.data_root) if args.data_root else None)
        print(json.dumps(result, ensure_ascii=False, default=str, allow_nan=False), flush=True)
        return 1 if isinstance(result, dict) and result.get("status") == "failed" else 0
    except Exception as error:
        if args.traceback:
            traceback.print_exc()
        print(json.dumps(dict(status="failed", command=args.command, error_type=type(error).__name__,
                              error=str(error)), ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
