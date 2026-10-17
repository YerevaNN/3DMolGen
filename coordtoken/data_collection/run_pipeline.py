#!/usr/bin/env python3
"""Replay the final stages from saved CSV snapshots, without changing the inputs."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
HISTORICAL = ROOT / "historical"
SPLITS = ("train", "val", "test")


def resolve_path(value, config_path):
    path = Path(value).expanduser()
    return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(config_path):
    config = json.loads(config_path.read_text())
    required = {"start_at", "source_root", "output_root", "workers", "expected_train_rows"}
    missing = required - config.keys()
    unknown = config.keys() - required - {"reference_csv"}
    if missing or unknown:
        raise ValueError(f"Invalid config keys: missing={sorted(missing)}, unknown={sorted(unknown)}")
    if config["start_at"] not in {"deduplicate", "filter", "merge"}:
        raise ValueError("start_at must be deduplicate, filter, or merge")
    if type(config["workers"]) is not int or config["workers"] < 1:
        raise ValueError("workers must be a positive integer")
    expected = config["expected_train_rows"]
    if expected is not None and (type(expected) is not int or expected < 0):
        raise ValueError("expected_train_rows must be a nonnegative integer or null")
    for name in ("source_root", "output_root", "reference_csv"):
        if config.get(name):
            config[name] = resolve_path(config[name], config_path)
    if config["start_at"] != "merge" and not config.get("reference_csv"):
        raise ValueError("reference_csv is required for deduplicate/filter")
    return config


def command(script, *arguments):
    return [sys.executable, "-u", str(HISTORICAL / script), *map(str, arguments)]


def build_steps(config):
    output = config["output_root"]
    source = config["source_root"]
    steps = []
    if config["start_at"] == "deduplicate":
        dedup = output / "deduplicated"
        steps.append(("deduplicate", command(
            "dedupe_samples_and_report_intersections.py",
            "--phase", "all", "--source-root", source, "--output-root", dedup,
            "--summary-csv", output / "duplicate_stats.csv",
            "--intersections-csv", output / "sample_intersections.csv",
            "--apply-summary-csv", output / "dedup_apply_summary.csv",
            "--sqlite-db", output / "state" / "sample_duplicate_stats.sqlite",
            "--drop-lists-root", output / "state" / "drop_lists",
        )))
        source = dedup
    if config["start_at"] != "merge":
        final = output / "filtered"
        steps.append(("filter", command(
            "remove_geom_revisited_test_smiles.py",
            "--input-root", source, "--reference-csv", config["reference_csv"],
            "--output-root", final,
            "--summary-csv", output / "geom_revisited_removal_summary.csv",
            "--workers", config["workers"],
        )))
        source = final
    steps.append(("merge", command(
        "merge_train_only_streaming.py", "--root", source,
        "--output-csv", output / "merged_train.csv",
    )))
    return steps


def preflight(config):
    source = config["source_root"]
    output = config["output_root"]
    if output.exists():
        raise ValueError(f"Output must be a new directory: {output}")
    inputs = []
    splits = ("train",) if config["start_at"] == "merge" else SPLITS
    for split in splits:
        directory = source / split
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing input split: {directory}")
        files = sorted(directory.glob("*.csv"))
        if not files:
            raise ValueError(f"No input CSVs in {directory}")
        inputs.extend(files)
    if config["start_at"] != "merge":
        inputs.append(config["reference_csv"])
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input file: {path}")
        if path.resolve().is_relative_to(output):
            raise ValueError(f"Output contains an input: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        if "enriched_text" not in header:
            raise ValueError(f"Missing enriched_text column: {path}")
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output and input roots must not overlap")
    return sorted(set(inputs))


def file_record(path, hash_contents):
    before = path.stat()
    record = {"path": str(path), "bytes": before.st_size, "mtime_ns": before.st_mtime_ns}
    if hash_contents:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Input changed while hashing: {path}")
        record["sha256"] = digest.hexdigest()
    return record


def save_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")


def run_step(name, argv, output):
    log_path = output / "logs" / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(argv, cwd=HISTORICAL, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}; see {log_path}")
    return log_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--plan", action="store_true", help="Print commands without requiring mounted data")
    actions.add_argument("--execute", action="store_true", help="Execute into a new output directory")
    parser.add_argument("--hash-inputs", action="store_true", help="Read all input bytes to record SHA-256 fingerprints")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    steps = build_steps(config)
    for name, argv in steps:
        print(f"{name}: {shlex.join(argv)}", flush=True)
    if args.plan:
        return
    inputs = preflight(config)
    print(f"Preflight passed: {len(inputs)} input files", flush=True)
    if not args.execute:
        print("No files written. Add --execute to run.")
        return
    output = config["output_root"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    manifest_path = output / "run_manifest.json"
    manifest = {
        "status": "running", "config": config,
        "python": platform.python_version(), "platform": platform.platform(),
        "tqdm": importlib.metadata.version("tqdm"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "input_fingerprints": "sha256" if args.hash_inputs else "size-and-mtime-only",
        "inputs": [], "steps": steps, "completed_steps": [],
        "code": [file_record(path, True) for path in [Path(__file__).resolve(), *sorted(HISTORICAL.rglob("*.py"))]],
    }
    save_manifest(manifest_path, manifest)
    try:
        manifest["inputs"] = [file_record(path, args.hash_inputs) for path in inputs]
        save_manifest(manifest_path, manifest)
        for name, argv in steps:
            print(f"Running {name}; log: {output / 'logs' / (name + '.log')}", flush=True)
            run_step(name, argv, output)
            manifest["completed_steps"].append(name)
            save_manifest(manifest_path, manifest)
        merge_log = (output / "logs" / "merge.log").read_text()
        match = re.search(r"^train total rows=([\d,]+) empty_name_rows=", merge_log, re.MULTILINE)
        if match is None:
            raise RuntimeError("Merge log contains no final row count")
        rows = int(match.group(1).replace(",", ""))
        manifest["train_rows"] = rows
        expected = config["expected_train_rows"]
        if expected is not None and rows != expected:
            raise ValueError(f"Expected {expected:,} training rows; got {rows:,}")
        for record in manifest["inputs"]:
            current = Path(record["path"]).stat()
            if (current.st_size, current.st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
                raise RuntimeError(f"Input changed during execution: {record['path']}")
        manifest["output"] = file_record(output / "merged_train.csv", args.hash_inputs)
        manifest["status"] = "complete"
        print(f"Completed: {rows:,} training rows. Manifest: {manifest_path}")
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        save_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
