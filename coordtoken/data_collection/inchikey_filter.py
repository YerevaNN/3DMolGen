#!/usr/bin/env python3
"""Apply a completed InChIKey audit to a new, byte-preserving training CSV.

All removal identities are regenerated and checked against reference keys before
filtering. Parallel chunks retain original order. Count reconciliation and hashes
are required before publishing merged_train.csv. Source files are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
from functools import lru_cache
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdinchi

COORDINATES = re.compile(r"<[^>]+>")
BRACKETS = re.compile(r"\[([^\]]+)\]")
ORGANIC_ATOMS = frozenset("B C N O P S F Cl Br I b c n o p s".split())
STANDARD_KEY = re.compile(r"[A-Z]{14}-[A-Z]{8}SA-[A-Z]")


def stripped_text(enriched_text: str) -> str:
    return COORDINATES.sub("", enriched_text)


def standard_inchikey(text: str) -> str:
    """Decode CoordToken atom brackets, preserving text-encoded stereochemistry.

    No stereochemistry is inferred from coordinates; no custom salt, tautomer,
    isotope or charge normalization is applied. Standard InChI normalizes its
    supported representations. Conversion errors are explicit exceptions.
    """
    base = stripped_text(text)
    smiles = BRACKETS.sub(lambda m: m[1] if m[1] in ORGANIC_ATOMS else m[0], base)
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("Cannot parse the decoded molecular structure")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    inchi, code, message, _, _ = rdinchi.MolToInchi(molecule)
    if code not in (0, 1) or not inchi.startswith("InChI=1S/"):
        raise ValueError(f"Standard InChI generation failed: {message}")
    key = rdinchi.InchiToInchiKey(inchi)
    if not STANDARD_KEY.fullmatch(key):
        raise ValueError("Invalid Standard InChIKey")
    return key


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def verify_identity_batch(items: list[tuple[str, str]]) -> int:
    RDLogger.DisableLog("rdApp.*")
    for base, expected in items:
        if standard_inchikey(base) != expected:
            raise ValueError(f"Audit key disagrees with regenerated key: {expected}")
    return len(items)


def load_audit(audit: Path, preserve: set[str], failure_policy: str, reference_labels: list[str]):
    """Read only completed audits, and reconcile their summary and database."""
    progress = json.loads((audit / "progress.json").read_text())
    summary = json.loads((audit / "summary.json").read_text())
    manifest = json.loads((audit / "manifest.json").read_text())
    if progress["status"] != "complete" or summary["status"] != "complete":
        raise ValueError("The audit must have completed successfully")
    if progress["counts"] != summary["counts"]:
        raise ValueError("Audit progress and summary counts disagree")
    if manifest["rdkit"] != rdBase.rdkitVersion:
        raise ValueError("Use the audited RDKit version")
    train = Path(manifest["train"]["path"])
    if fingerprint(train) != manifest["train"]:
        raise ValueError("The training input changed since the audit")
    for reference in manifest["references"]:
        if fingerprint(Path(reference["path"])) != reference:
            raise ValueError("A reference input changed since the audit")
    reference_summary = json.loads((audit / "reference_summary.json").read_text())
    if reference_summary["failed_unique_texts"]:
        raise ValueError("Reference conversion failures must be resolved before filtering")
    available = {item["label"]: item["bit"] for item in reference_summary["files"]}
    unknown = set(reference_labels) - available.keys()
    if not reference_labels or unknown:
        raise ValueError(f"Select explicit reference labels from {sorted(available)}; unknown={sorted(unknown)}")
    reference_mask = 0
    for label in reference_labels:
        reference_mask |= available[label]
    failures = progress["counts"].get("conversion_failure_rows", 0)
    if failures and failure_policy == "error":
        raise ValueError(f"{failures} training conversion failures; select quarantine or keep-reported explicitly")

    database = sqlite3.connect(f"file:{audit.resolve() / 'audit.sqlite'}?mode=ro", uri=True)
    try:
        stored_manifest = json.loads(database.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        state = json.loads(database.execute("SELECT value FROM metadata WHERE key='state'").fetchone()[0])
        if stored_manifest != manifest or state["rows"] != progress["rows"] or state["offset"] != manifest["train"]["size"]:
            raise ValueError("Audit database does not describe the completed input scan")
        total_key_rows = database.execute("SELECT COALESCE(SUM(n),0) FROM hits WHERE key_mask!=0").fetchone()[0]
        if total_key_rows != progress["counts"].get("inchikey_match_rows", 0):
            raise ValueError("Audit match totals disagree")
        references = {row[0] for row in database.execute("SELECT DISTINCT key FROM reference WHERE key!='' AND (bit & ?) != 0", (reference_mask,))}
        expected = Counter()
        identities = {}
        for base, source, key, count in database.execute("SELECT base,source,key,n FROM hits WHERE (key_mask & ?) != 0", (reference_mask,)):
            if source in preserve:
                continue
            if key not in references:
                raise ValueError("Audit match key is absent from the references")
            if base in identities and identities[base] != key:
                raise ValueError("One text has inconsistent keys in the audit")
            identities[base] = key
            expected[(source, base)] += count
        expected_by_source = Counter()
        for (source, _), count in expected.items():
            expected_by_source[source] += count
        failed_rows = Counter({
            (source, base): count
            for base, source, count in database.execute("SELECT base,source,n FROM failures")
            if source not in preserve
        })
        if database.execute("SELECT COALESCE(SUM(n),0) FROM failures").fetchone()[0] != failures:
            raise ValueError("Failure ledger does not match audit failure count")
        quarantine = failed_rows if failure_policy == "quarantine" else Counter()
        if set(quarantine) & set(expected):
            raise ValueError("A failed conversion unexpectedly has a successful key match")
    finally:
        database.close()
    return train, manifest, progress, expected, identities, failures, quarantine, reference_mask


# Read-only lookup state inherited by forked chunk workers.
REMOVE = set()
SOURCES = []
TRAIN = None
OUTPUT = None
PRESERVE = set()
QUARANTINE = set()


@lru_cache(maxsize=100000)
def matching_source(name: str) -> str:
    for source in SOURCES:
        if name.startswith(source + "_"):
            return source
    raise ValueError(f"Unrecognized dataset prefix: {name}")


def filter_chunk(task: tuple[int, int, int]) -> dict:
    index, start, end = task
    prefix = OUTPUT / "chunks" / f"{index:04d}"
    result_path = prefix.with_suffix(".json")
    part = prefix.with_suffix(".csv")
    removals = prefix.with_suffix(".removed.csv")
    quarantine_path = prefix.with_suffix(".quarantine.csv")
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if part.stat().st_size != result["output_bytes"] or sha256_file(part) != result["sha256"]:
            raise ValueError("A completed output chunk changed")
        if sha256_file(removals) != result["removals_sha256"]:
            raise ValueError("A completed removal ledger changed")
        if sha256_file(quarantine_path) != result["quarantine_sha256"]:
            raise ValueError("A completed quarantine chunk changed")
        return result
    temporary = prefix.with_suffix(".partial")
    removed = Counter()
    total = Counter()
    digest = hashlib.sha256()
    kept = 0
    quarantined = 0
    input_digest = hashlib.sha256()
    with TRAIN.open("rb", buffering=4 * 1024 * 1024) as source, temporary.open("wb", buffering=4 * 1024 * 1024) as target, quarantine_path.open("wb") as rejected:
        if start:
            source.seek(start - 1)
            if source.read(1) != b"\n":
                source.readline()
        else:
            header = source.readline()
            input_digest.update(header)
            if next(csv.reader([header.decode()])) != ["name", "enriched_text"]:
                raise ValueError("Unexpected CSV header")
        while source.tell() < end:
            raw = source.readline()
            if not raw:
                break
            input_digest.update(raw)
            row = next(csv.reader([raw.decode("utf-8")], strict=True))
            if len(row) != 2:
                raise ValueError("Expected one complete two-column CSV record per line")
            name, enriched = row
            dataset = matching_source(name)
            total[dataset] += 1
            match = None if dataset in PRESERVE else (dataset, stripped_text(enriched))
            if match in QUARANTINE:
                rejected.write(raw)
                removed[match] += 1
                quarantined += 1
            elif match in REMOVE:
                removed[match] += 1
            else:
                target.write(raw)
                digest.update(raw)
                kept += 1
        target.flush()
        os.fsync(target.fileno())
        rejected.flush()
        os.fsync(rejected.fileno())
    temporary.replace(part)
    with removals.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "stripped_smiles", "removed_rows"])
        for (dataset, base), count in sorted(removed.items()):
            writer.writerow([dataset, base, count])
        handle.flush()
        os.fsync(handle.fileno())
    result = {
        "index": index, "start": start, "end": end, "source_rows": dict(total),
        "input_rows": sum(total.values()), "kept_rows": kept,
        "removed_rows": sum(removed.values()), "output_bytes": part.stat().st_size,
        "sha256": digest.hexdigest(), "removals_sha256": sha256_file(removals),
        "quarantine_sha256": sha256_file(quarantine_path), "quarantined_rows": quarantined,
        "input_chunk_sha256": input_digest.hexdigest(),
    }
    atomic_json(result_path, result)
    return result


def run(args) -> dict:
    global REMOVE, SOURCES, TRAIN, OUTPUT, PRESERVE, QUARANTINE
    OUTPUT = args.output.resolve()
    if OUTPUT.exists() and not args.resume:
        raise ValueError("Output directory exists; use a new path or --resume")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    PRESERVE = set(args.preserve_source)
    TRAIN, audit_manifest, audit_progress, expected, identities, failures, quarantine, reference_mask = load_audit(args.audit, PRESERVE, args.failure_policy, args.reference_label)
    if TRAIN.resolve().is_relative_to(OUTPUT) or args.audit.resolve().is_relative_to(OUTPUT):
        raise ValueError("The output directory must not contain source or audit inputs")
    SOURCES = sorted(
        {p.stem for directory in (TRAIN.parent / "test", TRAIN.parent / "val") for p in directory.glob("*.csv")}
        | {source for source, _ in expected} | PRESERVE,
        key=len, reverse=True,
    )
    manifest = {
        "audit": str(args.audit.resolve()), "audit_manifest": audit_manifest,
        "audit_summary_sha256": sha256_file(args.audit / "summary.json"),
        "script_sha256": sha256_file(Path(__file__)), "preserve_sources": sorted(PRESERVE),
        "failure_policy": args.failure_policy, "chunks": args.chunks, "rdkit": rdBase.rdkitVersion,
        "reference_labels": sorted(set(args.reference_label)), "reference_mask": reference_mask,
    }
    manifest_path = OUTPUT / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume configuration, audit, code, or inputs changed")
    else:
        atomic_json(manifest_path, manifest)
    if (OUTPUT / "report.json").exists():
        report = json.loads((OUTPUT / "report.json").read_text())
        if sha256_file(OUTPUT / "merged_train.csv") != report["sha256"]:
            raise ValueError("Published output hash changed")
        return report
    if (OUTPUT / "merged_train.csv").exists():
        raise ValueError("An output CSV exists without its completion report; inspect before proceeding")

    started = time.monotonic()
    atomic_json(OUTPUT / "progress.json", {"status": "verifying_removal_keys", "unique_texts": len(identities)})
    items = list(identities.items())
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        verified = sum(pool.map(verify_identity_batch, (items[i:i+500] for i in range(0, len(items), 500))))
    del items, identities
    print(f"Verified {verified:,} unique removal strings; expected {sum(expected.values()):,} removed rows", flush=True)
    REMOVE = set(expected)
    QUARANTINE = set(quarantine)
    overlap_expected = sum(expected.values())
    expected.update(quarantine)
    (OUTPUT / "chunks").mkdir(exist_ok=True)
    size = TRAIN.stat().st_size
    tasks = [(i, size*i//args.chunks, size*(i+1)//args.chunks) for i in range(args.chunks)]
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        for future in as_completed([pool.submit(filter_chunk, task) for task in tasks]):
            result = future.result()
            results[result["index"]] = result
            progress = {
                "status": "filtering", "completed_chunks": len(results), "total_chunks": args.chunks,
                "removed_rows": sum(r["removed_rows"] for r in results.values()),
                "input_rows": sum(r["input_rows"] for r in results.values()),
                "elapsed_seconds": time.monotonic() - started,
            }
            atomic_json(OUTPUT / "progress.json", progress)
            if len(results) % 8 == 0:
                print(json.dumps(progress), flush=True)
    observed = Counter()
    source_rows = Counter()
    for i in range(args.chunks):
        source_rows.update(results[i]["source_rows"])
        with (OUTPUT / "chunks" / f"{i:04d}.removed.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                observed[(row["source"], row["stripped_smiles"])] += int(row["removed_rows"])
    if observed != expected or sum(source_rows.values()) != audit_progress["rows"]:
        raise ValueError("Full per-source/per-molecule counts disagree with the audit; output will not be published")
    if fingerprint(TRAIN) != audit_manifest["train"]:
        raise ValueError("Training input changed during filtering")
    atomic_json(OUTPUT / "progress.json", {"status": "assembling", "removed_rows": sum(observed.values())})
    final_temp = OUTPUT / "merged_train.csv.partial"
    digest = hashlib.sha256()
    with final_temp.open("wb") as target, TRAIN.open("rb") as source:
        header = source.readline()
        target.write(header)
        digest.update(header)
        for i in range(args.chunks):
            path = OUTPUT / "chunks" / f"{i:04d}.csv"
            chunk_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    target.write(block)
                    digest.update(block)
                    chunk_digest.update(block)
            if chunk_digest.hexdigest() != results[i]["sha256"]:
                raise ValueError("Output chunk hash mismatch during assembly")
        target.flush()
        os.fsync(target.fileno())
    if fingerprint(TRAIN) != audit_manifest["train"]:
        raise ValueError("Training input changed during assembly")
    with (OUTPUT / "quarantined.csv").open("wb") as rejected, TRAIN.open("rb") as source:
        rejected.write(source.readline())
        for i in range(args.chunks):
            with (OUTPUT / "chunks" / f"{i:04d}.quarantine.csv").open("rb") as handle:
                shutil.copyfileobj(handle, rejected)
    removed_by_source = Counter()
    for (source, _), n in observed.items():
        removed_by_source[source] += n
    report = {
        "status": "complete", "input": str(TRAIN), "output": str(OUTPUT / "merged_train.csv"),
        "input_rows": sum(source_rows.values()), "removed_rows": sum(observed.values()),
        "remaining_rows": sum(source_rows.values()) - sum(observed.values()),
        "preserved_sources": sorted(PRESERVE), "conversion_failure_rows_retained": failures - sum(quarantine.values()),
        "quarantined_rows": sum(quarantine.values()), "overlap_removed_rows": overlap_expected,
        "reference_labels": sorted(set(args.reference_label)), "reference_mask": reference_mask,
        "removed_by_source": dict(sorted(removed_by_source.items())),
        "unique_removal_texts_verified": verified, "sha256": digest.hexdigest(),
        "output_bytes": final_temp.stat().st_size, "elapsed_seconds": time.monotonic()-started,
        "validation": "Regenerated removal keys; reconciled every source/text count with completed audit; verified chunk hashes during ordered assembly.",
    }
    shutil.copyfile(args.audit / "conversion_failures.csv", OUTPUT / "conversion_failures.csv")
    # No overwrite: hard-link publication fails if another file has this name.
    os.link(final_temp, OUTPUT / "merged_train.csv")
    atomic_json(OUTPUT / "report.json", report)
    final_temp.unlink()
    atomic_json(OUTPUT / "progress.json", report)
    # Only disposable data chunks created by this run are cleaned up.
    for i in range(args.chunks):
        (OUTPUT / "chunks" / f"{i:04d}.csv").unlink()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True, help="Completed audit_inchikey_overlap.py output directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory for filtered CSV and provenance")
    parser.add_argument("--preserve-source", action="append", default=[], help="Merged-name dataset prefix to leave untouched; repeatable")
    parser.add_argument("--reference-label", action="append", required=True, help="Reference label from reference_summary.json; repeatable")
    parser.add_argument("--failure-policy", choices=["error", "quarantine", "keep-reported"], default="error")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunks", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.chunks < 1:
        parser.error("workers and chunks must be positive")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
