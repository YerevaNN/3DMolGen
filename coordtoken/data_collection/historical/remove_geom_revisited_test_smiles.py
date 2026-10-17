#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

from tqdm.auto import tqdm


COORD_PATTERN = re.compile(r"<[^>]+>")
DEFAULT_INPUT_ROOT = Path("/nfs/h100/raid/chem/3D_big_data_new_dedup")
DEFAULT_REFERENCE_CSV = Path("/nfs/h100/raid/chem/3D_big_data/grp_d/test/geom_revisited_test.csv")
DEFAULT_OUTPUT_ROOT = Path("/nfs/h100/raid/chem/3D_big_data_new_dedup_no_geom_revisited")
DEFAULT_EXCLUDE_DATASETS = ("geom_revisited_test",)


def log(msg: str) -> None:
    print(msg, flush=True)


def alias_candidates(path: Path) -> list[Path]:
    candidates = [path]
    s = str(path)

    if s.startswith("/home/"):
        candidates.append(Path("/auto") / s.lstrip("/"))
    elif s.startswith("/auto/home/"):
        candidates.append(Path(s[len("/auto") :]))

    if s.startswith("/raid/"):
        candidates.append(Path("/nfs/h100") / s.lstrip("/"))
    elif s.startswith("/nfs/h100/raid/"):
        candidates.append(Path(s[len("/nfs/h100") :]))

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def resolve_existing_path(path: Path) -> Path:
    for candidate in alias_candidates(path):
        if candidate.exists():
            return candidate
    return path


def resolve_output_path(path: Path) -> Path:
    for candidate in alias_candidates(path):
        ancestor = candidate
        while ancestor != ancestor.parent and not ancestor.exists():
            ancestor = ancestor.parent
        if ancestor.exists():
            return candidate
    return path


def extract_smiles(enriched_text: str) -> str:
    return COORD_PATTERN.sub("", enriched_text)


def load_reference_smiles(reference_csv: Path) -> Set[str]:
    resolved_csv = resolve_existing_path(reference_csv)
    if not resolved_csv.exists():
        raise ValueError(f"Reference CSV not found: {reference_csv}")

    smiles_set: Set[str] = set()
    with resolved_csv.open("r", encoding="utf-8", buffering=1048576) as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading Group D reference SMILES", mininterval=5):
            smiles = (row.get("smiles") or "").strip()
            if not smiles:
                enriched_text = row.get("enriched_text", "")
                if enriched_text:
                    smiles = extract_smiles(enriched_text)
            if smiles:
                smiles_set.add(smiles)

    if not smiles_set:
        raise ValueError(f"No reference SMILES could be loaded from {resolved_csv}")
    return smiles_set


def list_tasks(input_root: Path, output_root: Path, exclude_datasets: Set[str]) -> list[tuple[Path, Path]]:
    tasks: list[tuple[Path, Path]] = []
    for split in ("train", "val", "test"):
        split_dir = input_root / split
        if not split_dir.is_dir():
            raise ValueError(f"Split directory not found: {split_dir}")
        for input_csv in sorted(split_dir.glob("*.csv")):
            if input_csv.stem in exclude_datasets:
                continue
            output_csv = output_root / split / input_csv.name
            tasks.append((input_csv, output_csv))
    return tasks


def process_file(args: Tuple[Path, Path, Set[str]]) -> Tuple[Path, Dict[str, int]]:
    input_csv, output_csv, forbidden_smiles = args

    stats = {
        "original_rows": 0,
        "removed_rows": 0,
        "kept_rows": 0,
        "unique_original_smiles": 0,
        "unique_removed_smiles": 0,
        "unique_kept_smiles": 0,
    }
    original_smiles: Set[str] = set()
    removed_smiles: Set[str] = set()
    kept_smiles: Set[str] = set()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open("r", encoding="utf-8", buffering=1048576) as in_f:
        reader = csv.DictReader(in_f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Missing header in {input_csv}")

        with output_csv.open("w", newline="", encoding="utf-8", buffering=1048576) as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                stats["original_rows"] += 1
                enriched_text = row.get("enriched_text", "")
                if not enriched_text:
                    writer.writerow(row)
                    stats["kept_rows"] += 1
                    continue

                smiles = extract_smiles(enriched_text)
                if not smiles:
                    writer.writerow(row)
                    stats["kept_rows"] += 1
                    continue

                original_smiles.add(smiles)
                if smiles in forbidden_smiles:
                    stats["removed_rows"] += 1
                    removed_smiles.add(smiles)
                    continue

                writer.writerow(row)
                stats["kept_rows"] += 1
                kept_smiles.add(smiles)

    stats["unique_original_smiles"] = len(original_smiles)
    stats["unique_removed_smiles"] = len(removed_smiles)
    stats["unique_kept_smiles"] = len(kept_smiles)
    return input_csv, stats


def save_summary(rows: Iterable[dict[str, int | str]], summary_csv: Path) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "dataset",
                "original_rows",
                "removed_rows",
                "kept_rows",
                "unique_original_smiles",
                "unique_removed_smiles",
                "unique_kept_smiles",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_comma_list(raw: str | None, fallback: tuple[str, ...]) -> Set[str]:
    if raw is None:
        return set(fallback)
    return {part.strip() for part in raw.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a flat train/val/test dataset root by removing any rows whose base SMILES "
            "exist in the geom revisited Group D test set."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Flat dataset root containing train/val/test CSV files.",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=DEFAULT_REFERENCE_CSV,
        help="Group D test CSV produced by build_group_d_geom_revisited_test.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Where to write the filtered train/val/test dataset root.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional summary CSV path. Defaults to <output-root>/geom_revisited_smiles_removal_summary.csv.",
    )
    parser.add_argument(
        "--exclude-datasets",
        type=str,
        default=None,
        help="Comma-separated dataset stems to skip while copying/filtering.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to cpu_count() - 2.",
    )
    args = parser.parse_args()

    input_root = resolve_existing_path(args.input_root)
    reference_csv = resolve_existing_path(args.reference_csv)
    output_root = resolve_output_path(args.output_root)
    if str(input_root) == str(output_root):
        raise ValueError("Input and output roots must be different.")

    if not input_root.is_dir():
        raise ValueError(f"Input root not found: {args.input_root}")

    summary_csv = args.summary_csv
    if summary_csv is None:
        summary_csv = output_root / "geom_revisited_smiles_removal_summary.csv"
    else:
        summary_csv = resolve_output_path(summary_csv)

    exclude_datasets = parse_comma_list(args.exclude_datasets, DEFAULT_EXCLUDE_DATASETS)
    workers = args.workers
    if workers is None:
        workers = max(1, cpu_count() - 2)

    forbidden_smiles = load_reference_smiles(reference_csv)
    log(f"Loaded {len(forbidden_smiles):,} unique Group D reference SMILES from {reference_csv}")

    tasks = list_tasks(input_root, output_root, exclude_datasets)
    log(f"Files to process: {len(tasks):,}")

    results: list[dict[str, int | str]] = []
    pool_args = [(input_csv, output_csv, forbidden_smiles) for input_csv, output_csv in tasks]

    with Pool(processes=workers) as pool:
        for input_csv, stats in tqdm(
            pool.imap_unordered(process_file, pool_args, chunksize=1),
            total=len(pool_args),
            desc="Filtering overlaps",
            mininterval=5,
        ):
            row: dict[str, int | str] = {
                "split": input_csv.parent.name,
                "dataset": input_csv.stem,
                **stats,
            }
            results.append(row)
            log(
                f"{row['split']}/{row['dataset']}: "
                f"removed_rows={row['removed_rows']:,}, "
                f"unique_removed_smiles={row['unique_removed_smiles']:,}"
            )

    results.sort(key=lambda row: (str(row["split"]), str(row["dataset"])))
    save_summary(results, summary_csv)
    log(f"Saved filtering summary to {summary_csv}")
    log(f"Saved filtered dataset root to {output_root}")


if __name__ == "__main__":
    main()
