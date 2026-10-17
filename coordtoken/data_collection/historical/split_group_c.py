#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

from tqdm.auto import tqdm

COORD_PATTERN = re.compile(r"<[^>]+>")
def log(msg: str) -> None:
    print(msg, flush=True)


def extract_smiles(enriched_text: str) -> str:
    return COORD_PATTERN.sub("", enriched_text)


def alias_candidates(path: Path) -> List[Path]:
    candidates = [path]
    s = str(path)
    if s.startswith("/raid/"):
        candidates.append(Path("/nfs/h100") / s.lstrip("/"))
    elif s.startswith("/nfs/h100/raid/"):
        candidates.append(Path(s[len("/nfs/h100") :]))

    seen: Set[str] = set()
    deduped: List[Path] = []
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
    if path.exists():
        return path
    for candidate in alias_candidates(path):
        if candidate.parent.exists():
            return candidate
    return path


def count_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    total = 0
    with path.open("r", encoding="utf-8", buffering=1048576) as f:
        reader = csv.reader(f)
        next(reader, None)
        for _ in reader:
            total += 1
    return total
def load_smiles(csv_path: Path) -> Set[str]:
    smiles: Set[str] = set()
    if not csv_path.exists():
        return smiles
    with csv_path.open("r", encoding="utf-8", buffering=1048576) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "enriched_text" not in reader.fieldnames:
            raise ValueError(f"Invalid header in {csv_path}")
        for row in reader:
            s = extract_smiles(row.get("enriched_text", ""))
            if s:
                smiles.add(s)
    return smiles


def load_smiles_from_dir(split_dir: Path, desc: str, exclude_prefixes: Tuple[str, ...] = ()) -> Set[str]:
    if not split_dir.is_dir():
        raise ValueError(f"Split directory not found: {split_dir}")
    files = sorted(
        path
        for path in split_dir.glob("*.csv")
        if not any(path.stem.startswith(prefix) for prefix in exclude_prefixes)
    )
    if not files:
        raise ValueError(f"No CSV files found in {split_dir}")
    smiles: Set[str] = set()
    for path in tqdm(files, desc=desc, mininterval=5):
        smiles.update(load_smiles(path))
    return smiles


def pick_smiles(
    pool: Set[str],
    target_rows: int,
    rows_per_smiles: Dict[str, int],
    rng: random.Random,
) -> Tuple[Set[str], int]:
    if target_rows <= 0:
        return set(), 0
    candidates = [s for s in pool if rows_per_smiles.get(s, 0) > 0]
    rng.shuffle(candidates)
    selected: Set[str] = set()
    moved_rows = 0
    for smiles in candidates:
        if moved_rows >= target_rows:
            break
        selected.add(smiles)
        moved_rows += rows_per_smiles[smiles]
    return selected, moved_rows
def scan_nabla_train(
    train_path: Path,
    other_val_smiles: Set[str],
    forbidden_train_smiles: Set[str],
) -> Tuple[Dict[str, int], Set[str], Set[str], int, int, int]:
    rows_per_smiles: Dict[str, int] = {}
    val_candidate_smiles: Set[str] = set()
    filtered_train_smiles: Set[str] = set()
    scanned_rows = 0
    forbidden_train_rows = 0
    skipped_rows = 0

    with train_path.open("r", encoding="utf-8", buffering=1048576) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "enriched_text" not in reader.fieldnames:
            raise ValueError(f"Invalid header in {train_path}")
        for row in tqdm(reader, desc="Scanning nablaDFT train", mininterval=5):
            enriched = row.get("enriched_text", "")
            if not enriched:
                skipped_rows += 1
                continue
            smiles = extract_smiles(enriched)
            if not smiles:
                skipped_rows += 1
                continue
            rows_per_smiles[smiles] = rows_per_smiles.get(smiles, 0) + 1
            scanned_rows += 1
            if smiles in other_val_smiles:
                val_candidate_smiles.add(smiles)
            elif smiles in forbidden_train_smiles:
                forbidden_train_rows += 1
            else:
                filtered_train_smiles.add(smiles)

    return (
        rows_per_smiles,
        val_candidate_smiles,
        filtered_train_smiles,
        scanned_rows,
        forbidden_train_rows,
        skipped_rows,
    )


def write_nabla_splits(
    source_train_path: Path,
    output_root: Path,
    dataset: str,
    train_smiles: Set[str],
    val_smiles: Set[str],
) -> Dict[str, int]:
    output_root.mkdir(parents=True, exist_ok=True)
    train_dir = output_root / "train"
    val_dir = output_root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    tmp_train = train_dir / f"{dataset}.tmp.csv"
    tmp_val = val_dir / f"{dataset}.tmp.csv"
    final_train = train_dir / f"{dataset}.csv"
    final_val = val_dir / f"{dataset}.csv"

    stats = {"train_rows": 0, "val_rows": 0, "dropped_rows": 0}

    with source_train_path.open("r", encoding="utf-8", buffering=1048576) as in_f, \
        tmp_train.open("w", newline="", encoding="utf-8", buffering=1048576) as out_train, \
        tmp_val.open("w", newline="", encoding="utf-8", buffering=1048576) as out_val:
        reader = csv.DictReader(in_f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {source_train_path}")
        fieldnames = reader.fieldnames
        train_writer = csv.DictWriter(out_train, fieldnames=fieldnames)
        val_writer = csv.DictWriter(out_val, fieldnames=fieldnames)
        train_writer.writeheader()
        val_writer.writeheader()

        for row in tqdm(reader, desc="Writing new nablaDFT splits", mininterval=5):
            enriched = row.get("enriched_text", "")
            smiles = extract_smiles(enriched) if enriched else ""
            if not smiles:
                stats["dropped_rows"] += 1
                continue
            if smiles in val_smiles:
                val_writer.writerow(row)
                stats["val_rows"] += 1
            elif smiles in train_smiles:
                train_writer.writerow(row)
                stats["train_rows"] += 1
            else:
                stats["dropped_rows"] += 1

    tmp_train.replace(final_train)
    tmp_val.replace(final_val)
    return stats
def write_stats_csv(stats: Dict[str, int | str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
        writer.writeheader()
        writer.writerow(stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split Group C (nablaDFT): first take Nabla val candidates whose SMILES appear in the "
            "reference val split, then filter Nabla train against reference val/test SMILES, then "
            "fill the remaining val quota from the filtered train if needed."
        )
    )
    parser.add_argument("--reference-root", type=Path, default=Path("/nfs/h100/raid/chem/3D_big_data_new"))
    parser.add_argument("--group-c-root", type=Path, default=Path("/nfs/h100/raid/chem/3D_big_data/grp_c"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dataset", default="nablaDFT")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-rows",
        type=int,
        default=63381,
        help="Target number of Nabla val rows. Default 63381 matches 0.5% of the whole NablaDFT data.",
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for split statistics (default: <output-root>/group_c_split_stats.csv).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    reference_root = resolve_existing_path(args.reference_root)
    group_c_root = resolve_existing_path(args.group_c_root)
    output_root = reference_root if args.output_root is None else resolve_output_path(args.output_root)
    stats_csv = args.stats_csv if args.stats_csv is not None else output_root / "group_c_split_stats.csv"
    stats_csv = resolve_output_path(stats_csv)

    if not reference_root.is_dir():
        raise ValueError(f"Reference root not found: {reference_root}")
    if not group_c_root.is_dir():
        raise ValueError(f"Group C root not found: {group_c_root}")

    source_train_path = group_c_root / "train" / f"{args.dataset}.csv"
    if not source_train_path.exists():
        raise FileNotFoundError(f"Missing group C train file: {source_train_path}")

    log(f"reference_root={reference_root}")
    log(f"group_c_root={group_c_root}")
    log(f"output_root={output_root}")
    log(f"dataset={args.dataset}")
    log(f"seed={args.seed}")
    log(f"target_rows={args.target_rows}")

    log("Loading reference val/test SMILES (Group A+B only)...")
    reference_val = load_smiles_from_dir(
        reference_root / "val",
        desc="reference val SMILES",
        exclude_prefixes=(args.dataset,),
    )
    reference_test = load_smiles_from_dir(
        reference_root / "test",
        desc="reference test SMILES",
        exclude_prefixes=(args.dataset,),
    )

    forbidden_train_smiles = reference_val.union(reference_test)

    original_train_rows = count_rows(source_train_path)
    target_val_rows = args.target_rows

    log(f"Original NablaDFT train rows: {original_train_rows:,}")
    log(f"Target NablaDFT val rows: {target_val_rows:,}")

    (
        rows_per_smiles,
        val_candidate_smiles,
        filtered_train_smiles,
        scanned_train_rows,
        forbidden_train_rows,
        skipped_rows,
    ) = scan_nabla_train(
        train_path=source_train_path,
        other_val_smiles=reference_val,
        forbidden_train_smiles=forbidden_train_smiles,
    )
    if not filtered_train_smiles and not val_candidate_smiles:
        raise ValueError("No candidate NablaDFT train SMILES remain after filtering against Group A+B val/test")

    selected_val: Set[str] = set()
    val_candidate_rows = sum(rows_per_smiles[s] for s in val_candidate_smiles)
    remaining_val_rows = target_val_rows
    if val_candidate_rows >= remaining_val_rows:
        selected_val, selected_val_rows = pick_smiles(
            pool=val_candidate_smiles,
            target_rows=remaining_val_rows,
            rows_per_smiles=rows_per_smiles,
            rng=rng,
        )
        remaining_val_rows -= selected_val_rows
    else:
        selected_val = set(val_candidate_smiles)
        selected_val_rows = val_candidate_rows
        remaining_val_rows -= val_candidate_rows

    extra_val: Set[str] = set()
    if remaining_val_rows > 0:
        extra_val, extra_val_rows = pick_smiles(
            pool=filtered_train_smiles,
            target_rows=remaining_val_rows,
            rows_per_smiles=rows_per_smiles,
            rng=rng,
        )
        selected_val.update(extra_val)
        selected_val_rows += extra_val_rows

    selected_train = filtered_train_smiles.difference(extra_val)
    filtered_train_rows = sum(rows_per_smiles[s] for s in selected_train)
    candidate_val_rows_used = sum(rows_per_smiles[s] for s in selected_val.intersection(val_candidate_smiles))
    extra_val_rows_used = selected_val_rows - candidate_val_rows_used

    log(f"Scanned valid NablaDFT train rows: {scanned_train_rows:,}")
    log(f"Nabla val candidate rows from reference val overlap: {val_candidate_rows:,}")
    log(f"Nabla train rows excluded by reference test overlap: {forbidden_train_rows:,}")
    log(f"Filtered NablaDFT train rows kept after val selection: {filtered_train_rows:,}")
    log(
        f"Selected NablaDFT val: {selected_val_rows:,} rows total "
        f"({candidate_val_rows_used:,} from val candidates, {extra_val_rows_used:,} filled from filtered train)"
    )

    row_stats = write_nabla_splits(
        source_train_path=source_train_path,
        output_root=output_root,
        dataset=args.dataset,
        train_smiles=selected_train,
        val_smiles=selected_val,
    )

    final_stats: Dict[str, int | str] = {
        "dataset": args.dataset,
        "source_train_rows": original_train_rows,
        "target_val_rows": target_val_rows,
        "reference_val_unique_smiles": len(reference_val),
        "reference_test_unique_smiles": len(reference_test),
        "forbidden_train_unique_smiles": len(forbidden_train_smiles),
        "val_candidate_unique_smiles": len(val_candidate_smiles),
        "scanned_train_rows": scanned_train_rows,
        "val_candidate_rows": val_candidate_rows,
        "forbidden_train_rows": forbidden_train_rows,
        "selected_val_unique_smiles": len(selected_val),
        "selected_train_unique_smiles": len(selected_train),
        "selected_val_rows": selected_val_rows,
        "selected_val_candidate_rows": candidate_val_rows_used,
        "selected_extra_val_rows": extra_val_rows_used,
        "filtered_train_unique_smiles": len(selected_train),
        "filtered_train_rows": filtered_train_rows,
        "skipped_train_rows": skipped_rows,
        "output_train_rows": row_stats["train_rows"],
        "output_val_rows": row_stats["val_rows"],
        "dropped_rows_while_writing": row_stats["dropped_rows"],
    }
    write_stats_csv(stats=final_stats, output_path=stats_csv)

    log(f"Final NablaDFT sizes written: train={row_stats['train_rows']:,}, val={row_stats['val_rows']:,}")
    log(f"Saved stats CSV to {stats_csv}")
    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"Error: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
