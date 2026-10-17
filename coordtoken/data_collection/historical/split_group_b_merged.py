#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from tqdm.auto import tqdm

COORD_PATTERN = re.compile(r"<[^>]+>")
DEFAULT_GROUP_B_ORDER: Tuple[str, ...] = (
    "KIBA-3D",
    "OMol25_small_mols",
    "pubchem3d",
    "zinc",
)
DEFAULT_REFERENCE_ROOT = Path("/nfs/h100/raid/chem/3D_big_data_new")
DEFAULT_GROUP_B_ROOT = Path("/nfs/h100/raid/chem/3D_big_data/grp_b")


def log(msg: str) -> None:
    print(msg, flush=True)


def extract_smiles(enriched_text: str) -> str:
    return COORD_PATTERN.sub("", enriched_text)


def pick_smiles(
    pool: Set[str],
    target_rows: int,
    rows_per_smiles: Dict[str, int],
    rng: random.Random,
    exclude: Set[str] | None = None,
) -> Tuple[Set[str], int]:
    if exclude is None:
        exclude = set()
    lst = [s for s in pool if s not in exclude and rows_per_smiles.get(s, 0) > 0]
    rng.shuffle(lst)
    selected: Set[str] = set()
    moved = 0
    for s in lst:
        if moved >= target_rows:
            break
        selected.add(s)
        moved += rows_per_smiles[s]
    return selected, moved


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
        parent = candidate.parent
        if parent.exists():
            return candidate
    return path


def load_smiles(csv_path: Path) -> Set[str]:
    out: Set[str] = set()
    if not csv_path.exists():
        return out
    with csv_path.open("r", encoding="utf-8", buffering=1048576) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "enriched_text" not in reader.fieldnames:
            raise ValueError(f"Invalid header in {csv_path}")
        for row in reader:
            s = extract_smiles(row.get("enriched_text", ""))
            if s:
                out.add(s)
    return out


def iter_csv_files(dirs: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for split_dir in dirs:
        if not split_dir.is_dir():
            continue
        files.extend(sorted(split_dir.glob("*.csv")))
    return files


def load_smiles_from_dirs(dirs: Iterable[Path], desc: str) -> Set[str]:
    files = iter_csv_files(dirs)
    smiles: Set[str] = set()
    for path in tqdm(files, desc=desc, mininterval=5):
        smiles.update(load_smiles(path))
    return smiles


def discover_reference_split_dirs(base_root: Path) -> Tuple[List[Path], List[Path], List[Path], str]:
    flat_train = base_root / "train"
    flat_val = base_root / "val"
    flat_test = base_root / "test"
    if flat_train.is_dir() and flat_val.is_dir() and flat_test.is_dir():
        return [flat_train], [flat_val], [flat_test], "base_root train/val/test"

    group_a_root = base_root / "grp_a"
    group_a_train = group_a_root / "train"
    group_a_val = group_a_root / "val"
    group_a_test = group_a_root / "test"
    if group_a_train.is_dir() and group_a_val.is_dir() and group_a_test.is_dir():
        return [group_a_train], [group_a_val], [group_a_test], "grp_a train/val/test"

    raise ValueError(
        f"Could not find reference split directories under {base_root}. "
        "Expected either train/val/test or grp_a/train|val|test."
    )


def discover_group_b_datasets(group_b_root: Path) -> List[str]:
    files = sorted(group_b_root.glob("*.csv"))
    discovered = [p.stem for p in files]
    if not discovered:
        raise ValueError(f"No merged Group B CSV files found in {group_b_root}")

    known = [d for d in DEFAULT_GROUP_B_ORDER if d in discovered]
    rest = sorted(d for d in discovered if d not in known)
    return known + rest


def count_rows_for_smiles(csv_path: Path) -> Tuple[int, Dict[str, int]]:
    total = 0
    rows_per_smiles: Dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8", buffering=1048576) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "enriched_text" not in reader.fieldnames:
            raise ValueError(f"Invalid header in {csv_path}")
        for row in tqdm(reader, desc=f"{csv_path.stem} read", mininterval=5):
            total += 1
            s = extract_smiles(row.get("enriched_text", ""))
            if not s:
                continue
            rows_per_smiles[s] = rows_per_smiles.get(s, 0) + 1
    return total, rows_per_smiles


def classify_smiles(
    rows_per_smiles: Dict[str, int],
    global_train: Set[str],
    global_val: Set[str],
    global_test: Set[str],
) -> Tuple[Set[str], Set[str], Set[str], Set[str], Set[str]]:
    fixed_test: Set[str] = set()
    fixed_val: Set[str] = set()
    fixed_train: Set[str] = set()
    new_smiles: Set[str] = set()
    contaminated: Set[str] = set()

    for s in rows_per_smiles:
        in_test = s in global_test
        in_val = s in global_val
        in_train = s in global_train
        count = int(in_test) + int(in_val) + int(in_train)
        if count > 1:
            contaminated.add(s)
            continue
        if in_test:
            fixed_test.add(s)
        elif in_val:
            fixed_val.add(s)
        elif in_train:
            fixed_train.add(s)
        else:
            new_smiles.add(s)

    return fixed_test, fixed_val, fixed_train, new_smiles, contaminated


def write_split_outputs(
    input_path: Path,
    output_root: Path,
    dataset: str,
    train_smiles: Set[str],
    val_smiles: Set[str],
    test_smiles: Set[str],
) -> Dict[str, int]:
    stats = {"train_rows": 0, "val_rows": 0, "test_rows": 0}
    train_dir = output_root / "train"
    val_dir = output_root / "val"
    test_dir = output_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    tmp_train = train_dir / f"{dataset}.tmp.csv"
    tmp_val = val_dir / f"{dataset}.tmp.csv"
    tmp_test = test_dir / f"{dataset}.tmp.csv"
    final_train = train_dir / f"{dataset}.csv"
    final_val = val_dir / f"{dataset}.csv"
    final_test = test_dir / f"{dataset}.csv"

    with input_path.open("r", encoding="utf-8", buffering=1048576) as inf:
        reader = csv.DictReader(inf)
        if reader.fieldnames is None:
            raise ValueError(f"No header in {input_path}")
        fieldnames = reader.fieldnames
        with tmp_train.open("w", newline="", encoding="utf-8", buffering=1048576) as ot, \
            tmp_val.open("w", newline="", encoding="utf-8", buffering=1048576) as ov, \
            tmp_test.open("w", newline="", encoding="utf-8", buffering=1048576) as oe:
            wt = csv.DictWriter(ot, fieldnames)
            wv = csv.DictWriter(ov, fieldnames)
            we = csv.DictWriter(oe, fieldnames)
            wt.writeheader()
            wv.writeheader()
            we.writeheader()

            for row in reader:
                s = extract_smiles(row.get("enriched_text", ""))
                if not s:
                    continue
                if s in test_smiles:
                    we.writerow(row)
                    stats["test_rows"] += 1
                elif s in val_smiles:
                    wv.writerow(row)
                    stats["val_rows"] += 1
                elif s in train_smiles:
                    wt.writerow(row)
                    stats["train_rows"] += 1

    tmp_train.rename(final_train)
    tmp_val.rename(final_val)
    tmp_test.rename(final_test)
    return stats


def load_existing_outputs(output_root: Path, dataset: str) -> Tuple[Set[str], Set[str], Set[str]]:
    train_smiles = load_smiles(output_root / "train" / f"{dataset}.csv")
    val_smiles = load_smiles(output_root / "val" / f"{dataset}.csv")
    test_smiles = load_smiles(output_root / "test" / f"{dataset}.csv")
    return train_smiles, val_smiles, test_smiles


def write_stats_csv(stats_rows: Sequence[Dict[str, int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "input_rows",
        "target_rows_per_split",
        "fixed_test_smiles",
        "fixed_val_smiles",
        "fixed_train_smiles",
        "new_smiles",
        "contaminated_smiles",
        "selected_test_smiles",
        "selected_val_smiles",
        "selected_train_smiles",
        "output_train_rows",
        "output_val_rows",
        "output_test_rows",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats_rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split merged Group B datasets into train/val/test using the same iterative "
            "0.5% logic as split_excluded_datasets.py, but reading directly from grp_b/*.csv "
            "and writing into a flat train/val/test root."
        )
    )
    parser.add_argument("--base-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--group-b-root", type=Path, default=DEFAULT_GROUP_B_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=None,
        help="Optional CSV path for per-dataset split stats (default: <output-root>/group_b_split_stats.csv).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    base_root = resolve_existing_path(args.base_root)
    if not base_root.exists():
        raise ValueError(f"Base root not found: {args.base_root}")

    group_b_root = resolve_existing_path(args.group_b_root)
    if not group_b_root.is_dir():
        raise ValueError(f"Group B merged root not found: {group_b_root}")

    output_root = resolve_output_path(args.output_root)
    stats_csv = args.stats_csv if args.stats_csv is not None else output_root / "group_b_split_stats.csv"
    stats_csv = resolve_output_path(stats_csv)

    datasets = args.datasets or discover_group_b_datasets(group_b_root)
    global_train_dirs, global_val_dirs, global_test_dirs, reference_desc = discover_reference_split_dirs(base_root)

    log(f"base_root={base_root}")
    log(f"group_b_root={group_b_root}")
    log(f"output_root={output_root}")
    log(f"datasets={datasets}")
    log(f"seed={args.seed}")

    log(f"Loading global train SMILES from {reference_desc}...")
    global_train = load_smiles_from_dirs(global_train_dirs, desc="global train SMILES")
    log(f"Loaded {len(global_train):,} unique global train SMILES")

    log(f"Loading global val SMILES from {reference_desc}...")
    global_val = load_smiles_from_dirs(global_val_dirs, desc="global val SMILES")
    log(f"Loaded {len(global_val):,} unique global val SMILES")

    log(f"Loading global test SMILES from {reference_desc}...")
    global_test = load_smiles_from_dirs(global_test_dirs, desc="global test SMILES")
    log(f"Loaded {len(global_test):,} unique global test SMILES")

    output_root.mkdir(parents=True, exist_ok=True)
    stats_rows: List[Dict[str, int | str]] = []

    for dataset in datasets:
        input_path = group_b_root / f"{dataset}.csv"
        if not input_path.exists():
            raise ValueError(f"Missing merged input for {dataset}: {input_path}")

        out_train = output_root / "train" / f"{dataset}.csv"
        out_val = output_root / "val" / f"{dataset}.csv"
        out_test = output_root / "test" / f"{dataset}.csv"
        if out_train.exists() and out_val.exists() and out_test.exists():
            log(f"Skipping {dataset}: already done")
            existing_train, existing_val, existing_test = load_existing_outputs(output_root=output_root, dataset=dataset)
            global_train.update(existing_train)
            global_val.update(existing_val)
            global_test.update(existing_test)
            stats_rows.append(
                {
                    "dataset": dataset,
                    "input_rows": 0,
                    "target_rows_per_split": 0,
                    "fixed_test_smiles": 0,
                    "fixed_val_smiles": 0,
                    "fixed_train_smiles": 0,
                    "new_smiles": 0,
                    "contaminated_smiles": 0,
                    "selected_test_smiles": len(existing_test),
                    "selected_val_smiles": len(existing_val),
                    "selected_train_smiles": len(existing_train),
                    "output_train_rows": 0,
                    "output_val_rows": 0,
                    "output_test_rows": 0,
                }
            )
            continue

        log(f"Processing {dataset}...")
        input_rows, rows_per_smiles = count_rows_for_smiles(input_path)
        if input_rows == 0:
            raise ValueError(f"No rows found in {input_path}")

        fixed_test, fixed_val, fixed_train, new_smiles, contaminated = classify_smiles(
            rows_per_smiles=rows_per_smiles,
            global_train=global_train,
            global_val=global_val,
            global_test=global_test,
        )

        target_rows = max(1, int(round(input_rows * 0.005)))

        selected_test: Set[str] = set()
        needed_test = target_rows
        fixed_test_rows = sum(rows_per_smiles[s] for s in fixed_test)
        if fixed_test_rows >= needed_test:
            selected_test, moved_test = pick_smiles(fixed_test, needed_test, rows_per_smiles, rng)
            needed_test -= moved_test
        else:
            selected_test = set(fixed_test)
            needed_test -= fixed_test_rows
        if needed_test > 0:
            extra_test, _ = pick_smiles(new_smiles, needed_test, rows_per_smiles, rng)
            selected_test.update(extra_test)
            new_smiles -= extra_test

        selected_val: Set[str] = set()
        needed_val = target_rows
        fixed_val_rows = sum(rows_per_smiles[s] for s in fixed_val)
        if fixed_val_rows >= needed_val:
            selected_val, moved_val = pick_smiles(fixed_val, needed_val, rows_per_smiles, rng)
            needed_val -= moved_val
        else:
            selected_val = set(fixed_val)
            needed_val -= fixed_val_rows
        if needed_val > 0:
            extra_val, _ = pick_smiles(new_smiles, needed_val, rows_per_smiles, rng, exclude=selected_test)
            selected_val.update(extra_val)
            new_smiles -= extra_val

        selected_train = fixed_train.union(new_smiles)

        row_stats = write_split_outputs(
            input_path=input_path,
            output_root=output_root,
            dataset=dataset,
            train_smiles=selected_train,
            val_smiles=selected_val,
            test_smiles=selected_test,
        )

        global_test.update(selected_test)
        global_val.update(selected_val)
        global_train.update(selected_train)

        stats_rows.append(
            {
                "dataset": dataset,
                "input_rows": input_rows,
                "target_rows_per_split": target_rows,
                "fixed_test_smiles": len(fixed_test),
                "fixed_val_smiles": len(fixed_val),
                "fixed_train_smiles": len(fixed_train),
                "new_smiles": len(selected_train - fixed_train),
                "contaminated_smiles": len(contaminated),
                "selected_test_smiles": len(selected_test),
                "selected_val_smiles": len(selected_val),
                "selected_train_smiles": len(selected_train),
                "output_train_rows": row_stats["train_rows"],
                "output_val_rows": row_stats["val_rows"],
                "output_test_rows": row_stats["test_rows"],
            }
        )
        log(
            f"Done {dataset}: train_rows={row_stats['train_rows']:,}, "
            f"val_rows={row_stats['val_rows']:,}, test_rows={row_stats['test_rows']:,}"
        )

    write_stats_csv(stats_rows=stats_rows, output_path=stats_csv)
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
