#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict


def count_lines(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    n = 0
    with path.open("r", encoding="utf-8", buffering=1048576) as f:
        for _ in f:
            n += 1
    return n


def collect_counts(root: Path) -> Dict[str, Dict[str, int]]:
    dataset_counts: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.glob("*.csv")):
            dataset = path.stem
            n_rows = max(0, count_lines(path) - 1)
            dataset_counts.setdefault(dataset, {"train": 0, "val": 0, "test": 0})
            dataset_counts[dataset][split] = n_rows
    return dataset_counts


def write_counts_csv(dataset_counts: Dict[str, Dict[str, int]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "train", "val", "test"])
        writer.writeheader()
        for dataset in sorted(dataset_counts):
            counts = dataset_counts[dataset]
            writer.writerow(
                {
                    "dataset": dataset,
                    "train": counts["train"],
                    "val": counts["val"],
                    "test": counts["test"],
                }
            )


def print_summary(dataset_counts: Dict[str, Dict[str, int]]) -> None:
    total_train = sum(counts["train"] for counts in dataset_counts.values())
    total_val = sum(counts["val"] for counts in dataset_counts.values())
    total_test = sum(counts["test"] for counts in dataset_counts.values())

    print("dataset,train,val,test")
    for dataset in sorted(dataset_counts):
        counts = dataset_counts[dataset]
        print(f"{dataset},{counts['train']},{counts['val']},{counts['test']}")
    print()
    print(f"Totals: train={total_train:,} val={total_val:,} test={total_test:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Save split sizes for /nfs/h100/raid/chem/3D_big_data_new to CSV.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/nfs/h100/raid/chem/3D_big_data_new"),
        help="Dataset root containing train/val/test directories.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/auto/home/filya/fsq/3d_big_data_new_split_counts.csv"),
        help="Where to save the dataset size CSV.",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        raise ValueError(f"Root directory not found: {args.root}")

    dataset_counts = collect_counts(root=args.root)
    write_counts_csv(dataset_counts=dataset_counts, output_csv=args.output_csv)
    print_summary(dataset_counts=dataset_counts)
    print(f"\nSaved CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()
