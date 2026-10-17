#!/usr/bin/env python3
from pathlib import Path
from typing import Dict
import csv
import argparse


def count_lines(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    n = 0
    with path.open("r", encoding="utf-8", buffering=1048576) as f:
        for _ in f:
            n += 1
    return n


def collect_counts(root: Path) -> Dict[str, Dict[str, int]]:
    splits = ("train", "val", "test")
    dataset_counts: Dict[str, Dict[str, int]] = {}
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.glob("*.csv")):
            dataset = path.stem
            n = count_lines(path=path)
            if dataset not in dataset_counts:
                dataset_counts[dataset] = {"train": 0, "val": 0, "test": 0}
            dataset_counts[dataset][split] = max(0, n - 1)
    return dataset_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/nfs/h100/raid/chem/3D_big_data"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/auto/home/filya/fsq/split_line_counts.csv"),
    )
    args = parser.parse_args()
    root: Path = args.root
    output_csv: Path = args.output_csv
    if not root.is_dir():
        raise ValueError(f"Root directory not found: {root}")
    dataset_counts = collect_counts(root=root)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "train", "val", "test"])
        writer.writeheader()
        for dataset in sorted(dataset_counts):
            counts = dataset_counts[dataset]
            writer.writerow({
                "dataset": dataset,
                "train": counts["train"],
                "val": counts["val"],
                "test": counts["test"],
            })


if __name__ == "__main__":
    main()
