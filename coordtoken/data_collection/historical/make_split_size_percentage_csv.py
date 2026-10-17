#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT = Path("/auto/home/filya/fsq/3d_big_data_new_dedup_apply_summary.csv")
DEFAULT_OUTPUT = Path("/auto/home/filya/fsq/3d_big_data_new_dedup_split_sizes_percentages.csv")
SPLITS = ("train", "val", "test")
DEFAULT_NABLA_TEST_SIZE = 3907073


def pct(part: int, total: int) -> str:
    if total == 0:
        return "0.0000"
    return f"{(100.0 * part / total):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a per-file apply summary into a dataset-by-split CSV with sizes and percentages."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--nabla-test-size",
        type=int,
        default=DEFAULT_NABLA_TEST_SIZE,
        help="Test size to use for nablaDFT in the percentage table.",
    )
    args = parser.parse_args()

    input_csv = args.input_csv
    if not input_csv.exists():
        raise ValueError(f"Input CSV not found: {input_csv}")

    dataset_sizes: dict[str, dict[str, int]] = {}
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"split", "dataset", "kept_rows"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Input CSV missing required columns: {input_csv}")

        for row in reader:
            dataset = row["dataset"]
            split = row["split"]
            kept_rows = int(row["kept_rows"])
            if split not in SPLITS:
                continue
            dataset_sizes.setdefault(dataset, {name: 0 for name in SPLITS})
            dataset_sizes[dataset][split] = kept_rows

    output_rows = []
    for dataset in sorted(dataset_sizes):
        train_size = dataset_sizes[dataset]["train"]
        val_size = dataset_sizes[dataset]["val"]
        test_size = dataset_sizes[dataset]["test"]
        if dataset == "nablaDFT":
            test_size = args.nabla_test_size
        total = train_size + val_size + test_size
        output_rows.append(
            {
                "dataset": dataset,
                "train_size": train_size,
                "train_percentage": pct(train_size, total),
                "test_size": test_size,
                "test_percentage": pct(test_size, total),
                "val_size": val_size,
                "val_percentage": pct(val_size, total),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "train_size",
                "train_percentage",
                "test_size",
                "test_percentage",
                "val_size",
                "val_percentage",
            ],
        )
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    print(f"Saved split size/percentage CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
