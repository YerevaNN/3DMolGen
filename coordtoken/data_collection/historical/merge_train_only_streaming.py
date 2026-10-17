#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def normalize_name(dataset: str, idx: int, row: list[str]) -> tuple[str, str]:
    if len(row) < 2:
        raise ValueError(f"Row {idx + 2} has fewer than 2 columns")
    raw_name = row[0].strip()
    enriched_text = row[1]
    if not enriched_text:
        return "", ""
    if raw_name:
        return f"{dataset}_{raw_name}", enriched_text
    return f"{dataset}_{idx}", enriched_text


def merge_train(root: Path, output_csv: Path) -> tuple[int, int]:
    train_dir = root / "train"
    if not train_dir.is_dir():
        raise ValueError(f"Train directory not found: {train_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    missing_name_rows = 0

    with output_csv.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["name", "enriched_text"])

        for csv_path in sorted(train_dir.glob("*.csv")):
            dataset = csv_path.stem
            with csv_path.open("r", encoding="utf-8", buffering=1048576) as in_f:
                reader = csv.reader(in_f)
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                if len(header) < 2:
                    raise ValueError(f"File has fewer than 2 columns: {csv_path}")

                dataset_rows = 0
                for idx, row in enumerate(reader):
                    name, enriched_text = normalize_name(dataset, idx, row)
                    if not enriched_text:
                        continue
                    if row[0].strip() == "":
                        missing_name_rows += 1
                    writer.writerow([name, enriched_text])
                    rows_written += 1
                    dataset_rows += 1
                print(f"train/{dataset}: merged_rows={dataset_rows:,}", flush=True)

    return rows_written, missing_name_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream-merge only the train split into one merged_train.csv without loading the full split into memory."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/nfs/h100/raid/chem/3D_big_data_new_dedup_no_geom_revisited"),
        help="Dataset root containing train/val/test directories.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/nfs/h100/raid/chem/3D_big_data_new_dedup_no_geom_revisited/merged_train.csv"),
        help="Merged train CSV path.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.is_dir():
        raise ValueError(f"Root directory not found: {root}")

    rows_written, missing_name_rows = merge_train(root=root, output_csv=args.output_csv)
    print(f"train total rows={rows_written:,} empty_name_rows={missing_name_rows:,}", flush=True)
    print(f"Saved merged train CSV to: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
