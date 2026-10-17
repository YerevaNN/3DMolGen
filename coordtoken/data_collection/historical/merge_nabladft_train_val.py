#!/usr/bin/env python3
from pathlib import Path
from typing import List
import csv
import argparse


def merge_files(input_paths: List[Path], output_path: Path) -> None:
    existing_inputs = [p for p in input_paths if p.exists()]
    if not existing_inputs:
        raise FileNotFoundError("No input files found for nablaDFT merge")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = None
    rows_written = 0

    with open(output_path, "w", newline="", encoding="utf-8", buffering=1048576) as out_f:
        writer = None

        for in_path in existing_inputs:
            with open(in_path, "r", encoding="utf-8", buffering=1048576) as in_f:
                reader = csv.DictReader(in_f)
                if fieldnames is None:
                    if reader.fieldnames is None:
                        raise ValueError(f"Missing header in {in_path}")
                    fieldnames = reader.fieldnames
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()
                else:
                    if reader.fieldnames != fieldnames:
                        raise ValueError(f"Header mismatch between files: {input_paths[0]} and {in_path}")

                for row in reader:
                    writer.writerow(row)
                    rows_written += 1

    print(f"Merged {len(existing_inputs)} files into {output_path}")
    print(f"Total rows written (excluding header): {rows_written}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/nfs/dgx/raid/chem/TokenizerData"),
    )
    args = parser.parse_args()

    root = args.root
    train_backup = root / "train_nabla_backup.csv"
    val_backup = root / "val_nabla_backup.csv"

    if not train_backup.exists():
        raise FileNotFoundError(f"Expected train backup not found: {train_backup}")
    if not val_backup.exists():
        raise FileNotFoundError(f"Expected val backup not found: {val_backup}")

    output_path = root / "train" / "nablaDFT.csv"
    merge_files([train_backup, val_backup], output_path)


if __name__ == "__main__":
    main()
