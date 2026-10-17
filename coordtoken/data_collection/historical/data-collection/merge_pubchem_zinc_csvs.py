#!/usr/bin/env python3
"""Append split CSVs into existing backup CSVs, skipping duplicate headers."""

import argparse
from pathlib import Path

from tqdm import tqdm


DATASETS = {
    "pubchem": "pubchem3d.csv",
    "zinc": "zinc.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append split CSVs onto existing backup CSVs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/nfs/dgx/raid/chem/TokenizerData"),
        help="TokenizerData root containing train/val/test subdirectories.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Directory containing the destination CSVs. Defaults to <data-root>/backup.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=["pubchem", "zinc"],
        help="Datasets to append.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Splits to append, in order.",
    )
    return parser.parse_args()


def append_split_file(source_path: Path, dest_path: Path) -> int:
    if not source_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {source_path}")
    if not dest_path.exists():
        raise FileNotFoundError(f"Missing destination CSV: {dest_path}")

    with source_path.open("r", newline="") as src:
        header = src.readline()
        if not header:
            return 0

        with dest_path.open("a", newline="") as dst:
            written = 0
            for line in tqdm(src, desc=source_path.name, unit="row"):
                dst.write(line)
                written += 1
    return written


def main() -> None:
    args = parse_args()
    backup_dir = args.backup_dir or (args.data_root / "backup")
    backup_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        filename = DATASETS[dataset]
        dest_path = backup_dir / filename
        total_written = 0

        for split in args.splits:
            source_path = args.data_root / split / filename
            print(f"Appending {source_path} -> {dest_path}")
            total_written += append_split_file(source_path, dest_path)

        print(f"Appended {total_written} rows to {dest_path}")


if __name__ == "__main__":
    main()
