#!/usr/bin/env python3
"""
Concatenate shuffled shards in random order into a single CSV.

Usage:
  python concat_shards.py --in-dir /path/to/shuffled --out-file /path/to/final.csv --seed 42 --pattern "zinc_w*.csv"
"""
import argparse
import csv
import random
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pattern", type=str, default="*.csv")
    args = parser.parse_args()

    args.out_file.parent.mkdir(parents=True, exist_ok=True)

    csv_files = list(args.in_dir.glob(args.pattern))
    if not csv_files:
        raise ValueError(f"No files matching '{args.pattern}' in {args.in_dir}")

    random.seed(args.seed)
    random.shuffle(csv_files)

    header_written = False
    with open(args.out_file, "w", newline="") as fout:
        writer = None
        for csv_path in tqdm(csv_files, desc="CONCAT", unit="shard"):
            with open(csv_path, "r", newline="") as fin:
                reader = csv.reader(fin)
                header = next(reader)
                if not header_written:
                    writer = csv.writer(fout)
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    writer.writerow(row)


if __name__ == "__main__":
    main()
