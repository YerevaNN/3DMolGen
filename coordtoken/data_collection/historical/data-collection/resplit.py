#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, required=True, help="Base dir containing train/val/test subdirs")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output base dir for new splits")
    ap.add_argument("--filename", type=str, required=True, help="Input CSV filename (e.g. zinc.csv)")
    ap.add_argument("--out-filename", type=str, default=None, help="Output CSV filename (defaults to --filename)")
    ap.add_argument("--val-size", type=int, default=500_000)
    ap.add_argument("--test-size", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_in = args.base_dir / "train" / args.filename
    val_in = args.base_dir / "val" / args.filename
    test_in = args.base_dir / "test" / args.filename

    out_filename = args.out_filename if args.out_filename else args.filename

    for p in [train_in, val_in, test_in]:
        try:
            exists = p.exists()
        except OSError as e:
            raise OSError(f"I/O error accessing {p}: {e}") from e

        if not exists:
            raise FileNotFoundError(f"Missing: {p}")

    print(f"Loading rows from all splits for {args.filename}...")
    rows = []
    header = None

    for split_file in [train_in, val_in, test_in]:
        print(f"Reading {split_file}...")
        with open(split_file, "r", newline="") as f:
            reader = csv.reader(f)
            h = next(reader)
            if header is None:
                header = h
            for row in tqdm(reader, desc=f"  {split_file.name}", unit=" rows"):
                rows.append(row)

    total = len(rows)
    print(f"Total rows: {total:,}")

    if total < args.val_size + args.test_size:
        raise ValueError(f"Not enough rows ({total:,}) for val ({args.val_size:,}) + test ({args.test_size:,})")

    print(f"Shuffling {total:,} rows with seed={args.seed}...")
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    val_rows = rows[:args.val_size]
    test_rows = rows[args.val_size : args.val_size + args.test_size]
    train_rows = rows[args.val_size + args.test_size :]

    print(f"New splits: train={len(train_rows):,}, val={len(val_rows):,}, test={len(test_rows):,}")

    for split_name, split_rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        out_path = args.out_dir / split_name / out_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing {out_path}...")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in tqdm(split_rows, desc=f"  Writing {split_name}", unit=" rows"):
                writer.writerow(row)
        print(f"Wrote {len(split_rows):,} rows to {out_path}")

    print("Done!")

if __name__ == "__main__":
    main()
