#!/usr/bin/env python3
"""
Shuffle each CSV shard independently.

Usage:
  python shuffle_shards.py --in-dir /path/to/in --out-dir /path/to/out --seed 42 --pattern "zinc_w*.csv"
"""
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pattern", type=str, default="*.csv")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(args.in_dir.glob(args.pattern))
    if not csv_files:
        raise ValueError(f"No files matching '{args.pattern}' in {args.in_dir}")

    for csv_path in tqdm(csv_files, desc="SHUFFLE", unit="shard"):
        df = pd.read_csv(csv_path)
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        out_path = args.out_dir / csv_path.name
        df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
