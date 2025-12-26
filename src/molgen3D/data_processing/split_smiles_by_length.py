#!/usr/bin/env python3
"""Split SMILES JSONL entries into short/long files based on character length."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream over a SMILES JSONL file and write each non-empty line to either "
            "a 'short' or 'long' JSONL output depending on its character length."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the source JSONL file (e.g. train_smiles.jsonl).",
    )
    parser.add_argument(
        "--short-output",
        required=True,
        type=Path,
        help="Destination JSONL path for entries whose character length is below the threshold.",
    )
    parser.add_argument(
        "--long-output",
        required=True,
        type=Path,
        help="Destination JSONL path for entries whose character length is greater than or equal to the threshold.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="Character-length cutoff between 'short' and 'long' entries (default: 100).",
    )
    return parser.parse_args()


def split_smiles_by_length(
    input_path: Path,
    short_output: Path,
    long_output: Path,
    threshold: int,
) -> tuple[int, int]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    short_output.parent.mkdir(parents=True, exist_ok=True)
    long_output.parent.mkdir(parents=True, exist_ok=True)

    short_count = 0
    long_count = 0

    with input_path.open("r") as src, short_output.open("w") as short_f, long_output.open(
        "w"
    ) as long_f:
        for raw_line in src:
            line = raw_line.rstrip("\n")
            if not line:
                continue

            if len(line) < threshold:
                short_f.write(line + "\n")
                short_count += 1
            else:
                long_f.write(line + "\n")
                long_count += 1

    return short_count, long_count


def main() -> None:
    args = parse_args()
    short_count, long_count = split_smiles_by_length(
        input_path=args.input,
        short_output=args.short_output,
        long_output=args.long_output,
        threshold=args.threshold,
    )
    total = short_count + long_count
    print(
        f"Processed {total:,} lines from {args.input} "
        f"(short: {short_count:,}, long: {long_count:,}, threshold: {args.threshold})"
    )


if __name__ == "__main__":
    main()

