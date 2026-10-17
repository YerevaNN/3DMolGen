#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple


def read_split_rows(roots: List[Path], split: str) -> Tuple[List[Tuple[str, str]], int]:
    rows: List[Tuple[str, str]] = []
    missing_name_count: int = 0
    for root in roots:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.glob("*.csv")):
            dataset = path.stem
            with path.open("r", encoding="utf-8", buffering=1048576) as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                if len(header) < 2:
                    raise ValueError(f"File has fewer than 2 columns: {path}")
                name_idx = 0
                text_idx = 1
                for idx, row in enumerate(reader):
                    if len(row) <= max(name_idx, text_idx):
                        raise ValueError(f"Row {idx + 2} in {path} has fewer columns than expected")
                    orig_name = row[name_idx].strip()
                    text = row[text_idx]
                    if not text:
                        continue
                    if orig_name:
                        name = f"{dataset}_{orig_name}"
                    else:
                        name = f"{dataset}_{idx}"
                        missing_name_count += 1
                    rows.append((name, text))
    return rows, missing_name_count


def write_split(rows: List[Tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "enriched_text"])
        for name, text in rows:
            writer.writerow([name, text])


def parse_splits(raw: str) -> List[str]:
    splits = [part.strip() for part in raw.split(",") if part.strip()]
    if not splits:
        raise ValueError("At least one split must be provided")
    allowed = {"train", "val", "test"}
    invalid = [split for split in splits if split not in allowed]
    if invalid:
        raise ValueError(f"Invalid split(s): {', '.join(invalid)}")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=[Path("/raid/chem/3D_big_data")],
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("/raid/chem/3D_big_data/merged"),
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to merge. Allowed: train,val,test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
    )
    args = parser.parse_args()
    roots: List[Path] = args.roots
    output_prefix: Path = args.output_prefix
    splits = parse_splits(args.splits)
    seed: int = args.seed
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Root directory not found: {root}")
    rng = random.Random(seed)
    for split in splits:
        rows, missing_name_count = read_split_rows(roots=roots, split=split)
        print(f"{split}: rows={len(rows)}, empty_name_rows={missing_name_count}")
        rng.shuffle(rows)
        out_path = output_prefix.with_name(output_prefix.name + f"_{split}.csv")
        write_split(rows=rows, path=out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
