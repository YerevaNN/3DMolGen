#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, List, Set, Tuple
import argparse
import csv

from tqdm.auto import tqdm

def scan_file(
    path: Path,
    global_seen: Set[str],
    global_duplicated: Set[str],
) -> Tuple[Dict[str, int | str], int]:
    file_total_rows = 0
    file_duplicate_rows = 0
    file_seen: Set[str] = set()
    file_duplicated: Set[str] = set()
    global_duplicate_rows = 0

    with path.open("r", encoding="utf-8", buffering=1048576) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "enriched_text" not in reader.fieldnames:
            raise ValueError(f"Invalid header in {path}")
        for row in reader:
            enriched_text = row.get("enriched_text", "")
            if not enriched_text:
                continue
            file_total_rows += 1
            if enriched_text in file_seen:
                file_duplicate_rows += 1
                file_duplicated.add(enriched_text)
            else:
                file_seen.add(enriched_text)

            if enriched_text in global_seen:
                global_duplicate_rows += 1
                global_duplicated.add(enriched_text)
            else:
                global_seen.add(enriched_text)

    return (
        {
            "dataset": path.stem,
            "total_rows": file_total_rows,
            "unique_samples": len(file_seen),
            "duplicate_rows": file_duplicate_rows,
            "duplicated_samples": len(file_duplicated),
        },
        global_duplicate_rows,
    )


def write_csv(rows: List[Dict[str, int | str]], fieldnames: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check duplicate enriched_text samples within each split of a flat train/val/test root. "
            "Duplicate rows count extra rows beyond the first occurrence of the exact same enriched_text."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/nfs/h100/raid/chem/3D_big_data_new"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("/auto/home/filya/fsq/3d_big_data_new_duplicate_sample_summary.csv"),
    )
    parser.add_argument(
        "--per-file-csv",
        type=Path,
        default=Path("/auto/home/filya/fsq/3d_big_data_new_duplicate_sample_per_file.csv"),
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.is_dir():
        raise ValueError(f"Root directory not found: {root}")

    summary_rows: List[Dict[str, int | str]] = []
    per_file_rows: List[Dict[str, int | str]] = []

    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            raise ValueError(f"{split} directory not found: {split_dir}")
        files = sorted(split_dir.glob("*.csv"))
        if not files:
            raise ValueError(f"No CSV files in {split_dir}")

        global_total_rows = 0
        global_duplicate_rows = 0
        global_seen: Set[str] = set()
        global_duplicated: Set[str] = set()

        for path in tqdm(files, desc=f"{split} duplicates", mininterval=5):
            file_stats, new_global_duplicate_rows = scan_file(
                path=path,
                global_seen=global_seen,
                global_duplicated=global_duplicated,
            )
            per_file_rows.append(
                {
                    "split": split,
                    "dataset": file_stats["dataset"],
                    "total_rows": file_stats["total_rows"],
                    "unique_samples": file_stats["unique_samples"],
                    "duplicate_rows": file_stats["duplicate_rows"],
                    "duplicated_samples": file_stats["duplicated_samples"],
                }
            )
            global_total_rows += int(file_stats["total_rows"])
            global_duplicate_rows += new_global_duplicate_rows

        summary_rows.append(
            {
                "split": split,
                "total_rows": global_total_rows,
                "unique_samples": len(global_seen),
                "duplicate_rows": global_duplicate_rows,
                "duplicated_samples": len(global_duplicated),
            }
        )

    write_csv(
        rows=summary_rows,
        fieldnames=["split", "total_rows", "unique_samples", "duplicate_rows", "duplicated_samples"],
        path=args.summary_csv,
    )
    write_csv(
        rows=per_file_rows,
        fieldnames=["split", "dataset", "total_rows", "unique_samples", "duplicate_rows", "duplicated_samples"],
        path=args.per_file_csv,
    )

    print("Duplicate sample summary by split:")
    for row in summary_rows:
        print(
            f"  {row['split']}: rows={row['total_rows']:,}, unique_samples={row['unique_samples']:,}, "
            f"duplicate_rows={row['duplicate_rows']:,}, duplicated_samples={row['duplicated_samples']:,}"
        )
    print(f"\nSaved summary CSV to: {args.summary_csv}")
    print(f"Saved per-file CSV to: {args.per_file_csv}")


if __name__ == "__main__":
    main()
