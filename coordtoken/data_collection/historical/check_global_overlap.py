#!/usr/bin/env python3
from pathlib import Path
from typing import Set
import csv
import argparse

from tqdm.auto import tqdm

from split_excluded_datasets import extract_smiles


def load_smiles_from_files(files: list[Path], desc: str) -> Set[str]:
    smiles: Set[str] = set()
    for path in tqdm(files, desc=desc, mininterval=5):
        with path.open("r", encoding="utf-8", buffering=1048576) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "enriched_text" not in reader.fieldnames:
                raise ValueError(f"Invalid header in {path}")
            for row in reader:
                s = extract_smiles(row["enriched_text"])
                if s:
                    smiles.add(s)
    return smiles


def get_split_files(root: Path, split: str) -> list[Path]:
    split_dir = root / split
    if not split_dir.is_dir():
        raise ValueError(f"{split} directory not found: {split_dir}")
    files = sorted(split_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files in {split_dir}")
    return files


def load_split_smiles(root: Path, split: str) -> Set[str]:
    files = get_split_files(root=root, split=split)
    return load_smiles_from_files(files=files, desc=f"{split} SMILES")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/nfs/h100/raid/chem/3D_big_data"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to output CSV file (default: global_overlap_sizes.csv in current directory)",
    )
    parser.add_argument(
        "--ignore-val-dataset-vs-other-train",
        type=str,
        default=None,
        help=(
            "Optional dataset name in val whose overlaps with non-matching train datasets "
            "should be excluded from the reported train∩val count."
        ),
    )
    args = parser.parse_args()
    root: Path = args.root
    if not root.is_dir():
        raise ValueError(f"Root directory not found: {root}")
    train_files = get_split_files(root=root, split="train")
    val_files = get_split_files(root=root, split="val")
    train_smiles = load_smiles_from_files(files=train_files, desc="train SMILES")
    val_smiles = load_smiles_from_files(files=val_files, desc="val SMILES")
    test_smiles = load_split_smiles(root=root, split="test")
    print(f"Unique SMILES counts:")
    print(f"  train: {len(train_smiles)}")
    print(f"  val:   {len(val_smiles)}")
    print(f"  test:  {len(test_smiles)}")
    train_val = train_smiles & val_smiles
    train_test = train_smiles & test_smiles
    val_test = val_smiles & test_smiles
    ignored_train_val: Set[str] = set()
    ignored_dataset = args.ignore_val_dataset_vs_other_train
    if ignored_dataset:
        val_dataset_path = root / "val" / f"{ignored_dataset}.csv"
        if not val_dataset_path.exists():
            raise ValueError(f"Val dataset file not found: {val_dataset_path}")
        other_train_files = [path for path in train_files if path.stem != ignored_dataset]
        if not other_train_files:
            raise ValueError(f"No non-{ignored_dataset} train files found in {root / 'train'}")
        val_dataset_smiles = load_smiles_from_files(
            files=[val_dataset_path],
            desc=f"{ignored_dataset} val SMILES",
        )
        other_train_smiles = load_smiles_from_files(
            files=other_train_files,
            desc=f"other train SMILES excluding {ignored_dataset}",
        )
        ignored_train_val = val_dataset_smiles & other_train_smiles
        train_val = train_val.difference(ignored_train_val)

    print("\nGlobal overlaps (all datasets in root):")
    print(f"  train ∩ val:   {len(train_val)} unique SMILES")
    print(f"  train ∩ test:  {len(train_test)} unique SMILES")
    print(f"  val ∩ test:    {len(val_test)} unique SMILES")
    if ignored_dataset:
        print(
            f"\nIgnored from train ∩ val: {len(ignored_train_val)} unique SMILES "
            f"(val/{ignored_dataset}.csv vs non-{ignored_dataset} train files)"
        )
    output_csv = args.output_csv
    if output_csv is None:
        output_csv = Path("/auto/home/filya/fsq/global_overlap_sizes.csv")
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "count"])
        writer.writeheader()
        writer.writerow({"metric": "train_unique_smiles", "count": len(train_smiles)})
        writer.writerow({"metric": "val_unique_smiles", "count": len(val_smiles)})
        writer.writerow({"metric": "test_unique_smiles", "count": len(test_smiles)})
        writer.writerow({"metric": "train_val_overlap", "count": len(train_val)})
        writer.writerow({"metric": "train_test_overlap", "count": len(train_test)})
        writer.writerow({"metric": "val_test_overlap", "count": len(val_test)})
        if ignored_dataset:
            writer.writerow({"metric": f"ignored_train_val_overlap_{ignored_dataset}", "count": len(ignored_train_val)})
    print(f"\nSaved sizes to: {output_csv}")


if __name__ == "__main__":
    main()
