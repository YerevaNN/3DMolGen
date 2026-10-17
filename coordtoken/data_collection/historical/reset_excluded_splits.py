#!/usr/bin/env python3
import sys
import shutil
from pathlib import Path
from typing import List, Optional
import argparse

from split_excluded_datasets import EXCLUDED_DATASETS


def ensure_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if not path.is_dir():
        raise NotADirectoryError(f"{path} is not a directory")


def backup_file(src: Path, dst_dir: Path) -> None:
    if not src.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        print(f"Backup already exists, skipping: {dst}")
        return
    shutil.copy2(src, dst)


def collect_tokenizer_inputs(
    tok_root: Path,
    dataset: str,
) -> List[Path]:
    train_dir = tok_root / "train"
    val_dir = tok_root / "val"
    test_dir = tok_root / "test"
    ensure_dir(train_dir)
    inputs: List[Path] = []
    train_path = train_dir / f"{dataset}.csv"
    if train_path.exists():
        inputs.append(train_path)
    if val_dir.exists():
        val_path = val_dir / f"{dataset}.csv"
        if val_path.exists():
            inputs.append(val_path)
    if test_dir.exists():
        test_path = test_dir / f"{dataset}.csv"
        if test_path.exists():
            inputs.append(test_path)
    return inputs


def backup_and_merge_tokenizer(
    tok_root: Path,
    backup_root: Path,
) -> None:
    ensure_dir(tok_root)
    train_dir = tok_root / "train"
    ensure_dir(train_dir)
    for dataset in sorted(EXCLUDED_DATASETS):
        inputs = collect_tokenizer_inputs(tok_root=tok_root, dataset=dataset)
        if not inputs:
            print(f"No TokenizerData files found for {dataset}, skipping")
            continue
        for path in inputs:
            split_name = path.parent.name
            backup_dir = backup_root / dataset / split_name
            backup_file(src=path, dst_dir=backup_dir)
            print(f"Backed up {path} to {backup_dir}")
        print(f"Skipping on-disk merge for {dataset}")


def merge_csv_files(inputs: List[Path], output_path: Path) -> None:
    existing_inputs = [p for p in inputs if p.exists()]
    if not existing_inputs:
        raise FileNotFoundError("No input files found for merge")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    header_line = None

    with open(output_path, "w", encoding="utf-8", buffering=1048576) as out_f:
        for in_path in existing_inputs:
            with open(in_path, "r", encoding="utf-8", buffering=1048576) as in_f:
                first_line = in_f.readline()
                if not first_line:
                    continue
                if header_line is None:
                    header_line = first_line
                    out_f.write(header_line)
                else:
                    if first_line.rstrip("\n") != header_line.rstrip("\n"):
                        raise ValueError(f"Header mismatch between files: {existing_inputs[0]} and {in_path}")
                for chunk in in_f:
                    out_f.write(chunk)
                    rows_written += 1

    print(f"merge_csv_files: merged {len(existing_inputs)} files into {output_path}")
    print(f"merge_csv_files: total rows written (excluding header): {rows_written}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer-data",
        type=Path,
        default=Path("/nfs/dgx/raid/chem/TokenizerData"),
    )
    parser.add_argument(
        "--backup-tokenizer",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    tok_root: Path = args.tokenizer_data
    backup_root: Optional[Path] = args.backup_tokenizer
    if backup_root is None:
        backup_root = tok_root / "backup_excluded_splits"
    print(f"tokenizer_data={tok_root}")
    print(f"backup_tokenizer={backup_root}")
    backup_and_merge_tokenizer(tok_root=tok_root, backup_root=backup_root)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise
