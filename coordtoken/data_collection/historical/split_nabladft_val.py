#!/usr/bin/env python3
from pathlib import Path
from typing import Set, Dict, Tuple
import csv
import argparse
import random

from tqdm.auto import tqdm

from split_excluded_datasets import extract_smiles, pick_smiles
from check_global_overlap import load_split_smiles


def count_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    count = 0
    with open(path, "r", encoding="utf-8", buffering=1048576) as f:
        reader = csv.reader(f)
        next(reader, None)
        for _ in reader:
            count += 1
    return count


def load_nabla_smiles(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    smiles: Set[str] = set()
    with path.open("r", encoding="utf-8", buffering=1048576) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "enriched_text" not in reader.fieldnames:
            raise ValueError(f"Invalid header in {path}")
        for row in reader:
            s = extract_smiles(row.get("enriched_text", ""))
            if s:
                smiles.add(s)
    return smiles


def select_nabla_val(
    train_path: Path,
    other_val_smiles: Set[str],
    forbidden_smiles: Set[str],
    target_rows: int,
    rng: random.Random,
) -> Tuple[Set[str], Set[str], Set[str]]:
    rows_per_smiles: Dict[str, int] = {}
    overlap_val_smiles: Set[str] = set()
    clean_train_smiles: Set[str] = set()
    total_rows = 0
    with train_path.open("r", encoding="utf-8", buffering=1048576) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "enriched_text" not in reader.fieldnames:
            raise ValueError(f"Invalid header in {train_path}")
        for row in tqdm(reader, desc="Scanning nablaDFT train", mininterval=5):
            enriched = row.get("enriched_text", "")
            if not enriched:
                continue
            s = extract_smiles(enriched)
            if not s:
                continue
            total_rows += 1
            rows_per_smiles[s] = rows_per_smiles.get(s, 0) + 1
            if s in other_val_smiles:
                overlap_val_smiles.add(s)
            elif s in forbidden_smiles:
                continue
            else:
                clean_train_smiles.add(s)
    if total_rows == 0:
        raise ValueError("No usable rows in nablaDFT train")
    if target_rows <= 0:
        raise ValueError("Requested validation size is non-positive")
    sel_val: Set[str] = set()
    overlap_rows = sum(rows_per_smiles[s] for s in overlap_val_smiles)
    remaining = target_rows
    if overlap_rows >= remaining:
        sel_val, moved_overlap = pick_smiles(overlap_val_smiles, remaining, rows_per_smiles, rng)
        remaining -= moved_overlap
    else:
        sel_val = set(overlap_val_smiles)
        remaining -= overlap_rows
    if remaining > 0 and clean_train_smiles:
        extra, moved_extra = pick_smiles(clean_train_smiles, remaining, rows_per_smiles, rng)
        sel_val.update(extra)
    train_smiles = clean_train_smiles.difference(sel_val)
    return sel_val, train_smiles, overlap_val_smiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/nfs/h100/raid/chem/3D_big_data"))
    parser.add_argument("--tokenizer-data", type=Path, default=Path("/nfs/dgx/raid/chem/TokenizerData"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-rows", type=int, default=63381)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    root: Path = args.root
    tok: Path = args.tokenizer_data
    src_train_path = tok / "train" / "nablaDFT.csv"
    dst_train_path = root / "train" / "nablaDFT.csv"
    val_path = root / "val" / "nablaDFT.csv"
    if not src_train_path.exists():
        raise FileNotFoundError(f"Missing nablaDFT tokenizer train: {src_train_path}")
    if val_path.exists() and count_rows(val_path) > 0:
        raise ValueError(f"nablaDFT val is not empty: {val_path}")
    initial_train = count_rows(src_train_path)
    print(f"Initial nablaDFT train rows: {initial_train:,}")
    if initial_train == 0:
        raise ValueError("nablaDFT train is empty")
    all_val_smiles = load_split_smiles(root=root, split="val")
    all_test_smiles = load_split_smiles(root=root, split="test")
    nabla_val_smiles = load_nabla_smiles(val_path)
    other_val_smiles = all_val_smiles.difference(nabla_val_smiles)
    forbidden_smiles = all_val_smiles.union(all_test_smiles)
    sel_val, clean_train_smiles, overlap_val_smiles = select_nabla_val(
        train_path=src_train_path,
        other_val_smiles=other_val_smiles,
        forbidden_smiles=forbidden_smiles,
        target_rows=args.target_rows,
        rng=rng,
    )
    train_smiles = clean_train_smiles.difference(sel_val).difference(other_val_smiles)
    print(f"Selected {len(sel_val):,} unique SMILES for nablaDFT val")
    print(f"Nabla train will contain {len(train_smiles):,} unique SMILES (all not in any global val/test)")
    tmp_train = root / "train" / "nablaDFT.tmp.csv"
    with src_train_path.open("r", encoding="utf-8", buffering=1048576) as in_f, \
            tmp_train.open("w", newline="", encoding="utf-8", buffering=1048576) as out_train, \
            val_path.open("w", newline="", encoding="utf-8", buffering=1048576) as out_val:
        reader = csv.DictReader(in_f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Missing header in {src_train_path}")
        train_writer = csv.DictWriter(out_train, fieldnames=fieldnames)
        val_writer = csv.DictWriter(out_val, fieldnames=fieldnames)
        train_writer.writeheader()
        val_writer.writeheader()
        moved_rows = 0
        kept_rows = 0
        for row in tqdm(reader, desc="Writing new nablaDFT splits", mininterval=5):
            enriched = row.get("enriched_text", "")
            s = extract_smiles(enriched) if enriched else ""
            if s and s in sel_val:
                val_writer.writerow(row)
                moved_rows += 1
            elif s and s in train_smiles:
                train_writer.writerow(row)
                kept_rows += 1
    tmp_train.rename(dst_train_path)
    final_train = count_rows(dst_train_path)
    final_val = count_rows(val_path)
    print(f"Final nablaDFT sizes:   train={final_train:,}, val={final_val:,}, total={final_train + final_val:,}")
    print(f"Rows moved to val: {moved_rows:,}, rows kept in train: {kept_rows:,}")


if __name__ == "__main__":
    main()
