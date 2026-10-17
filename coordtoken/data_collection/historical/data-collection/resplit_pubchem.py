#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/nfs/dgx/raid/chem/TokenizerData"))
    parser.add_argument("--output-dir", type=Path, default=Path("pubchem_resplit"))
    parser.add_argument("--test-size", type=int, default=1_000_000)
    parser.add_argument("--val-size", type=int, default=500_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    isotopic_tokens = [
        '[113I]', '[11C-]', '[11C@@H]', '[11C@H]', '[11CH2]', '[11CH3]', '[11CH]',
        '[11C]', '[11c]', '[123I]', '[124I]', '[125I]', '[126IH]', '[127I]', '[12C@@H]', '[12C@H]', '[12C@]',
        '[12CH2]', '[12CH3]', '[12CH]', '[12C]', '[12cH]', '[12c]', '[131I]', '[13C-]', '[13C@@H]', '[13C@@]',
        '[13C@H]', '[13C@]', '[13CH2]', '[13CH3]', '[13CH]', '[13C]', '[13NH2]', '[13NH]', '[13N]', '[13O]',
        '[13cH]', '[13c]', '[14C@@H]', '[14C@@]', '[14C@H]', '[14C@]', '[14CH2]', '[14CH3]', '[14CH4]', '[14CH]',
        '[14C]', '[14NH2]', '[14NH]', '[14N]', '[14cH]', '[14c]', '[14n]', '[15C]', '[15N+]', '[15NH+]',
        '[15NH2+]', '[15NH2]', '[15NH]', '[15N]', '[15O]', '[15n+]', '[15nH+]', '[15nH]', '[15n]', '[16OH]',
        '[16O]', '[16n]', '[17F]', '[17O-]', '[17OH]', '[17O]', '[18F]', '[18O-]', '[18OH]', '[18O]', '[19F]',
        '[1H]', '[20CH2]', '[20CH3]', '[20OH]', '[22N]', '[22OH2]', '[28SiH4]', '[2H]', '[32P@@]', '[32P@]',
        '[32P]', '[32SH]', '[34S]', '[35Cl]', '[35SH2]', '[35SH]', '[35S]', '[36Cl]', '[36S]', '[37Cl]',
        '[38PH3]', '[39ClH]', '[3H]', '[41PH3]', '[76Br]', '[77Br]', '[79Br]', '[80Br]', '[81Br]', '[82Br]',
        '[83BrH]', '[9cH]'
    ]
    pattern = '|'.join(map(re.escape, isotopic_tokens))
    regex = re.compile(pattern)

    all_dfs = []
    removed_ids = []

    splits = ['train', 'val', 'test']
    for split in splits:
        csv_path = args.data_dir / split / "pubchem3d.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found.")
            continue

        print(f"Loading and filtering {csv_path}...")
        # Read in chunks to manage memory and filter on the fly
        for chunk in tqdm(pd.read_csv(csv_path, usecols=["name", "enriched_text"], chunksize=500_000)):
            # Find matches
            is_isotopic = chunk["enriched_text"].str.contains(regex, na=False)

            # Save removed IDs
            removed_chunk_ids = chunk[is_isotopic]["name"].tolist()
            removed_ids.extend(removed_chunk_ids)

            # Keep clean rows
            clean_chunk = chunk[~is_isotopic]
            all_dfs.append(clean_chunk)

    print("Concatenating dataframes...")
    df = pd.concat(all_dfs, ignore_index=True)
    del all_dfs # Free memory

    print(f"Total clean rows: {len(df):,}")
    print(f"Total removed rows: {len(removed_ids):,}")

    # Save removed IDs
    removed_ids_path = args.output_dir / "removed_isotopic_ids.txt"
    with removed_ids_path.open("w") as f:
        for rid in removed_ids:
            f.write(f"{rid}\n")
    print(f"Saved removed IDs to {removed_ids_path}")

    print("Shuffling...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Resplit
    print(f"Splitting (Test: {args.test_size:,}, Val: {args.val_size:,})...")

    if len(df) < args.test_size + args.val_size:
        print("Error: Not enough data for the requested split sizes.")
        return

    test_df = df.iloc[:args.test_size]
    val_df = df.iloc[args.test_size : args.test_size + args.val_size]
    train_df = df.iloc[args.test_size + args.val_size :]

    print(f"Final split sizes: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    # Save
    print("Saving new splits...")
    test_df.to_csv(args.output_dir / "test_pubchem3d.csv", index=False)
    val_df.to_csv(args.output_dir / "val_pubchem3d.csv", index=False)
    train_df.to_csv(args.output_dir / "train_pubchem3d.csv", index=False)

    print("Done.")

if __name__ == "__main__":
    main()
