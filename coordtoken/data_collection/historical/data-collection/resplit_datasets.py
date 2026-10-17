#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/nfs/dgx/raid/chem/TokenizerData"))
    args = parser.parse_args()

    # Isotopic tokens identified from the vocabulary
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

    def filter_isotopic(csv_path):
        print(f"Filtering {csv_path}...")
        all_chunks = []
        removed_ids = []
        for chunk in tqdm(pd.read_csv(csv_path, chunksize=500_000)):
            is_isotopic = chunk["enriched_text"].str.contains(regex, na=False)
            removed_ids.extend(chunk[is_isotopic]["name"].tolist())
            all_chunks.append(chunk[~is_isotopic])
        return pd.concat(all_chunks, ignore_index=True), removed_ids

    # 1. Handle chembl3d (Move originals to root, Filter + Resplit)
    print("\n--- Processing chembl3d ---")

    backup_paths = []
    for split in ['train', 'val', 'test']:
        orig = args.data_dir / split / "chembl3d.csv"
        dest = args.data_dir / f"chembl_{split}.csv"
        if orig.exists():
            print(f"Moving {orig} to {dest}")
            os.rename(orig, dest)
            backup_paths.append(dest)
        elif dest.exists():
            print(f"Found existing backup: {dest}")
            backup_paths.append(dest)

    if not backup_paths:
        print("Error: No ChEMBL files found to process.")
    else:
        chembl_dfs = []
        chembl_removed = []
        for p in backup_paths:
            df, removed = filter_isotopic(p)
            chembl_dfs.append(df)
            chembl_removed.extend(removed)

        chembl_all = pd.concat(chembl_dfs, ignore_index=True)
        print(f"Total ChEMBL clean rows: {len(chembl_all):,}")

        print("Shuffling ChEMBL...")
        chembl_all = chembl_all.sample(frac=1, random_state=42).reset_index(drop=True)

        test_size = 37_000
        val_size = 37_000

        chembl_test = chembl_all.iloc[:test_size]
        chembl_val = chembl_all.iloc[test_size : test_size + val_size]
        chembl_train = chembl_all.iloc[test_size + val_size :]

        print("Saving new ChEMBL splits...")
        chembl_test.to_csv(args.data_dir / "test" / "chembl3d.csv", index=False)
        chembl_val.to_csv(args.data_dir / "val" / "chembl3d.csv", index=False)
        chembl_train.to_csv(args.data_dir / "train" / "chembl3d.csv", index=False)

        with open(args.data_dir / "chembl_removed_ids.txt", "w") as f:
            for rid in chembl_removed:
                f.write(f"{rid}\n")

    # 2. Handle KIBA-3D (Filter only, preserve splits)
    print("\n--- Processing KIBA-3D ---")
    kiba_removed_all = []
    for split in ['train', 'val', 'test']:
        p = args.data_dir / split / "KIBA-3D.csv"
        backup_kiba = args.data_dir / f"KIBA-3D_{split}_orig.csv"

        if p.exists():
            print(f"Backing up {p} to {backup_kiba}")
            os.rename(p, backup_kiba)

            df, removed = filter_isotopic(backup_kiba)
            df.to_csv(p, index=False)
            kiba_removed_all.extend(removed)
        elif backup_kiba.exists():
            print(f"Found existing backup {backup_kiba}, refiltering...")
            df, removed = filter_isotopic(backup_kiba)
            df.to_csv(p, index=False)
            kiba_removed_all.extend(removed)

    with open(args.data_dir / "kiba_removed_ids.txt", "w") as f:
        for rid in kiba_removed_all:
            f.write(f"{rid}\n")

    print("\nDone.")

if __name__ == "__main__":
    main()
