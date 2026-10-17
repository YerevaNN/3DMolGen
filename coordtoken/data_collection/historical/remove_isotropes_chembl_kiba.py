#!/usr/bin/env python3
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/nfs/dgx/raid/chem/TokenizerData"))
    args = parser.parse_args()

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
        for chunk in tqdm(pd.read_csv(csv_path, chunksize=500_000), desc=f"Processing {csv_path.name}", mininterval=5):
            is_isotopic = chunk["enriched_text"].str.contains(regex, na=False)
            removed_ids.extend(chunk[is_isotopic]["name"].tolist())
            all_chunks.append(chunk[~is_isotopic])
        result_df = pd.concat(all_chunks, ignore_index=True)
        return result_df, removed_ids

    print("\n--- Processing chembl3d ---")
    chembl_removed_all = []
    for split in ['train', 'val', 'test']:
        backup_path = args.data_dir / f"chembl_{split}.csv"
        output_path = args.data_dir / split / "chembl3d.csv"

        if not backup_path.exists():
            print(f"Warning: {backup_path} not found, skipping {split}")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df, removed = filter_isotopic(backup_path)
        df.to_csv(output_path, index=False)
        chembl_removed_all.extend(removed)
        print(f"Saved {len(df):,} rows to {output_path}")
        print(f"Removed {len(removed):,} isotopic rows from {split}")

    if chembl_removed_all:
        removed_path = args.data_dir / "chembl_removed_ids.txt"
        with open(removed_path, "w") as f:
            for rid in chembl_removed_all:
                f.write(f"{rid}\n")
        print(f"Saved {len(chembl_removed_all):,} removed IDs to {removed_path}")

    print("\n--- Processing KIBA-3D ---")
    kiba_removed_all = []
    for split in ['train', 'val', 'test']:
        backup_path = args.data_dir / f"KIBA-3D_{split}_orig.csv"
        output_path = args.data_dir / split / "KIBA-3D.csv"

        if not backup_path.exists():
            print(f"Warning: {backup_path} not found, skipping {split}")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df, removed = filter_isotopic(backup_path)
        df.to_csv(output_path, index=False)
        kiba_removed_all.extend(removed)
        print(f"Saved {len(df):,} rows to {output_path}")
        print(f"Removed {len(removed):,} isotopic rows from {split}")

    if kiba_removed_all:
        removed_path = args.data_dir / "kiba_removed_ids.txt"
        with open(removed_path, "w") as f:
            for rid in kiba_removed_all:
                f.write(f"{rid}\n")
        print(f"Saved {len(kiba_removed_all):,} removed IDs to {removed_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
