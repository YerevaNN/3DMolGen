#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


ISOTOPIC_TOKENS = [
    "[113I]", "[11C-]", "[11C@@H]", "[11C@H]", "[11CH2]", "[11CH3]", "[11CH]",
    "[11C]", "[11c]", "[123I]", "[124I]", "[125I]", "[126IH]", "[127I]", "[12C@@H]", "[12C@H]", "[12C@]",
    "[12CH2]", "[12CH3]", "[12CH]", "[12C]", "[12cH]", "[12c]", "[131I]", "[13C-]", "[13C@@H]", "[13C@@]",
    "[13C@H]", "[13C@]", "[13CH2]", "[13CH3]", "[13CH]", "[13C]", "[13NH2]", "[13NH]", "[13N]", "[13O]",
    "[13cH]", "[13c]", "[14C@@H]", "[14C@@]", "[14C@H]", "[14C@]", "[14CH2]", "[14CH3]", "[14CH4]", "[14CH]",
    "[14C]", "[14NH2]", "[14NH]", "[14N]", "[14cH]", "[14c]", "[14n]", "[15C]", "[15N+]", "[15NH+]",
    "[15NH2+]", "[15NH2]", "[15NH]", "[15N]", "[15O]", "[15n+]", "[15nH+]", "[15nH]", "[15n]", "[16OH]",
    "[16O]", "[16n]", "[17F]", "[17O-]", "[17OH]", "[17O]", "[18F]", "[18O-]", "[18OH]", "[18O]", "[19F]",
    "[1H]", "[20CH2]", "[20CH3]", "[20OH]", "[22N]", "[22OH2]", "[28SiH4]", "[2H]", "[32P@@]", "[32P@]",
    "[32P]", "[32SH]", "[34S]", "[35Cl]", "[35SH2]", "[35SH]", "[35S]", "[36Cl]", "[36S]", "[37Cl]",
    "[38PH3]", "[39ClH]", "[3H]", "[41PH3]", "[76Br]", "[77Br]", "[79Br]", "[80Br]", "[81Br]", "[82Br]",
    "[83BrH]", "[9cH]",
]


def filter_split(csv_path: Path, removed_path: Path, regex: re.Pattern[str]) -> tuple[int, int]:
    filtered_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    removed_tmp = removed_path.with_suffix(removed_path.suffix + ".tmp")
    filtered_tmp.parent.mkdir(parents=True, exist_ok=True)
    removed_tmp.parent.mkdir(parents=True, exist_ok=True)

    keep_header = True
    removed_header = True
    kept_rows = 0
    removed_rows = 0

    for chunk in tqdm(pd.read_csv(csv_path, chunksize=500_000), desc=f"Filtering {csv_path.name}", mininterval=5):
        is_isotopic = chunk["enriched_text"].str.contains(regex, na=False)
        kept_chunk = chunk[~is_isotopic]
        removed_chunk = chunk[is_isotopic]

        kept_chunk.to_csv(filtered_tmp, mode="w" if keep_header else "a", header=keep_header, index=False)
        removed_chunk.to_csv(removed_tmp, mode="w" if removed_header else "a", header=removed_header, index=False)

        keep_header = False
        removed_header = False
        kept_rows += len(kept_chunk)
        removed_rows += len(removed_chunk)

    filtered_tmp.replace(csv_path)
    removed_tmp.replace(removed_path)
    return kept_rows, removed_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/nfs/dgx/raid/chem/TokenizerData"))
    args = parser.parse_args()

    regex = re.compile("|".join(map(re.escape, ISOTOPIC_TOKENS)))

    total_removed = 0
    for split in ("train", "val", "test"):
        csv_path = args.data_dir / split / "chembl3d.csv"
        removed_path = args.data_dir / f"chembl_rm_{split}.csv"
        if not csv_path.exists():
            print(f"Missing {csv_path}, skipping {split}")
            continue

        kept_rows, removed_rows = filter_split(csv_path, removed_path, regex)
        total_removed += removed_rows
        print(f"{split}: kept {kept_rows:,}, removed {removed_rows:,}")
        print(f"{split}: removed rows saved to {removed_path}")

    print(f"Total removed across splits: {total_removed:,}")


if __name__ == "__main__":
    main()
