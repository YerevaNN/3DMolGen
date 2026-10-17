#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Dict, Set, Tuple
import csv
import argparse
import random
import re
from tqdm.auto import tqdm


def log(msg: str) -> None:
    print(msg, flush=True)

EXCLUDED_DATASETS: Set[str] = {
    "KIBA-3D",
    "OMol25_small_mols",
    "chembl3d",
    "zinc",
    "pubchem3d",
}
COORD_PATTERN = re.compile(r"<[^>]+>")


def extract_smiles(enriched_text: str) -> str:
    return COORD_PATTERN.sub("", enriched_text)


def load_smiles(csv_path: Path) -> Set[str]:
    out: Set[str] = set()
    if not csv_path.exists():
        return out
    with open(csv_path, "r", encoding="utf-8", buffering=1048576) as f:
        for row in csv.DictReader(f):
            s = extract_smiles(row.get("enriched_text", ""))
            if s:
                out.add(s)
    return out


def pick_smiles(
    pool: Set[str],
    target_rows: int,
    rows_per_smiles: Dict[str, int],
    rng: random.Random,
    exclude: Set[str] = None,
) -> Tuple[Set[str], int]:
    if exclude is None:
        exclude = set()
    lst = [s for s in pool if s not in exclude and rows_per_smiles.get(s, 0) > 0]
    rng.shuffle(lst)
    selected, moved = set(), 0
    for s in lst:
        if moved >= target_rows:
            break
        selected.add(s)
        moved += rows_per_smiles[s]
    return selected, moved


def main() -> None:
    log("split_excluded_datasets: start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/nfs/h100/raid/chem/3D_big_data"))
    parser.add_argument("--tokenizer-data", type=Path, default=Path("/nfs/dgx/raid/chem/TokenizerData"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    root, tok = args.root, args.tokenizer_data
    log(f"root={root} exists={root.exists()}, tok={tok} exists={tok.exists()}")
    if not root.exists():
        log(f"root {root} not found")
        sys.exit(1)
    train_dir, val_dir, test_dir = root / "train", root / "val", root / "test"

    global_train: Set[str] = set()
    for p in tqdm(list(train_dir.glob("*.csv")), desc="train SMILES", mininterval=5):
        global_train.update(load_smiles(p))

    global_test = set()
    for p in tqdm(list(test_dir.glob("*.csv")), desc="test SMILES", mininterval=5):
        global_test.update(load_smiles(p))

    global_val = set()
    for p in tqdm(list(val_dir.glob("*.csv")), desc="val SMILES", mininterval=5):
        global_val.update(load_smiles(p))

    for dataset in sorted(EXCLUDED_DATASETS):
        out_test = test_dir / f"{dataset}.csv"
        out_val = val_dir / f"{dataset}.csv"
        out_train = train_dir / f"{dataset}.csv"
        if out_test.exists() and out_val.exists() and out_train.exists():
            log(f"Skipping {dataset}: already done")
            global_test.update(load_smiles(out_test))
            global_train.update(load_smiles(out_train))
            global_val.update(load_smiles(out_val))
            continue

        if dataset in {"zinc", "pubchem3d"}:
            source_paths = []
            for split_name in ("train", "val", "test"):
                p = tok / split_name / f"{dataset}.csv"
                if p.exists():
                    source_paths.append(p)
            if not source_paths:
                log(f"Skipping {dataset}: no TokenizerData files")
                continue
        else:
            p = tok / "train" / f"{dataset}.csv"
            if not p.exists():
                log(f"Skipping {dataset}: no TokenizerData train")
                continue
            source_paths = [p]

        log(f"Processing {dataset}...")

        rows_per_smiles: Dict[str, int] = {}
        fixed_test: Set[str] = set()
        fixed_val: Set[str] = set()
        fixed_train: Set[str] = set()
        new_smiles: Set[str] = set()
        contaminated: Set[str] = set()
        total = 0

        for path in source_paths:
            with open(path, "r", encoding="utf-8", buffering=1048576) as f:
                reader = csv.DictReader(f)
                for row in tqdm(reader, desc=f"{dataset} read", mininterval=5):
                    total += 1
                    s = extract_smiles(row.get("enriched_text", ""))
                    if not s:
                        continue
                    rows_per_smiles[s] = rows_per_smiles.get(s, 0) + 1
                    in_test = s in global_test
                    in_val = s in global_val
                    in_train = s in global_train
                    count = int(in_test) + int(in_val) + int(in_train)
                    if count > 1:
                        contaminated.add(s)
                        continue
                    if in_test:
                        fixed_test.add(s)
                    elif in_val:
                        fixed_val.add(s)
                    elif in_train:
                        fixed_train.add(s)
                    else:
                        new_smiles.add(s)

        if total == 0:
            raise ValueError(f"No rows in {dataset}")
        target = max(1, int(round(total * 0.005)))
        test_rows_fixed = sum(rows_per_smiles[s] for s in fixed_test)
        sel_test: Set[str] = set()
        need_test = target
        if test_rows_fixed >= need_test:
            sel_test, moved_test = pick_smiles(fixed_test, need_test, rows_per_smiles, rng)
            need_test -= moved_test
        else:
            sel_test = set(fixed_test)
            need_test -= test_rows_fixed
        if need_test > 0:
            extra_test, moved_extra_test = pick_smiles(new_smiles, need_test, rows_per_smiles, rng)
            sel_test.update(extra_test)
            new_smiles -= extra_test

        val_rows_fixed = sum(rows_per_smiles[s] for s in fixed_val)
        sel_val: Set[str] = set()
        need_val = target
        if val_rows_fixed >= need_val:
            sel_val, moved_val = pick_smiles(fixed_val, need_val, rows_per_smiles, rng)
            need_val -= moved_val
        else:
            sel_val = set(fixed_val)
            need_val -= val_rows_fixed
        if need_val > 0:
            exclude_for_val = sel_test.union(sel_val)
            extra_val, moved_extra_val = pick_smiles(new_smiles, need_val, rows_per_smiles, rng, exclude_for_val)
            sel_val.update(extra_val)
            new_smiles -= extra_val

        train_smiles: Set[str] = fixed_train.union(new_smiles)
        global_test.update(sel_test)
        global_val.update(sel_val)
        global_train.update(train_smiles)
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        tmp_t = train_dir / f"{dataset}.tmp.csv"
        tmp_v = val_dir / f"{dataset}.tmp.csv"
        tmp_e = test_dir / f"{dataset}.tmp.csv"
        out_train = train_dir / f"{dataset}.csv"

        with open(tmp_t, "w", newline="", encoding="utf-8", buffering=1048576) as ot, \
                open(tmp_v, "w", newline="", encoding="utf-8", buffering=1048576) as ov, \
                open(tmp_e, "w", newline="", encoding="utf-8", buffering=1048576) as oe:
            wt = wv = we = None
            fieldnames = None
            for path in source_paths:
                with open(path, "r", encoding="utf-8", buffering=1048576) as inf:
                    reader = csv.DictReader(inf)
                    if reader.fieldnames is None:
                        raise ValueError(f"No header: {path}")
                    if fieldnames is None:
                        fieldnames = reader.fieldnames
                        wt = csv.DictWriter(ot, fieldnames)
                        wv = csv.DictWriter(ov, fieldnames)
                        we = csv.DictWriter(oe, fieldnames)
                        wt.writeheader()
                        wv.writeheader()
                        we.writeheader()
                    else:
                        if reader.fieldnames != fieldnames:
                            raise ValueError(f"Header mismatch between files: {source_paths[0]} and {path}")
                    for row in reader:
                        s = extract_smiles(row.get("enriched_text", ""))
                        if not s:
                            continue
                        if s in sel_test:
                            we.writerow(row)
                        elif s in sel_val:
                            wv.writerow(row)
                        elif s in train_smiles:
                            wt.writerow(row)

        tmp_t.rename(out_train)
        tmp_v.rename(val_dir / f"{dataset}.csv")
        tmp_e.rename(test_dir / f"{dataset}.csv")
        log(f"Done {dataset}")

    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
