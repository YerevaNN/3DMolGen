#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Set, Tuple

from tqdm.auto import tqdm

SPLITS: Tuple[str, str, str] = ("train", "val", "test")
DEFAULT_DATASETS: List[str] = [
    "BindingMoad",
    "BindingNet-High",
    "BindingNet-Low",
    "BindingNet-Mid",
    "CrossDocked2020",
    "DAVIS-3D",
    "HiQBind",
    "Kinodata-3D",
    "OMol25_bio_mols",
    "Plinder",
    "SAIR",
    "SPINDR",
    "chembl3d",
    "geom",
    "nablaDFT",
]
COORD_PATTERN = re.compile(r"<[^>]+>")

WORKER_TEST_SMILES: Set[str] = set()
WORKER_VAL_SMILES: Set[str] = set()


def extract_smiles_from_enriched(enriched_text: str) -> str:
    return COORD_PATTERN.sub("", enriched_text)


def load_smiles_from_file(csv_path: Path) -> Set[str]:
    smiles_set: Set[str] = set()
    with csv_path.open("r", encoding="utf-8", buffering=1048576) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "enriched_text" not in reader.fieldnames:
            raise ValueError(f"Invalid header in {csv_path}")
        for row in tqdm(reader, desc=f"  {csv_path.name}", mininterval=5, leave=False):
            enriched_text = row.get("enriched_text", "")
            if enriched_text:
                smiles = extract_smiles_from_enriched(enriched_text)
                if smiles:
                    smiles_set.add(smiles)
    return smiles_set


def source_path_for_dataset(
    base_root: Path,
    chembl_root: Path,
    split: str,
    dataset: str,
    chembl_dataset: str,
) -> Path:
    if dataset == chembl_dataset:
        return chembl_root / split / f"{dataset}.csv"
    candidates = [
        base_root / split / f"{dataset}.csv",
        base_root / "grp_a" / split / f"{dataset}.csv",
        base_root / "grp_c" / split / f"{dataset}.csv",
        base_root / "grp_b" / split / f"{dataset}.csv",
        base_root / "grp_b" / f"{dataset}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def build_source_map(
    base_root: Path,
    chembl_root: Path,
    datasets: List[str],
    chembl_dataset: str,
) -> Dict[str, Dict[str, Path]]:
    source_map: Dict[str, Dict[str, Path]] = {split: {} for split in SPLITS}
    missing_datasets: List[str] = []

    for dataset in datasets:
        found_any_split = False
        for split in SPLITS:
            path = source_path_for_dataset(
                base_root=base_root,
                chembl_root=chembl_root,
                split=split,
                dataset=dataset,
                chembl_dataset=chembl_dataset,
            )
            if path.exists():
                source_map[split][dataset] = path
                found_any_split = True
        if not found_any_split:
            missing_datasets.append(dataset)

    if missing_datasets:
        raise ValueError(f"Missing all split files for datasets: {', '.join(sorted(missing_datasets))}")

    for split in SPLITS:
        chembl_path = chembl_root / split / f"{chembl_dataset}.csv"
        if not chembl_path.exists():
            raise ValueError(f"Missing chembl source file: {chembl_path}")

    return source_map


def collect_all_smiles(source_map: Dict[str, Dict[str, Path]], split: str) -> Set[str]:
    all_smiles: Set[str] = set()
    files = [source_map[split][dataset] for dataset in sorted(source_map[split].keys())]

    for csv_path in tqdm(files, desc=f"Collecting {split} SMILES", mininterval=5):
        smiles = load_smiles_from_file(csv_path)
        all_smiles.update(smiles)

    return all_smiles


def ensure_output_root(output_root: Path, overwrite_output: bool) -> None:
    if not output_root.exists():
        return

    existing_csvs = []
    for split in SPLITS:
        split_dir = output_root / split
        if split_dir.is_dir():
            existing_csvs.extend(split_dir.glob("*.csv"))

    if existing_csvs and not overwrite_output:
        raise ValueError(
            f"Output directory already contains CSV files: {output_root}. "
            "Use --overwrite-output to overwrite target dataset files."
        )


def init_worker(test_smiles_set: Set[str], val_smiles_set: Set[str]) -> None:
    global WORKER_TEST_SMILES, WORKER_VAL_SMILES
    WORKER_TEST_SMILES = test_smiles_set
    WORKER_VAL_SMILES = val_smiles_set


def process_file(args: Tuple[Path, Path]) -> Tuple[Path, Dict[str, int]]:
    input_file, output_file = args
    split = output_file.parent.name

    stats = {
        "original": 0,
        "removed": 0,
        "unique_original": 0,
        "unique_removed": 0,
        "unique_remaining": 0,
    }
    original_smiles: Set[str] = set()
    remaining_smiles: Set[str] = set()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with input_file.open("r", encoding="utf-8", buffering=1048576) as in_f:
        reader = csv.DictReader(in_f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {input_file}")
        fieldnames = reader.fieldnames

        with output_file.open("w", newline="", encoding="utf-8", buffering=1048576) as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                stats["original"] += 1

                enriched_text = row.get("enriched_text", "")
                if not enriched_text:
                    writer.writerow(row)
                    continue

                smiles = extract_smiles_from_enriched(enriched_text)
                if not smiles:
                    writer.writerow(row)
                    continue

                original_smiles.add(smiles)

                should_remove = False
                if split == "train":
                    if smiles in WORKER_TEST_SMILES or smiles in WORKER_VAL_SMILES:
                        should_remove = True
                elif split == "val":
                    if smiles in WORKER_TEST_SMILES:
                        should_remove = True

                if should_remove:
                    stats["removed"] += 1
                else:
                    remaining_smiles.add(smiles)
                    writer.writerow(row)

    if original_smiles:
        stats["unique_original"] = len(original_smiles)
        stats["unique_remaining"] = len(remaining_smiles)
        stats["unique_removed"] = stats["unique_original"] - stats["unique_remaining"]

    return output_file, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild Group A data with fresh chembl3d from TokenizerData, then apply the "
            "same global train/val/test overlap filtering used for 3D_big_data."
        )
    )
    parser.add_argument(
        "--base-root",
        type=Path,
        default=Path("/nfs/h100/raid/chem/3D_big_data"),
        help="Existing filtered Group A root with train/val/test folders.",
    )
    parser.add_argument(
        "--chembl-root",
        type=Path,
        default=Path("/nfs/dgx/raid/chem/TokenizerData"),
        help="TokenizerData root that contains train/val/test chembl3d.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/nfs/h100/raid/chem/3D_big_data_new"),
        help="Output root for the rebuilt filtered dataset.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to include in the rebuild.",
    )
    parser.add_argument(
        "--chembl-dataset",
        default="chembl3d",
        help="Dataset name to source from TokenizerData instead of the base root.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker count for filtering output files.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite target dataset files in the output root if they already exist.",
    )
    args = parser.parse_args()

    datasets = list(dict.fromkeys(args.datasets))
    if args.chembl_dataset not in datasets:
        datasets.append(args.chembl_dataset)

    if args.workers is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus:
            args.workers = max(1, int(slurm_cpus))
        else:
            args.workers = max(1, cpu_count() - 2)

    if not args.base_root.is_dir():
        raise ValueError(f"Base root not found: {args.base_root}")
    if not args.chembl_root.is_dir():
        raise ValueError(f"Chembl root not found: {args.chembl_root}")

    ensure_output_root(output_root=args.output, overwrite_output=args.overwrite_output)

    source_map = build_source_map(
        base_root=args.base_root,
        chembl_root=args.chembl_root,
        datasets=datasets,
        chembl_dataset=args.chembl_dataset,
    )

    print(f"Datasets to rebuild ({len(datasets)}): {', '.join(datasets)}")
    print(f"Base root:   {args.base_root}")
    print(f"Chembl root: {args.chembl_root}")
    print(f"Output root: {args.output}")
    if (args.base_root / "grp_a").is_dir() or (args.base_root / "grp_c").is_dir():
        print("Detected grouped base layout under grp_a/grp_c/grp_b")
    for split in SPLITS:
        print(f"Using fresh {args.chembl_dataset} {split}: {args.chembl_root / split / f'{args.chembl_dataset}.csv'}")

    print("\nStep 1: Collecting all test SMILES...")
    test_smiles_set = collect_all_smiles(source_map=source_map, split="test")
    print(f"Collected {len(test_smiles_set):,} unique test SMILES")

    print("\nStep 2: Collecting all val SMILES...")
    val_smiles_set = collect_all_smiles(source_map=source_map, split="val")
    print(f"Collected {len(val_smiles_set):,} unique val SMILES")

    print("\nStep 3: Filtering combined dataset...")
    tasks: List[Tuple[Path, Path]] = []
    for split in SPLITS:
        for dataset in datasets:
            input_file = source_map[split].get(dataset)
            if input_file is None:
                continue
            output_file = args.output / split / f"{dataset}.csv"
            tasks.append((input_file, output_file))

    all_stats = defaultdict(
        lambda: defaultdict(
            lambda: {
                "original": 0,
                "removed": 0,
                "unique_original": 0,
                "unique_removed": 0,
                "unique_remaining": 0,
            }
        )
    )

    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(test_smiles_set, val_smiles_set),
    ) as pool:
        results_dict: Dict[Path, Dict[str, int]] = {}
        with tqdm(total=len(tasks), desc="Processing files", mininterval=3) as pbar:
            for output_file, stats in pool.imap_unordered(process_file, tasks, chunksize=1):
                results_dict[output_file] = stats
                pbar.set_postfix(file=output_file.name)
                pbar.update(1)

    for _, output_file in tasks:
        stats = results_dict.get(output_file)
        if stats is None:
            continue
        split = output_file.parent.name
        dataset = output_file.stem
        all_stats[dataset][split] = stats

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_train_original = 0
    total_train_removed = 0
    total_val_original = 0
    total_val_removed = 0
    total_test_original = 0
    total_test_removed = 0

    for dataset in datasets:
        stats = all_stats[dataset]
        train_stats = stats.get(
            "train",
            {"original": 0, "removed": 0, "unique_original": 0, "unique_removed": 0, "unique_remaining": 0},
        )
        val_stats = stats.get(
            "val",
            {"original": 0, "removed": 0, "unique_original": 0, "unique_removed": 0, "unique_remaining": 0},
        )
        test_stats = stats.get(
            "test",
            {"original": 0, "removed": 0, "unique_original": 0, "unique_removed": 0, "unique_remaining": 0},
        )

        print(f"\n{dataset}:")
        print(
            f"  Train: rows {train_stats['original']:,} -> -{train_stats['removed']:,} = "
            f"{train_stats['original'] - train_stats['removed']:,}; unique SMILES "
            f"{train_stats['unique_original']:,} -> -{train_stats['unique_removed']:,} = "
            f"{train_stats['unique_remaining']:,}"
        )
        print(
            f"  Val:   rows {val_stats['original']:,} -> -{val_stats['removed']:,} = "
            f"{val_stats['original'] - val_stats['removed']:,}; unique SMILES "
            f"{val_stats['unique_original']:,} -> -{val_stats['unique_removed']:,} = "
            f"{val_stats['unique_remaining']:,}"
        )
        print(
            f"  Test:  rows {test_stats['original']:,} -> -{test_stats['removed']:,} = "
            f"{test_stats['original'] - test_stats['removed']:,}; unique SMILES "
            f"{test_stats['unique_original']:,} -> -{test_stats['unique_removed']:,} = "
            f"{test_stats['unique_remaining']:,}"
        )

        total_train_original += train_stats["original"]
        total_train_removed += train_stats["removed"]
        total_val_original += val_stats["original"]
        total_val_removed += val_stats["removed"]
        total_test_original += test_stats["original"]
        total_test_removed += test_stats["removed"]

    print("\n" + "=" * 80)
    print("TOTALS")
    print("=" * 80)
    print(
        f"Train: {total_train_original:,} original, -{total_train_removed:,} removed = "
        f"{total_train_original - total_train_removed:,} remaining"
    )
    print(
        f"Val:   {total_val_original:,} original, -{total_val_removed:,} removed = "
        f"{total_val_original - total_val_removed:,} remaining"
    )
    print(
        f"Test:  {total_test_original:,} original, -{total_test_removed:,} removed = "
        f"{total_test_original - total_test_removed:,} remaining"
    )
    print(f"\nOutput directory: {args.output}")


if __name__ == "__main__":
    main()
