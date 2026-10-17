#!/usr/bin/env python3
from pathlib import Path
from typing import Set, Dict, Tuple
import csv
import argparse
import re
from tqdm.auto import tqdm
from collections import defaultdict
from multiprocessing import Pool, cpu_count

EXCLUDE_DATASETS = {"KIBA-3D", "OMol25_small_mols", "chembl3d", "zinc", "pubchem3d"}

COORD_PATTERN = re.compile(r'<[^>]+>')

def extract_smiles_from_enriched(enriched_text: str) -> str:
    return COORD_PATTERN.sub('', enriched_text)

def load_smiles_from_file(csv_path: Path) -> Set[str]:
    smiles_set = set()
    if not csv_path.exists():
        return smiles_set

    with open(csv_path, 'r', encoding='utf-8', buffering=1048576) as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc=f"  {csv_path.name}", mininterval=5, leave=False):
            enriched_text = row.get('enriched_text', '')
            if enriched_text:
                smiles = extract_smiles_from_enriched(enriched_text)
                if smiles:
                    smiles_set.add(smiles)
    return smiles_set

def collect_all_smiles(input_dir: Path, split: str, datasets: list) -> Set[str]:
    all_smiles = set()
    files = [input_dir / split / f'{dataset}.csv' for dataset in datasets]

    for csv_path in tqdm(files, desc=f"Collecting {split} SMILES", mininterval=5):
        smiles = load_smiles_from_file(csv_path)
        all_smiles.update(smiles)

    return all_smiles

def process_file(args: Tuple[Path, Path, Set[str], Set[str]]) -> Tuple[Path, Dict[str, int]]:
    input_file, output_file, test_smiles_set, val_smiles_set = args

    stats = {
        'original': 0,
        'removed': 0,
        'unique_original': 0,
        'unique_removed': 0,
        'unique_remaining': 0,
    }
    original_smiles: Set[str] = set()
    remaining_smiles: Set[str] = set()

    if not input_file.exists():
        return input_file, stats

    split = input_file.parent.name
    output_file.parent.mkdir(parents=True, exist_ok=True)

    remove_from_train_val = test_smiles_set
    remove_from_train = val_smiles_set

    with open(input_file, 'r', encoding='utf-8', buffering=1048576) as in_f:
        reader = csv.DictReader(in_f)
        fieldnames = reader.fieldnames

        with open(output_file, 'w', newline='', encoding='utf-8', buffering=1048576) as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                stats['original'] += 1

                enriched_text = row.get('enriched_text', '')
                if not enriched_text:
                    writer.writerow(row)
                    continue

                smiles = extract_smiles_from_enriched(enriched_text)
                if not smiles:
                    writer.writerow(row)
                    continue

                original_smiles.add(smiles)
                should_remove = False
                if split == 'train':
                    if smiles in remove_from_train_val or smiles in remove_from_train:
                        should_remove = True
                elif split == 'val':
                    if smiles in remove_from_train_val:
                        should_remove = True

                if should_remove:
                    stats['removed'] += 1
                else:
                    remaining_smiles.add(smiles)
                    writer.writerow(row)
    if original_smiles:
        stats['unique_original'] = len(original_smiles)
        stats['unique_remaining'] = len(remaining_smiles)
        stats['unique_removed'] = stats['unique_original'] - stats['unique_remaining']
    return input_file, stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/nfs/dgx/raid/chem/TokenizerData"))
    parser.add_argument("--output", type=Path, default=Path("/nfs/h100/raid/chem/3D_big_data"))
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if args.workers is None:
        args.workers = max(1, cpu_count() - 2)

    input_dir = args.input
    output_dir = args.output

    train_dir = input_dir / 'train'
    if not train_dir.exists():
        print(f"Error: {train_dir} does not exist")
        return

    all_datasets = sorted([f.stem for f in train_dir.glob('*.csv')])
    datasets = [d for d in all_datasets if d not in EXCLUDE_DATASETS]

    print(f"Found {len(datasets)} datasets to process (excluding {len(EXCLUDE_DATASETS)} excluded)")
    print(f"Excluded: {', '.join(sorted(EXCLUDE_DATASETS))}")

    print("\nStep 1: Collecting all test SMILES...")
    test_smiles_set = collect_all_smiles(input_dir, 'test', datasets)
    print(f"Collected {len(test_smiles_set):,} unique test SMILES")

    print("\nStep 2: Collecting all val SMILES...")
    val_smiles_set = collect_all_smiles(input_dir, 'val', datasets)
    print(f"Collected {len(val_smiles_set):,} unique val SMILES")

    print("\nStep 3: Processing files with multiprocessing...")
    tasks = []
    for split in ['train', 'val', 'test']:
        for dataset in datasets:
            input_file = input_dir / split / f'{dataset}.csv'
            output_file = output_dir / split / f'{dataset}.csv'
            if input_file.exists():
                tasks.append((input_file, output_file, test_smiles_set, val_smiles_set))

    all_stats = defaultdict(lambda: defaultdict(lambda: {
        'original': 0,
        'removed': 0,
        'unique_original': 0,
        'unique_removed': 0,
        'unique_remaining': 0,
    }))

    with Pool(processes=args.workers) as pool:
        results_dict = {}
        with tqdm(total=len(tasks), desc="Processing files", mininterval=3) as pbar:
            for result in pool.imap_unordered(process_file, tasks, chunksize=1):
                input_file, stats = result
                results_dict[input_file] = stats
                pbar.set_postfix(file=input_file.name)
                pbar.update(1)

    for (input_file, _, _, _) in tasks:
        if input_file in results_dict:
            stats = results_dict[input_file]
            split = input_file.parent.name
            dataset = input_file.stem
            all_stats[dataset][split] = stats

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total_train_original = 0
    total_train_removed = 0
    total_val_original = 0
    total_val_removed = 0
    total_test_original = 0
    total_test_removed = 0

    for dataset in sorted(all_stats.keys()):
        stats = all_stats[dataset]
        train_stats = stats.get('train', {
            'original': 0,
            'removed': 0,
            'unique_original': 0,
            'unique_removed': 0,
            'unique_remaining': 0,
        })
        val_stats = stats.get('val', {
            'original': 0,
            'removed': 0,
            'unique_original': 0,
            'unique_removed': 0,
            'unique_remaining': 0,
        })
        test_stats = stats.get('test', {
            'original': 0,
            'removed': 0,
            'unique_original': 0,
            'unique_removed': 0,
            'unique_remaining': 0,
        })

        print(f"\n{dataset}:")
        print(
            f"  Train: rows {train_stats['original']:,} -> -{train_stats['removed']:,} = {train_stats['original'] - train_stats['removed']:,}; "
            f"unique SMILES {train_stats['unique_original']:,} -> -{train_stats['unique_removed']:,} = {train_stats['unique_remaining']:,}"
        )
        print(
            f"  Val:   rows {val_stats['original']:,} -> -{val_stats['removed']:,} = {val_stats['original'] - val_stats['removed']:,}; "
            f"unique SMILES {val_stats['unique_original']:,} -> -{val_stats['unique_removed']:,} = {val_stats['unique_remaining']:,}"
        )
        print(
            f"  Test:  rows {test_stats['original']:,} -> -{test_stats['removed']:,} = {test_stats['original'] - test_stats['removed']:,}; "
            f"unique SMILES {test_stats['unique_original']:,} -> -{test_stats['unique_removed']:,} = {test_stats['unique_remaining']:,}"
        )

        total_train_original += train_stats['original']
        total_train_removed += train_stats['removed']
        total_val_original += val_stats['original']
        total_val_removed += val_stats['removed']
        total_test_original += test_stats['original']
        total_test_removed += test_stats['removed']

    print("\n" + "="*80)
    print("TOTALS")
    print("="*80)
    print(f"Train: {total_train_original:,} original, -{total_train_removed:,} removed = {total_train_original - total_train_removed:,} remaining")
    print(f"Val: {total_val_original:,} original, -{total_val_removed:,} removed = {total_val_original - total_val_removed:,} remaining")
    print(f"Test: {total_test_original:,} original, -{total_test_removed:,} removed = {total_test_original - total_test_removed:,} remaining")
    print(f"\nOutput directory: {output_dir}")

if __name__ == "__main__":
    main()
