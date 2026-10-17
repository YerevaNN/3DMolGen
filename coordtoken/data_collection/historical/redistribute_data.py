from pathlib import Path
from typing import Dict, Set, Tuple
import csv
import argparse
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm


def extract_smiles_from_enriched(enriched_text: str) -> str:
    import re
    return re.sub(r'<[^>]+>', '', enriched_text)


def load_train_smiles(data_dir: Path) -> Dict[str, Set[str]]:
    train_dir = data_dir / 'train'
    train_smiles = defaultdict(set)

    if not train_dir.exists():
        return train_smiles

    csv_files = sorted(train_dir.glob('*.csv'))
    for csv_path in tqdm(csv_files, desc="Loading training SMILES"):
        dataset = csv_path.stem

        with open(csv_path, 'r', encoding='utf-8', buffering=1048576) as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc=f"  {dataset}", leave=False, unit="rows"):
                enriched_text = row.get('enriched_text', '')
                if enriched_text:
                    smiles = extract_smiles_from_enriched(enriched_text)
                    if smiles:
                        train_smiles[dataset].add(smiles)

    return train_smiles


def redistribute_dataset(
    data_dir: Path,
    output_dir: Path,
    dataset: str,
    train_smiles_set: Set[str]
) -> Dict[str, int]:
    stats = {
        'train_original': 0,
        'train_added': 0,
        'val_original': 0,
        'val_removed': 0,
        'test_original': 0,
        'test_removed': 0,
    }

    # First, load val and test SMILES to check for val/test co-occurrences
    val_smiles_set = set()
    test_smiles_set = set()
    header = None

    for split in ['val', 'test']:
        input_path = data_dir / split / f'{dataset}.csv'
        if not input_path.exists():
            continue

        with open(input_path, 'r', encoding='utf-8', buffering=1048576) as f:
            reader = csv.DictReader(f)
            if header is None:
                header = reader.fieldnames
            for row in reader:
                enriched_text = row.get('enriched_text', '')
                if enriched_text:
                    smiles = extract_smiles_from_enriched(enriched_text)
                    if smiles:
                        if split == 'val':
                            val_smiles_set.add(smiles)
                        else:
                            test_smiles_set.add(smiles)

    # Find val/test co-occurrences
    val_test_cooccur = val_smiles_set & test_smiles_set

    all_rows_to_move = []

    for split in ['train', 'val', 'test']:
        input_path = data_dir / split / f'{dataset}.csv'
        if not input_path.exists():
            continue

        output_path = output_dir / split
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / f'{dataset}.csv'

        with open(input_path, 'r', encoding='utf-8', buffering=1048576) as f:
            reader = csv.DictReader(f)
            if header is None:
                header = reader.fieldnames
            total_rows = sum(1 for _ in reader)

        if split == 'train':
            with open(input_path, 'r', encoding='utf-8', buffering=1048576) as f:
                reader = csv.DictReader(f)

                with open(output_path, 'w', encoding='utf-8', newline='') as out_f:
                    writer = csv.DictWriter(out_f, fieldnames=header)
                    writer.writeheader()

                    for row in tqdm(reader, total=total_rows, desc=f"  {dataset} {split}", leave=False, unit="rows"):
                        enriched_text = row.get('enriched_text', '')
                        if not enriched_text:
                            continue

                        stats['train_original'] += 1
                        writer.writerow(row)
        else:
            rows_to_move = []

            with open(input_path, 'r', encoding='utf-8', buffering=1048576) as f:
                reader = csv.DictReader(f)

                with open(output_path, 'w', encoding='utf-8', newline='') as out_f:
                    writer = csv.DictWriter(out_f, fieldnames=header)
                    writer.writeheader()

                    for row in tqdm(reader, total=total_rows, desc=f"  {dataset} {split}", leave=False, unit="rows"):
                        enriched_text = row.get('enriched_text', '')
                        if not enriched_text:
                            continue

                        if split == 'val':
                            stats['val_original'] += 1
                        else:
                            stats['test_original'] += 1

                        smiles = extract_smiles_from_enriched(enriched_text)
                        # Move to train if: co-occurs with train OR co-occurs with test/val
                        should_move = (smiles in train_smiles_set) or (smiles in val_test_cooccur)

                        if should_move:
                            if split == 'val':
                                stats['val_removed'] += 1
                            else:
                                stats['test_removed'] += 1
                            rows_to_move.append(row)
                        else:
                            writer.writerow(row)

            all_rows_to_move.extend(rows_to_move)

    if all_rows_to_move:
        train_output = output_dir / 'train' / f'{dataset}.csv'
        with open(train_output, 'a', encoding='utf-8', newline='') as out_f:
            writer = csv.DictWriter(out_f, fieldnames=header)
            writer.writerows(all_rows_to_move)

        stats['train_added'] = len(all_rows_to_move)

    return stats


def _redistribute_dataset_wrapper(args: Tuple[Path, Path, str, Set[str]]) -> Tuple[str, Dict[str, int]]:
    data_dir, output_dir, dataset, train_smiles_set = args
    stats = redistribute_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        dataset=dataset,
        train_smiles_set=train_smiles_set,
    )
    return dataset, stats


def main(input_dir: Path, output_dir: Path) -> None:
    print("Loading training SMILES...")
    train_smiles = load_train_smiles(input_dir)

    print(f"\nFound {len(train_smiles)} datasets in training set")
    print(f"Total unique SMILES in training: {sum(len(s) for s in train_smiles.values()):,}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    datasets = sorted(train_smiles.keys())
    if not datasets:
        print("No training datasets found, nothing to redistribute")
        return

    max_procs = cpu_count()
    num_procs = min(len(datasets), max_procs)
    args_list = [
        (input_dir, output_dir, dataset, train_smiles[dataset])
        for dataset in datasets
    ]

    print("\nRedistributing data...")
    with Pool(processes=num_procs) as pool:
        for dataset, stats in tqdm(
            pool.imap_unordered(_redistribute_dataset_wrapper, args_list),
            total=len(args_list),
            desc="Processing datasets",
        ):
            all_stats[dataset] = stats

    print(f"\n{'='*80}")
    print("REDISTRIBUTION SUMMARY")
    print(f"{'='*80}")

    total_train_added = 0
    total_val_removed = 0
    total_test_removed = 0

    for dataset in sorted(all_stats.keys()):
        stats = all_stats[dataset]
        print(f"\n{dataset}:")
        print(f"  Train: {stats['train_original']:,} original, +{stats['train_added']:,} added = {stats['train_original'] + stats['train_added']:,} total")
        print(f"  Val: {stats['val_original']:,} original, -{stats['val_removed']:,} removed = {stats['val_original'] - stats['val_removed']:,} remaining")
        print(f"  Test: {stats['test_original']:,} original, -{stats['test_removed']:,} removed = {stats['test_original'] - stats['test_removed']:,} remaining")

        total_train_added += stats['train_added']
        total_val_removed += stats['val_removed']
        total_test_removed += stats['test_removed']

    print(f"\n{'='*80}")
    print("OVERALL TOTALS")
    print(f"{'='*80}")
    print(f"Total moved to train: {total_train_added:,}")
    print(f"Total removed from val: {total_val_removed:,}")
    print(f"Total removed from test: {total_test_removed:,}")
    print(f"\nData saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Redistribute data to remove train/val, train/test, and val/test co-occurrences"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/nfs/dgx/raid/chem/TokenizerData"),
        help="Input directory with train/test/val folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/nfs/dgx/raid/chem/3D_big_data"),
        help="Output directory for redistributed data",
    )
    args = parser.parse_args()

    main(input_dir=args.input, output_dir=args.output)
