from pathlib import Path
from typing import Dict, List, Tuple
import csv
import argparse
from collections import defaultdict
from multiprocessing import Pool
from tqdm.auto import tqdm
import re


def extract_smiles_from_enriched(enriched_text: str) -> str:
    return re.sub(r'<[^>]+>', '', enriched_text)


def load_train_conformers_by_smiles(data_dir: Path, dataset: str) -> Tuple[Dict[str, List[Dict[str, str]]], List[str]]:
    train_path = data_dir / 'train' / f'{dataset}.csv'
    if not train_path.exists():
        return {}, []

    conformers_by_smiles = defaultdict(list)
    header = None

    with open(train_path, 'r', encoding='utf-8', buffering=1048576) as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames) if reader.fieldnames else []
        for row in reader:
            enriched_text = row.get('enriched_text', '')
            if enriched_text:
                smiles = extract_smiles_from_enriched(enriched_text)
                if smiles:
                    conformers_by_smiles[smiles].append(row)

    return dict(conformers_by_smiles), header


def get_current_sizes(data_dir: Path, dataset: str) -> Tuple[int, int]:
    val_path = data_dir / 'val' / f'{dataset}.csv'
    test_path = data_dir / 'test' / f'{dataset}.csv'

    val_count = 0
    test_count = 0

    if val_path.exists():
        with open(val_path, 'r', encoding='utf-8', buffering=1048576) as f:
            reader = csv.reader(f)
            next(reader, None)
            val_count = sum(1 for _ in reader)

    if test_path.exists():
        with open(test_path, 'r', encoding='utf-8', buffering=1048576) as f:
            reader = csv.reader(f)
            next(reader, None)
            test_count = sum(1 for _ in reader)

    return val_count, test_count


def parse_redistribution_stats(stats_file: Path) -> Dict[str, Dict[str, int]]:
    stats = {}

    if not stats_file.exists():
        return stats

    with open(stats_file, 'r') as f:
        lines = f.readlines()

    current_dataset = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('Loading') or line.startswith('Found') or line.startswith('Total') or line.startswith('Redistributing') or line.startswith('Data saved'):
            continue

        if line.endswith(':'):
            current_dataset = line[:-1]
            stats[current_dataset] = {'val_removed': 0, 'test_removed': 0, 'val_original': 0, 'test_original': 0}
        elif current_dataset and 'Val:' in line:
            val_match = re.search(r'Val:\s*([\d,]+)\s+original', line)
            if val_match:
                stats[current_dataset]['val_original'] = int(val_match.group(1).replace(',', ''))
            removed_match = re.search(r'-([\d,]+)\s+removed', line)
            if removed_match:
                stats[current_dataset]['val_removed'] = int(removed_match.group(1).replace(',', ''))
        elif current_dataset and 'Test:' in line:
            test_match = re.search(r'Test:\s*([\d,]+)\s+original', line)
            if test_match:
                stats[current_dataset]['test_original'] = int(test_match.group(1).replace(',', ''))
            removed_match = re.search(r'-([\d,]+)\s+removed', line)
            if removed_match:
                stats[current_dataset]['test_removed'] = int(removed_match.group(1).replace(',', ''))

    return stats


def add_conformers_to_dataset(
    data_dir: Path,
    output_dir: Path,
    dataset: str,
    val_to_add: int,
    test_to_add: int,
) -> Dict[str, int]:
    stats = {
        'val_added': 0,
        'test_added': 0,
        'train_removed': 0,
    }

    val_path = output_dir / 'val' / f'{dataset}.csv'
    test_path = output_dir / 'test' / f'{dataset}.csv'
    train_path = output_dir / 'train' / f'{dataset}.csv'

    val_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    same_dir = data_dir.resolve() == output_dir.resolve()
    src_val_path = data_dir / 'val' / f'{dataset}.csv'
    src_test_path = data_dir / 'test' / f'{dataset}.csv'
    src_train_path = data_dir / 'train' / f'{dataset}.csv'

    conformers_by_smiles, header = load_train_conformers_by_smiles(data_dir, dataset)
    if not conformers_by_smiles or not header:
        return stats

    if not same_dir:
        if src_val_path.exists():
            with open(src_val_path, 'r', encoding='utf-8', buffering=1048576) as f_in, open(val_path, 'w', encoding='utf-8', newline='') as f_out:
                reader = csv.DictReader(f_in)
                writer = csv.DictWriter(f_out, fieldnames=header)
                writer.writeheader()
                for row in reader:
                    writer.writerow(row)
        if src_test_path.exists():
            with open(src_test_path, 'r', encoding='utf-8', buffering=1048576) as f_in, open(test_path, 'w', encoding='utf-8', newline='') as f_out:
                reader = csv.DictReader(f_in)
                writer = csv.DictWriter(f_out, fieldnames=header)
                writer.writeheader()
                for row in reader:
                    writer.writerow(row)

    val_needed = val_to_add
    test_needed = test_to_add

    if val_needed == 0 and test_needed == 0:
        return stats

    smiles_list = [
        (smiles, conformers)
        for smiles, conformers in conformers_by_smiles.items()
        if len(conformers) <= 100
    ]
    smiles_list.sort(key=lambda x: len(x[1]))

    val_rows_to_add = []
    test_rows_to_add = []

    for smiles, conformers in smiles_list:
        n_conf = len(conformers)
        if val_needed >= n_conf:
            val_rows_to_add.extend(conformers)
            val_needed -= n_conf
            stats['val_added'] += n_conf
        elif test_needed >= n_conf:
            test_rows_to_add.extend(conformers)
            test_needed -= n_conf
            stats['test_added'] += n_conf
        if val_needed <= 0 and test_needed <= 0:
            break

    moved_enriched = set()
    for row in val_rows_to_add + test_rows_to_add:
        et = row.get('enriched_text', '')
        if et:
            moved_enriched.add(et)

    if val_rows_to_add:
        file_exists = val_path.exists()
        with open(val_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerows(val_rows_to_add)

    if test_rows_to_add:
        file_exists = test_path.exists()
        with open(test_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerows(test_rows_to_add)

    if src_train_path.exists():
        if moved_enriched:
            with open(src_train_path, 'r', encoding='utf-8', buffering=1048576) as f_in:
                reader = csv.DictReader(f_in)
                rows_kept = [row for row in reader if row.get('enriched_text', '') not in moved_enriched]
            stats['train_removed'] = len(val_rows_to_add) + len(test_rows_to_add)
            with open(train_path, 'w', encoding='utf-8', newline='') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows_kept)
        elif not same_dir:
            with open(src_train_path, 'r', encoding='utf-8', buffering=1048576) as f_in, open(train_path, 'w', encoding='utf-8', newline='') as f_out:
                reader = csv.DictReader(f_in)
                writer = csv.DictWriter(f_out, fieldnames=header)
                writer.writeheader()
                for row in reader:
                    writer.writerow(row)

    return stats


def process_dataset(
    args: Tuple[Path, Path, str, Dict[str, int]]
) -> Tuple[str, Dict[str, int]]:
    data_dir, output_dir, dataset, target_sizes = args

    stats = add_conformers_to_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        dataset=dataset,
        val_to_add=target_sizes['val_removed'],
        test_to_add=target_sizes['test_removed'],
    )

    return dataset, stats


def main(input_dir: Path, output_dir: Path, stats_file: Path, num_workers: int) -> None:
    print("Parsing redistribution stats...")
    if not stats_file.exists():
        print(f"Error: Stats file not found: {stats_file}")
        return
    redistribution_stats = parse_redistribution_stats(stats_file)

    if not redistribution_stats:
        print("Warning: Could not parse stats file. Will use current sizes as targets.")
    else:
        sample_ds = next(iter(redistribution_stats))
        s = redistribution_stats[sample_ds]
        print(f"Sample stats {sample_ds}: val_removed={s['val_removed']}, test_removed={s['test_removed']}")

    print(f"\nFound {len(redistribution_stats)} datasets in stats")

    datasets = sorted(redistribution_stats.keys())
    if not datasets:
        print("No datasets found in stats file")
        return

    train_dir = input_dir / 'train'
    if not train_dir.exists():
        print(f"Error: Train directory does not exist: {train_dir}")
        return

    sample = next((d for d in datasets if (train_dir / f"{d}.csv").exists()), None)
    if sample:
        p = train_dir / f"{sample}.csv"
        n = sum(1 for _ in open(p, encoding='utf-8')) - 1
        print(f"Sample: {p} exists, {n:,} rows")
    else:
        print(f"Error: No train CSV found under {train_dir}")

    print(f"\nProcessing {len(datasets)} datasets...")

    all_stats = {}
    args_list = []

    for dataset in datasets:
        target_sizes = redistribution_stats.get(dataset, {'val_removed': 0, 'test_removed': 0})
        args_list.append((input_dir, output_dir, dataset, target_sizes))

    if not args_list:
        print("No datasets to process")
        return

    with Pool(processes=num_workers) as pool:
        for dataset, stats in tqdm(
            pool.imap_unordered(process_dataset, args_list),
            total=len(args_list),
            desc="Adding conformers",
        ):
            all_stats[dataset] = stats

    print(f"\n{'='*80}")
    print("ADDITION SUMMARY")
    print(f"{'='*80}")

    total_val_added = 0
    total_test_added = 0

    total_train_removed = 0
    for dataset in sorted(all_stats.keys()):
        stats = all_stats[dataset]
        print(f"\n{dataset}:")
        print(f"  Val: +{stats['val_added']:,} added")
        print(f"  Test: +{stats['test_added']:,} added")
        print(f"  Train: -{stats['train_removed']:,} removed")

        total_val_added += stats['val_added']
        total_test_added += stats['test_added']
        total_train_removed += stats['train_removed']

    print(f"\n{'='*80}")
    print("OVERALL TOTALS")
    print(f"{'='*80}")
    print(f"Total added to val: {total_val_added:,}")
    print(f"Total added to test: {total_test_added:,}")
    print(f"Total removed from train: {total_train_removed:,}")
    print(f"\nData saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add conformers back to val and test sets to restore original sizes"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/nfs/h100/raid/chem/3D_big_data"),
        help="Input directory with train/test/val folders (redistributed data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/nfs/h100/raid/chem/3D_big_data"),
        help="Output directory for augmented data",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("/auto/home/filya/fsq/redistribute_data_464796.out"),
        help="Path to redistribution stats output file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker processes",
    )
    args = parser.parse_args()

    main(
        input_dir=args.input,
        output_dir=args.output,
        stats_file=args.stats,
        num_workers=args.workers,
    )
