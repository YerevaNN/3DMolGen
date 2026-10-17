import sys
import pickle
import csv
import gc
import random
import numpy as np
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm
from utils import encode_cartesian_v2

if len(sys.argv) != 2:
    print("Usage: python csv_flowr_single_dataset.py <dataset_name>")
    sys.exit(1)

dataset_name = sys.argv[1]

input_dir = Path("/nfs/h100/raid/chem/Flowr")
base_output_dir = Path("/nfs/dgx/raid/chem/TokenizerData")

random.seed(42)

print(f"\nProcessing {dataset_name}...")

pkl_file = f"{dataset_name}_rdkit_mols.pkl"
pkl_path = input_dir / pkl_file
splits_file = input_dir / f"{dataset_name}_splits.npz"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

data_len = len(data)
print(f"Loaded {data_len} items")

splits = np.load(splits_file)
idx_train = splits['idx_train'].tolist()
idx_val = splits['idx_val'].tolist()
idx_test = splits['idx_test'].tolist()

def filter_valid_indices(indices, data_len):
    valid = [i for i in indices if 0 <= i < data_len]
    invalid_count = len(indices) - len(valid)
    if invalid_count > 0:
        print(f"  Filtered out {invalid_count} out-of-range indices")
    return valid

idx_train_orig = len(idx_train)
idx_val_orig = len(idx_val)
idx_test_orig = len(idx_test)

idx_train = filter_valid_indices(idx_train, data_len)
idx_val = filter_valid_indices(idx_val, data_len)
idx_test = filter_valid_indices(idx_test, data_len)

print(f"Train: {len(idx_train)} (was {idx_train_orig}), Val: {len(idx_val)} (was {idx_val_orig}), Test: {len(idx_test)} (was {idx_test_orig})")

random.shuffle(idx_train)
print(f"Shuffled train indices")

has_names = "OMol25_small" in dataset_name or "KIBA-3D" in dataset_name

splits_data = {'train': idx_train, 'val': idx_val, 'test': idx_test}

for split_name, indices in splits_data.items():
    split_dir = base_output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    output_csv = split_dir / f"{dataset_name}.csv"

    print(f"\nProcessing {split_name} split...")

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'enriched_text'])

        processed = 0
        skipped = 0
        flush_interval = 1000

        for enum_idx, orig_idx in enumerate(tqdm(indices, desc=f"Writing {dataset_name} {split_name}")):
            if orig_idx < 0 or orig_idx >= len(data):
                print(f"Skipped index {orig_idx}: out of range")
                skipped += 1
                continue

            try:
                item = data[orig_idx]
            except IndexError:
                print(f"Skipped index {orig_idx}: IndexError")
                skipped += 1
                continue

            try:
                if has_names:
                    blob, name = item
                    mol = Chem.Mol(blob)
                else:
                    mol = Chem.Mol(item)
                    name = dataset_name

                if mol is None:
                    print(f"Skipped index {orig_idx}: mol is None")
                    skipped += 1
                    continue

                if mol.GetNumConformers() == 0:
                    print(f"Skipped index {orig_idx}: no conformers")
                    skipped += 1
                    continue

                enriched_text, _ = encode_cartesian_v2(mol, precision=4)
                writer.writerow([name, enriched_text])
                processed += 1

                if (enum_idx + 1) % flush_interval == 0:
                    csvfile.flush()
                    gc.collect()

            except Exception as e:
                print(f"Skipped index {orig_idx}: {type(e).__name__}: {e}")
                skipped += 1
                continue

        print(f"{split_name}: Processed: {processed}, Skipped: {skipped}")
        print(f"Saved to {output_csv}")

del data
gc.collect()

print(f"\n{dataset_name} processed successfully!")
