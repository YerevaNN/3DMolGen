import csv
import gc
import random
import pandas as pd
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm
from utils import encode_cartesian_v2
import multiprocessing as mp
from functools import partial

nabla_dir = Path("/nfs/h100/raid/chem/nablaDFT")
xyz_root = nabla_dir / "conformers_v2"
summary_csv = nabla_dir / "summary.csv"
base_output_dir = Path("/nfs/h100/raid/chem/TokenizerData")

N_PROCESSES = 41
random.seed(42)

print("Loading summary.csv...")
df = pd.read_csv(summary_csv)
print(f"Total conformers: {len(df):,}")

def map_split(raw):
    if isinstance(raw, str) and 'scaffolds' in raw:
        return 'test_scaffolds'
    elif isinstance(raw, str) and 'conformations' in raw:
        return 'test_conformations'
    elif isinstance(raw, str) and 'structures' in raw:
        return 'test_structures'
    else:
        return 'train'

df['split'] = df['SPLITS'].apply(map_split)

df_train_full = df[df['split'] == 'train'].reset_index(drop=True)
df_test_scaffolds = df[df['split'] == 'test_scaffolds'].reset_index(drop=True)
df_test_conformations = df[df['split'] == 'test_conformations'].reset_index(drop=True)
df_test_structures = df[df['split'] == 'test_structures'].reset_index(drop=True)

print(f"Original - Train: {len(df_train_full):,}")
print(f"Test - Scaffolds: {len(df_test_scaffolds):,}, Conformations: {len(df_test_conformations):,}, Structures: {len(df_test_structures):,}")

train_indices_full = list(range(len(df_train_full)))
random.shuffle(train_indices_full)

val_indices_from_train = train_indices_full[:10000]
train_indices = train_indices_full[10000:]

df_val = df_train_full.iloc[val_indices_from_train].reset_index(drop=True)
df_train = df_train_full.iloc[train_indices].reset_index(drop=True)

train_indices = list(range(len(df_train)))
random.shuffle(train_indices)
val_indices = list(range(len(df_val)))
test_scaffolds_indices = list(range(len(df_test_scaffolds)))
test_conformations_indices = list(range(len(df_test_conformations)))
test_structures_indices = list(range(len(df_test_structures)))

print(f"Final - Train: {len(df_train):,}, Val: {len(df_val):,}")
print(f"Shuffled train indices only (val and test kept in original order)")

def parse_xyz_conformer(xyz_path: Path, target_idx: int):
    with open(xyz_path) as f:
        idx = 0
        while True:
            header = f.readline()
            if not header:
                break
            nat = int(header.strip())
            f.readline()
            coords_lines = [tuple(f.readline().split()) for _ in range(nat)]
            coords = [(atom, float(x), float(y), float(z)) for atom, x, y, z in coords_lines if atom != 'H']
            if idx == target_idx:
                return coords
            idx += 1
    raise IndexError(f"Conformer {target_idx} not in {xyz_path}")

def process_conformer(row_data, xyz_root):
    try:
        moses_id, conf_id, smiles = row_data
        moses_id = str(moses_id)
        conf_id = int(conf_id)

        name = f"{moses_id}_{conf_id}"

        xyz_file = xyz_root / f"{moses_id}_centroids.xyz"
        if not xyz_file.exists():
            return None, f"Skipped {name}: XYZ file not found"

        coords = parse_xyz_conformer(xyz_file, conf_id)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, f"Skipped {name}: invalid SMILES"

        mol = Chem.RemoveHs(mol)

        if len(coords) != mol.GetNumAtoms():
            return None, f"Skipped {name}: atom count mismatch"

        atom_symbols_match = all(
            coords[i][0] == mol.GetAtomWithIdx(i).GetSymbol()
            for i in range(len(coords))
        )
        if not atom_symbols_match:
            return None, f"Skipped {name}: atom symbols don't match"

        conf = Chem.Conformer(mol.GetNumAtoms())
        for atom_idx, (_, x, y, z) in enumerate(coords):
            conf.SetAtomPosition(atom_idx, Chem.rdGeometry.Point3D(x, y, z))

        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)

        enriched_text, _ = encode_cartesian_v2(mol, precision=4)
        return (name, enriched_text), None

    except Exception as e:
        name = f"unknown"
        return None, f"Skipped {name}: {type(e).__name__}: {e}"

def worker_wrapper(row_data, xyz_root_str):
    xyz_root = Path(xyz_root_str)
    return process_conformer(row_data, xyz_root)

splits_data = {
    'train': (df_train, train_indices, 'train', 'nablaDFT.csv'),
    'val': (df_val, val_indices, 'val', 'nablaDFT.csv'),
    'test_scaffolds': (df_test_scaffolds, test_scaffolds_indices, 'test', 'nablaDFT_scaffolds.csv'),
    'test_conformations': (df_test_conformations, test_conformations_indices, 'test', 'nablaDFT_conformations.csv'),
    'test_structures': (df_test_structures, test_structures_indices, 'test', 'nablaDFT_structures.csv')
}

for split_name, (df_split, indices, folder_name, filename) in splits_data.items():
    split_dir = base_output_dir / folder_name
    split_dir.mkdir(parents=True, exist_ok=True)
    output_csv = split_dir / filename

    print(f"\nProcessing {split_name} split with {N_PROCESSES} processes...")
    print(f"Preparing {len(indices)} rows...")

    xyz_root_str = str(xyz_root)
    rows_data = [(row['MOSES id'], row['CONFORMER id'], row['SMILES']) for _, row in df_split.iloc[indices].iterrows()]

    print(f"Starting multiprocessing pool...")
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'enriched_text'])

        processed = 0
        skipped = 0

        with mp.Pool(N_PROCESSES) as pool:
            worker_fn = partial(worker_wrapper, xyz_root_str=xyz_root_str)

            for result, error in tqdm(pool.imap_unordered(worker_fn, rows_data, chunksize=10), total=len(rows_data), desc=f"NablaDFT {split_name}", mininterval=60):
                if result:
                    writer.writerow(result)
                    processed += 1
                else:
                    if error:
                        print(error)
                    skipped += 1

                if (processed + skipped) % 10000 == 0:
                    csvfile.flush()
                    gc.collect()

        print(f"{split_name}: Processed: {processed:,}, Skipped: {skipped:,}")
        print(f"Saved to {output_csv}")

    gc.collect()

print("\nDone!")
