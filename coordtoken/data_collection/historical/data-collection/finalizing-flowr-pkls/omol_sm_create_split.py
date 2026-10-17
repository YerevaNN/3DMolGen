import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

pkl_file = Path("/nfs/h100/raid/chem/Flowr/OMol25_small_mols_rdkit_mols.pkl")
output_file = Path("/nfs/h100/raid/chem/Flowr/OMol25_small_mols_splits.npz")

print("Loading pickle file...")
with open(pkl_file, 'rb') as f:
    mol_data = pickle.load(f)

total_molecules = len(mol_data)
print(f"Total molecules: {total_molecules}")

train_pct = 0.9876
val_pct = 0.0028
test_pct = 0.0096

train_size = int(total_molecules * train_pct)
val_size = int(total_molecules * val_pct)
test_size = total_molecules - train_size - val_size

print(f"Train size: {train_size} ({100 * train_size / total_molecules:.2f}%)")
print(f"Val size: {val_size} ({100 * val_size / total_molecules:.2f}%)")
print(f"Test size: {test_size} ({100 * test_size / total_molecules:.2f}%)")

print("\nShuffling indices...")
all_indices = np.arange(total_molecules, dtype=np.int64)
np.random.seed(42)
np.random.shuffle(all_indices)

print("Sorting split indices...")
with tqdm(total=3, desc="Creating splits") as pbar:
    idx_train = np.sort(all_indices[:train_size])
    pbar.update(1)
    idx_val = np.sort(all_indices[train_size:train_size + val_size])
    pbar.update(1)
    idx_test = np.sort(all_indices[train_size + val_size:])
    pbar.update(1)

np.savez(output_file, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

print(f"\nSplit saved to {output_file}")
print(f"Verification:")
print(f"  idx_train shape: {idx_train.shape}")
print(f"  idx_val shape: {idx_val.shape}")
print(f"  idx_test shape: {idx_test.shape}")

#   idx_train shape: (5754252,)
#   idx_val shape: (16314,)
#   idx_test shape: (55935,)
# OMol25-data.tar.gz - total: 5826501
# OMol25-biomolecules.tar.gz:
#   Train: 5179908 (99.99%)
#   Val:      100 ( 0.00%)
#   Test:     225 ( 0.00%)
#   Total: 5180233
# BindingMoad:
#   Train:  32628 (98.03%)
#   Val:      187 ( 0.56%)
#   Test:     468 ( 1.41%)
#   Total:  33283
# BindingNet-High:
#   Train: 228165 (98.34%)
#   Val:      132 ( 0.06%)
#   Test:    3729 ( 1.61%)
#   Total: 232026
# BindingNet-Low:
#   Train: 279584 (96.94%)
#   Val:     1984 ( 0.69%)
#   Test:    6850 ( 2.38%)
#   Total: 288418
# BindingNet-Mid:
#   Train: 161263 (97.79%)
#   Val:      338 ( 0.20%)
#   Test:    3312 ( 2.01%)
#   Total: 164913
# CrossDocked2020:
#   Train:  99567 (99.60%)
#   Val:      299 ( 0.30%)
#   Test:     100 ( 0.10%)
#   Total:  99966
# DAVIS-3D:
#   Train:  12657 (97.50%)
#   Val:      100 ( 0.77%)
#   Test:     225 ( 1.73%)
#   Total:  12982
# HiQBind:
#   Train:  31197 (98.88%)
#   Val:       74 ( 0.23%)
#   Test:     279 ( 0.88%)
#   Total:  31550
# Kinodata-3D:
#   Train:  70026 (99.54%)
#   Val:      100 ( 0.14%)
#   Test:     225 ( 0.32%)
#   Total:  70351
# Plinder:
#   Train: 248929 (99.32%)
#   Val:      753 ( 0.30%)
#   Test:     950 ( 0.38%)
#   Total: 250632
# SAIR:
#   Train: 1564352 (99.98%)
#   Val:      100 ( 0.01%)
#   Test:     225 ( 0.01%)
#   Total: 1564677
# SPINDR:
#   Train:  35334 (99.18%)
#   Val:       68 ( 0.19%)
#   Test:     225 ( 0.63%)
