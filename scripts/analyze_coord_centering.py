"""
Analyze GEOM conformer coordinates:
1. Center all conformers (subtract centroid) and save centered pickle files
2. Verify centering (centroid displacement should be ~0)
3. Compute L/H bounds on centered train data (xyz-pooled and x-only)
   at quantile levels: 1%, 0.5%, 0.1%
4. Count overflow for all thresholds on centered train and test
"""

import pickle
import numpy as np
from rdkit.Geometry import Point3D

DATA_DIR = "/data/molgen/geom_revisited"


def get_coords(mol):
    """Extract N×3 coordinate array from an RDKit Mol with a conformer."""
    conf = mol.GetConformer()
    return conf.GetPositions().astype(np.float64)


def load_split(name):
    path = f"{DATA_DIR}/{name}_data.pickle"
    if 'centered' in name:
        name = name.replace("_centered", "")
        path = f"{DATA_DIR}/{name}_data_centered.pickle"
    print(f"Loading {path} ...", flush=True)
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  → {len(data)} molecules", flush=True)
    return data


# ──────────────────────────────────────────────────────────────────────────────
# 1. Center conformers in-place and save
# ──────────────────────────────────────────────────────────────────────────────

def center_split(data, split_name):
    """
    Center every conformer in-place (subtract centroid from atom positions).
    Saves the result to DATA_DIR/{split_name}_data_centered.pickle.
    Returns the modified data list.
    """
    print(f"\nCentering {split_name} ({len(data)} mols)...", flush=True)
    n_confs = 0
    for smiles, confs in data:
        for mol in confs:
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            centroid = pos.mean(axis=0)
            new_pos = pos - centroid
            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, Point3D(*new_pos[i].tolist()))
            n_confs += 1
    print(f"  Centered {n_confs} conformers.", flush=True)

    out_path = f"{DATA_DIR}/{split_name}_data_centered.pickle"
    print(f"  Saving to {out_path} ...", flush=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Saved.", flush=True)
    return data


# ──────────────────────────────────────────────────────────────────────────────
# 2. Verify centering
# ──────────────────────────────────────────────────────────────────────────────

def centering_stats(data, label, max_mols=None):
    """Compute centroid L2-norm for each conformer (should be ~0 after centering)."""
    norms = []
    mols_done = 0
    for smiles, confs in data:
        for mol in confs:
            X = get_coords(mol)
            norms.append(np.linalg.norm(X.mean(axis=0)))
        mols_done += 1
        if max_mols and mols_done >= max_mols:
            break
    norms = np.array(norms)
    print(f"\n[{label}] centroid displacement over {len(norms)} conformers"
          f" (from {mols_done} mols)")
    print(f"  mean   = {norms.mean():.2e} Å")
    print(f"  median = {np.median(norms):.2e} Å")
    print(f"  max    = {norms.max():.2e} Å")
    print(f"  >1e-9Å : {(norms > 1e-9).sum()} ({100*(norms > 1e-9).mean():.4f}%)")
    return norms


# ──────────────────────────────────────────────────────────────────────────────
# 3. Collect coordinates (data already centered, no re-centering needed)
# ──────────────────────────────────────────────────────────────────────────────

def collect_coords(data, label, axis=None):
    """
    Returns a 1D array of raw scalar coordinate values (no centering).
    axis=None  → pool all three axes (x, y, z)
    axis=0/1/2 → x / y / z only
    """
    axis_label = {None: "xyz pooled", 0: "x only", 1: "y only", 2: "z only"}
    all_vals = []
    print(f"\nCollecting coords from {label} ({len(data)} mols)"
          f" [{axis_label[axis]}]...", flush=True)
    for smiles, confs in data:
        for mol in confs:
            X = get_coords(mol)
            vals = X.flatten() if axis is None else X[:, axis]
            all_vals.append(vals)
    all_vals = np.concatenate(all_vals)
    print(f"  total scalar values: {len(all_vals):,}", flush=True)
    return all_vals


# ──────────────────────────────────────────────────────────────────────────────
# 4. Overflow counting (data already centered)
# ──────────────────────────────────────────────────────────────────────────────

def count_overflow(data, L, H, label):
    """Count mols and confs with at least one raw coordinate outside [L, H]."""
    n_mols = len(data)
    n_confs_total = 0
    n_confs_overflow = 0
    n_mols_overflow = 0

    for smiles, confs in data:
        mol_has_overflow = False
        for mol in confs:
            flat = get_coords(mol).flatten()   # raw, no centering
            has_out = bool((flat < L).any() or (flat > H).any())
            n_confs_total += 1
            if has_out:
                n_confs_overflow += 1
                mol_has_overflow = True
        if mol_has_overflow:
            n_mols_overflow += 1

    print(f"\n[{label}]  L={L:.4f}, H={H:.4f}")
    print(f"  Mols  with overflow: {n_mols_overflow:>6} / {n_mols:>6}"
          f"  ({100*n_mols_overflow/n_mols:.2f}%)")
    print(f"  Confs with overflow: {n_confs_overflow:>6} / {n_confs_total:>6}"
          f"  ({100*n_confs_overflow/n_confs_total:.2f}%)")
    return n_mols_overflow, n_confs_overflow, n_mols, n_confs_total


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Load raw (uncentered) data ──
    train_data = load_split("train")
    test_data  = load_split("test")

    # ── Step 1: Collect raw coords ──
    print("\n" + "="*60)
    print("STEP 1 — COORDINATE COLLECTION (raw, uncentered)")
    print("="*60)
    train_vals_xyz = collect_coords(train_data, "train", axis=None)
    train_vals_x   = collect_coords(train_data, "train", axis=0)

    # Tail quantiles: 1%, 0.1%, 0.01%, 0.001%, 0.0001%
    q_pairs = {
        "1%    / 99%":       (0.01,     0.99),
        "0.1%  / 99.9%":     (0.001,    0.999),
        "0.01% / 99.99%":    (0.0001,   0.9999),
        "0.001%/ 99.999%":   (0.00001,  0.99999),
        "0.0001%/99.9999%":  (0.000001, 0.999999),
    }

    print("\n" + "="*60)
    print("STEP 2 — L / H BOUNDS (raw train): xyz-pooled vs x-only")
    print("="*60)

    configs = {}
    for q_name, (qlo, qhi) in q_pairs.items():
        for src_name, src_vals in [("xyz", train_vals_xyz), ("x-only", train_vals_x)]:
            L = float(np.quantile(src_vals, qlo))
            H = float(np.quantile(src_vals, qhi))
            key = f"{q_name} [{src_name}]"
            configs[key] = (L, H)
            print(f"\n  {key}")
            print(f"    L = {L:.4f} Å,  H = {H:.4f} Å")

    # ── Step 2: Count overflow ──
    print("\n" + "="*60)
    print("STEP 3 — OVERFLOW COUNTS (raw, uncentered)")
    print("="*60)

    results = {}
    for thresh_name, (L, H) in configs.items():
        for split_label, split_data in [("TRAIN", train_data), ("TEST", test_data)]:
            key = f"{thresh_name} | {split_label}"
            nm_ov, nc_ov, nm_tot, nc_tot = count_overflow(
                split_data, L, H, f"{thresh_name} — {split_label}")
            results[key] = (nm_ov, nc_ov, nm_tot, nc_tot)

    # ── Summary table ──
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    header = (f"{'Threshold + source':<32} {'Split':<7}"
              f" {'Mol overflow':>22} {'Conf overflow':>22}")
    print(header)
    print("-" * len(header))
    for thresh_name in configs:
        for split_label in ["TRAIN", "TEST"]:
            key = f"{thresh_name} | {split_label}"
            nm_ov, nc_ov, nm_tot, nc_tot = results[key]
            print(f"{thresh_name:<32} {split_label:<7}"
                  f"  {nm_ov:>7}/{nm_tot:<7} ({100*nm_ov/nm_tot:5.2f}%)"
                  f"  {nc_ov:>8}/{nc_tot:<8} ({100*nc_ov/nc_tot:5.2f}%)")
