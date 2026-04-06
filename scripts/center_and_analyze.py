"""
Center all GEOM conformers, report centering stats, and save coordinate
distributions (as numpy arrays and plots) before and after centering.

Outputs go to: scripts/centering_analysis/
  - centroid_norms_before.npy       centroid displacement per conformer (before)
  - centroid_norms_after.npy        centroid displacement per conformer (after)
  - coords_before_{xyz,x,y,z}.npy  coordinate values before centering
  - coords_after_{xyz,x,y,z}.npy   coordinate values after centering
  - distributions.png               before/after coordinate histograms
  - centroid_norms.png              before/after centroid displacement histograms

Centered data saved to DATA_DIR as:
  train_data_centered.pickle / val_data_centered.pickle / test_data_centered.pickle
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit.Geometry import Point3D

DATA_DIR  = "/data/molgen/geom_revisited"
OUT_DIR   = os.path.join(os.path.dirname(__file__), "centering_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_MOLS = 5000   # mols used for distribution snapshots and centroid stats


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_coords(mol):
    return mol.GetConformer().GetPositions().astype(np.float64)


def load_split(split):
    path = f"{DATA_DIR}/{split}_data.pickle"
    print(f"Loading {path} ...", flush=True)
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  → {len(data)} molecules", flush=True)
    return data


def save_split(data, split):
    path = f"{DATA_DIR}/{split}_data_centered.pickle"
    print(f"  Saving → {path} ...", flush=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Done.", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Centering
# ──────────────────────────────────────────────────────────────────────────────

def center_in_place(data, split):
    """Subtract per-conformer centroid from every atom. Mutates data in-place."""
    print(f"\nCentering {split} ({len(data)} mols)...", flush=True)
    n_confs = 0
    for smiles, confs in data:
        for mol in confs:
            conf = mol.GetConformer()
            pos  = conf.GetPositions()
            mu   = pos.mean(axis=0)
            new  = pos - mu
            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, Point3D(*new[i].tolist()))
            n_confs += 1
    print(f"  {n_confs:,} conformers centered.", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Stats collection
# ──────────────────────────────────────────────────────────────────────────────

def collect_centroid_norms(data, max_mols=None):
    """Centroid L2-norm for each conformer."""
    norms, done = [], 0
    for smiles, confs in data:
        for mol in confs:
            X = get_coords(mol)
            norms.append(np.linalg.norm(X.mean(axis=0)))
        done += 1
        if max_mols and done >= max_mols:
            break
    return np.array(norms)


def print_norm_stats(norms, label):
    print(f"\n[{label}]  n={len(norms):,} conformers")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"  p{p:3d} = {np.percentile(norms, p):.6f} Å")
    print(f"  mean  = {norms.mean():.6f} Å")
    print(f"  >1e-9 : {(norms > 1e-9).sum():,}  ({100*(norms > 1e-9).mean():.4f}%)")
    print(f"  >0.01 : {(norms > 0.01).sum():,}  ({100*(norms > 0.01).mean():.2f}%)")
    print(f"  >0.1  : {(norms > 0.1).sum():,}  ({100*(norms > 0.1).mean():.2f}%)")


def collect_coord_distributions(data, max_mols=None):
    """Pool all coordinate values (optionally limited to max_mols molecules)."""
    xyz, x, y, z = [], [], [], []
    done = 0
    for smiles, confs in data:
        for mol in confs:
            X = get_coords(mol)
            xyz.append(X.flatten())
            x.append(X[:, 0])
            y.append(X[:, 1])
            z.append(X[:, 2])
        done += 1
        if max_mols and done >= max_mols:
            break
    return {
        "xyz": np.concatenate(xyz),
        "x":   np.concatenate(x),
        "y":   np.concatenate(y),
        "z":   np.concatenate(z),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_distributions(before, after, out_path):
    keys = ["xyz", "x", "y", "z"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(
        f"Coordinate distributions — before vs after centering "
        f"(first {SAMPLE_MOLS:,} train mols)", fontsize=13)

    bins = np.linspace(-30, 30, 300)
    for col, key in enumerate(keys):
        b = np.clip(before[key], -30, 30)
        a = np.clip(after[key],  -30, 30)

        for row, (arr, raw, color, tag) in enumerate([
            (b, before[key], "steelblue", "BEFORE"),
            (a, after[key],  "coral",     "AFTER"),
        ]):
            ax = axes[row, col]
            ax.hist(arr, bins=bins, color=color, alpha=0.85)
            ax.set_title(f"{tag} — {key}", fontsize=11)
            ax.set_xlabel("Å")
            ax.set_ylabel("count")
            ax.text(0.97, 0.95,
                    f"μ={raw.mean():.3f}\nσ={raw.std():.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


def plot_centroid_norms(before, after, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Centroid displacement |μ| per conformer", fontsize=13)

    ax1.hist(np.clip(before, 0, 5), bins=200, color="steelblue", alpha=0.85)
    ax1.set_title("BEFORE centering")
    ax1.set_xlabel("|centroid| (Å)")
    ax1.set_ylabel("count")
    ax1.text(0.97, 0.95,
             f"mean={before.mean():.3f}\nmedian={np.median(before):.3f}\n"
             f"p99={np.percentile(before,99):.3f}\nmax={before.max():.3f}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=9)

    ax2.hist(after, bins=50, color="coral", alpha=0.85)
    ax2.set_title("AFTER centering")
    ax2.set_xlabel("|centroid| (Å)")
    ax2.set_ylabel("count")
    ax2.text(0.97, 0.95,
             f"mean={after.mean():.2e}\nmax={after.max():.2e}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    train_data = load_split("train")
    val_data   = load_split("val")
    test_data  = load_split("test")

    # ── BEFORE ──
    print("\n" + "="*60)
    print(f"BEFORE CENTERING (sample {SAMPLE_MOLS:,} train mols)")
    print("="*60)

    norms_before = collect_centroid_norms(train_data, max_mols=SAMPLE_MOLS)
    print_norm_stats(norms_before, "centroid |μ| BEFORE")
    np.save(os.path.join(OUT_DIR, "centroid_norms_before.npy"), norms_before)
    print(f"  Saved centroid_norms_before.npy  ({len(norms_before):,} values)")

    print(f"\nCollecting coord distributions...", flush=True)
    dists_before = collect_coord_distributions(train_data, max_mols=SAMPLE_MOLS)
    for key, arr in dists_before.items():
        np.save(os.path.join(OUT_DIR, f"coords_before_{key}.npy"), arr)
        print(f"  Saved coords_before_{key}.npy  "
              f"({len(arr):,} vals,  mean={arr.mean():.3f}, std={arr.std():.3f})")

    # ── CENTER + SAVE ──
    print("\n" + "="*60)
    print("CENTERING AND SAVING ALL SPLITS")
    print("="*60)
    for data, split in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
        center_in_place(data, split)
        save_split(data, split)

    # ── AFTER ──
    print("\n" + "="*60)
    print(f"AFTER CENTERING (same {SAMPLE_MOLS:,} train mols)")
    print("="*60)

    norms_after = collect_centroid_norms(train_data, max_mols=SAMPLE_MOLS)
    print_norm_stats(norms_after, "centroid |μ| AFTER")
    np.save(os.path.join(OUT_DIR, "centroid_norms_after.npy"), norms_after)
    print(f"  Saved centroid_norms_after.npy  ({len(norms_after):,} values)")

    print(f"\nCollecting coord distributions after centering...", flush=True)
    dists_after = collect_coord_distributions(train_data, max_mols=SAMPLE_MOLS)
    for key, arr in dists_after.items():
        np.save(os.path.join(OUT_DIR, f"coords_after_{key}.npy"), arr)
        print(f"  Saved coords_after_{key}.npy  "
              f"({len(arr):,} vals,  mean={arr.mean():.2e}, std={arr.std():.3f})")

    # ── PLOTS ──
    print("\n" + "="*60)
    print("SAVING PLOTS")
    print("="*60)
    plot_distributions(dists_before, dists_after,
                       os.path.join(OUT_DIR, "distributions.png"))
    plot_centroid_norms(norms_before, norms_after,
                        os.path.join(OUT_DIR, "centroid_norms.png"))

    print(f"\nAll outputs → {OUT_DIR}")
