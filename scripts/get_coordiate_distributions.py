from rdkit import Chem
from rdkit.Chem import AllChem
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os

CLIP = 13.0
OUT_DIR = "/auto/home/vover/3DMolGen/outputs/coord_distribution"
os.makedirs(OUT_DIR, exist_ok=True)


def collect_coords(smiles_list):
    """Return arrays of all x/y/z atom positions and per-molecule outside-clip counts."""
    xs, ys, zs = [], [], []
    n_mols = 0
    n_mols_outside = 0
    for entry in smiles_list:
        item = entry[1] if isinstance(entry, tuple) else entry
        confs = item if isinstance(item, list) else item.get("confs", [])
        for conf in confs:
            n_mols += 1
            conformer = conf.GetConformer()
            mol_outside = False
            for i in range(conformer.GetNumAtoms()):
                pos = conformer.GetAtomPosition(i)
                xs.append(pos.x)
                ys.append(pos.y)
                zs.append(pos.z)
                if abs(pos.x) > CLIP or abs(pos.y) > CLIP or abs(pos.z) > CLIP:
                    mol_outside = True
            if mol_outside:
                n_mols_outside += 1
    return np.array(xs), np.array(ys), np.array(zs), n_mols, n_mols_outside


def plot_coord_distribution(xs, ys, zs, n_mols, n_mols_outside, split_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"{split_name}  |  {n_mols_outside}/{n_mols} conformers with any coord outside "
        f"[{-CLIP}, {CLIP}]  ({100*n_mols_outside/max(n_mols,1):.1f}%)",
        fontsize=11,
    )
    for ax, coords, label in zip(axes, [xs, ys, zs], ["X", "Y", "Z"]):
        ax.hist(coords, bins=200, color="steelblue", edgecolor="none")
        ax.axvline(-CLIP, color="red", linestyle="--", linewidth=1.2, label=f"±{CLIP}")
        ax.axvline( CLIP, color="red", linestyle="--", linewidth=1.2)
        n_out = int(np.sum(np.abs(coords) > CLIP))
        ax.set_title(f"{label}  |  {n_out} atoms outside  (min {coords.min():.2f}, max {coords.max():.2f})")
        ax.set_xlabel("Coordinate (Å)")
        ax.set_ylabel("Atom count")
        ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{split_name}_coord_dist.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


with open("/nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_revisit/train_data.pickle", "rb") as f:
    xl_smi = pkl.load(f)

with open("/nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_revisit/val_data.pickle", "rb") as f:
    distinct_smiles = pkl.load(f)

with open("/nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_revisit/test_data.pickle", "rb") as f:
    qm9_smi = pkl.load(f)

splits = [("val", distinct_smiles), ("train", xl_smi), ("test", qm9_smi)]
for split_name, data in splits:
    print(f"\nProcessing {split_name}...")
    xs, ys, zs, n_mols, n_mols_outside = collect_coords(data)
    print(f"  X min/max: {xs.min():.3f} / {xs.max():.3f}")
    print(f"  Y min/max: {ys.min():.3f} / {ys.max():.3f}")
    print(f"  Z min/max: {zs.min():.3f} / {zs.max():.3f}")
    print(f"  Conformers outside [{-CLIP},{CLIP}]: {n_mols_outside}/{n_mols}")
    plot_coord_distribution(xs, ys, zs, n_mols, n_mols_outside, split_name)
