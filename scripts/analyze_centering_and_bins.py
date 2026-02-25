import pickle as pkl
import numpy as np

TRAIN_PATH = "/nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_revisit/train_data.pickle"
TEST_PATH  = "/nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_revisit/test_data.pickle"


def iter_conformers(data):
    for mol_idx, entry in enumerate(data):
        item = entry[1] if isinstance(entry, tuple) else entry
        confs = item if isinstance(item, list) else item.get("confs", [])
        for conf_idx, conf in enumerate(confs):
            conformer = conf.GetConformer()
            n = conformer.GetNumAtoms()
            pos = np.array(
                [[*conformer.GetAtomPosition(i)] for i in range(n)],
                dtype=np.float64,
            )
            yield mol_idx, conf_idx, pos



def centering_shift_stats(data, label, max_report=5):

    print(f"\n=== Centering shift analysis: {label} ===")
    shifts = []
    for _, _, pos in iter_conformers(data):
        centroid = pos.mean(axis=0)          # (3,)
        shift = np.linalg.norm(centroid)     # scalar
        shifts.append(shift)

    shifts = np.array(shifts)
    print(f"  Conformers analysed : {len(shifts):,}")
    print(f"  Centroid ‖μ‖  mean  : {shifts.mean():.4f} Å")
    print(f"  Centroid ‖μ‖  std   : {shifts.std():.4f} Å")
    print(f"  Centroid ‖μ‖  median: {np.median(shifts):.4f} Å")
    print(f"  Centroid ‖μ‖  max   : {shifts.max():.4f} Å")
    print(f"  Already (near-)zero : {(shifts < 0.001).sum():,} / {len(shifts):,} "
          f"({100*(shifts < 0.001).mean():.1f}%)")


def collect_centered_coords(data):
    """Return all centered scalar coordinates (x,y,z pooled) as 1-D array."""
    parts = []
    for _, _, pos in iter_conformers(data):
        centered = pos - pos.mean(axis=0)
        parts.append(centered.ravel())
    return np.concatenate(parts)


def count_overflow(data, L, H, label, threshold_label):

    n_confs_total = 0
    n_confs_overflow = 0
    overflow_mol_ids = set()

    for mol_idx, _, pos in iter_conformers(data):
        n_confs_total += 1
        centered = pos - pos.mean(axis=0)
        if np.any(centered < L) or np.any(centered > H):
            n_confs_overflow += 1
            overflow_mol_ids.add(mol_idx)

    # total unique molecules
    unique_mol_ids = set()
    for mol_idx, _, _ in iter_conformers(data):
        unique_mol_ids.add(mol_idx)
    n_mols_total = len(unique_mol_ids)
    n_mols_overflow = len(overflow_mol_ids)

    print(
        f"  [{threshold_label}] {label}: "
        f"confs {n_confs_overflow:,}/{n_confs_total:,} "
        f"({100*n_confs_overflow/max(n_confs_total,1):.3f}%)  |  "
        f"mols {n_mols_overflow:,}/{n_mols_total:,} "
        f"({100*n_mols_overflow/max(n_mols_total,1):.3f}%)"
    )
    return n_confs_overflow, n_confs_total, n_mols_overflow, n_mols_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading data…")
with open(TRAIN_PATH, "rb") as f:
    train_data = pkl.load(f)
with open(TEST_PATH, "rb") as f:
    test_data = pkl.load(f)
print(f"Train entries: {len(train_data):,}  |  Test entries: {len(test_data):,}")

# --- Step 1: centering shifts ---
centering_shift_stats(train_data, "TRAIN")
centering_shift_stats(test_data,  "TEST")

# --- Step 2: fit L/H from train ---
print("\n=== Collecting centered coordinates from TRAIN (all axes pooled) ===")
V = collect_centered_coords(train_data)
print(f"  Total scalar values: {len(V):,}")
print(f"  Overall min / max  : {V.min():.4f} / {V.max():.4f} Å")
print(f"  Mean ± std         : {V.mean():.4f} ± {V.std():.4f} Å")

thresholds = {
    "1.0%": (0.01, 0.99),
    "0.5%": (0.005, 0.995),
}

results = {}
for label, (qlo, qhi) in thresholds.items():
    L = np.quantile(V, qlo)
    H = np.quantile(V, qhi)
    results[label] = (L, H)
    print(f"\n  [{label} tail] L = {L:.4f} Å,  H = {H:.4f} Å")

# --- Step 3: overflow counts ---
print("\n=== Overflow counts (any atom coord of centered conformer outside [L, H]) ===")
for thr_label, (L, H) in results.items():
    count_overflow(train_data, L, H, "TRAIN", thr_label)
    count_overflow(test_data,  L, H, "TEST",  thr_label)
