import argparse
import math
import pickle

import numpy as np


DATA_DIR = "/data/molgen/geom_revisited"


def load_split(data_dir, name):
    path = f"{data_dir}/{name}_data.pickle"
    print(f"Loading {path} ...", flush=True)
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  -> {len(data)} molecules", flush=True)
    return data


def conformer_radius_proxies(data):
    radii = []
    for _smiles, confs in data:
        for mol in confs:
            pos = mol.GetConformer().GetPositions()
            radii.append(np.abs(pos).max())
    return np.array(radii, dtype=np.float64)


def round_up(value, step=0.5):
    return math.ceil(value / step) * step


def count_overflow(data, R, label):
    n_confs = 0
    n_overflow = 0
    for _smiles, confs in data:
        for mol in confs:
            pos = mol.GetConformer().GetPositions()
            n_confs += 1
            if np.abs(pos).max() > R:
                n_overflow += 1
    pct = 100 * n_overflow / n_confs if n_confs else 0
    print(f"  [{label}]  overflow: {n_overflow:>7} / {n_confs:<7}  ({pct:.4f}%)")
    return n_overflow, n_confs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", type=str, default=DATA_DIR,
        help="Directory with {train,val,test}_data_centered.pickle",
    )
    parser.add_argument(
        "--quantile", type=float, default=0.9999,
        help="Quantile level for choosing R (default: 0.9999 = 99.99%%)",
    )
    parser.add_argument(
        "--round_step", type=float, default=0.5,
        help="Round R up to this multiple (default: 0.5 A)",
    )
    args = parser.parse_args()

    # -- Load centered data --
    train_data = load_split(args.data_dir, "train")

    # -- Conformer radius proxies on train --
    print("\nComputing conformer radius proxies (m = max|Xc|) on train...", flush=True)
    m_train = conformer_radius_proxies(train_data)
    print(f"  {len(m_train):,} conformers")

    # -- Percentile table --
    print("\nPercentile table (train):")
    for p in [50, 75, 90, 95, 99, 99.5, 99.9, 99.95, 99.99, 99.999, 100]:
        val = np.percentile(m_train, p)
        print(f"  p{p:<8} = {val:.4f} A")

    # -- Pick R --
    R_raw = float(np.quantile(m_train, args.quantile))
    R = round_up(R_raw, args.round_step)
    print(f"\nR_raw (quantile {args.quantile})  = {R_raw:.4f} A")
    print(f"R     (rounded up to {args.round_step})  = {R:.1f} A")

    # -- Overflow counts --
    print(f"\nOverflow counts with R = {R:.1f}  (coords outside [-{R:.1f}, {R:.1f}]):")
    count_overflow(train_data, R, "train")

    # Load val/test if available
    for split in ("val", "test"):
        try:
            split_data = load_split(args.data_dir, split)
            count_overflow(split_data, R, split)
        except FileNotFoundError:
            print(f"  [{split}]  skipped (file not found)")

    print(f"\n>>> Recommended range: [-{R:.1f}, {R:.1f}]")
    print(f'>>> CLI flag:  --ranges "[-{R:.1f}, {R:.1f}], [-{R:.1f}, {R:.1f}], [-{R:.1f}, {R:.1f}]"')
