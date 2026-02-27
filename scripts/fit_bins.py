"""
Fit bin configurations from training data.

Produces two BinConfig JSON files (uniform + quantile) that can be passed
to ``encode_cartesian_with_config`` / ``decode_cartesian_with_config``.

Usage:
    python scripts/fit_bins.py \
        --data_dir /data/molgen/geom_revisited \
        --out_dir  src/molgen3D/config/bin_configs \
        --n_bins 256 --q_low 0.01 --q_high 0.99
"""

import argparse
import os
import pickle

import numpy as np

# Append project root so the import works when running as a script
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from molgen3D.data_processing.smiles_encoder_decoder import (
    fit_uniform_bins,
    fit_quantile_bins,
)


DATA_DIR = "/data/molgen/geom_revisited"


def load_split(data_dir, name):
    path = f"{data_dir}/{name}_data.pickle"
    print(f"Loading {path} ...", flush=True)
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  -> {len(data)} molecules", flush=True)
    return data


def pool_coords(data, x_only=True):
    """Pool scalar coordinate values into a single 1-D array.

    Args:
        data: list of (smiles, [mol, ...]) tuples.
        x_only: if True, pool only the X coordinate (column 0).
            The X-axis distribution is representative of all axes
            (random molecular orientations), so quantile edges
            computed from X alone can be reused for Y and Z.
    """
    all_vals = []
    for _smiles, confs in data:
        for mol in confs:
            pos = mol.GetConformer().GetPositions()
            if x_only:
                all_vals.append(pos[:, 0])
            else:
                all_vals.append(pos.flatten())
    return np.concatenate(all_vals)


def overflow_stats(data, L, H, label):
    """Count conformers with any coordinate outside [L, H]."""
    n_confs = n_overflow = 0
    for _smiles, confs in data:
        for mol in confs:
            flat = mol.GetConformer().GetPositions().flatten()
            n_confs += 1
            if (flat < L).any() or (flat > H).any():
                n_overflow += 1
    pct = 100 * n_overflow / n_confs if n_confs else 0
    print(f"  [{label}]  overflow: {n_overflow:>7} / {n_confs:<7}  ({pct:.4f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--out_dir", type=str, default="../src/molgen3D/config/bin_configs")
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument("--q_low", type=float, default=0.0001)
    parser.add_argument("--q_high", type=float, default=0.9999)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -- Load and pool coordinates --
    train_data = load_split(args.data_dir, "train")

    # Pool all xyz for uniform bins and distribution summary
    print("\nPooling all scalar coordinates from train (xyz)...", flush=True)
    V_all = pool_coords(train_data)
    print(f"  {len(V_all):,} scalar values")

    # Pool X-only for quantile bins (X is representative of all axes
    # since molecular orientations are random)
    print("Pooling X-only coordinates from train...", flush=True)
    V_x = pool_coords(train_data, x_only=True)
    print(f"  {len(V_x):,} scalar values")

    # -- Distribution summary --
    print("\nCoordinate distribution (train, pooled xyz):")
    for p in [0.01, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 99.99]:
        print(f"  p{p:<6} = {np.percentile(V_all, p):+.4f} A")
    print(f"  min    = {V_all.min():+.4f} A")
    print(f"  max    = {V_all.max():+.4f} A")

    # -- Fit uniform (from pooled xyz) --
    print(f"\n{'='*60}")
    print(f"Fitting UNIFORM bins  (B={args.n_bins}, q=[{args.q_low}, {args.q_high}])")
    print(f"{'='*60}")
    uniform_cfg = fit_uniform_bins(V_all, n_bins=args.n_bins,
                                   q_low=args.q_low, q_high=args.q_high)
    print(f"  L = {uniform_cfg.L:+.4f} A")
    print(f"  H = {uniform_cfg.H:+.4f} A")
    print(f"  w = {(uniform_cfg.H - uniform_cfg.L) / uniform_cfg.n_bins:.6f} A")
    print(f"  digit_width = {uniform_cfg.digit_width}")

    uniform_path = os.path.join(args.out_dir, "uniform_bins.json")
    uniform_cfg.save(uniform_path)
    print(f"  Saved -> {uniform_path}")

    # -- Fit quantile (from X-only) --
    print(f"\n{'='*60}")
    print(f"Fitting QUANTILE bins  (B={args.n_bins}, q=[{args.q_low}, {args.q_high}], X-only)")
    print(f"{'='*60}")
    quantile_cfg = fit_quantile_bins(V_x, n_bins=args.n_bins,
                                     q_low=args.q_low, q_high=args.q_high)
    print(f"  L = {quantile_cfg.L:+.4f} A")
    print(f"  H = {quantile_cfg.H:+.4f} A")
    print(f"  edge range: [{quantile_cfg.edges[0]:+.4f}, {quantile_cfg.edges[-1]:+.4f}]")
    print(f"  median bin width = {np.median(np.diff(quantile_cfg.edges)):.6f} A")
    print(f"  digit_width = {quantile_cfg.digit_width}")

    quantile_path = os.path.join(args.out_dir, "quantile_bins.json")
    quantile_cfg.save(quantile_path)
    print(f"  Saved -> {quantile_path}")

    # -- Overflow stats --
    print(f"\nOverflow counts (coords outside [L, H]):")
    overflow_stats(train_data, uniform_cfg.L, uniform_cfg.H, "train-uniform")
    overflow_stats(train_data, quantile_cfg.L, quantile_cfg.H, "train-quantile")

    for split in ("val", "test"):
        try:
            split_data = load_split(args.data_dir, split)
            overflow_stats(split_data, uniform_cfg.L, uniform_cfg.H, f"{split}-uniform")
            overflow_stats(split_data, quantile_cfg.L, quantile_cfg.H, f"{split}-quantile")
        except FileNotFoundError:
            print(f"  [{split}]  skipped (file not found)")

    print(f"\nConfigs saved to {args.out_dir}/")
    print(f"  uniform_bins.json   — use with encode_cartesian_with_config")
    print(f"  quantile_bins.json  — use with encode_cartesian_with_config")
