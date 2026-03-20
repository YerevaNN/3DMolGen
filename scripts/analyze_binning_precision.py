"""
Measure coordinate precision loss from binning on the validation set.

Compares three encoding methods:
  1. Uniform bins (256 bins, raw/non-centered)
  2. Quantile bins (256 bins, raw/non-centered)
  3. Raw text encoding (truncate to 4 decimal places)

For each conformer, computes RMSD between original and round-tripped coordinates.

Usage:
    python scripts/analyze_binning_precision.py \
        --data /data/molgen/geom_revisited/val_data.pickle \
        --n_mols 10000
"""

import argparse
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from molgen3D.data_processing.smiles_encoder_decoder import (
    BinConfig,
    _encode_scalar,
    _decode_scalar,
)

BIN_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "bin_configs")


def truncate_coord(x, precision=4):
    """Replicate the raw text encoding's truncation."""
    factor = 10 ** precision
    truncated = math.trunc(x * factor) / factor
    if abs(truncated) < 10 ** (-precision):
        truncated = 0.0
    return truncated


def roundtrip_bins(coords_flat, config):
    """Bin and unbin a flat array of coordinate values."""
    decoded = np.empty_like(coords_flat)
    for i, c in enumerate(coords_flat):
        idx = _encode_scalar(float(c), config)
        decoded[i] = _decode_scalar(idx, config)
    return decoded


def roundtrip_raw(coords_flat, precision=4):
    """Truncate to `precision` decimal places (raw text encoding)."""
    return np.array([truncate_coord(float(c), precision) for c in coords_flat])


def conformer_rmsd(original, decoded):
    """RMSD between two (N, 3) coordinate arrays."""
    diff = original - decoded.reshape(original.shape)
    return np.sqrt(np.mean(diff ** 2))


def conformer_max_error(original, decoded):
    """Max absolute error across all coordinates."""
    return np.max(np.abs(original - decoded.reshape(original.shape)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=str,
                        default="/data/molgen/geom_revisited/val_data.pickle")
    parser.add_argument("--n_mols", type=int, default=10000)
    parser.add_argument("--precision", type=int, default=4,
                        help="Decimal precision for raw text encoding")
    args = parser.parse_args()

    # Load bin configs (raw = fit on non-centered data)
    uniform_cfg = BinConfig.load(os.path.join(BIN_CONFIGS_DIR, "uniform_bins.json"))
    quantile_cfg = BinConfig.load(os.path.join(BIN_CONFIGS_DIR, "quantile_bins.json"))

    print(f"Uniform bins: L={uniform_cfg.L:.4f}, H={uniform_cfg.H:.4f}, "
          f"n_bins={uniform_cfg.n_bins}, "
          f"bin_width={(uniform_cfg.H - uniform_cfg.L) / uniform_cfg.n_bins:.6f} A")
    print(f"Quantile bins: L={quantile_cfg.L:.4f}, H={quantile_cfg.H:.4f}, "
          f"n_bins={quantile_cfg.n_bins}, "
          f"median_bin_width={np.median(np.diff(quantile_cfg.edges)):.6f} A")
    print(f"Raw text precision: {args.precision} decimal places")

    # Load validation data
    print(f"\nLoading {args.data} ...")
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    print(f"  {len(data)} molecules in validation set")

    # Collect conformers from first n_mols molecules
    n_mols = min(args.n_mols, len(data))
    print(f"  Using first {n_mols} molecules\n")

    methods = ["uniform", "quantile", "raw"]
    stats = {m: {"rmsd": [], "max_err": [], "overflow": 0, "total_coords": 0} for m in methods}

    n_conformers = 0
    for mol_idx, (smiles, confs) in enumerate(data[:n_mols]):
        for mol in confs:
            try:
                pos = mol.GetConformer().GetPositions()  # (n_atoms, 3)
            except Exception:
                continue

            flat = pos.flatten()
            n_conformers += 1

            # Uniform bins
            decoded_u = roundtrip_bins(flat, uniform_cfg)
            stats["uniform"]["rmsd"].append(conformer_rmsd(pos, decoded_u))
            stats["uniform"]["max_err"].append(conformer_max_error(pos, decoded_u))
            stats["uniform"]["overflow"] += int(np.sum((flat < uniform_cfg.L) | (flat > uniform_cfg.H)))
            stats["uniform"]["total_coords"] += len(flat)

            # Quantile bins
            decoded_q = roundtrip_bins(flat, quantile_cfg)
            stats["quantile"]["rmsd"].append(conformer_rmsd(pos, decoded_q))
            stats["quantile"]["max_err"].append(conformer_max_error(pos, decoded_q))
            stats["quantile"]["overflow"] += int(np.sum((flat < quantile_cfg.L) | (flat > quantile_cfg.H)))
            stats["quantile"]["total_coords"] += len(flat)

            # Raw text truncation
            decoded_r = roundtrip_raw(flat, args.precision)
            stats["raw"]["rmsd"].append(conformer_rmsd(pos, decoded_r))
            stats["raw"]["max_err"].append(conformer_max_error(pos, decoded_r))
            stats["raw"]["total_coords"] += len(flat)

        if (mol_idx + 1) % 2000 == 0:
            print(f"  Processed {mol_idx + 1}/{n_mols} molecules "
                  f"({n_conformers} conformers so far)")

    print(f"\nTotal: {n_mols} molecules, {n_conformers} conformers\n")

    # Print results
    print("=" * 72)
    print(f"{'Method':<12} {'Mean RMSD':>10} {'Median RMSD':>12} {'p95 RMSD':>10} "
          f"{'p99 RMSD':>10} {'Max RMSD':>10} {'Mean MaxErr':>12}")
    print("=" * 72)

    for method in methods:
        rmsds = np.array(stats[method]["rmsd"])
        max_errs = np.array(stats[method]["max_err"])

        print(f"{method:<12} "
              f"{np.mean(rmsds):>10.6f} "
              f"{np.median(rmsds):>12.6f} "
              f"{np.percentile(rmsds, 95):>10.6f} "
              f"{np.percentile(rmsds, 99):>10.6f} "
              f"{np.max(rmsds):>10.6f} "
              f"{np.mean(max_errs):>12.6f}")

    print("=" * 72)

    # Overflow stats (only for binned methods)
    print(f"\nOverflow statistics (coords outside [L, H]):")
    for method in ["uniform", "quantile"]:
        s = stats[method]
        pct = 100.0 * s["overflow"] / s["total_coords"] if s["total_coords"] else 0
        print(f"  {method:<12}: {s['overflow']:>8} / {s['total_coords']:<10} ({pct:.4f}%)")

    # Per-axis error distribution (sample from last batch)
    print(f"\nPer-scalar absolute error distribution:")
    print(f"{'Method':<12} {'Mean':>10} {'Median':>10} {'p95':>10} {'p99':>10} {'Max':>10}")
    print("-" * 64)

    # Recompute on a sample for per-scalar stats
    sample_flat = []
    for mol_idx, (smiles, confs) in enumerate(data[:min(1000, n_mols)]):
        for mol in confs:
            try:
                pos = mol.GetConformer().GetPositions()
                sample_flat.append(pos.flatten())
            except Exception:
                continue
    sample_flat = np.concatenate(sample_flat)

    for method, cfg in [("uniform", uniform_cfg), ("quantile", quantile_cfg)]:
        decoded = roundtrip_bins(sample_flat, cfg)
        errs = np.abs(sample_flat - decoded)
        print(f"{method:<12} {np.mean(errs):>10.6f} {np.median(errs):>10.6f} "
              f"{np.percentile(errs, 95):>10.6f} {np.percentile(errs, 99):>10.6f} "
              f"{np.max(errs):>10.6f}")

    decoded_r = roundtrip_raw(sample_flat, args.precision)
    errs_r = np.abs(sample_flat - decoded_r)
    print(f"{'raw':<12} {np.mean(errs_r):>10.6f} {np.median(errs_r):>10.6f} "
          f"{np.percentile(errs_r, 95):>10.6f} {np.percentile(errs_r, 99):>10.6f} "
          f"{np.max(errs_r):>10.6f}")


if __name__ == "__main__":
    main()
