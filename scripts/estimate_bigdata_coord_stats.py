#!/usr/bin/env python3
"""Quickly estimate coordinate statistics by sampling conformers from shard pickles.

Randomly samples up to --max_confs conformers across all shards, centers each
one, and computes statistics directly on the sampled values (no histogram needed
at this scale).

Usage
-----
python scripts/estimate_bigdata_coord_stats.py \
    --input_dir /mnt/weka/vtarasov/3DBigData_formatted/grouped_conformers \
    --max_confs 500000
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molgen3D.data_processing.smiles_encoder_decoder import fit_quantile_bins  # noqa: E402


def sample_shard(
    shard_path: str,
    target: int,
    rng: random.Random,
) -> np.ndarray:
    """Return a flat float32 array of up to *target* randomly sampled centered
    conformer coordinates from one shard.

    Uses reservoir sampling so we never load all coordinates at once.
    """
    with open(shard_path, "rb") as fh:
        shard: dict = pickle.load(fh)

    # Collect all (smiles, mol_list) pairs in shuffled order
    items = list(shard.items())
    rng.shuffle(items)

    coords_list: list[np.ndarray] = []
    collected = 0

    for _smiles, mols in items:
        if collected >= target:
            break
        rng.shuffle(mols)
        for mol in mols:
            if collected >= target:
                break
            if mol is None:
                continue
            try:
                pos = mol.GetConformer().GetPositions()
            except Exception:
                continue
            pos = pos - pos.mean(axis=0)          # center
            coords_list.append(pos.ravel().astype(np.float32))
            collected += len(coords_list[-1])

    return np.concatenate(coords_list) if coords_list else np.empty(0, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory with shard_NNNN.pkl files.",
    )
    parser.add_argument(
        "--max_confs",
        type=int,
        default=500_000,
        help="Total number of conformers to sample (default: 500 000).",
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=1,
        help=(
            "Maximum number of shards to load (default: 1).  "
            "Each shard is ~16 GB on disk; loading more improves coverage "
            "but takes proportionally longer."
        ),
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=256,
        help="Quantile bins to fit and report (default: 256).",
    )
    parser.add_argument(
        "--q_low",
        type=float,
        default=0.0001,
        help="Lower quantile cut-point (default 0.01%% = 0.0001).",
    )
    parser.add_argument(
        "--q_high",
        type=float,
        default=0.999,
        help="Upper quantile cut-point (default 99.9%% = 0.999).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(os.path.join(args.input_dir, "shard_*.pkl")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.pkl files in {args.input_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(shard_paths)

    shard_paths = shard_paths[: args.max_shards]
    per_shard = max(1, args.max_confs // len(shard_paths))
    print(f"Shards      : {len(shard_paths)} / {len(sorted(glob.glob(os.path.join(args.input_dir, 'shard_*.pkl'))))}")
    print(f"Target confs: {args.max_confs:,}  ({per_shard:,} per shard)")
    print(flush=True)

    t0 = time.time()
    chunks: list[np.ndarray] = []

    for path in tqdm(shard_paths, desc="Sampling", unit="shard", dynamic_ncols=True):
        chunk = sample_shard(path, per_shard, rng)
        chunks.append(chunk)
        tqdm.write(f"  {os.path.basename(path)}: {len(chunk):,} coords")

    values = np.concatenate(chunks)
    elapsed = time.time() - t0

    print(f"\nSampled {len(values):,} coordinate scalars in {elapsed:.1f}s", flush=True)

    # Statistics
    percentiles_q = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999, 0.9999]
    pct_labels    = ["p0.001", "p0.01", "p0.1", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99.9", "p99.99"]
    pct_values = np.quantile(values, percentiles_q)

    print(f"\n{'='*58}")
    print(f"  Coordinate stats  (centered, pooled xyz, n={len(values):,})")
    print(f"{'='*58}")
    print(f"  min        : {values.min():>+14.4f} Å")
    print(f"  max        : {values.max():>+14.4f} Å")
    print(f"  mean       : {values.mean():>+14.6f} Å")
    print(f"  std        : {values.std():>14.6f} Å")
    for label, val in zip(pct_labels, pct_values):
        marker = " ◄" if label in ("p0.01", "p99.9") else ""
        print(f"  {label:<9}: {val:>+14.4f} Å{marker}")
    print(f"{'='*58}")

    # Quantile bin fit
    cfg = fit_quantile_bins(values, n_bins=args.n_bins, q_low=args.q_low, q_high=args.q_high)
    print(f"\n  Quantile BinConfig ({args.n_bins} bins, q=[{args.q_low}, {args.q_high}])")
    print(f"  L              : {cfg.L:>+14.4f} Å")
    print(f"  H              : {cfg.H:>+14.4f} Å")
    print(f"  median bin w   : {np.median(np.diff(cfg.edges)):>14.6f} Å")
    print(f"  digit_width    : {cfg.digit_width}")
    print()


if __name__ == "__main__":
    main()
