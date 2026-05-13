#!/usr/bin/env python3
"""Compute exact coordinate statistics over all centered big-data shard pickles.

Memory-efficient design
-----------------------
Coordinate values are **never stored**.  A fine-grained histogram (default 1 M
bins, 8 MB) is accumulated; all statistics and quantile edges are derived from
its CDF.

Two-phase workflow (recommended for large datasets)
---------------------------------------------------
Phase 1 — run as a SLURM array, one task per shard:

    sbatch --array=0-7 scripts/submit_bigdata_coord_stats.sh

Each task processes one shard and writes a partial histogram to
``{out_dir}/partial_hist/shard_NNNN.npz``.

Phase 2 — merge partial histograms (single job, seconds):

    python scripts/compute_bigdata_coord_stats.py \\
        --merge --out_dir /path/to/processed

Single-process mode (slow on large shards — use array mode instead):

    python scripts/compute_bigdata_coord_stats.py \\
        --input_dir /path/to/shards --out_dir /path/to/out
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molgen3D.data_processing.smiles_encoder_decoder import BinConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Histogram-based quantile helpers
# ---------------------------------------------------------------------------

def _histogram_quantile(cum_norm: np.ndarray, edges: np.ndarray, q: float | np.ndarray):
    return np.interp(q, cum_norm, edges)


def _fit_quantile_bins_from_histogram(
    counts: np.ndarray,
    edges: np.ndarray,
    n_bins: int = 256,
    q_low: float = 0.0001,
    q_high: float = 0.999,
) -> BinConfig:
    """Exact equivalent of fit_quantile_bins() but works from a histogram."""
    cum = np.empty(len(edges), dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(counts, out=cum[1:])
    total = cum[-1]
    if total == 0:
        raise ValueError("Empty histogram — no coordinates collected.")
    cum /= total

    L = float(_histogram_quantile(cum, edges, q_low))
    H = float(_histogram_quantile(cum, edges, q_high))

    q_points = np.linspace(0.0, 1.0, n_bins + 1)
    raw_edges = _histogram_quantile(cum, edges, q_points).astype(np.float64)
    bin_edges = np.where(q_points <= q_low, L, np.where(q_points >= q_high, H, raw_edges))

    return BinConfig(mode="quantile", L=L, H=H, n_bins=n_bins, edges=bin_edges)


# ---------------------------------------------------------------------------
# Phase 1: process one shard → partial .npz
# Perf notes:
#   • Coordinates are buffered (BATCH_COORDS) before calling np.histogram so
#     the 1 M-bin histogram is computed in large batches (~10 M values) rather
#     than once per conformer — reduces histogram call count by ~10 000×.
#   • The shard dict is loaded once in the main process; workers inherit it
#     via fork (copy-on-write, zero IPC cost for the Mol objects).
# ---------------------------------------------------------------------------

_SHARD_GLOBAL: dict | None = None  # populated before Pool fork
_HIST_EDGES_GLOBAL: np.ndarray | None = None
_BATCH_COORDS = 10_000_000  # flush histogram after this many scalars


def _worker_chunk(args: tuple) -> dict:
    """Process a list of (smiles, mols) pairs from the fork-inherited global."""
    keys_chunk: list[str] = args
    hist = np.zeros(len(_HIST_EDGES_GLOBAL) - 1, dtype=np.int64)
    total = 0
    coord_sum = 0.0
    coord_min = np.inf
    coord_max = -np.inf
    n_confs = n_skip = 0

    buf: list[np.ndarray] = []
    buf_size = 0

    def _flush():
        nonlocal buf, buf_size
        if not buf:
            return
        batch = np.concatenate(buf).astype(np.float64)
        h, _ = np.histogram(batch, bins=_HIST_EDGES_GLOBAL)
        hist[:] += h
        buf = []
        buf_size = 0

    for key in keys_chunk:
        mols = _SHARD_GLOBAL[key]
        for mol in mols:
            if mol is None:
                n_skip += 1
                continue
            try:
                pos = mol.GetConformer().GetPositions()
            except Exception:
                n_skip += 1
                continue
            pos = pos - pos.mean(axis=0)
            coords = pos.ravel().astype(np.float32)
            n_confs += 1
            total += len(coords)
            coord_sum += float(coords.sum())
            lmin, lmax = float(coords.min()), float(coords.max())
            if lmin < coord_min:
                coord_min = lmin
            if lmax > coord_max:
                coord_max = lmax
            buf.append(coords)
            buf_size += len(coords)
            if buf_size >= _BATCH_COORDS:
                _flush()

    _flush()
    return {
        "hist": hist,
        "total": total,
        "coord_sum": coord_sum,
        "coord_min": coord_min,
        "coord_max": coord_max,
        "n_confs": n_confs,
        "n_skip": n_skip,
    }


def process_shard(
    shard_path: str,
    out_dir: str,
    hist_bins: int,
    hist_range: tuple[float, float],
    num_workers: int = 1,
) -> None:
    global _SHARD_GLOBAL, _HIST_EDGES_GLOBAL

    shard_name = os.path.splitext(os.path.basename(shard_path))[0]
    partial_dir = os.path.join(out_dir, "partial_hist")
    os.makedirs(partial_dir, exist_ok=True)
    out_path = os.path.join(partial_dir, f"{shard_name}.npz")

    if os.path.exists(out_path):
        print(f"Partial result already exists, skipping: {out_path}", flush=True)
        return

    print(f"Loading {shard_path}  ({os.path.getsize(shard_path)/1e9:.1f} GB) …", flush=True)
    t0 = time.time()
    with open(shard_path, "rb") as fh:
        shard: dict = pickle.load(fh)
    n_unique = len(shard)
    print(f"Loaded in {time.time()-t0:.1f}s — {n_unique:,} unique SMILES", flush=True)

    # Make available to forked workers
    _SHARD_GLOBAL = shard
    _HIST_EDGES_GLOBAL = np.linspace(hist_range[0], hist_range[1], hist_bins + 1)

    keys = list(shard.keys())
    chunks = np.array_split(keys, max(1, num_workers))
    chunks = [list(c) for c in chunks if len(c)]

    print(f"Processing {n_unique:,} SMILES with {len(chunks)} workers …", flush=True)
    t1 = time.time()

    if len(chunks) == 1:
        results = [_worker_chunk(chunks[0])]
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=len(chunks)) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_worker_chunk, chunks),
                    total=len(chunks),
                    desc=shard_name,
                    unit="chunk",
                    dynamic_ncols=True,
                )
            )

    _SHARD_GLOBAL = None  # release memory

    # Merge worker results
    hist = np.zeros(hist_bins, dtype=np.int64)
    total = 0
    coord_sum = 0.0
    coord_min = np.inf
    coord_max = -np.inf
    n_confs = n_skip = 0
    for r in results:
        hist += r["hist"]
        total += r["total"]
        coord_sum += r["coord_sum"]
        if r["coord_min"] < coord_min:
            coord_min = r["coord_min"]
        if r["coord_max"] > coord_max:
            coord_max = r["coord_max"]
        n_confs += r["n_confs"]
        n_skip += r["n_skip"]

    elapsed = time.time() - t0
    rate = n_confs / max(time.time() - t1, 1e-9)
    print(
        f"{shard_name}: {n_unique:,} mols  {n_confs:,} confs  "
        f"{total:,} coords  skip={n_skip}  {elapsed:.1f}s  {rate:,.0f} confs/s",
        flush=True,
    )

    np.savez_compressed(
        out_path,
        hist=hist,
        hist_range=np.array(hist_range),
        total=np.array(total, dtype=np.int64),
        coord_sum=np.array(coord_sum),
        coord_min=np.array(coord_min),
        coord_max=np.array(coord_max),
        n_mols=np.array(n_unique, dtype=np.int64),
        n_confs=np.array(n_confs, dtype=np.int64),
        n_skip=np.array(n_skip, dtype=np.int64),
    )
    print(f"Saved → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Phase 2: merge partial .npz files → stats + BinConfig
# ---------------------------------------------------------------------------

def merge(
    out_dir: str,
    n_bins: int = 256,
    q_low: float = 0.0001,
    q_high: float = 0.999,
) -> None:
    partial_dir = os.path.join(out_dir, "partial_hist")
    npz_paths = sorted(glob.glob(os.path.join(partial_dir, "shard_*.npz")))
    if not npz_paths:
        raise FileNotFoundError(f"No partial histograms found in {partial_dir}")

    print(f"Merging {len(npz_paths)} partial histograms …", flush=True)

    global_hist = None
    global_edges = None
    global_total = 0
    global_sum = 0.0
    global_min = np.inf
    global_max = -np.inf
    global_mols = global_confs = global_skip = 0

    for p in npz_paths:
        d = np.load(p)
        if global_hist is None:
            global_hist = d["hist"].astype(np.int64)
            hist_range = tuple(d["hist_range"].tolist())
            n_hist_bins = len(global_hist)
            global_edges = np.linspace(hist_range[0], hist_range[1], n_hist_bins + 1)
        else:
            global_hist += d["hist"].astype(np.int64)
        global_total += int(d["total"])
        global_sum += float(d["coord_sum"])
        v = float(d["coord_min"])
        if v < global_min:
            global_min = v
        v = float(d["coord_max"])
        if v > global_max:
            global_max = v
        global_mols += int(d["n_mols"])
        global_confs += int(d["n_confs"])
        global_skip += int(d["n_skip"])
        print(f"  {os.path.basename(p)}: {int(d['n_confs']):>12,} confs  {int(d['total']):>15,} coords")

    if global_total == 0:
        raise RuntimeError("No coordinates collected across all partial histograms.")

    n_out_of_range = global_total - int(global_hist.sum())
    if n_out_of_range > 0:
        pct = 100.0 * n_out_of_range / global_total
        print(f"WARNING: {n_out_of_range:,} coords ({pct:.4f}%) outside histogram range.")

    # CDF
    cum = np.empty(len(global_edges), dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(global_hist, out=cum[1:])
    cum_norm = cum / cum[-1]

    percentiles_q  = [0.00001, q_low,  0.001,  0.01,  0.05,  0.25,  0.50,
                      0.75,    0.95,   0.99,   q_high, 0.9999]
    percentile_lbl = ["p0.001","p0.01","p0.1","p1","p5","p25","p50",
                      "p75",   "p95",  "p99",  "p99.9","p99.99"]
    pct_values = {lbl: float(_histogram_quantile(cum_norm, global_edges, q))
                  for lbl, q in zip(percentile_lbl, percentiles_q)}

    stats = {
        "n_shards": len(npz_paths),
        "n_molecules": global_mols,
        "n_conformers": global_confs,
        "n_conformers_skipped": global_skip,
        "n_coord_scalars": global_total,
        "centered": True,
        "global_min": float(global_min),
        "global_max": float(global_max),
        "mean": float(global_sum / global_total),
        "q_low_used": q_low,
        "q_high_used": q_high,
        "percentiles": pct_values,
        "histogram": {
            "n_bins": n_hist_bins,
            "range": list(hist_range),
            "n_out_of_range": n_out_of_range,
        },
    }

    stats_path = os.path.join(out_dir, "bigdata_coord_stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"\n{'='*60}")
    print(f"  Coordinate statistics  (centered, pooled xyz)")
    print(f"{'='*60}")
    print(f"  molecules  : {global_mols:>15,}")
    print(f"  conformers : {global_confs:>15,}")
    print(f"  coord vals : {global_total:>15,}")
    print(f"  min        : {global_min:>+15.4f} Å")
    print(f"  max        : {global_max:>+15.4f} Å")
    print(f"  mean       : {stats['mean']:>+15.6f} Å")
    for lbl, val in pct_values.items():
        marker = " ◄" if lbl in ("p0.01", "p99.9") else ""
        print(f"  {lbl:<9}: {val:>+15.4f} Å{marker}")
    print(f"{'='*60}")
    print(f"Stats saved → {stats_path}")

    # BinConfig
    cfg = _fit_quantile_bins_from_histogram(
        global_hist, global_edges, n_bins=n_bins, q_low=q_low, q_high=q_high,
    )
    bin_path = os.path.join(out_dir, f"bigdata_quantile_bins_{n_bins}.json")
    cfg.save(bin_path)

    print(f"\n{'='*60}")
    print(f"  Quantile BinConfig  ({n_bins} bins)")
    print(f"{'='*60}")
    print(f"  L (q={q_low})  : {cfg.L:>+15.4f} Å")
    print(f"  H (q={q_high}) : {cfg.H:>+15.4f} Å")
    print(f"  median bin w   : {np.median(np.diff(cfg.edges)):>15.6f} Å")
    print(f"  digit_width    : {cfg.digit_width}")
    print(f"BinConfig saved → {bin_path}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory with shard_NNNN.pkl files.  Requires --shard_id.",
    )
    mode.add_argument(
        "--merge",
        action="store_true",
        help="Merge partial histograms from a previous array run.  Requires --out_dir.",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=None,
        help="Index of shard to process (matches shard_NNNN.pkl).  Required with --input_dir.",
    )
    parser.add_argument("--out_dir", type=str, default=".",
                        help="Directory for partial histograms and final outputs.")
    parser.add_argument("--n_bins",  type=int,   default=256)
    parser.add_argument("--q_low",   type=float, default=0.0001)
    parser.add_argument("--q_high",  type=float, default=0.999)
    parser.add_argument("--hist_bins",  type=int,   default=1_000_000)
    parser.add_argument("--hist_range", type=float, nargs=2, default=(-50.0, 50.0),
                        metavar=("LO", "HI"))
    parser.add_argument(
        "--workers", "-nw",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel workers for coordinate extraction within a shard (default: cpu_count-1).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.merge:
        merge(out_dir=args.out_dir, n_bins=args.n_bins, q_low=args.q_low, q_high=args.q_high)
    else:
        if args.shard_id is None:
            raise SystemExit("--input_dir requires --shard_id.  Use the SLURM array script.")
        shard_paths = sorted(glob.glob(os.path.join(args.input_dir, "shard_*.pkl")))
        target = f"shard_{args.shard_id:04d}.pkl"
        matched = [p for p in shard_paths if os.path.basename(p) == target]
        if not matched:
            raise SystemExit(f"{target} not found in {args.input_dir}")
        process_shard(
            shard_path=matched[0],
            out_dir=args.out_dir,
            hist_bins=args.hist_bins,
            hist_range=tuple(args.hist_range),
            num_workers=args.workers,
        )
