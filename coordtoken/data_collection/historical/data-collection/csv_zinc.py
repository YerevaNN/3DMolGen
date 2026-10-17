#!/usr/bin/env python3
"""
FAST ZINC shards -> CSV (streaming multiprocessing; no in-RAM list; no gzip; no bucket shuffle)

Input:
  shard_dir/zinc_shard_*.pkl  (pickle stream of (mol_id, blob))

Output (parts; safe parallel write):
  out_dir/train/zinc_w000.csv
  out_dir/val/zinc_w000.csv
  out_dir/test/zinc_w000.csv
  ... one file per worker per split

What you get:
  - Very fast: each worker streams shards, encodes, writes directly (no IPC of huge strings)
  - Low RAM: no 75M list in RAM
  - "Global-ish" shuffle: deterministic hash(mol_id) makes split assignment uniformly random across shards
  - Exact val/test sizes (as long as enough valid molecules exist), using shared counters + batched locking
  - Clean progress bar: one monitor thread in parent, low contention progress updates from workers

IMPORTANT:
  This does NOT produce a single globally permuted order in one file.
  It produces a globally-random SPLIT assignment (train/val/test) without global ordering.
  If you later want one file per split, just concatenate parts.
"""

import argparse
import csv
import hashlib
import multiprocessing as mp
import os
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple

from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

from utils import encode_cartesian_v2

# -------------------- shared globals (set via Pool initializer) --------------------
G_SHARED_VAL = None
G_SHARED_TEST = None
G_SHARED_PROCESSED = None
G_LOCK = None
G_VAL_TARGET = 0
G_TEST_TARGET = 0

# progress updates: reduce contention (10k is usually smooth enough)
PROGRESS_UPDATE_INTERVAL = 10_000


def init_worker(shared_val, shared_test, shared_processed, lock, val_target, test_target):
    """Initializer runs in each worker once; sets up inherited shared state."""
    global G_SHARED_VAL, G_SHARED_TEST, G_SHARED_PROCESSED, G_LOCK, G_VAL_TARGET, G_TEST_TARGET
    G_SHARED_VAL = shared_val
    G_SHARED_TEST = shared_test
    G_SHARED_PROCESSED = shared_processed
    G_LOCK = lock
    G_VAL_TARGET = int(val_target)
    G_TEST_TARGET = int(test_target)
    RDLogger.DisableLog("rdApp.*")


# -------------------- pickle streaming --------------------
def iter_pickle_stream(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)  # (mol_id, blob)
            except EOFError:
                return


def count_records(pkl_path: Path) -> int:
    """Counts top-level pickle objects in the stream file (robust-ish)."""
    n = 0
    with open(pkl_path, "rb") as f:
        while True:
            try:
                pickle.load(f)
                n += 1
            except EOFError:
                break
            except Exception:
                # skip corrupt record; keep going
                continue
    return n


# -------------------- deterministic 64-bit hash --------------------
def hash64(s: str, seed: int) -> int:
    """Deterministic uint64 hash of mol_id; seed controls assignment."""
    h = hashlib.blake2b(
        s.encode("utf-8", "ignore"),
        digest_size=8,
        person=int(seed).to_bytes(8, "little", signed=False),
    )
    return int.from_bytes(h.digest(), "little", signed=False)


def choose_preferred_split(h: int, val_thresh: int, test_thresh: int) -> str:
    """Given uint64 hash, choose preferred split by thresholds."""
    if h < val_thresh:
        return "val"
    if h < test_thresh:
        return "test"
    return "train"


# -------------------- worker --------------------
def worker_task(task):
    """
    Each worker gets a list of shard paths; writes to its own per-split CSVs.
    """
    (
        wid,
        shard_paths,
        out_dir,
        precision,
        seed,
        val_thresh,
        test_thresh,
        batch_size,
    ) = task

    RDLogger.DisableLog("rdApp.*")

    out_dir = Path(out_dir)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    # per-worker files (no write contention)
    f_train = open(out_dir / "train" / f"zinc_w{wid:03d}.csv", "w", newline="")
    f_val = open(out_dir / "val" / f"zinc_w{wid:03d}.csv", "w", newline="")
    f_test = open(out_dir / "test" / f"zinc_w{wid:03d}.csv", "w", newline="")

    w_train = csv.writer(f_train)
    w_val = csv.writer(f_val)
    w_test = csv.writer(f_test)

    for w in (w_train, w_val, w_test):
        w.writerow(["name", "enriched_text"])

    processed = 0
    wrote_rows = 0
    skipped = 0
    last_reported = 0

    # buffer rows to reduce lock overhead
    batch_val: List[Tuple[str, str]] = []
    batch_test: List[Tuple[str, str]] = []
    batch_train: List[Tuple[str, str]] = []

    def flush_batches():
        nonlocal batch_val, batch_test, batch_train, wrote_rows
        if not (batch_val or batch_test or batch_train):
            return

        # allocate exact val/test quotas with ONE lock per batch
        with G_LOCK:
            rem_val = max(0, G_VAL_TARGET - G_SHARED_VAL.value)
            rem_test = max(0, G_TEST_TARGET - G_SHARED_TEST.value)

            take_val = min(len(batch_val), rem_val)
            G_SHARED_VAL.value += take_val
            val_out = batch_val[:take_val]
            overflow_val = batch_val[take_val:]

            take_test = min(len(batch_test), rem_test)
            G_SHARED_TEST.value += take_test
            test_out = batch_test[:take_test]
            overflow_test = batch_test[take_test:]

        # write outside lock
        for name, txt in val_out:
            w_val.writerow([name, txt])
            wrote_rows += 1
        for name, txt in test_out:
            w_test.writerow([name, txt])
            wrote_rows += 1

        # train gets: preferred train + overflows
        for name, txt in batch_train:
            w_train.writerow([name, txt])
            wrote_rows += 1
        for name, txt in overflow_val:
            w_train.writerow([name, txt])
            wrote_rows += 1
        for name, txt in overflow_test:
            w_train.writerow([name, txt])
            wrote_rows += 1

        batch_val.clear()
        batch_test.clear()
        batch_train.clear()

    try:
        for pkl_path_str in shard_paths:
            pkl_path = Path(pkl_path_str)
            try:
                for mol_id, blob in iter_pickle_stream(pkl_path):
                    processed += 1
                    try:
                        mol = Chem.Mol(blob)
                        if mol is None or mol.GetNumConformers() == 0:
                            skipped += 1
                            continue

                        # your encode_cartesian_v2 returns ONLY the text
                        txt = encode_cartesian_v2(mol, precision=precision)

                        h = hash64(mol_id, seed)
                        pref = choose_preferred_split(h, val_thresh, test_thresh)

                        if pref == "val":
                            batch_val.append((mol_id, txt))
                        elif pref == "test":
                            batch_test.append((mol_id, txt))
                        else:
                            batch_train.append((mol_id, txt))

                        if (len(batch_val) + len(batch_test) + len(batch_train)) >= batch_size:
                            flush_batches()

                    except Exception:
                        skipped += 1

                    # progress accounting outside inner try (counts skipped too)
                    if processed - last_reported >= PROGRESS_UPDATE_INTERVAL:
                        with G_SHARED_PROCESSED.get_lock():
                            G_SHARED_PROCESSED.value += processed - last_reported
                        last_reported = processed

            except Exception as e:
                print(
                    f"[w{wid:03d}] ERROR reading shard {pkl_path}: {type(e).__name__}: {e}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

        flush_batches()

        # final progress bump
        delta = processed - last_reported
        if delta:
            with G_SHARED_PROCESSED.get_lock():
                G_SHARED_PROCESSED.value += delta

    finally:
        try:
            f_train.close()
        except Exception:
            pass
        try:
            f_val.close()
        except Exception:
            pass
        try:
            f_test.close()
        except Exception:
            pass

    return wid, processed, wrote_rows, skipped


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=int, required=True)
    ap.add_argument("--test-size", type=int, required=True)
    ap.add_argument("--n-workers", type=int, default=50)
    ap.add_argument("--precision", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2000, help="rows per lock allocation batch")
    ap.add_argument("--count-only", action="store_true", help="Only count records then exit")
    ap.add_argument("--oversample", type=float, default=1.0,
                    help="Optional >1.0 to oversample val/test candidates if skip rate is high (e.g., 1.2)")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    RDLogger.DisableLog("rdApp.*")

    shard_files = sorted(args.shard_dir.glob("zinc_shard_*.pkl"))
    if not shard_files:
        raise SystemExit(f"No zinc_shard_*.pkl found in {args.shard_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "val").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "test").mkdir(parents=True, exist_ok=True)

    # PASS 0: count records (for threshold calibration)
    print(f"Counting records across {len(shard_files)} shards...", flush=True)
    n_total = 0
    for p in tqdm(shard_files, desc="COUNT shards", unit="shard", mininterval=30, dynamic_ncols=True):
        n_total += count_records(p)

    print(f"Total records (top-level pickles): {n_total:,}", flush=True)
    if args.count_only:
        return

    if args.val_size + args.test_size >= n_total:
        raise SystemExit(f"val+test ({args.val_size + args.test_size}) must be < total_records ({n_total})")

    # thresholds on uint64 range
    U64 = 1 << 64
    oversample = float(args.oversample)
    if oversample < 1.0:
        oversample = 1.0

    val_frac = min(1.0, oversample * (args.val_size / n_total))
    test_frac = min(1.0, oversample * (args.test_size / n_total))

    val_thresh = int(val_frac * U64)
    test_thresh = int((val_frac + test_frac) * U64)

    print(
        f"Targets: val={args.val_size:,} test={args.test_size:,} train≈{(n_total - args.val_size - args.test_size):,}\n"
        f"Workers: {args.n_workers}  precision: {args.precision}  batch_size={args.batch_size}\n"
        f"Threshold oversample: {oversample:.2f}\n"
        f"Writing per-worker part CSVs to {args.out_dir}/{{train,val,test}}",
        flush=True,
    )

    # shard assignment: round-robin
    per_worker: List[List[str]] = [[] for _ in range(args.n_workers)]
    for i, p in enumerate(shard_files):
        per_worker[i % args.n_workers].append(str(p))

    ctx = mp.get_context("fork")
    shared_val = ctx.Value("Q", 0)
    shared_test = ctx.Value("Q", 0)
    shared_processed = ctx.Value("Q", 0)
    lock = ctx.Lock()

    tasks = []
    for wid in range(args.n_workers):
        if not per_worker[wid]:
            continue
        tasks.append((
            wid,
            per_worker[wid],
            str(args.out_dir),
            int(args.precision),
            int(args.seed),
            val_thresh,
            test_thresh,
            int(args.batch_size),
        ))

    stop_monitor = threading.Event()

    def progress_monitor():
        with tqdm(total=n_total, desc="PROCESSING", unit="mol", mininterval=1, dynamic_ncols=True) as pbar:
            last = 0
            while not stop_monitor.is_set():
                cur = shared_processed.value
                if cur > last:
                    pbar.update(cur - last)
                    last = cur
                time.sleep(1)
            cur = shared_processed.value
            if cur > last:
                pbar.update(cur - last)

    monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
    monitor_thread.start()

    results = []
    with ctx.Pool(
        processes=min(args.n_workers, len(tasks)),
        initializer=init_worker,
        initargs=(shared_val, shared_test, shared_processed, lock, int(args.val_size), int(args.test_size)),
        maxtasksperchild=1,
    ) as pool:
        for r in pool.imap_unordered(worker_task, tasks):
            results.append(r)

    stop_monitor.set()
    monitor_thread.join(timeout=2)

    for wid, proc, wrote, skip in sorted(results):
        print(f"[w{wid:03d}] processed={proc:,} wrote={wrote:,} skipped={skip:,}", flush=True)

    print(
        f"\nDONE.\n"
        f"Filled targets (valid rows): val={shared_val.value:,} test={shared_test.value:,} train=rest\n"
        f"Part files are in: {args.out_dir}/train , {args.out_dir}/val , {args.out_dir}/test\n"
        f"Tip: if val/test are underfilled, re-run with --oversample 1.2 (or higher) to oversample candidates.",
        flush=True,
    )


if __name__ == "__main__":
    main()
