#!/usr/bin/env python3
"""
External global shuffle for large JSONL datasets.

Algorithm (two-pass external shuffle):
  Pass 1 (parallel): each worker reads a subset of input files and routes
          every line to one of NUM_BUCKETS temp files using a random float key.
          Workers write to separate files (bucket_{b}_w{w}.txt) to avoid
          concurrent-write conflicts.
  Pass 2 (serial):   for each bucket, merge all worker files, sort by key,
          and stream lines to the output shards.

Peak memory ≈ total_size / num_buckets (e.g. 80 GB / 200 = ~400 MB).

Usage:
    python shuffle_jsonl_dataset.py INPUT_DIR OUTPUT_DIR [options]

    --num-buckets INT      Temp bucket count (default 200). Increase if RAM is tight.
    --num-output-shards INT  Output file count (default = number of input files).
    --workers INT          Parallel workers for Pass 1 (default: cpu_count).
    --bucket-dir PATH      Where to store temp bucket files (default: OUTPUT_DIR/_buckets).
    --seed INT             RNG seed (default 42).
    --skip-pass1           Resume: skip Pass 1 and use existing bucket files.
    --cleanup              Delete bucket dir after Pass 2.
"""

import os
import sys
import glob
import random
import logging
import argparse
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_WRITE_BUF = 1 << 23  # 8 MB write buffer per bucket handle


# ---------------------------------------------------------------------------
# Pass 1
# ---------------------------------------------------------------------------

def _pass1_worker(args):
    # indexed_file_list: list of (global_file_index, path)
    # Each file gets its own RNG seeded by (seed + global_file_index) so the
    # output is identical regardless of how many workers are used.
    worker_id, indexed_file_list, bucket_dir, num_buckets, seed = args

    handles = [
        open(os.path.join(bucket_dir, f"bucket_{b:04d}_w{worker_id:03d}.txt"), "w", buffering=_WRITE_BUF)
        for b in range(num_buckets)
    ]
    try:
        for file_idx, path in indexed_file_list:
            rng = random.Random(seed + file_idx)
            with open(path, buffering=1 << 20) as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    key = rng.random()
                    b = min(int(key * num_buckets), num_buckets - 1)
                    handles[b].write(f"{key:.15f}\t{line}\n")
    finally:
        for h in handles:
            h.close()
    return worker_id, len(indexed_file_list)


def run_pass1(input_files, bucket_dir, num_buckets, num_workers, seed):
    os.makedirs(bucket_dir, exist_ok=True)
    indexed_files = list(enumerate(input_files))
    chunks = [indexed_files[i::num_workers] for i in range(num_workers)]
    chunks = [c for c in chunks if c]  # drop empty chunks

    log.info(f"Pass 1: {len(input_files)} files → {num_buckets} buckets × {len(chunks)} workers")

    worker_args = [
        (w, chunks[w], bucket_dir, num_buckets, seed)
        for w in range(len(chunks))
    ]
    with Pool(processes=len(chunks)) as pool:
        for worker_id, n_files in pool.imap_unordered(_pass1_worker, worker_args):
            log.info(f"  Worker {worker_id} done ({n_files} files)")

    log.info("Pass 1 complete.")


# ---------------------------------------------------------------------------
# Pass 2
# ---------------------------------------------------------------------------

def run_pass2(bucket_dir, output_dir, num_output_shards, num_buckets, num_workers):
    os.makedirs(output_dir, exist_ok=True)

    # Collect bucket files grouped by bucket index
    all_bucket_files = sorted(glob.glob(os.path.join(bucket_dir, "bucket_*.txt")))
    if not all_bucket_files:
        log.error(f"No bucket files found in {bucket_dir}. Did Pass 1 run?")
        sys.exit(1)

    bucket_groups = {}
    for path in all_bucket_files:
        b = int(os.path.basename(path).split("_")[1])
        bucket_groups.setdefault(b, []).append(path)

    # Count total lines (fast: byte-level scan)
    log.info("Counting total lines...")
    total_lines = 0
    for path in all_bucket_files:
        with open(path, "rb") as f:
            total_lines += sum(1 for _ in f)
    log.info(f"Total lines: {total_lines:,}")

    lines_per_shard = (total_lines + num_output_shards - 1) // num_output_shards
    log.info(f"Output: {num_output_shards} shards × ~{lines_per_shard:,} lines")

    shard_idx = 0
    shard_count = 0
    out = open(
        os.path.join(output_dir, f"shuffled_{shard_idx:04d}.jsonl"),
        "w", buffering=1 << 22,
    )

    def roll_shard():
        nonlocal shard_idx, shard_count, out
        out.close()
        shard_idx += 1
        shard_count = 0
        out = open(
            os.path.join(output_dir, f"shuffled_{shard_idx:04d}.jsonl"),
            "w", buffering=1 << 22,
        )

    for b in sorted(bucket_groups):
        log.info(f"  Bucket {b}/{num_buckets - 1}: merging {len(bucket_groups[b])} worker files...")
        items = []
        for path in bucket_groups[b]:
            with open(path, buffering=1 << 20) as f:
                for raw in f:
                    raw = raw.rstrip("\n")
                    if not raw:
                        continue
                    tab = raw.index("\t")
                    items.append((float(raw[:tab]), raw[tab + 1:]))

        items.sort()

        for _, content in items:
            out.write(content + "\n")
            shard_count += 1
            if shard_count >= lines_per_shard and shard_idx < num_output_shards - 1:
                roll_shard()

        del items  # release memory before next bucket

    out.close()
    log.info(f"Pass 2 complete. Wrote {shard_idx + 1} output shards.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="External global shuffle for large JSONL datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", help="Directory containing input .jsonl files")
    parser.add_argument("output_dir", help="Directory for shuffled output .jsonl files")
    parser.add_argument("--num-buckets", type=int, default=200,
                        help="Number of temp bucket files per worker. "
                             "Increase if you're OOM in Pass 2.")
    parser.add_argument("--num-output-shards", type=int, default=None,
                        help="Number of output .jsonl files (default: same as input)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers for Pass 1 (default: cpu_count)")
    parser.add_argument("--bucket-dir", default=None,
                        help="Temp directory for bucket files (default: OUTPUT_DIR/_buckets)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-pass1", action="store_true",
                        help="Skip Pass 1 and resume from existing bucket files")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete bucket dir after Pass 2 finishes")
    args = parser.parse_args()

    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.jsonl")))
    if not input_files:
        log.error(f"No .jsonl files found in: {args.input_dir}")
        sys.exit(1)
    log.info(f"Input: {len(input_files)} .jsonl files in {args.input_dir}")

    num_output_shards = args.num_output_shards or len(input_files)
    num_workers = args.workers or cpu_count()
    bucket_dir = args.bucket_dir or os.path.join(args.output_dir, "_buckets")

    if not args.skip_pass1:
        run_pass1(input_files, bucket_dir, args.num_buckets, num_workers, args.seed)

    run_pass2(bucket_dir, args.output_dir, num_output_shards, args.num_buckets, num_workers)

    if args.cleanup:
        log.info(f"Removing bucket dir: {bucket_dir}")
        shutil.rmtree(bucket_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
