#!/usr/bin/env python3
"""
ZINC: ids.txt  ->  sharded pickle streams (mol_id, rdkit_binary_blob)

Goals:
- ~1/k UNBIASED sampling of UNIQUE IDs using block sampling with unbiased tail handling
- Good mixing: balanced shards by input gz size + shuffled file order per shard
- Multiprocessing: 1 shard = 1 output file (safe, no write contention)
- Efficient membership: store chosen IDs as stable 64-bit hashes (huge RAM win with many workers)
- Optional within-shard shuffle using a bounded buffer (streaming-friendly)
"""

import argparse
import gzip
import hashlib
import pickle
import random
import tempfile
import time
from pathlib import Path

import multiprocessing as mp
from rdkit import Chem
from tqdm import tqdm


def hash64(s: str) -> int:
    # Stable 64-bit hash (blake2b). Collision probability is negligible for practical sizes.
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def choose_ids_unbiased_blocks(ids: list[str], k: int, rng: random.Random) -> set[int]:
    # unbiased sampling of unique IDs using block sampling with unbiased tail handling
    chosen: set[int] = set()
    n = len(ids)
    full = (n // k) * k

    for i in range(0, full, k):
        chosen.add(hash64(ids[i + rng.randrange(k)]))

    m = n - full
    if m > 0 and rng.random() < (m / k):
        chosen.add(hash64(ids[full + rng.randrange(m)]))

    return chosen


def balanced_partition_by_size(gz_files: list[Path], n_shards: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    items = []
    for p in gz_files:
        try:
            sz = p.stat().st_size
        except OSError:
            sz = 0
        items.append((sz, str(p)))

    rng.shuffle(items)
    items.sort(key=lambda t: t[0], reverse=True)

    shards = [[] for _ in range(n_shards)]
    load = [0] * n_shards
    for sz, fp in items:
        i = min(range(n_shards), key=lambda j: load[j])
        shards[i].append(fp)
        load[i] += sz

    return shards


# ---------------- worker globals ----------------
CHOSEN_HASHES = None
SANITIZE = None
REMOVE_HS = None
SHUFFLE_BUFFER = 0
BASE_SEED = 0
PROG_Q = None
PROG_EVERY = 1  # how many files to batch before sending progress


def init_worker(chosen_hashes, sanitize, remove_hs, shuffle_buffer, seed, prog_q, prog_every: int):
    global CHOSEN_HASHES, SANITIZE, REMOVE_HS, SHUFFLE_BUFFER, BASE_SEED, PROG_Q, PROG_EVERY
    CHOSEN_HASHES = chosen_hashes
    SANITIZE = sanitize
    REMOVE_HS = remove_hs
    SHUFFLE_BUFFER = shuffle_buffer
    BASE_SEED = seed
    PROG_Q = prog_q
    PROG_EVERY = max(1, int(prog_every))


def iter_mols(gz_path: Path):
    # Stream first; if gzip is truncated, do NOT fallback (saves time)
    try:
        with gzip.open(gz_path, "rb") as fb:
            sup = Chem.ForwardSDMolSupplier(fb, sanitize=SANITIZE, removeHs=REMOVE_HS)
            for mol in sup:
                yield mol
        return
    except EOFError:
        raise
    except Exception:
        pass

    # Fallback decompress (for weird RDKit streaming issues)
    with tempfile.TemporaryDirectory() as td:
        sdf = Path(td) / gz_path.with_suffix("").name
        with gzip.open(gz_path, "rb") as fin, open(sdf, "wb") as fout:
            while True:
                chunk = fin.read(1024 * 1024)
                if not chunk:
                    break
                fout.write(chunk)
        sup = Chem.SDMolSupplier(str(sdf), sanitize=SANITIZE, removeHs=REMOVE_HS)
        for mol in sup:
            yield mol


def flush_buffer(buf, out_f, rng: random.Random) -> int:
    if not buf:
        return 0
    rng.shuffle(buf)
    for rec in buf:
        pickle.dump(rec, out_f, protocol=pickle.HIGHEST_PROTOCOL)
    n = len(buf)
    buf.clear()
    return n


def process_one_shard(task):
    shard_idx, files, out_path = task
    rng = random.Random(BASE_SEED + shard_idx * 1000003)

    files = list(files)
    rng.shuffle(files)

    written = 0
    corrupted = []
    buf = []

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # local batching for progress updates (reduces queue traffic)
    batch_files = 0
    batch_kept = 0

    with open(out_p, "wb") as out:
        for gz_str in files:
            gz = Path(gz_str)
            kept_in_file = 0
            try:
                for mol in iter_mols(gz):
                    if mol is None or not mol.HasProp("_Name"):
                        continue
                    mol_id = mol.GetProp("_Name").strip()
                    if not mol_id:
                        continue
                    if hash64(mol_id) not in CHOSEN_HASHES:
                        continue

                    rec = (mol_id, mol.ToBinary())
                    kept_in_file += 1

                    if SHUFFLE_BUFFER > 0:
                        buf.append(rec)
                        if len(buf) >= SHUFFLE_BUFFER:
                            written += flush_buffer(buf, out, rng)
                    else:
                        pickle.dump(rec, out, protocol=pickle.HIGHEST_PROTOCOL)
                        written += 1

            except Exception as e:
                corrupted.append(f"{gz.name}: {type(e).__name__}: {e}")

            # progress (1 update per file, but batched before sending)
            batch_files += 1
            batch_kept += kept_in_file
            if PROG_Q is not None and batch_files >= PROG_EVERY:
                PROG_Q.put((batch_files, batch_kept))
                batch_files = 0
                batch_kept = 0

        if SHUFFLE_BUFFER > 0 and buf:
            written += flush_buffer(buf, out, rng)

    # flush remaining progress batch
    if PROG_Q is not None and batch_files:
        PROG_Q.put((batch_files, batch_kept))

    return shard_idx, written, corrupted


def load_ids(ids_file: Path) -> list[str]:
    return [ln.strip() for ln in ids_file.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf-dir", type=Path, required=True)
    ap.add_argument("--ids-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sanitize", action="store_true", default=False)

    ap.add_argument("--remove-hs", action="store_true", default=True)
    ap.add_argument("--keep-hs", dest="remove_hs", action="store_false")

    ap.add_argument("--n-shards", type=int, default=50)
    ap.add_argument("--n-workers", type=int, default=50)
    ap.add_argument("--shuffle-buffer", type=int, default=0)

    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--progress-every", type=int, default=5,
                    help="Batch N files before sending a progress update (reduces overhead).")
    ap.add_argument("--tqdm-mininterval", type=float, default=30.0)
    args = ap.parse_args()

    gz_files = sorted(args.sdf_dir.glob("*.sdf.gz"))
    if not gz_files:
        raise SystemExit(f"No *.sdf.gz found in {args.sdf_dir}")

    ids = load_ids(args.ids_file)
    if not ids:
        raise SystemExit("ids-file is empty")

    rng = random.Random(args.seed)
    chosen_hashes = choose_ids_unbiased_blocks(ids, args.k, rng)

    print(f"Total gz files:     {len(gz_files):,}")
    print(f"IDs loaded:         {len(ids):,}")
    print(f"Chosen (expected):  ~{len(ids)/args.k:,.0f}")
    print(f"Chosen (actual):    {len(chosen_hashes):,}")
    print(f"sanitize={args.sanitize}  remove_hs={args.remove_hs}")
    print(f"shards={args.n_shards}  workers={args.n_workers}  shuffle_buffer={args.shuffle_buffer}")
    print(f"progress_every={args.progress_every}  tqdm_mininterval={args.tqdm_mininterval}")

    shards = balanced_partition_by_size(gz_files, args.n_shards, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    corrupted_log = args.out_dir / "corrupted_extract.txt"
    corrupted_log.unlink(missing_ok=True)

    tasks = []
    for si, files in enumerate(shards):
        out_path = args.out_dir / f"zinc_shard_{si:05d}.pkl"
        tasks.append((si, files, str(out_path)))

    ctx = mp.get_context("fork")  # good on Linux; avoids big pickling overhead
    mgr = ctx.Manager()
    prog_q = mgr.Queue()

    tmp_ctx = tempfile.TemporaryDirectory(dir=str(args.work_dir) if args.work_dir else None)
    with tmp_ctx:
        total_written = 0
        total_kept_signal = 0
        total_files_done = 0

        pbar = tqdm(total=len(gz_files), desc="Files processed", mininterval=args.tqdm_mininterval)

        with ctx.Pool(
            processes=args.n_workers,
            initializer=init_worker,
            initargs=(chosen_hashes, args.sanitize, args.remove_hs, args.shuffle_buffer, args.seed, prog_q, args.progress_every),
            maxtasksperchild=1,
        ) as pool:
            async_results = [pool.apply_async(process_one_shard, (task,)) for task in tasks]

            # Drain progress while shards run
            finished = 0
            while finished < len(async_results):
                # drain queue quickly
                drained = 0
                while True:
                    try:
                        df, dk = prog_q.get_nowait()
                    except Exception:
                        break
                    total_files_done += df
                    total_kept_signal += dk
                    drained += 1
                    pbar.update(df)
                if drained:
                    pbar.set_postfix(kept=total_kept_signal)

                finished = sum(r.ready() for r in async_results)
                time.sleep(1)

            # ensure queue fully drained at end
            while True:
                try:
                    df, dk = prog_q.get_nowait()
                except Exception:
                    break
                total_files_done += df
                total_kept_signal += dk
                pbar.update(df)
            pbar.set_postfix(kept=total_kept_signal)
            pbar.close()

            # gather results
            for r in async_results:
                shard_idx, written, corrupted = r.get()
                total_written += written
                if corrupted:
                    with corrupted_log.open("a", encoding="utf-8") as log:
                        for line in corrupted:
                            log.write(f"[shard {shard_idx:05d}] {line}\n")

    print(f"\nDone. Total written: {total_written:,} records")
    print(f"Shards at: {args.out_dir}")
    if corrupted_log.exists():
        lines = [ln for ln in corrupted_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines:
            print(f"Failures logged to {corrupted_log} ({len(lines)} lines)")
        else:
            corrupted_log.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
