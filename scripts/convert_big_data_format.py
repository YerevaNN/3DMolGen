"""Convert enriched conformer strings from merged_train.csv into grouped pickle shards.

Output format: {canonical_smiles: [Mol, Mol, ...]} per shard file.

Two-phase pipeline
------------------
Phase 1 — Stream decode (parallel, low memory):
  * A ``multiprocessing.Pool`` decodes CSV rows and returns ``(smiles, mol_binary)``
    pairs (RDKit ``mol.ToBinary()`` — compact, no Mol reconstruction in main process).
  * The main process hashes each SMILES to a shard bucket and appends the raw bytes
    to one of N shard stream files.  No Mol objects ever live in the main process
    during this phase → memory is bounded by the worker IPC buffer, not the dataset.

Phase 2 — Merge shards (sequential or parallel, bounded memory):
  * Each shard stream file is processed independently: read, group by SMILES,
    reconstruct ``Chem.Mol`` objects, write ``{smiles: [Mol, ...]}`` pkl.
  * Only ``ceil(total_conformers / num_shards)`` Mol objects are in RAM at once.
  * ``--merge-workers M`` runs M shards simultaneously; tune so
    ``M × (total_mols/num_shards × mol_size)`` fits in available RAM.

Memory guide (200 M conformers, ~5 KB/mol)
  num_shards=16, merge-workers=1  →  ~50 GB peak  (64 GB node)
  num_shards=8,  merge-workers=2  →  ~100 GB peak (128 GB node)
  num_shards=4,  merge-workers=4  →  ~200 GB peak (256 GB node)

Output layout (when --num-shards > 1)
  {output_dir}/
    shard_0000.pkl  …  shard_NNNN.pkl   ← final grouped pkls
    metadata.json
    failures.json
    .streams/                            ← temp; deleted after phase 2 by default
      stream_0000.bin  …  stream_NNNN.bin
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

from rdkit import Chem
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molgen3D.data_processing.smiles_encoder_decoder import decode_cartesian_v2, strip_smiles


DEFAULT_INPUT = Path("/mnt/weka/vtarasov/3DBigData/merged_train.csv")
DEFAULT_OUTPUT = Path("/mnt/weka/vtarasov/3DBigData_formatted/grouped_conformers.pkl")
DEFAULT_LOG_EVERY = 50_000
DEFAULT_CHUNK_SIZE = 500
DEFAULT_NUM_SHARDS = 16
_IO_BUFFER = 32 * 1024 * 1024  # 32 MB read buffer


# ---------------------------------------------------------------------------
# Worker — runs in subprocess, no shared state
# ---------------------------------------------------------------------------

def _canonicalize_smiles(decoded_mol: Chem.Mol, enriched_text: str) -> str:
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(decoded_mol), canonical=True, isomericSmiles=True)
    except Exception:  # noqa: BLE001
        stripped = strip_smiles(enriched_text)
        parsed = Chem.MolFromSmiles(stripped)
        if parsed is None:
            raise ValueError(f"Failed to canonicalize: {enriched_text[:200]}")
        return Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=True)


def _process_chunk(
    chunk: list[tuple[int, str, str]],
) -> tuple[list[tuple[str, bytes]], list[dict]]:
    """Decode and canonicalize (row_idx, enriched_text, name) tuples.

    Returns (successes, failures) where successes = [(canonical_smiles, mol_binary)].
    mol_binary is compact RDKit binary; no Mol objects cross process boundaries.
    """
    successes: list[tuple[str, bytes]] = []
    failures: list[dict] = []
    for row_idx, enriched_text, name in chunk:
        try:
            mol = decode_cartesian_v2(enriched_text)
            smiles = _canonicalize_smiles(mol, enriched_text)
            successes.append((smiles, mol.ToBinary()))
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "line": str(row_idx),
                    "name": name,
                    "error": repr(exc),
                    "text_prefix": enriched_text[:200],
                }
            )
    return successes, failures


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def _iter_chunks(
    path: Path,
    start_line: int,
    max_lines: int,
    chunk_size: int,
):
    """Yield chunks of (row_idx, enriched_text, name) from a large CSV.

    Uses a 32 MB I/O buffer and ``csv.reader`` (avoids per-row dict allocation).
    """
    raw_fh = io.open(path, "rb", buffering=_IO_BUFFER)
    text_fh = io.TextIOWrapper(raw_fh, encoding="utf-8", newline="")
    try:
        reader = csv.reader(text_fh)
        header = next(reader)
        try:
            enriched_idx = header.index("enriched_text")
        except ValueError:
            raise ValueError(
                f"'enriched_text' column not found in CSV. "
                f"Available columns: {header[:20]}"
            )
        name_idx = header.index("name") if "name" in header else None

        chunk: list[tuple[int, str, str]] = []
        total_fed = 0

        for row_idx, row in enumerate(reader, start=1):
            if row_idx <= start_line:
                continue
            if max_lines > 0 and total_fed >= max_lines:
                break

            enriched_text = row[enriched_idx] if enriched_idx < len(row) else ""
            name = (
                row[name_idx]
                if (name_idx is not None and name_idx < len(row))
                else ""
            )
            chunk.append((row_idx, enriched_text, name))
            total_fed += 1

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        if chunk:
            yield chunk
    finally:
        text_fh.close()
        raw_fh.close()


# ---------------------------------------------------------------------------
# Phase 1 — stream decode → shard stream files
# ---------------------------------------------------------------------------

def phase1_stream(
    args: argparse.Namespace,
    num_workers: int,
    num_shards: int,
    streams_dir: Path,
) -> tuple[int, int, int, list[dict]]:
    """Decode CSV in parallel; write (smiles, mol_binary) records to shard stream files.

    Returns (total_processed, conformer_count, total_failures, failures).
    Stream files: streams_dir/stream_{i:04d}.bin — sequential pickle dumps.
    """
    streams_dir.mkdir(parents=True, exist_ok=True)

    stream_paths = [streams_dir / f"stream_{i:04d}.bin" for i in range(num_shards)]
    stream_fhs = [open(p, "wb") for p in stream_paths]
    stream_pklrs = [
        pickle.Pickler(fh, protocol=pickle.HIGHEST_PROTOCOL) for fh in stream_fhs
    ]

    failures: list[dict] = []
    conformer_count = 0
    total_processed = 0
    total_failures = 0
    run_start = time.time()

    progress_total = args.max_lines if args.max_lines > 0 else None

    try:
        with mp.Pool(
            processes=num_workers,
            maxtasksperchild=500,
        ) as pool, tqdm(
            total=progress_total,
            desc="Phase 1 – decode",
            unit="rows",
            dynamic_ncols=True,
            mininterval=2.0,
            file=sys.stdout,
        ) as pbar:

            for successes, chunk_failures in pool.imap_unordered(
                _process_chunk,
                _iter_chunks(
                    args.input,
                    start_line=args.start_line,
                    max_lines=args.max_lines,
                    chunk_size=args.chunk_size,
                ),
                chunksize=1,
            ):
                for smiles, mol_binary in successes:
                    shard_id = hash(smiles) % num_shards
                    stream_pklrs[shard_id].dump((smiles, mol_binary))
                    conformer_count += 1

                batch_size = len(successes) + len(chunk_failures)
                total_processed += batch_size
                total_failures += len(chunk_failures)
                failures.extend(chunk_failures)
                pbar.update(batch_size)

                if total_processed % args.log_every < args.chunk_size:
                    elapsed = time.time() - run_start
                    rate = total_processed / max(elapsed, 1e-9)
                    stats = (
                        f"decoded={total_processed:,} "
                        f"conformers={conformer_count:,} "
                        f"failures={total_failures:,} "
                        f"rate={rate:,.0f} rows/s"
                    )
                    pbar.set_postfix_str(stats, refresh=False)
                    tqdm.write(stats, file=sys.stdout)
    finally:
        for fh in stream_fhs:
            fh.close()

    elapsed = time.time() - run_start
    print(
        f"Phase 1 done: {total_processed:,} rows, {conformer_count:,} conformers "
        f"in {elapsed:.1f}s ({total_processed / max(elapsed, 1e-9):,.0f} rows/s)"
    )
    return total_processed, conformer_count, total_failures, failures


# ---------------------------------------------------------------------------
# Phase 2 — merge shard stream files → grouped pkl per shard
# ---------------------------------------------------------------------------

def _merge_one_shard(args_tuple: tuple) -> tuple[int, int, int]:
    """Merge one shard stream file into a grouped {smiles: [Mol, ...]} pkl.

    Runs in a subprocess so memory is released after each shard.
    Returns (shard_id, unique_smiles, conformer_count).
    """
    shard_id, stream_path, output_path, sort_keys = args_tuple

    grouped: DefaultDict[str, list[Chem.Mol]] = defaultdict(list)
    conformer_count = 0

    with open(stream_path, "rb") as fh:
        unpkl = pickle.Unpickler(fh)
        while True:
            try:
                smiles, mol_binary = unpkl.load()
                grouped[smiles].append(Chem.Mol(mol_binary))
                conformer_count += 1
            except EOFError:
                break

    out = dict(sorted(grouped.items())) if sort_keys else dict(grouped)
    with open(output_path, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return shard_id, len(out), conformer_count


def phase2_merge(
    streams_dir: Path,
    output_dir: Path,
    num_shards: int,
    merge_workers: int,
    sort_keys: bool,
) -> list[dict]:
    """Merge all shard stream files into final grouped pkls.

    Processes ``merge_workers`` shards in parallel; tune so their combined
    memory fits within available RAM.
    """
    tasks = [
        (
            i,
            streams_dir / f"stream_{i:04d}.bin",
            output_dir / f"shard_{i:04d}.pkl",
            sort_keys,
        )
        for i in range(num_shards)
    ]

    shard_stats = []
    run_start = time.time()

    with mp.Pool(processes=merge_workers) as pool, tqdm(
        total=num_shards,
        desc="Phase 2 – merge",
        unit="shards",
        dynamic_ncols=True,
        mininterval=2.0,
        file=sys.stdout,
    ) as pbar:
        for shard_id, unique_smiles, confs in pool.imap_unordered(
            _merge_one_shard, tasks, chunksize=1
        ):
            shard_stats.append(
                {"shard": shard_id, "unique_smiles": unique_smiles, "conformers": confs}
            )
            pbar.update(1)
            tqdm.write(
                f"  shard {shard_id:04d}: {unique_smiles:,} smiles, {confs:,} conformers",
                file=sys.stdout,
            )

    elapsed = time.time() - run_start
    total_smiles = sum(s["unique_smiles"] for s in shard_stats)
    total_confs = sum(s["conformers"] for s in shard_stats)
    print(
        f"Phase 2 done: {num_shards} shards, {total_smiles:,} unique SMILES, "
        f"{total_confs:,} conformers in {elapsed:.1f}s"
    )
    return shard_stats


# ---------------------------------------------------------------------------
# Single-file path (--num-shards 1 or 0) — original behaviour
# ---------------------------------------------------------------------------

def run_single(args: argparse.Namespace, num_workers: int) -> None:
    """Original single-file accumulation for smaller datasets."""
    args.output.parent.mkdir(parents=True, exist_ok=True)

    grouped_conformers: DefaultDict[str, list[Chem.Mol]] = defaultdict(list)
    failures: list[dict] = []
    conformer_count = 0
    total_processed = 0
    total_failures = 0
    run_start = time.time()

    progress_total = args.max_lines if args.max_lines > 0 else None

    with mp.Pool(processes=num_workers, maxtasksperchild=500) as pool, tqdm(
        total=progress_total,
        desc="Converting",
        unit="rows",
        dynamic_ncols=True,
        mininterval=2.0,
        file=sys.stdout,
    ) as pbar:
        for successes, chunk_failures in pool.imap_unordered(
            _process_chunk,
            _iter_chunks(
                args.input,
                start_line=args.start_line,
                max_lines=args.max_lines,
                chunk_size=args.chunk_size,
            ),
            chunksize=1,
        ):
            for smiles, mol_binary in successes:
                grouped_conformers[smiles].append(Chem.Mol(mol_binary))
                conformer_count += 1

            batch_size = len(successes) + len(chunk_failures)
            total_processed += batch_size
            total_failures += len(chunk_failures)
            failures.extend(chunk_failures)
            pbar.update(batch_size)

            if total_processed % args.log_every < args.chunk_size:
                elapsed = time.time() - run_start
                rate = total_processed / max(elapsed, 1e-9)
                tqdm.write(
                    f"processed={total_processed:,} conformers={conformer_count:,} "
                    f"unique_smiles={len(grouped_conformers):,} "
                    f"failures={total_failures:,} rate={rate:,.0f} rows/s",
                    file=sys.stdout,
                )

    elapsed = time.time() - run_start
    out = (
        dict(sorted(grouped_conformers.items())) if args.sort_output
        else dict(grouped_conformers)
    )
    with args.output.open("wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "input_csv": str(args.input),
        "output": str(args.output),
        "num_shards": 1,
        "conformers": conformer_count,
        "unique_smiles": len(grouped_conformers),
        "failures": len(failures),
        "elapsed_seconds": elapsed,
        "rows_per_second": total_processed / max(elapsed, 1e-9),
    }
    with args.output.with_suffix(".json").open("w") as fh:
        json.dump(meta, fh, indent=2)
    if failures:
        with args.output.with_name(f"{args.output.stem}_failures.json").open("w") as fh:
            json.dump(failures, fh, indent=2)

    print(
        f"saved {args.output} — "
        f"{conformer_count:,} conformers, {len(grouped_conformers):,} unique SMILES "
        f"in {elapsed:.1f}s"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert enriched conformer strings from merged_train.csv into grouped "
            "pickle shards of the form {smiles: [Mol, Mol, ...]}."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "For --num-shards 1: path to output .pkl file. "
            "For --num-shards > 1: treated as a directory "
            "(e.g. grouped_conformers.pkl → grouped_conformers/shard_NNNN.pkl)."
        ),
    )
    parser.add_argument("--max-lines", type=int, default=0, help="Max rows to process (0=all).")
    parser.add_argument("--start-line", type=int, default=0, help="Skip N data rows (resume).")
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Decode worker processes (0 = cpu_count - 1).",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=DEFAULT_NUM_SHARDS,
        help=(
            "Number of output shard pkl files. Each shard holds 1/N of the unique "
            "SMILES so peak RAM ≈ total_mol_ram / num_shards during merge. "
            "Use 1 to write a single pkl (original behaviour)."
        ),
    )
    parser.add_argument(
        "--streams-dir",
        type=Path,
        default=None,
        help=(
            "Directory for temporary shard stream files written in phase 1. "
            "Defaults to {output_dir}/.streams/. "
            "Put this on a fast local disk if available."
        ),
    )
    parser.add_argument(
        "--merge-workers",
        type=int,
        default=1,
        help=(
            "Shards to merge in parallel in phase 2. "
            "Each parallel shard needs ~(total_mols/num_shards × mol_size) RAM. "
            "Increase only if you have the headroom."
        ),
    )
    parser.add_argument(
        "--keep-streams",
        action="store_true",
        default=False,
        help="Keep temporary stream files after phase 2 (useful for debugging).",
    )
    parser.add_argument(
        "--sort-output",
        action="store_true",
        default=False,
        help="Sort each shard pkl by SMILES key (adds time, no functional benefit).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    num_workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    num_shards = max(1, args.num_shards)

    print(f"Input CSV:      {args.input}")
    max_label = f"{args.max_lines:,}" if args.max_lines > 0 else "EOF"
    print(f"Range:          skip={args.start_line:,}  max={max_label}")
    print(f"Decode workers: {num_workers}  chunk_size={args.chunk_size}")
    print(f"Num shards:     {num_shards}")

    if num_shards == 1:
        print(f"Output PKL:     {args.output}")
        print(flush=True)
        run_single(args, num_workers)
        return

    # Sharded path -------------------------------------------------------
    # Derive output directory from --output (strip .pkl suffix if present)
    if args.output.suffix.lower() == ".pkl":
        output_dir = args.output.parent / args.output.stem
    else:
        output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    streams_dir = args.streams_dir or output_dir / ".streams"

    print(f"Output dir:     {output_dir}")
    print(f"Streams dir:    {streams_dir}")
    print(f"Merge workers:  {args.merge_workers}")
    print(flush=True)

    total_start = time.time()

    # Phase 1
    total_processed, conformer_count, total_failures, failures = phase1_stream(
        args, num_workers, num_shards, streams_dir
    )

    # Phase 2
    shard_stats = phase2_merge(
        streams_dir, output_dir, num_shards, args.merge_workers, args.sort_output
    )

    # Write metadata and failures
    total_elapsed = time.time() - total_start
    total_unique = sum(s["unique_smiles"] for s in shard_stats)
    metadata = {
        "input_csv": str(args.input),
        "output_dir": str(output_dir),
        "num_shards": num_shards,
        "start_line_arg": args.start_line,
        "max_lines_arg": args.max_lines,
        "total_rows_processed": total_processed,
        "conformers": conformer_count,
        "unique_smiles": total_unique,
        "failures": total_failures,
        "elapsed_seconds": total_elapsed,
        "rows_per_second": total_processed / max(total_elapsed, 1e-9),
        "shards": sorted(shard_stats, key=lambda s: s["shard"]),
    }
    with (output_dir / "metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)
    if failures:
        with (output_dir / "failures.json").open("w") as fh:
            json.dump(failures, fh, indent=2)

    # Optionally clean up stream files
    if not args.keep_streams and streams_dir.exists():
        shutil.rmtree(streams_dir)
        print(f"Removed temp streams dir: {streams_dir}")

    print(
        f"\nAll done: {total_processed:,} rows, {conformer_count:,} conformers, "
        f"{total_unique:,} unique SMILES across {num_shards} shards "
        f"in {total_elapsed:.1f}s "
        f"({total_processed / max(total_elapsed, 1e-9):,.0f} rows/s total)"
    )


if __name__ == "__main__":
    main()
