#!/usr/bin/env python3
"""
Fast PubChem SDF.GZ sampler (single pass, single process).

Assumption for speed: conformers are grouped consecutively by CID (CID-runs).
We treat each CID-run as one "molecule". Keep every k-th molecule and save all its
conformers, BUT only if conformer 0 has any non-zero Z (3D check).

Outputs:
- pubchem_shard_XXXXX.pkl : pickle stream of records (cid_str, [blob1, blob2, ...])
- meta.json
- corrupted.txt

Counters:
- seen_molecules (CID-runs)
- chosen_molecules (selected by k/start)
- seen_conformers (records)
- saved_molecules (chosen molecules with >=1 saved conformer)
- saved_conformers
"""

import argparse
import gzip
import hashlib
import io
import json
import pickle
from pathlib import Path
from typing import Optional, List

from rdkit import Chem
from tqdm import tqdm

SEP = b"$$$$"


def hash64(b: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


def open_gz_buffered(path: Path, bufsize: int) -> io.BufferedReader:
    # IMPORTANT: use gzip.open so underlying file handle is closed (no FD leak)
    gz = gzip.open(path, "rb")
    return io.BufferedReader(gz, buffer_size=bufsize)


def read_next_nonempty_line(f: io.BufferedReader) -> Optional[bytes]:
    while True:
        line = f.readline()
        if not line:
            return None
        s = line.strip()
        if s:
            return s


def skip_to_record_end(f: io.BufferedReader) -> bool:
    while True:
        line = f.readline()
        if not line:
            return False
        if line[:1] == b"$" and line.startswith(SEP):
            return True


def read_record_bytes_after_name(f: io.BufferedReader, name_line: bytes) -> Optional[bytes]:
    buf = bytearray()
    buf += name_line
    buf += b"\n"
    while True:
        line = f.readline()
        if not line:
            return None
        if line[:1] == b"$" and line.startswith(SEP):
            return bytes(buf)
        buf += line


def has_3d_conformer_fast(mol: Chem.Mol) -> bool:
    if mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer(0)

    # Fast path: GetPositions() usually returns an Nx3 numpy array if numpy is available
    try:
        pos = conf.GetPositions()  # numpy array
        # check if any |z| > eps
        # (pos[:,2] creates a view; abs+max is fast in numpy)
        return (abs(pos[:, 2]).max() > 1e-6)
    except Exception:
        # Fallback: pure-python scan
        for i in range(mol.GetNumAtoms()):
            if abs(conf.GetAtomPosition(i).z) > 1e-6:
                return True
        return False


def molblock_to_binary_if_3d(molblock_bytes: bytes, sanitize: bool, remove_hs: bool) -> Optional[bytes]:
    block = molblock_bytes.decode("utf-8", errors="replace")
    mol = Chem.MolFromMolBlock(block, sanitize=sanitize, removeHs=remove_hs, strictParsing=False)
    if mol is None:
        return None
    if not has_3d_conformer_fast(mol):
        return None
    return mol.ToBinary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10, help="Keep every k-th molecule (CID-run).")
    ap.add_argument(
        "--start",
        type=int,
        default=None,
        help="Which position in each k-block to keep (0..k-1). Default: k-1 (keeps 10th/20th/...)",
    )
    ap.add_argument("--n-shards", type=int, default=50)
    ap.add_argument("--sanitize", action="store_true", default=False)
    ap.add_argument("--keep-hs", action="store_true", default=False)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--buf-mb", type=int, default=8, help="gzip read buffer in MB (default 8)")
    # ap.add_argument("--tqdm-mininterval", type=float, default=30.0)  # commented out per request
    args = ap.parse_args()

    if args.k <= 0:
        raise SystemExit("--k must be >= 1")
    start = args.start if args.start is not None else (args.k - 1)
    if not (0 <= start < args.k):
        raise SystemExit("--start must be in [0, k-1]")

    gz_files = sorted(args.sdf_dir.rglob("*.sdf.gz") if args.recursive else args.sdf_dir.glob("*.sdf.gz"))
    if not gz_files:
        raise SystemExit(f"No *.sdf.gz found in {args.sdf_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "meta.json"
    corrupted_path = args.out_dir / "corrupted.txt"
    corrupted_path.unlink(missing_ok=True)

    shard_paths = [args.out_dir / f"pubchem_shard_{i:05d}.pkl" for i in range(args.n_shards)]
    shard_fhs = [open(p, "wb") for p in shard_paths]

    def shard_idx(cid_b: bytes) -> int:
        return hash64(cid_b) % args.n_shards

    remove_hs = not args.keep_hs
    bufsize = max(1, args.buf_mb) * (1 << 20)

    # Current CID-run state
    current_cid: Optional[bytes] = None
    current_keep = False
    current_blobs: List[bytes] = []

    # Counters
    seen_molecules = 0
    chosen_molecules = 0
    seen_conformers = 0
    saved_molecules = 0
    saved_conformers = 0
    corrupted: List[str] = []

    def flush_current():
        nonlocal current_cid, current_keep, current_blobs
        nonlocal saved_molecules, saved_conformers
        if current_keep and current_cid is not None and current_blobs:
            cid_str = current_cid.decode("utf-8", errors="replace")
            si = shard_idx(current_cid)
            pickle.dump((cid_str, current_blobs), shard_fhs[si], protocol=pickle.HIGHEST_PROTOCOL)
            saved_molecules += 1
            saved_conformers += len(current_blobs)
        current_cid = None
        current_keep = False
        current_blobs = []

    pbar = tqdm(total=len(gz_files), desc="Files", dynamic_ncols=True)

    try:
        for gz in gz_files:
            try:
                with open_gz_buffered(gz, bufsize) as f:
                    while True:
                        cid = read_next_nonempty_line(f)
                        if cid is None:
                            break

                        seen_conformers += 1

                        # New molecule when CID changes
                        if current_cid is None:
                            current_cid = cid
                            current_keep = ((seen_molecules % args.k) == start)
                            seen_molecules += 1
                            if current_keep:
                                chosen_molecules += 1
                            current_blobs = []
                        elif cid != current_cid:
                            flush_current()
                            current_cid = cid
                            current_keep = ((seen_molecules % args.k) == start)
                            seen_molecules += 1
                            if current_keep:
                                chosen_molecules += 1
                            current_blobs = []

                        # Skip path (no RDKit)
                        if not current_keep:
                            if not skip_to_record_end(f):
                                break
                            continue

                        # Keep path (RDKit only here)
                        rec_bytes = read_record_bytes_after_name(f, cid)
                        if rec_bytes is None:
                            break

                        blob = molblock_to_binary_if_3d(rec_bytes, sanitize=args.sanitize, remove_hs=remove_hs)
                        if blob is not None:
                            current_blobs.append(blob)

            except Exception as e:
                corrupted.append(f"{gz}: {type(e).__name__}: {e}")

            pbar.update(1)
            pbar.set_postfix(
                chosen=chosen_molecules,
                saved_confs=saved_conformers,
                saved_mols=saved_molecules,
                seen_confs=seen_conformers,
                seen_mols=seen_molecules,
            )

        flush_current()

    finally:
        pbar.close()
        for fh in shard_fhs:
            fh.close()

    if corrupted:
        corrupted_path.write_text("\n".join(corrupted) + "\n", encoding="utf-8")

    meta = {
        "sdf_dir": str(args.sdf_dir),
        "recursive": bool(args.recursive),
        "k": args.k,
        "start": start,
        "n_shards": args.n_shards,
        "sanitize": bool(args.sanitize),
        "remove_hs": bool(remove_hs),
        "buf_mb": args.buf_mb,
        "files_total": len(gz_files),
        "seen_molecules": seen_molecules,
        "chosen_molecules": chosen_molecules,
        "seen_conformers": seen_conformers,
        "saved_molecules": saved_molecules,
        "saved_conformers": saved_conformers,
        "corrupted_files": len(corrupted),
        "shards": [str(p) for p in shard_paths],
        "corrupted_log": str(corrupted_path),
        "assumption": "conformers are consecutive (CID-grouped); molecule count increments on CID change",
        "note": "Saved conformers require at least one atom with |z|>1e-6 in conformer 0.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSeen mols (CID-runs): {seen_molecules:,}")
    print(f"Chosen mols: {chosen_molecules:,} (every {args.k}th, start={start})")
    print(f"Seen confs (records): {seen_conformers:,}")
    print(f"Saved mols: {saved_molecules:,}")
    print(f"Saved confs (3D only): {saved_conformers:,}")
    print(f"Meta: {meta_path}")
    if corrupted:
        print(f"Corrupted log: {corrupted_path} ({len(corrupted)} files)")


if __name__ == "__main__":
    main()
