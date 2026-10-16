#!/usr/bin/env python3
"""Lay out a generation pickle for PoseBusters scoring.

Writes one record per molecule in the shape ``scripts/setup3_posebusters.py``
consumes, with a single ``generated_all`` condition. Use it to score a complete
generation set, rather than the interpolation-eligible subset Setup 3 covers.

Usage:
  prepare_generation_pb.py <generation_pickle> <out_dir> [--exclude-complete-from RUN_DIR]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation_pickle", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--only-missing-from",
        type=Path,
        help="a Setup 3 run directory; keep only molecules it did NOT mark complete",
    )
    args = parser.parse_args()
    (args.out_dir / "molecules").mkdir(parents=True, exist_ok=True)

    with args.generation_pickle.open("rb") as handle:
        generated = pickle.load(handle)

    keys = list(generated)
    if args.only_missing_from is not None:
        complete = set()
        for path in glob.glob(str(args.only_missing_from / "molecules" / "*.pickle")):
            with open(path, "rb") as handle:
                record = pickle.load(handle)
            if record.get("status") == "complete":
                complete.add(record["molecule_key"])
        keys = [key for key in keys if key not in complete]

    conformers = 0
    for key in keys:
        mols = list(generated[key])
        conformers += len(mols)
        name = hashlib.sha256(key.encode()).hexdigest() + ".pickle"
        with (args.out_dir / "molecules" / name).open("wb") as handle:
            pickle.dump(
                {
                    "molecule_key": key,
                    "status": "complete",
                    "conformers": {"generated_all": mols},
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    print(f"wrote {len(keys)} molecules, {conformers:,} conformers to {args.out_dir}")


if __name__ == "__main__":
    main()
