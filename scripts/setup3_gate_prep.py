#!/usr/bin/env python3
"""Pair each Setup 3 interpolant with its refined version, for the gate.

Gated refinement keeps a conformer's original coordinates when the original
already passes PoseBusters and takes the refined ones otherwise. That decision
is per conformer, so it needs the *joint* of before- and after-validity — which
per-molecule pass counts cannot give, only the two marginals. So the paired
interpolant sets are written out here and busted once with per-conformer flags.

Only the interpolants are needed: the gate leaves the anchors alone, and their
validity is already known from the original PoseBusters run.

Records hold, per source condition, ``<cond>_interp`` and ``<cond>_interp_rt``
as index-aligned lists, so flag position i refers to the same interpolant in
both.

Usage: setup3_gate_prep.py <source_dir> <refine_dir> <out_dir> [--conditions ...]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

SOURCE_CONDITIONS = ("latent", "random_token", "cartesian")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("refine_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--conditions", nargs="+", default=list(SOURCE_CONDITIONS))
    args = parser.parse_args()

    (args.out_dir / "molecules").mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    mismatched = 0
    for source_path in sorted((args.source_dir / "molecules").glob("*.pickle")):
        refine_path = args.refine_dir / "molecules" / source_path.name
        out_path = args.out_dir / "molecules" / source_path.name
        if out_path.exists():
            written += 1
            continue
        with source_path.open("rb") as handle:
            source = pickle.load(handle)
        refined = None
        if refine_path.exists():
            with refine_path.open("rb") as handle:
                refined = pickle.load(handle)
        if source.get("status") != "complete" or (refined or {}).get("status") != "complete":
            with out_path.open("wb") as handle:
                pickle.dump(
                    {
                        "molecule_key": source.get("molecule_key"),
                        "status": source.get("status", "missing_refinement"),
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            skipped += 1
            continue

        n_anchor = source["n_anchor"]
        conformers = {}
        for condition in args.conditions:
            before = (source["conformers"].get(condition) or [])[n_anchor:]
            after = (refined["conformers"].get(f"{condition}_rt_all") or [])[n_anchor:]
            if not before or len(before) != len(after):
                mismatched += 1
                continue
            conformers[f"{condition}_interp"] = list(before)
            conformers[f"{condition}_interp_rt"] = list(after)
        with out_path.open("wb") as handle:
            pickle.dump(
                {
                    "molecule_key": source["molecule_key"],
                    "status": "complete" if conformers else "no_paired_conditions",
                    "k": source["k"],
                    "n_anchor": n_anchor,
                    "variant": source["variant"],
                    "conformers": conformers,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        written += 1

    print(
        f"wrote {written} records, {skipped} not interpolable, "
        f"{mismatched} condition pairs dropped for a length mismatch"
    )


if __name__ == "__main__":
    main()
