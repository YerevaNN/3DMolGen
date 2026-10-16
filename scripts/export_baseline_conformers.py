#!/usr/bin/env python3
"""Export a baseline refinement run back to the input pickle format.

The per-molecule records are convenient for resuming and for per-conformer
scoring, but awkward to consume. This writes one pickle per condition in the
same shape the baselines ship — ``{molecule_key: [Mol, ...]}`` — so existing
scoring code reads them without changes.

Usage:
  export_baseline_conformers.py <run_dir> <out_dir> [--conditions before after]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--conditions", nargs="+", default=["before", "after"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    method = args.run_dir.name
    exported: dict[str, dict] = {c: {} for c in args.conditions}
    missing: dict[str, int] = {c: 0 for c in args.conditions}
    molecules = 0
    for path in sorted((args.run_dir / "molecules").glob("*.pickle")):
        with path.open("rb") as handle:
            record = pickle.load(handle)
        if record.get("status") != "complete":
            continue
        molecules += 1
        key = record["molecule_key"]
        for condition in args.conditions:
            mols = record["conformers"].get(condition)
            if mols is None:
                missing[condition] += 1
                continue
            exported[condition][key] = list(mols)

    for condition in args.conditions:
        if not exported[condition]:
            print(f"{condition}: nothing to write ({missing[condition]} records lack it)")
            continue
        out_path = args.out_dir / f"{method}_{condition}.pkl"
        with out_path.open("wb") as handle:
            pickle.dump(exported[condition], handle, protocol=pickle.HIGHEST_PROTOCOL)
        conformers = sum(len(v) for v in exported[condition].values())
        note = f", {missing[condition]} records lack it" if missing[condition] else ""
        print(
            f"{out_path}  {len(exported[condition])} molecules, "
            f"{conformers:,} conformers{note}"
        )
    print(f"({molecules} complete records read from {args.run_dir})")


if __name__ == "__main__":
    main()
