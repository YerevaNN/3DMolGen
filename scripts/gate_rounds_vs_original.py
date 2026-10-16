#!/usr/bin/env python3
"""Write a `gated` condition into a later round's records, gated on the original.

`refine-broken` at round r means: keep each conformer the baseline itself
produced where that conformer already passed PoseBusters, and use the r-times
refined one only where it failed. Round r's own `before` is round r-1's output,
not the original, so the gate has to reach back to the round-1 directory — which
is why the module's own `gate` phase cannot do this.

Adds `conformers["gated"]` in place (never touches the existing conditions), so
`--phase evaluate` then computes its COV/MAT like any other condition.

Usage:
  gate_rounds_vs_original.py <method> <round> [--flags-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

BASE = Path("/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/baseline_refinement")


def load_flags(pb_dir: Path, condition: str) -> dict[str, str]:
    out = {}
    for path in pb_dir.glob("*.json"):
        if path.name.startswith("summary"):
            continue
        rec = json.loads(path.read_text())
        if rec.get("status") != "complete":
            continue
        entry = (rec.get("conditions") or {}).get(condition)
        if entry and entry.get("flags", {}).get("__all__"):
            out[rec["molecule_key"]] = entry["flags"]["__all__"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("method")
    ap.add_argument("round", type=int)
    ap.add_argument("--flags-dir", type=Path)
    args = ap.parse_args()

    origin = BASE / args.method                      # round 1 dir holds `before` = the originals
    target = BASE / f"{args.method}_r{args.round}"
    flags = load_flags(args.flags_dir or BASE / f"pb_{args.method}_before", "before")
    print(f"{args.method} r{args.round}: {len(flags)} molecules have original PoseBusters flags", flush=True)

    written = skipped = kept = swapped = 0
    for path in sorted((target / "molecules").glob("*.pickle")):
        with path.open("rb") as handle:
            rec = pickle.load(handle)
        if rec.get("status") != "complete" or "gated" in rec["conformers"]:
            skipped += 1
            continue
        key = rec["molecule_key"]
        bits = flags.get(key)
        after = rec["conformers"]["after"]
        with (origin / "molecules" / path.name).open("rb") as handle:
            original = pickle.load(handle)["conformers"]["before"]
        if bits is None or len(bits) != len(after) or len(original) != len(after):
            skipped += 1
            continue
        gated = []
        for i, ok in enumerate(bits):
            if ok == "1":
                gated.append(original[i])
                kept += 1
            else:
                gated.append(after[i])
                swapped += 1
        rec["conformers"]["gated"] = gated
        rec.setdefault("gate_statistics", {})[f"vs_original_r{args.round}"] = {
            "kept": sum(1 for b in bits if b == "1"), "refined": sum(1 for b in bits if b == "0")
        }
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(rec, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        written += 1
    print(f"  wrote gated into {written} records ({skipped} skipped); "
          f"conformers kept from original {kept:,}, taken from round {args.round} {swapped:,}", flush=True)


if __name__ == "__main__":
    main()
