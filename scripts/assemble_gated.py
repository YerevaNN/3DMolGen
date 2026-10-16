#!/usr/bin/env python3
"""Assemble the gated ensemble for one round of *iterative* gating.

Iterative gating differs from gating against the original: at round r the pass is
applied to round r-1's **gated** ensemble, and the gate then keeps whichever
conformer is valid. So the trajectory itself depends on the gate, and the rounds
must run in sequence.

    gated_r[i] = gated_{r-1}[i]   if gated_{r-1}[i] passed PoseBusters
                 after_r[i]       otherwise

Validity carries forward exactly without re-busting, because a gated conformer is
whichever of the two was scored: gated_r is valid iff gated_{r-1} was valid or
after_r is. Flags per round are cached in gated_flags/ so the next round can gate
without another PoseBusters run over its input.

Usage:
  assemble_gated.py <method> <round>          # round >= 2; writes `gated` in <method>_g<round>
  assemble_gated.py <method> 1 --bootstrap    # cache round-1 flags from the existing run
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

BASE = Path("/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/baseline_refinement")
FLAGS = BASE / "gated_flags"


def pb_flags(pb_dir: Path, condition: str) -> dict[str, str]:
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


def merge(prev: str, new: str) -> str:
    """A gated conformer is the previous one if it passed, else the refined one."""
    return "".join(p if p == "1" else n for p, n in zip(prev, new))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("method")
    ap.add_argument("round", type=int)
    ap.add_argument("--bootstrap", action="store_true")
    args = ap.parse_args()
    FLAGS.mkdir(parents=True, exist_ok=True)
    out_flags = FLAGS / f"{args.method}_g{args.round}.json"

    if args.bootstrap:
        before = pb_flags(BASE / f"pb_{args.method}_before", "before")
        after = pb_flags(BASE / f"pb_{args.method}_after", "after")
        keys = sorted(set(before) & set(after))
        cached = {k: merge(before[k], after[k]) for k in keys if len(before[k]) == len(after[k])}
        out_flags.write_text(json.dumps(cached))
        valid = sum(b.count("1") for b in cached.values())
        total = sum(len(b) for b in cached.values())
        print(f"round 1 gated flags cached for {len(cached)} molecules: "
              f"{valid:,}/{total:,} conformers valid ({valid/total*100:.2f}%)")
        return

    prev_flags = json.loads((FLAGS / f"{args.method}_g{args.round - 1}.json").read_text())
    target = BASE / f"{args.method}_g{args.round}"
    after_flags = pb_flags(BASE / f"pb_{args.method}_g{args.round}", "after")

    written = skipped = kept = swapped = 0
    new_flags = {}
    for path in sorted((target / "molecules").glob("*.pickle")):
        with path.open("rb") as handle:
            rec = pickle.load(handle)
        key = rec.get("molecule_key")
        if rec.get("status") != "complete" or key not in prev_flags or key not in after_flags:
            skipped += 1
            continue
        prev, new = prev_flags[key], after_flags[key]
        before, after = rec["conformers"]["before"], rec["conformers"]["after"]
        if not (len(prev) == len(new) == len(before) == len(after)):
            skipped += 1
            continue
        gated = []
        for i, ok in enumerate(prev):
            gated.append(before[i] if ok == "1" else after[i])
            kept += ok == "1"
            swapped += ok != "1"
        rec["conformers"]["gated"] = gated
        new_flags[key] = merge(prev, new)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(rec, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        written += 1

    out_flags.write_text(json.dumps(new_flags))
    valid = sum(b.count("1") for b in new_flags.values())
    total = sum(len(b) for b in new_flags.values())
    print(f"round {args.round}: gated written into {written} records ({skipped} skipped); "
          f"kept {kept:,}, refined {swapped:,}; validity now {valid:,}/{total:,} "
          f"({valid/total*100:.2f}%)")


if __name__ == "__main__":
    main()
