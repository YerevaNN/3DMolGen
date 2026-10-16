#!/usr/bin/env python3
"""Iterative-gating ensembles for every round, derived from the blanket chain.

Under iterative gating the pass is applied to the previous round's gated
ensemble, so a conformer is refined only until it first passes PoseBusters and is
then frozen. Because the tokenizer pass is deterministic, that makes round r's
gated conformer exactly

    refine^min(k, r)(original),   k = first round whose blanket conformer passes

and every one of those conformers already exists in the blanket run directories.
Verified against a real sequential run at round 2: identical flags for all 998
molecules and a maximum coordinate difference of 0.0 Å.

Writes <method>_it<r>/molecules/*.pickle holding the `gated` condition only.

Usage: assemble_iterative_gated.py <method> [--max-round 10] [--workers 16]
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BASE = Path("/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/baseline_refinement")


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


def one(args) -> str | None:
    name, method, max_round, flags = args
    src = [BASE / method] + [BASE / f"{method}_r{r}" for r in range(2, max_round + 1)]
    with (src[0] / "molecules" / name).open("rb") as h:
        head = pickle.load(h)
    key = head.get("molecule_key")
    if head.get("status") != "complete" or key not in flags[0]:
        return None
    # pools[k] holds refine^k: k=0 the original, k=1 the first pass, and so on
    pools = [head["conformers"]["before"], head["conformers"]["after"]]
    for d in src[1:]:
        with (d / "molecules" / name).open("rb") as h:
            pools.append(pickle.load(h)["conformers"]["after"])
    n = len(pools[0])
    bits = [flags[k].get(key) for k in range(max_round + 1)]
    if any(b is None or len(b) != n for b in bits) or any(len(p) != n for p in pools):
        return None
    first = [next((k for k in range(max_round + 1) if bits[k][i] == "1"), None) for i in range(n)]
    for r in range(2, max_round + 1):
        out_dir = BASE / f"{method}_it{r}" / "molecules"
        out_dir.mkdir(parents=True, exist_ok=True)
        gated = [pools[min(first[i], r)][i] if first[i] is not None else pools[r][i]
                 for i in range(n)]
        rec = {"molecule_key": key, "status": "complete", "method": method,
               "n_conformers": n, "conformers": {"gated": gated}}
        with (out_dir / name).open("wb") as h:
            pickle.dump(rec, h, protocol=pickle.HIGHEST_PROTOCOL)
    return key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("method")
    ap.add_argument("--max-round", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    flags = [pb_flags(BASE / f"pb_{args.method}_before", "before"),
             pb_flags(BASE / f"pb_{args.method}_after", "after")]
    for r in range(2, args.max_round + 1):
        flags.append(pb_flags(BASE / f"pb_{args.method}_r{r}", "after"))
    print(f"flags loaded for rounds 0..{args.max_round}", flush=True)

    names = sorted(p.name for p in (BASE / args.method / "molecules").glob("*.pickle"))
    tasks = [(n, args.method, args.max_round, flags) for n in names]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for got in ex.map(one, tasks, chunksize=8):
            done += got is not None
    print(f"assembled {done}/{len(names)} molecules into "
          f"{args.method}_it2..{args.method}_it{args.max_round}", flush=True)
    for r in range(2, args.max_round + 1):
        shutil.copy2(BASE / args.method / "config.json", BASE / f"{args.method}_it{r}" / "config.json")


if __name__ == "__main__":
    main()
