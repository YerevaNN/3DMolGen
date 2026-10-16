#!/usr/bin/env python3
"""Rounds 0-10 for the four GEOM baselines on the common 938-molecule set.

Torsional Diffusion has usable COV/MAT for only 938 of its 998 molecules (the
rest disagree with the GEOM reference on tautomer or resonance form, so RMSD is
undefined), and that 938 is a strict subset of the other methods' 1,000. Scoring
every method and every round on it makes the table comparable in both directions
— across methods and across rounds.

`original` is the baseline's own conformers; round r is r tokenizer passes.

Usage: baseline_rounds_938.py [--max-round 10] [--workers 16] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

BASE = Path("/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/baseline_refinement")
METHODS = ("torsional", "mcf_s", "mcf_b", "etflow")
I75 = 14  # thresholds run 0.05..3.00 in 0.05 steps


def round_dir(method: str, r: int) -> Path:
    return BASE / method if r <= 1 else BASE / f"{method}_r{r}"


def pb_dir(method: str, r: int) -> Path:
    return BASE / f"pb_{method}_after" if r == 1 else BASE / f"pb_{method}_r{r}"


def covmat_of(args) -> tuple:
    """(method, round, condition) -> per-molecule COV/MAT, read from the records."""
    method, r, condition = args
    out = {}
    for p in (round_dir(method, r) / "molecules").glob("*.pickle"):
        try:
            rec = pickle.load(p.open("rb"))
        except Exception:
            continue
        if rec.get("status") != "complete":
            continue
        v = (rec.get("metrics") or {}).get(condition)
        if not v:
            continue
        cr = np.asarray(v["coverage_recall"], float)
        cp = np.asarray(v["coverage_precision"], float)
        mr, mp = float(v["matching_recall"]), float(v["matching_precision"])
        if not np.isfinite(cr[I75]) or not np.isfinite(mr):
            continue
        out[rec["molecule_key"]] = (cr[I75], cp[I75], mr, mp)
    return method, r, condition, out


def pb_of(method: str, r: int, keys: set[str]) -> tuple[float, float, int]:
    n = ok = 0
    fr = []
    d = pb_dir(method, r)
    for p in d.glob("*.json"):
        if p.name.startswith("summary"):
            continue
        j = json.loads(p.read_text())
        if j.get("status") != "complete" or j["molecule_key"] not in keys:
            continue
        v = (j.get("conditions") or {}).get("after")
        if not v or "pass" not in v or not v.get("n"):
            continue
        n += v["n"]
        ok += v["pass"]["__all__"]
        fr.append(v["pass"]["__all__"] / v["n"])
    return (ok / n * 100 if n else float("nan"),
            sum(fr) / len(fr) * 100 if fr else float("nan"), len(fr))


def pb_before(method: str, keys: set[str]) -> tuple[float, float, int]:
    n = ok = 0
    fr = []
    for p in (BASE / f"pb_{method}_before").glob("*.json"):
        if p.name.startswith("summary"):
            continue
        j = json.loads(p.read_text())
        if j.get("status") != "complete" or j["molecule_key"] not in keys:
            continue
        v = (j.get("conditions") or {}).get("before")
        if not v or "pass" not in v or not v.get("n"):
            continue
        n += v["n"]
        ok += v["pass"]["__all__"]
        fr.append(v["pass"]["__all__"] / v["n"])
    return (ok / n * 100 if n else float("nan"),
            sum(fr) / len(fr) * 100 if fr else float("nan"), len(fr))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-round", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--csv", type=Path, default=BASE / "rounds_938.csv")
    args = ap.parse_args()

    tasks = [(m, 1, "before") for m in METHODS]
    tasks += [(m, r, "after") for m in METHODS for r in range(1, args.max_round + 1)]
    print(f"reading {len(tasks)} (method, round) record sets with {args.workers} workers", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for method, r, cond, out in ex.map(covmat_of, tasks):
            results[(method, r, cond)] = out
            print(f"  {method} r{r} {cond}: {len(out)} molecules", flush=True)

    common = set.intersection(*(set(v) for v in results.values()))
    print(f"\ncommon molecule set across every method and round: {len(common)}", flush=True)

    rows = []
    for m in METHODS:
        for r in range(0, args.max_round + 1):
            cond = "before" if r == 0 else "after"
            key = (m, 1, "before") if r == 0 else (m, r, cond)
            per = results.get(key)
            if not per:
                continue
            vals = np.array([per[k] for k in common])
            cr, cp, mr, mp = vals.mean(axis=0)
            micro, macro, nmol = pb_before(m, common) if r == 0 else pb_of(m, r, common)
            rows.append({"method": m, "round": "original" if r == 0 else r,
                         "cov_r": cr * 100, "cov_p": cp * 100, "mat_r": mr, "mat_p": mp,
                         "pb_micro": micro, "pb_macro": macro,
                         "molecules": len(common), "pb_molecules": nmol})

    for m in METHODS:
        sel = [x for x in rows if x["method"] == m]
        if not sel:
            continue
        print(f"\n| **{m}** | " + " | ".join(str(x["round"]) for x in sel) + " |")
        for label, k, f in (("PB micro", "pb_micro", 2), ("PB macro", "pb_macro", 2),
                            ("COV-R", "cov_r", 2), ("COV-P", "cov_p", 2),
                            ("MAT-R", "mat_r", 3), ("MAT-P", "mat_p", 3)):
            print(f"| | {label} | " + " | ".join(f"{x[k]:.{f}f}" for x in sel) + " |")

    import csv as _csv
    with args.csv.open("w", newline="") as h:
        w = _csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
