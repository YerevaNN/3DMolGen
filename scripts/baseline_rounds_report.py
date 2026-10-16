#!/usr/bin/env python3
"""The multi-round refinement table for the four GEOM baselines.

Round 1 lives in outputs/baseline_refinement/<method>; round r>=2 in
<method>_r<r>. Each round's `after` is the state after r passes, and each
round's `before` repeats the previous round's `after`, which is a free
consistency check.

Usage:
  baseline_rounds_report.py [--methods etflow ...] [--max-round 10] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = Path("/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/baseline_refinement")
METHODS = ("torsional", "mcf_s", "mcf_b", "etflow")


def round_dir(method: str, r: int) -> Path:
    return BASE / method if r == 1 else BASE / f"{method}_r{r}"


def pb_dir(method: str, r: int) -> Path:
    return BASE / f"pb_{method}_after" if r == 1 else BASE / f"pb_{method}_r{r}"


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def collect(method: str, max_round: int) -> list[dict]:
    rows = []
    for r in range(1, max_round + 1):
        d = round_dir(method, r)
        if not d.exists():
            continue
        row = {"method": method, "round": r}
        covmat = _load(d / "covmat_summary.json")
        if covmat:
            after = covmat.get("conditions", {}).get("after")
            if after:
                row |= {
                    "cov_r": after["COV-R"] * 100,
                    "cov_p": after["COV-P"] * 100,
                    "mat_r": after["MAT-R"],
                    "mat_p": after["MAT-P"],
                }
        pb = _load(pb_dir(method, r) / "summary.json")
        if pb:
            entry = pb.get("conditions", {}).get("after")
            if entry:
                row |= {
                    "pb_micro": entry["pb_valid_pct"],
                    "pb_macro": entry["pb_valid_macro_pct"],
                    "conformers": entry["n_conformers"],
                }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--max-round", type=int, default=10)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    rows = [r for m in args.methods for r in collect(m, args.max_round)]
    cols = ("cov_r", "cov_p", "mat_r", "mat_p", "pb_micro", "pb_macro")
    head = f"{'method':10s} {'round':>5s} {'COV-R':>7s} {'COV-P':>7s} {'MAT-R':>7s} {'MAT-P':>7s} {'PB micro':>9s} {'PB macro':>9s}"
    print(head)
    print("-" * len(head))
    for row in rows:
        cells = []
        for c in cols:
            v = row.get(c)
            width = 7 if not c.startswith("pb") else 9
            if v is None:
                cells.append(f"{'-':>{width}s}")
            elif c.startswith("mat"):
                cells.append(f"{v:{width}.3f}")
            else:
                cells.append(f"{v:{width}.2f}")
        print(f"{row['method']:10s} {row['round']:5d} " + " ".join(cells))

    if args.csv:
        import csv as _csv

        fields = ["method", "round", *cols, "conformers"]
        with args.csv.open("w", newline="") as handle:
            writer = _csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
