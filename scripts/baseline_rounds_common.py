#!/usr/bin/env python3
"""Multi-round refinement table on one molecule set common to every phase.

PoseBusters loses the occasional molecule to a hang, and which molecule it loses
varies by round. COV/MAT can also exclude molecules for which atom correspondence
is undefined. Comparing phases on different sets would therefore manufacture a
trend. Every value here -- original input, rounds 1..N, PB, and COV/MAT -- is
re-aggregated from per-molecule records on one per-method intersection.

Usage:
  baseline_rounds_common.py [--methods etflow ...] [--max-round 10] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

BASE = Path("/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/baseline_refinement")
METHODS = ("torsional", "mcf_s", "mcf_b", "etflow")


def pb_dir(method: str, r: int) -> Path:
    if r == 0:
        return BASE / f"pb_{method}_before"
    return BASE / f"pb_{method}_after" if r == 1 else BASE / f"pb_{method}_r{r}"


def round_dir(method: str, r: int) -> Path:
    return BASE / method if r <= 1 else BASE / f"{method}_r{r}"


def condition(r: int) -> str:
    return "before" if r == 0 else "after"


def per_molecule(pb: Path, condition: str = "after") -> dict[str, tuple[int, int]]:
    """molecule key -> (conformers, conformers passing every check)."""
    out = {}
    for path in pb.glob("*.json"):
        if path.name.startswith("summary"):
            continue
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("status") != "complete":
            continue
        v = (d.get("conditions") or {}).get(condition)
        if not v or "pass" not in v or not v.get("n"):
            continue
        out[d["molecule_key"]] = (v["n"], v["pass"]["__all__"])
    return out


def metric_record(run: Path, molecule_key: str, condition_name: str) -> dict | None:
    name = hashlib.sha256(molecule_key.encode()).hexdigest() + ".pickle"
    try:
        with (run / "molecules" / name).open("rb") as handle:
            record = pickle.load(handle)
    except Exception:
        return None
    return (record.get("metrics") or {}).get(condition_name)


def metric_values(row: dict, threshold: float = 0.75) -> dict[str, float]:
    thresholds = np.arange(0.05, 3.05, 0.05)
    index = int(np.argmin(np.abs(thresholds - threshold)))
    return {
        "cov_r": float(row["coverage_recall"][index]) * 100,
        "cov_p": float(row["coverage_precision"][index]) * 100,
        "mat_r": float(row["matching_recall"]),
        "mat_p": float(row["matching_precision"]),
    }


def common_covmat(
    run: Path,
    condition_name: str,
    common: set[str],
    full_keys: set[str],
) -> dict[str, float]:
    """Remove excluded molecules from a stored full-set mean by additivity."""
    summary = json.loads((run / "covmat_summary.json").read_text())["conditions"][condition_name]
    full_n = int(summary["molecule_count"])
    excluded = full_keys - common
    totals = {
        "cov_r": summary["COV-R"] * 100 * full_n,
        "cov_p": summary["COV-P"] * 100 * full_n,
        "mat_r": summary["MAT-R"] * full_n,
        "mat_p": summary["MAT-P"] * full_n,
    }
    removed = 0
    for key in excluded:
        metric = metric_record(run, key, condition_name)
        if metric is None:
            continue
        removed += 1
        values = metric_values(metric)
        for name, value in values.items():
            totals[name] -= value
    denominator = full_n - removed
    if denominator != len(common):
        raise RuntimeError(
            f"{run.name}/{condition_name}: adjusted COV/MAT denominator {denominator}, "
            f"expected common set {len(common)}"
        )
    return {name: value / denominator for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--max-round", type=int, default=10)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    rows = []
    head = (f"{'method':10s} {'round':>5s} {'COV-R':>7s} {'COV-P':>7s} {'MAT-R':>7s} {'MAT-P':>7s}"
            f" {'PB micro':>9s} {'PB macro':>9s}")
    for method in args.methods:
        scored = {}
        for r in range(0, args.max_round + 1):
            d = pb_dir(method, r)
            if d.exists():
                m = per_molecule(d, condition(r))
                if m:
                    scored[r] = m
        if not scored:
            continue
        full_keys = set(scored[0])
        common = set.intersection(*(set(m) for m in scored.values()))

        # Torsional's release has 60 molecules with no atom correspondence, so
        # COV/MAT is undefined for them. Discover that stable support once from
        # round 1, then apply it to original and every subsequent round.
        first_summary = json.loads((round_dir(method, 1) / "covmat_summary.json").read_text())
        covmat_count = int(first_summary["conditions"]["after"]["molecule_count"])
        if covmat_count != len(full_keys):
            supported = {
                key for key in common
                if metric_record(round_dir(method, 1), key, "after") is not None
            }
            common &= supported
        print(
            f"\n{method}: original + {args.max_round} rounds, common molecule set = {len(common)} "
            f"(PB counts {min(len(m) for m in scored.values())}–{max(len(m) for m in scored.values())}; "
            f"full COV/MAT support {covmat_count})"
        )
        print(head)
        print("-" * len(head))
        for r in range(0, args.max_round + 1):
            row = {"method": method, "round": r, "molecules": len(common)}
            row |= common_covmat(
                round_dir(method, r), condition(r), common, full_keys
            )
            if r in scored:
                sel = [scored[r][k] for k in common]
                n = sum(x[0] for x in sel)
                ok = sum(x[1] for x in sel)
                row["pb_micro"] = ok / n * 100
                row["pb_macro"] = sum(x[1] / x[0] for x in sel) / len(sel) * 100
                row["conformers"] = n
            cells = []
            for c, w, f in (("cov_r", 7, 2), ("cov_p", 7, 2), ("mat_r", 7, 3), ("mat_p", 7, 3),
                            ("pb_micro", 9, 2), ("pb_macro", 9, 2)):
                v = row.get(c)
                cells.append(f"{'-':>{w}s}" if v is None else f"{v:{w}.{f}f}")
            print(f"{method:10s} {r:5d} " + " ".join(cells))
            rows.append(row)

    if args.csv:
        import csv as _csv
        fields = ["method", "round", "cov_r", "cov_p", "mat_r", "mat_p",
                  "pb_micro", "pb_macro", "molecules", "conformers"]
        with args.csv.open("w", newline="") as h:
            w = _csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
