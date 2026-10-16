#!/usr/bin/env python3
"""The baseline tokenizer-refinement table: before, refine-all and gated.

Reads the two PoseBusters runs for each method — one over the ``before``
conditions, one over ``after`` — and joins them on the per-conformer flag
strings both persist. The gated policy keeps a conformer's original coordinates
where the original already passed and takes the refined ones otherwise, which
needs the joint of before- and after-validity, so it is computed here from the
flags rather than from the pass counts.

Usage:
  baseline_refine_report.py <refinement_root> [--methods torsional etflow ...]
                           [--basis before|before_with_h]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METHODS = ("torsional", "etflow", "mcf_s", "mcf_b")
LABELS = {
    "torsional": "Torsional Diffusion",
    "etflow": "ETFlow",
    "mcf_s": "MCF-S",
    "mcf_b": "MCF-B",
}


def _flags(directory: Path, name: str, condition: str) -> str | None:
    path = directory / name
    if not path.exists():
        return None
    entry = json.loads(path.read_text()).get("conditions", {}).get(condition, {})
    return entry.get("flags", {}).get("__all__")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--basis", choices=("before", "before_with_h"), default="before")
    args = parser.parse_args()

    rows = []
    for method in args.methods:
        before_dir = args.root / f"pb_{method}_before"
        after_dir = args.root / f"pb_{method}_after"
        totals = {k: 0 for k in ("n", "before", "after", "gated", "fixed", "broken", "refined")}
        macro = {k: 0.0 for k in ("before", "after", "gated")}
        molecules = 0
        unpaired = 0
        for path in sorted(before_dir.glob("[0-9a-f]*.json")):
            b = _flags(before_dir, path.name, args.basis)
            a = _flags(after_dir, path.name, "after")
            if b is None or a is None or len(a) != len(b):
                unpaired += 1
                continue
            gated = "".join(
                bi if bi == "1" else ai for bi, ai in zip(b, a)
            )
            totals["n"] += len(b)
            totals["before"] += b.count("1")
            totals["after"] += a.count("1")
            totals["gated"] += gated.count("1")
            totals["fixed"] += sum(1 for x, y in zip(b, a) if x == "0" and y == "1")
            totals["broken"] += sum(1 for x, y in zip(b, a) if x == "1" and y == "0")
            totals["refined"] += b.count("0")
            for key, flags in (("before", b), ("after", a), ("gated", gated)):
                macro[key] += flags.count("1") / len(flags)
            molecules += 1
        if not molecules:
            print(f"{method}: no paired molecules found")
            continue
        rows.append(
            {
                "method": method,
                "molecules": molecules,
                "unpaired": unpaired,
                "n": totals["n"],
                **{f"micro_{k}": totals[k] / totals["n"] * 100 for k in ("before", "after", "gated")},
                **{f"macro_{k}": macro[k] / molecules * 100 for k in ("before", "after", "gated")},
                "fixed": totals["fixed"],
                "broken": totals["broken"],
                "refined": totals["refined"],
            }
        )

    print(f"basis for `before`: {args.basis}")
    print()
    print("| Method | PB before | refine-all | gated | Δ gated | macro before → gated | conformers fixed/broken |")
    print("|---|---:|---:|---:|---:|---|---|")
    for r in rows:
        print(
            f"| {LABELS.get(r['method'], r['method'])} | {r['micro_before']:.2f} | "
            f"{r['micro_after']:.2f} | **{r['micro_gated']:.2f}** | "
            f"{r['micro_gated'] - r['micro_before']:+.2f} | "
            f"{r['macro_before']:.2f} → {r['macro_gated']:.2f} | "
            f"{r['fixed']:,}/{r['broken']:,} |"
        )
    print()

    # combined COV/MAT + PB view, one block per condition
    names = {"before": "before", "after": "refine-all", "gated": "refine-broken"}
    print("| Method | Condition | COV-R | COV-P | MAT-R | MAT-P | PB-valid | COV/MAT mols |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        path = args.root / r["method"] / "covmat_summary.json"
        cm = json.loads(path.read_text())["conditions"] if path.exists() else {}
        for condition, label in names.items():
            c = cm.get(condition)
            pb = r[f"micro_{condition}"]
            if c is None:
                print(f"| {LABELS.get(r['method'])} | {label} | – | – | – | – | {pb:.2f} | – |")
                continue
            print(
                f"| {LABELS.get(r['method'])} | {label} | {c['COV-R']*100:.1f} | "
                f"{c['COV-P']*100:.1f} | {c['MAT-R']:.3f} | {c['MAT-P']:.3f} | "
                f"{pb:.2f} | {c['molecule_count']} |"
            )
    print()
    for r in rows:
        note = f", {r['unpaired']} unpaired" if r["unpaired"] else ""
        print(
            f"{r['method']}: {r['molecules']} molecules, {r['n']:,} conformers, "
            f"gate refined {r['refined']:,} ({r['refined'] / r['n'] * 100:.1f}%){note}"
        )


if __name__ == "__main__":
    main()
