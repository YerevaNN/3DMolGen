#!/usr/bin/env python3
"""The Setup 3 tokenizer-refinement table: COV/MAT and PoseBusters in one place.

COV/MAT comes from ``molgen3D.evaluation.setup3_tokenizer_refinement``'s
summary. Validity is assembled from two PoseBusters runs rather than one,
because busting every refined ensemble directly would have spent hours on the
few largest molecules for rows that are recoverable by arithmetic: anchors and
interpolants are disjoint conformer sets, so their pass counts add.

  generated_all, <cond>          the original Setup 3 run
  anchors_rt                     the refined-anchor run
  <cond>_rt_interp / _rt_all     original / refined anchors + refined interpolants
  <cond>_gated                   refined only where the original failed

Usage:
  setup3_refine_report.py <refine_dir> <gate_pb_dir> <baseline_pb_dir>
                          [--basis raw|fallback]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SOURCE_CONDITIONS = ("latent", "random_token", "cartesian")


def row_order(source_conditions=SOURCE_CONDITIONS) -> list[str]:
    """Baselines first, then each source condition next to its refined variants."""
    order = ["generated_all", "anchors", "anchors_rt"]
    for condition in source_conditions:
        order += [
            condition,
            f"{condition}_rt_all",
            f"{condition}_rt_interp",
            f"{condition}_gated",
        ]
    return order


def _load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def validity_by_condition(gate_summary: dict, baseline_summary: dict) -> dict[str, dict]:
    """Map every condition name onto its PoseBusters micro/macro pair."""
    validity: dict[str, dict] = {}
    for name in ("generated_all", "anchors"):
        entry = baseline_summary["conditions"].get(name)
        if entry:
            validity[name] = entry
    if gate_summary.get("anchors_rt"):
        validity["anchors_rt"] = gate_summary["anchors_rt"]
    policy_suffix = {"before": "", "rt_all": "_rt_all", "after": "_rt_interp", "gated": "_gated"}
    for condition, values in gate_summary["full_ensemble"].items():
        for policy, suffix in policy_suffix.items():
            if policy not in values:
                continue
            validity[f"{condition}{suffix}"] = {
                "pb_valid_pct": values[policy]["pb_valid_pct"],
                "pb_valid_macro_pct": values[policy]["pb_valid_macro_pct"],
                "n_conformers": values["n_conformers"],
                "molecules": values[policy].get("molecules", values["molecules"]),
            }
    return validity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("refine_dir", type=Path)
    parser.add_argument("gate_pb_dir", type=Path)
    parser.add_argument("baseline_pb_dir", type=Path)
    parser.add_argument("--basis", choices=("raw", "fallback"), default="fallback")
    args = parser.parse_args()

    covmat = _load(args.refine_dir / "summary.json")
    metrics = covmat["with_fallback" if args.basis == "fallback" else "raw"]
    validity = validity_by_condition(
        _load(args.gate_pb_dir / "gate_summary.json"),
        _load(args.baseline_pb_dir / "summary.json"),
    )

    print(
        f"COV/MAT basis: {args.basis} "
        f"({covmat['fallback_molecules']} molecules fall back to plain 2k generation)"
    )
    print()
    print("| Condition | COV-R | COV-P | MAT-R | MAT-P | PB micro | PB macro | PB molecules |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for condition in row_order(covmat["config"]["source_conditions"]):
        row = metrics.get(condition)
        if row is None:
            continue
        pb = validity.get(condition)
        print(
            "| "
            + " | ".join(
                [
                    condition,
                    f"{row['COV-R'] * 100:.1f}",
                    f"{row['COV-P'] * 100:.1f}",
                    f"{row['MAT-R']:.3f}",
                    f"{row['MAT-P']:.3f}",
                    f"{pb['pb_valid_pct']:.2f}" if pb else "–",
                    f"{pb['pb_valid_macro_pct']:.2f}" if pb else "–",
                    str(pb["molecules"]) if pb else "–",
                ]
            )
            + " |"
        )

    gate = covmat.get("gate_statistics") or {}
    if gate:
        print()
        print("Interpolants the gate chose to refine:")
        for name, stats in sorted(gate.items()):
            share = stats["refined"] / stats["n_interpolants"] * 100
            print(
                f"  {name}: {stats['refined']:,}/{stats['n_interpolants']:,} ({share:.1f}%)"
            )


if __name__ == "__main__":
    main()
