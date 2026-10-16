#!/usr/bin/env python3
"""PoseBusters validity of the gated policy, computed exactly from flags.

The gate keeps a conformer's original coordinates when the original already
passes PoseBusters and takes the refined ones otherwise, so its validity is

    valid_gated = valid_before + |{invalid before} and {valid after}|

The second term is a joint, not a marginal, which is why per-conformer flags
are needed and per-molecule pass counts do not suffice. Those flags come from
busting the paired interpolant sets written by ``setup3_gate_prep.py``.

Anchor validity is taken from the original Setup 3 PoseBusters run, since the
gate never touches the anchors. Ensemble counts are additive over the disjoint
anchor and interpolant sets.

Usage:
  setup3_gate_report.py <gate_pb_dir> <baseline_pb_dir> [--conditions ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SOURCE_CONDITIONS = ("latent", "random_token", "cartesian")

# before  = the interpolants as Setup 3 produced them
# after   = every interpolant refined (the `_rt_interp` ensemble)
# gated   = refined only where the original failed PoseBusters
POLICIES = ("before", "after", "gated")

# With --anchors-rt-pb-dir, `after` is also reported over refined anchors,
# which is the `_rt_all` ensemble.
RT_ALL_POLICY = "after"


def _counts(before: str, after: str) -> dict[str, int]:
    """Per-conformer transition counts between two equal-length flag strings."""
    if len(before) != len(after):
        raise ValueError(f"Flag lengths differ: {len(before)} != {len(after)}")
    fixed = sum(1 for b, a in zip(before, after) if b == "0" and a == "1")
    broken = sum(1 for b, a in zip(before, after) if b == "1" and a == "0")
    return {
        "n": len(before),
        "valid_before": before.count("1"),
        "valid_after": after.count("1"),
        # The gate only ever adopts a refined conformer where the original failed.
        "valid_gated": before.count("1") + fixed,
        "fixed": fixed,
        "broken": broken,
        "refined": before.count("0"),
    }


def _anchors_rt_totals(directory: Path) -> dict[str, float | int] | None:
    """Validity of the refined anchors on their own: the reconstruction control."""
    n = valid = molecules = 0
    macro = 0.0
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("summary"):
            continue
        entry = (
            json.loads(path.read_text()).get("conditions", {}).get("anchors_rt", {})
        )
        if "pass" not in entry or not entry.get("n"):
            continue
        n += entry["n"]
        valid += entry["pass"]["__all__"]
        macro += entry["pass"]["__all__"] / entry["n"]
        molecules += 1
    if not n:
        return None
    return {
        "n_conformers": n,
        "molecules": molecules,
        "pb_valid_pct": valid / n * 100,
        "pb_valid_macro_pct": macro / molecules * 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate_pb_dir", type=Path)
    parser.add_argument("baseline_pb_dir", type=Path)
    parser.add_argument("--conditions", nargs="+", default=list(SOURCE_CONDITIONS))
    parser.add_argument(
        "--anchors-rt-pb-dir",
        type=Path,
        help="PoseBusters output for the refined anchors, to add the `_rt_all` rows",
    )
    args = parser.parse_args()

    interpolants: dict[str, dict[str, int]] = {}
    checks: dict[str, dict[str, dict[str, int]]] = {}
    ensembles: dict[str, dict[str, float | int]] = {}
    molecules = 0
    missing_anchors = 0
    missing_flags = 0

    for path in sorted(args.gate_pb_dir.glob("*.json")):
        if path.name.startswith("summary"):
            continue
        validity = json.loads(path.read_text())
        if validity.get("status") != "complete":
            continue
        baseline_path = args.baseline_pb_dir / path.name
        baseline = (
            json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
        )
        anchors = baseline.get("conditions", {}).get("anchors", {})
        anchors_rt = {}
        if args.anchors_rt_pb_dir is not None:
            anchors_rt_path = args.anchors_rt_pb_dir / path.name
            if anchors_rt_path.exists():
                anchors_rt = (
                    json.loads(anchors_rt_path.read_text())
                    .get("conditions", {})
                    .get("anchors_rt", {})
                )
        counted_this_molecule = False
        for condition in args.conditions:
            before_entry = validity["conditions"].get(f"{condition}_interp", {})
            after_entry = validity["conditions"].get(f"{condition}_interp_rt", {})
            before = before_entry.get("flags", {}).get("__all__")
            after = after_entry.get("flags", {}).get("__all__")
            if not before or not after:
                missing_flags += 1
                continue
            totals = interpolants.setdefault(
                condition,
                {k: 0 for k in ("n", "valid_before", "valid_after", "valid_gated", "fixed", "broken", "refined")},
            )
            row = _counts(before, after)
            for key, value in row.items():
                totals[key] += value

            per_check = checks.setdefault(condition, {})
            shared = set(before_entry["flags"]) & set(after_entry["flags"])
            for check in shared:
                bucket = per_check.setdefault(
                    check, {"n": 0, "valid_before": 0, "valid_after": 0, "valid_gated": 0, "fixed": 0, "broken": 0}
                )
                # A gated conformer's per-check outcome follows the composite decision.
                gated_flags = "".join(
                    (before_entry["flags"][check][i] if before[i] == "1" else after_entry["flags"][check][i])
                    for i in range(len(before))
                )
                bucket["n"] += len(before)
                bucket["valid_before"] += before_entry["flags"][check].count("1")
                bucket["valid_after"] += after_entry["flags"][check].count("1")
                bucket["valid_gated"] += gated_flags.count("1")
                pair = _counts(before_entry["flags"][check], gated_flags)
                bucket["fixed"] += pair["fixed"]
                bucket["broken"] += pair["broken"]

            if "pass" not in anchors:
                missing_anchors += 1
                continue
            # The anchors are the same conformers in all three policies, and the
            # anchor and interpolant sets are disjoint, so ensemble counts add.
            ensemble = ensembles.setdefault(
                condition,
                {
                    "n": 0,
                    "molecules": 0,
                    **{f"valid_{p}": 0 for p in POLICIES},
                    **{f"macro_sum_{p}": 0.0 for p in POLICIES},
                },
            )
            n = anchors["n"] + row["n"]
            ensemble["n"] += n
            ensemble["molecules"] += 1
            for policy in POLICIES:
                valid = anchors["pass"]["__all__"] + row[f"valid_{policy}"]
                ensemble[f"valid_{policy}"] += valid
                ensemble[f"macro_sum_{policy}"] += valid / n
            if "pass" in anchors_rt:
                valid = anchors_rt["pass"]["__all__"] + row[f"valid_{RT_ALL_POLICY}"]
                ensemble["valid_rt_all"] = ensemble.get("valid_rt_all", 0) + valid
                ensemble["macro_sum_rt_all"] = (
                    ensemble.get("macro_sum_rt_all", 0.0) + valid / n
                )
                ensemble["molecules_rt_all"] = ensemble.get("molecules_rt_all", 0) + 1
                ensemble["n_rt_all"] = ensemble.get("n_rt_all", 0) + n
            counted_this_molecule = True
        molecules += counted_this_molecule

    summary = {
        "molecules": molecules,
        "missing_anchor_counts": missing_anchors,
        "missing_flag_pairs": missing_flags,
        "interpolants_only": {
            condition: {
                **totals,
                "pct_before": totals["valid_before"] / totals["n"] * 100,
                "pct_after": totals["valid_after"] / totals["n"] * 100,
                "pct_gated": totals["valid_gated"] / totals["n"] * 100,
            }
            for condition, totals in interpolants.items()
        },
        "full_ensemble": {
            condition: {
                "n_conformers": values["n"],
                "molecules": values["molecules"],
                **{
                    policy: {
                        "pb_valid_pct": values[f"valid_{policy}"] / values["n"] * 100,
                        "pb_valid_macro_pct": values[f"macro_sum_{policy}"]
                        / values["molecules"]
                        * 100,
                    }
                    for policy in POLICIES
                },
                **(
                    {
                        "rt_all": {
                            "pb_valid_pct": values["valid_rt_all"] / values["n_rt_all"] * 100,
                            "pb_valid_macro_pct": values["macro_sum_rt_all"]
                            / values["molecules_rt_all"]
                            * 100,
                            "molecules": values["molecules_rt_all"],
                        }
                    }
                    if values.get("molecules_rt_all")
                    else {}
                ),
            }
            for condition, values in ensembles.items()
        },
        "anchors_rt": _anchors_rt_totals(args.anchors_rt_pb_dir)
        if args.anchors_rt_pb_dir is not None
        else None,
        "per_check_interpolants": {
            condition: {
                check: {
                    **bucket,
                    "pct_before": bucket["valid_before"] / bucket["n"] * 100,
                    "pct_after": bucket["valid_after"] / bucket["n"] * 100,
                    "pct_gated": bucket["valid_gated"] / bucket["n"] * 100,
                }
                for check, bucket in sorted(per_check.items())
            }
            for condition, per_check in checks.items()
        },
    }
    (args.gate_pb_dir / "gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"molecules: {molecules}  (anchor counts missing for {missing_anchors})")
    print()
    print("Interpolants only — PoseBusters valid (%)")
    print("| Condition | before | refine-all | gated | refined/n | tests fixed/broken |")
    print("|---|---:|---:|---:|---:|---|")
    for condition, values in summary["interpolants_only"].items():
        print(
            f"| {condition} | {values['pct_before']:.2f} | {values['pct_after']:.2f} | "
            f"{values['pct_gated']:.2f} | {values['refined']:,}/{values['n']:,} | "
            f"{values['fixed']:,}/{values['broken']:,} |"
        )
    print()
    if summary["anchors_rt"]:
        control = summary["anchors_rt"]
        print(
            f"anchors_rt (round trip, no interpolation): "
            f"PB micro {control['pb_valid_pct']:.2f}, "
            f"macro {control['pb_valid_macro_pct']:.2f}, "
            f"over {control['n_conformers']:,} conformers in {control['molecules']} molecules"
        )
        print()
    print("Full ensemble (original anchors + interpolants under each policy)")
    print("| Condition | PB micro: before → refine-all → gated | PB macro: same | rt_all micro | conformers | molecules |")
    print("|---|---|---|---|---:|---:|")
    for condition, values in summary["full_ensemble"].items():
        micro = " → ".join(f"{values[p]['pb_valid_pct']:.2f}" for p in POLICIES)
        macro = " → ".join(f"{values[p]['pb_valid_macro_pct']:.2f}" for p in POLICIES)
        rt_all = (
            f"{values['rt_all']['pb_valid_pct']:.2f}" if "rt_all" in values else "–"
        )
        print(
            f"| {condition} | {micro} | {macro} | {rt_all} | "
            f"{values['n_conformers']:,} | {values['molecules']} |"
        )


if __name__ == "__main__":
    main()
