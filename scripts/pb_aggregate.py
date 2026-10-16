#!/usr/bin/env python3
"""Build a PoseBusters summary from whatever per-molecule outputs exist.

The sharded scorer only writes `summary.json` when the last shard finishes, so a
single slow molecule holds the whole round hostage. Every molecule that *has*
been scored is already on disk, so aggregate those and record the coverage.

Micro pools conformers; macro gives every molecule equal weight — the same
arithmetic setup3_posebusters.py does.

Usage:
  pb_aggregate.py <pb_dir> [<pb_dir> ...] [--expect 1000] [--write]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def aggregate(pb_dir: Path, expect: int | None) -> dict:
    agg: dict[str, dict] = {}
    mols = 0
    errors = 0
    for path in pb_dir.glob("*.json"):
        if path.name.startswith("summary"):
            continue
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("status") != "complete":
            continue
        mols += 1
        for c, v in d.get("conditions", {}).items():
            if not v.get("n") or "pass" not in v:
                errors += "error" in v
                continue
            a = agg.setdefault(c, {"n": 0, "all": 0, "checks": {}, "fractions": []})
            a["n"] += v["n"]
            a["all"] += v["pass"]["__all__"]
            a["fractions"].append(v["pass"]["__all__"] / v["n"])
            for k, n in v["pass"].items():
                if k != "__all__":
                    a["checks"][k] = a["checks"].get(k, 0) + n
    summary = {
        "run": pb_dir.name,
        "molecules": mols,
        "condition_errors": errors,
        "missing_molecule_outputs": (expect - mols) if expect else None,
        "partial": bool(expect and mols < expect),
        "conditions": {
            c: {
                "n_conformers": v["n"],
                "molecules": len(v["fractions"]),
                "pb_valid_pct": v["all"] / v["n"] * 100,
                "pb_valid_macro_pct": sum(v["fractions"]) / len(v["fractions"]) * 100,
                "per_check_pct": {k: n / v["n"] * 100 for k, n in sorted(v["checks"].items())},
            }
            for c, v in agg.items()
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pb_dirs", nargs="+", type=Path)
    parser.add_argument("--expect", type=int, default=1000)
    parser.add_argument("--write", action="store_true", help="write summary.json into each directory")
    args = parser.parse_args()

    for pb_dir in args.pb_dirs:
        if not pb_dir.exists():
            continue
        s = aggregate(pb_dir, args.expect)
        if not s["conditions"]:
            print(f"{pb_dir.name:26s} no scored molecules yet")
            continue
        after = s["conditions"].get("after") or next(iter(s["conditions"].values()))
        flag = "" if not s["partial"] else f"  (partial: {s['molecules']}/{args.expect})"
        print(
            f"{pb_dir.name:26s} molecules {s['molecules']:4d}  "
            f"conformers {after['n_conformers']:>8,}  "
            f"micro {after['pb_valid_pct']:6.2f}  macro {after['pb_valid_macro_pct']:6.2f}{flag}"
        )
        if args.write:
            target = pb_dir / "summary.json"
            if target.exists():
                existing = json.loads(target.read_text())
                if not existing.get("partial") and existing.get("molecules", 0) >= s["molecules"]:
                    continue  # a complete summary already stands
            target.write_text(json.dumps(s, indent=2) + "\n")


if __name__ == "__main__":
    main()
