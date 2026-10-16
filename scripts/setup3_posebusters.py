#!/usr/bin/env python3
"""PoseBusters validity for every Setup 3 condition.

These are free conformers with no protein frame, so PoseBusters runs in
molecule-only config — the same 12 checks used for Platinum Diverse, none of
which need a reference or a pocket.

Per molecule and per condition we bust every conformer and store the per-check
pass counts, which is all the aggregation needs. Conditions are whatever the
record holds. Results are written per molecule so the run resumes, and
``--shard`` lets several jobs split one run directory without racing.

Usage: setup3_posebusters.py <run_dir> <out_dir> [workers] [--shard i --num-shards n]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

_BUSTER = None


def buster():
    global _BUSTER
    if _BUSTER is None:
        from posebusters import PoseBusters

        _BUSTER = PoseBusters(config="mol")
    return _BUSTER


# PoseBusters' energy-ratio check can spin for hours on a single molecule: it
# retries ensemble generation per conformer and some topologies never succeed.
# One thiadiazole/naphthalene with 18 conformers has burned >2h in three separate
# runs. Each worker is its own process, so SIGALRM caps a molecule and lets the
# shard move on with `error: timeout` recorded instead of the flags.
PB_TIMEOUT = int(os.environ.get("PB_MOLECULE_TIMEOUT", "600"))
PB_PER_CONFORMER = float(os.environ.get("PB_SECONDS_PER_CONFORMER", "2"))


class _Timeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _Timeout()


def one_molecule(args) -> str | None:
    record_path, out_path, only = args
    out = Path(out_path)
    if out.exists():
        return out_path
    with Path(record_path).open("rb") as handle:
        record = pickle.load(handle)
    if record.get("status") != "complete":
        out.write_text(json.dumps({"status": record.get("status")}) + "\n")
        return out_path

    result = {"status": "complete", "molecule_key": record["molecule_key"], "conditions": {}}
    for condition, stored in record["conformers"].items():
        if only and condition not in only:
            continue
        mols = stored or []
        if not mols:
            result["conditions"][condition] = {"n": 0}
            continue
        try:
            # The alarm covers one bust() call over the whole ensemble, so the
            # budget has to scale with it: a 3,000-conformer molecule needs far
            # more than a 20-conformer one. Flat 600s silently dropped the
            # largest ensembles.
            budget = max(PB_TIMEOUT, int(PB_PER_CONFORMER * len(mols))) if PB_TIMEOUT > 0 else 0
            if budget > 0:
                signal.signal(signal.SIGALRM, _raise_timeout)
                signal.alarm(budget)
            try:
                table = buster().bust(list(mols), None, None)
            finally:
                if budget > 0:
                    signal.alarm(0)
        except _Timeout:
            result["conditions"][condition] = {
                "n": len(mols),
                "error": f"timeout after {budget}s",
            }
            continue
        except Exception as exc:  # keep going; record the failure
            result["conditions"][condition] = {"n": len(mols), "error": repr(exc)[:200]}
            continue
        checks = [c for c in table.columns]
        flags = table[checks].fillna(False).astype(bool)
        passes = {c: int(flags[c].sum()) for c in checks}
        passes["__all__"] = int(flags.all(axis=1).sum())
        entry = {"n": int(len(table)), "pass": passes}
        if len(table) != len(mols):
            # Flags are only usable if each row still maps to its input conformer.
            entry["error"] = f"posebusters returned {len(table)} rows for {len(mols)} conformers"
        else:
            columns = {c: flags[c] for c in checks}
            columns["__all__"] = flags.all(axis=1)
            entry["flags"] = {
                name: "".join("1" if value else "0" for value in series)
                for name, series in columns.items()
            }
        result["conditions"][condition] = entry
    out.write_text(json.dumps(result) + "\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("workers", type=int, nargs="?", default=32)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--conditions",
        nargs="+",
        help="bust only these conditions (default: every condition in the record)",
    )
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error("--shard must be in [0, --num-shards)")
    run_dir, out_dir, workers = args.run_dir, args.out_dir, args.workers
    out_dir.mkdir(parents=True, exist_ok=True)

    only = frozenset(args.conditions) if args.conditions else frozenset()
    jobs = []
    for p in sorted((run_dir / "molecules").glob("*.pickle")):
        jobs.append((str(p), str(out_dir / (p.stem + ".json")), only))
    mine = jobs[args.shard :: args.num_shards]
    todo = [j for j in mine if not Path(j[1]).exists()]
    print(
        f"{run_dir.name}: {len(jobs)} molecules, shard {args.shard}/{args.num_shards} "
        f"holds {len(mine)}, {len(todo)} to do, {workers} workers",
        flush=True,
    )

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one_molecule, j) for j in todo]
        for f in as_completed(futures):
            f.result()
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}", flush=True)

    # aggregate: micro pools conformers, macro gives every molecule equal weight
    agg: dict[str, dict] = {}
    mols = 0
    errors = 0
    missing = 0
    for _, out_path, _only in jobs:
        if not Path(out_path).exists():
            # Another shard has not caught up yet; its summary will be the complete one.
            missing += 1
            continue
        d = json.loads(Path(out_path).read_text())
        if d.get("status") != "complete":
            continue
        mols += 1
        for c, v in d["conditions"].items():
            if not v.get("n") or "pass" not in v:
                errors += "error" in v
                continue
            agg.setdefault(c, {"n": 0, "all": 0, "checks": {}, "fractions": []})
            agg[c]["n"] += v["n"]
            agg[c]["all"] += v["pass"]["__all__"]
            agg[c]["fractions"].append(v["pass"]["__all__"] / v["n"])
            for k, n in v["pass"].items():
                if k == "__all__":
                    continue
                agg[c]["checks"][k] = agg[c]["checks"].get(k, 0) + n
    summary = {
        "run": run_dir.name,
        "molecules": mols,
        "condition_errors": errors,
        "missing_molecule_outputs": missing,
        "conditions": {},
    }
    for c, v in agg.items():
        if not v["n"]:
            continue
        summary["conditions"][c] = {
            "n_conformers": v["n"],
            "molecules": len(v["fractions"]),
            "pb_valid_pct": v["all"] / v["n"] * 100,
            "pb_valid_macro_pct": sum(v["fractions"]) / len(v["fractions"]) * 100,
            "per_check_pct": {k: n / v["n"] * 100 for k, n in sorted(v["checks"].items())},
        }
    name = "summary.json" if not missing else f"summary_partial_shard{args.shard}.json"
    (out_dir / name).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({c: round(x["pb_valid_pct"], 2) for c, x in summary["conditions"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
