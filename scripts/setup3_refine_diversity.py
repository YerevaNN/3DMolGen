#!/usr/bin/env python3
"""Is the refinement pass's coverage gain hollow?

Refinement pulls conformers toward the tokenizer's manifold, so a coverage gain
would mean little if the interpolants simply collapsed — onto each other, or
onto the anchors they were built from, which are already in the ensemble. Three
measures per condition, before and after the pass:

``duplicate``   fraction of interpolants matching an earlier one (near-exact)
``spread``      mean pairwise best RMSD among the interpolants
``to_anchor``   mean RMSD from each interpolant to its nearest anchor

A collapse onto the anchors would show as ``to_anchor`` falling toward zero; a
collapse onto each other as ``spread`` falling.

Usage:
  setup3_refine_diversity.py <source_dir> <refine_dir> [--sample N] [--workers N]
                             [--threshold 0.1] [--min-k 20] [--max-k 200]
"""

from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from rdkit import Chem

from molgen3D.evaluation import rdkit_utils
from molgen3D.evaluation.conformer_interpolation import duplicate_rate

CONDITIONS = ("latent", "random_token", "cartesian")


def _mean_pairwise_rmsd(molecules, *, limit: int = 40) -> float:
    """Mean best RMSD over all pairs of the first `limit` conformers."""
    sample = [Chem.RemoveHs(molecule) for molecule in molecules[:limit]]
    values = [
        rdkit_utils._best_rmsd(sample[i], sample[j], use_alignmol=False)
        for i in range(len(sample))
        for j in range(i + 1, len(sample))
    ]
    return float(np.nanmean(values)) if values else float("nan")


def _mean_nearest_anchor_rmsd(molecules, anchors, *, limit: int = 40) -> float:
    """Mean over interpolants of the RMSD to their nearest anchor."""
    probes = [Chem.RemoveHs(molecule) for molecule in molecules[:limit]]
    references = [Chem.RemoveHs(molecule) for molecule in anchors]
    nearest = [
        np.nanmin(
            [
                rdkit_utils._best_rmsd(probe, reference, use_alignmol=False)
                for reference in references
            ]
        )
        for probe in probes
    ]
    return float(np.nanmean(nearest)) if nearest else float("nan")


def _one_molecule(args) -> dict | None:
    source_path, refine_path, threshold = args
    with Path(source_path).open("rb") as handle:
        source = pickle.load(handle)
    with Path(refine_path).open("rb") as handle:
        refined = pickle.load(handle)
    if source.get("status") != "complete" or refined.get("status") != "complete":
        return None
    n_anchor = source["n_anchor"]
    anchors = list(source["conformers"]["anchors"])
    row = {"molecule_key": source["molecule_key"], "k": source["k"], "n_anchor": n_anchor}
    for condition in CONDITIONS:
        stored = source["conformers"].get(condition)
        after = refined["conformers"].get(f"{condition}_rt_all")
        if not stored or not after:
            continue
        before_interpolants = list(stored[n_anchor:])
        after_interpolants = list(after[n_anchor:])
        row[f"{condition}_duplicate_before"] = duplicate_rate(
            before_interpolants, rmsd_threshold=threshold
        )
        row[f"{condition}_duplicate_after"] = duplicate_rate(
            after_interpolants, rmsd_threshold=threshold
        )
        row[f"{condition}_spread_before"] = _mean_pairwise_rmsd(before_interpolants)
        row[f"{condition}_spread_after"] = _mean_pairwise_rmsd(after_interpolants)
        row[f"{condition}_to_anchor_before"] = _mean_nearest_anchor_rmsd(
            before_interpolants, anchors
        )
        row[f"{condition}_to_anchor_after"] = _mean_nearest_anchor_rmsd(
            after_interpolants, anchors
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("refine_dir", type=Path)
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--min-k", type=int, default=20)
    parser.add_argument("--max-k", type=int, default=200)
    args = parser.parse_args()

    jobs = []
    for source_path in sorted((args.source_dir / "molecules").glob("*.pickle")):
        refine_path = args.refine_dir / "molecules" / source_path.name
        if not refine_path.exists():
            continue
        with source_path.open("rb") as handle:
            record = pickle.load(handle)
        if record.get("status") != "complete":
            continue
        if not args.min_k <= record["k"] <= args.max_k:
            continue
        jobs.append((str(source_path), str(refine_path), args.threshold))
        if len(jobs) == args.sample:
            break
    print(f"{len(jobs)} molecules, k in [{args.min_k}, {args.max_k}]", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_one_molecule, job) for job in jobs]
        for position, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            if row is not None:
                rows.append(row)
            if position % 10 == 0 or position == len(futures):
                print(f"  {position}/{len(futures)}", flush=True)

    summary = {"molecules": len(rows), "duplicate_threshold": args.threshold, "measures": {}}
    for measure in ("duplicate", "spread", "to_anchor"):
        summary["measures"][measure] = {}
        for condition in CONDITIONS:
            for phase in ("before", "after"):
                key = f"{condition}_{measure}_{phase}"
                values = [row[key] for row in rows if key in row]
                if values:
                    summary["measures"][measure][f"{condition}_{phase}"] = float(
                        np.nanmean(values)
                    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
