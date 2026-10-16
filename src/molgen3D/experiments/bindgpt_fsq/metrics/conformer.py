"""Evaluation 2: the regular GEOM-DRUGS conformer protocol.

Reuses the project's already-validated evaluator -- symmetry-aware RDKit
best-alignment RMSD, heavy atoms only, strict ``<`` threshold -- and reports
Recall/Precision COV and AMR at the single headline threshold of 0.75 A. No
extra threshold grid, no coverage curve, no AUC.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Geometry import Point3D

from molgen3D.evaluation import rdkit_utils
from molgen3D.evaluation.utils import covmat_metrics
from molgen3D.experiments.bindgpt_fsq import config as experiment_config

RDLogger.DisableLog("rdApp.*")

HEADLINE_THRESHOLD = 0.75


def load_generated_conformers(
    attempts_paths: Sequence[str | Path],
) -> tuple[Dict[str, List[Chem.Mol]], Dict[str, object]]:
    """Rebuild accepted conformers per test molecule from saved raw attempts."""
    per_molecule: Dict[str, List[Chem.Mol]] = {}
    attempts = 0
    accepted = 0
    failures: Counter = Counter()

    for path in attempts_paths:
        with open(path) as handle:
            for line in handle:
                payload = json.loads(line)
                attempts += 1
                if not payload.get("accepted"):
                    failures[payload.get("failure_reason") or "other"] += 1
                    continue
                coordinates = payload.get("coordinates")
                smiles = payload.get("right_smiles")
                if not coordinates or not smiles:
                    failures["other"] += 1
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None or mol.GetNumAtoms() != len(coordinates):
                    failures["atom_count_mismatch"] += 1
                    continue
                conformer = Chem.Conformer(mol.GetNumAtoms())
                for index, (x, y, z) in enumerate(coordinates):
                    conformer.SetAtomPosition(
                        index, Point3D(float(x), float(y), float(z))
                    )
                mol.RemoveAllConformers()
                mol.AddConformer(conformer, assignId=True)
                per_molecule.setdefault(str(payload["key"]), []).append(mol)
                accepted += 1

    diagnostics = {
        "attempts": attempts,
        "accepted": accepted,
        "decode_yield": accepted / max(1, attempts),
        "failure_reasons": dict(failures),
        "molecules_with_conformers": len(per_molecule),
    }
    return per_molecule, diagnostics


def load_references() -> Dict[str, Dict[str, object]]:
    """Load the frozen test molecules and their reference conformers."""
    import cloudpickle

    with open(experiment_config.BENCHMARK_TEST_PICKLE, "rb") as handle:
        return cloudpickle.load(handle)


def _rmsd_matrix(
    references: Sequence[Chem.Mol],
    generated: Sequence[Chem.Mol],
    use_alignmol: bool = False,
) -> np.ndarray:
    matrix = np.full((len(references), len(generated)), np.nan, dtype=float)
    for row, reference in enumerate(references):
        normalized = rdkit_utils._normalize_coords(reference)
        matrix[row] = [
            rdkit_utils._best_rmsd(candidate, normalized, use_alignmol)
            for candidate in generated
        ]
    return matrix


#: Cap on RMSD pairs per work unit. Parallelising per molecule is not enough:
#: the test set is heavy-tailed (one molecule has 2,497 references, which alone
#: is 14.4% of all 86.3M pairs), so a whole-molecule task pins one worker for
#: hours while the rest idle. Reference conformers are therefore split into row
#: blocks and the matrix is reassembled.
PAIRS_PER_TASK = 200_000


def _score_block(task: tuple) -> tuple:
    """Compute one row block of a molecule's RMSD matrix."""
    key, block_index, reference_block, candidates, use_alignmol = task
    return key, block_index, _rmsd_matrix(reference_block, candidates, use_alignmol)


def _summarize_molecule(
    key: str,
    reference_confs: Sequence[Chem.Mol],
    candidates: Sequence[Chem.Mol],
    matrix: np.ndarray,
    threshold: float,
) -> Optional[Dict[str, object]]:
    """Turn a complete RMSD matrix into the per-molecule row."""
    if np.isnan(matrix).all():
        return None
    cov_r, amr_r, cov_p, amr_p = covmat_metrics(
        matrix, np.asarray([threshold], dtype=float)
    )
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RemoveHs(reference_confs[0])))
    return {
        "key": key,
        "n_references": len(reference_confs),
        "n_generated": len(candidates),
        "cov_r": float(cov_r[0]),
        "amr_r": float(amr_r),
        "cov_p": float(cov_p[0]),
        "amr_p": float(amr_p),
        "heavy_atoms": mol.GetNumHeavyAtoms() if mol else None,
        "rotatable_bonds": (
            rdMolDescriptors.CalcNumRotatableBonds(mol) if mol else None
        ),
        "rings": mol.GetRingInfo().NumRings() if mol else None,
    }


def evaluate(
    generated: Dict[str, List[Chem.Mol]],
    threshold: float = HEADLINE_THRESHOLD,
    use_alignmol: bool = False,
    keys: Optional[Sequence[str]] = None,
    num_workers: int = 0,
    pairs_per_task: int = PAIRS_PER_TASK,
) -> Dict[str, object]:
    """Per-molecule and aggregate COV/AMR at one threshold.

    Work is split into row blocks capped at ``pairs_per_task`` RMSD pairs, so a
    single huge molecule cannot serialise the whole run. Blocks are dispatched
    largest-molecule-first so the long tail starts early rather than last.
    """
    references = load_references()
    rows: List[Dict[str, object]] = []
    missing: List[str] = []

    molecules: Dict[str, tuple] = {}
    for key in keys if keys is not None else sorted(references):
        entry = references.get(key)
        if entry is None:
            continue
        candidates = generated.get(key) or []
        if not candidates:
            missing.append(key)
            continue
        reference_confs = rdkit_utils.process_molecules_remove_hs(
            {key: entry["confs"]}
        )[key]
        molecules[key] = (reference_confs, candidates)

    tasks: List[tuple] = []
    for key, (reference_confs, candidates) in molecules.items():
        rows_per_block = max(1, pairs_per_task // max(1, len(candidates)))
        for block_index, offset in enumerate(
            range(0, len(reference_confs), rows_per_block)
        ):
            tasks.append(
                (
                    key,
                    block_index,
                    reference_confs[offset : offset + rows_per_block],
                    candidates,
                    use_alignmol,
                )
            )
    # Longest-processing-time-first: start the biggest molecules immediately.
    tasks.sort(key=lambda t: -len(molecules[t[0]][0]))

    workers = num_workers or max(1, min(64, os.cpu_count() or 4))
    if workers == 1 or len(tasks) <= 1:
        results = [_score_block(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_score_block, tasks, chunksize=1))

    blocks: Dict[str, List[tuple]] = {}
    for key, block_index, block in results:
        blocks.setdefault(key, []).append((block_index, block))

    for key, (reference_confs, candidates) in molecules.items():
        parts = sorted(blocks.get(key, []), key=lambda item: item[0])
        if not parts:
            missing.append(key)
            continue
        matrix = np.vstack([block for _, block in parts])
        if matrix.shape[0] != len(reference_confs):
            raise RuntimeError(
                f"reassembled {matrix.shape[0]} reference rows for {key}, "
                f"expected {len(reference_confs)}"
            )
        row = _summarize_molecule(
            key, reference_confs, candidates, matrix, threshold
        )
        if row is None:
            missing.append(key)
        else:
            rows.append(row)
    rows.sort(key=lambda row: str(row["key"]))

    def aggregate(field: str) -> Dict[str, float]:
        values = np.asarray([row[field] for row in rows], dtype=float)
        return {
            "mean": float(np.nanmean(values)) if values.size else float("nan"),
            "median": float(np.nanmedian(values)) if values.size else float("nan"),
        }

    return {
        "threshold": threshold,
        "threshold_convention": "strict less-than, project evaluator",
        "workers": workers,
        "work_units": len(tasks),
        "molecules_scored": len(rows),
        "molecules_missing_generations": len(missing),
        "missing_keys": sorted(missing)[:50],
        "cov_r": aggregate("cov_r"),
        "amr_r": aggregate("amr_r"),
        "cov_p": aggregate("cov_p"),
        "amr_p": aggregate("amr_p"),
        "per_molecule": rows,
    }


def posebusters(
    generated: Dict[str, List[Chem.Mol]],
    num_workers: int = 8,
    config: str = "mol",
) -> Dict[str, object]:
    """PoseBusters validity over the generated conformers.

    Uses the project's own ``bust_full_gens`` so the number is the same PB score
    the existing Qwen FSQ evaluations report, on the same test molecules.
    """
    from molgen3D.evaluation.posebusters_check import bust_full_gens

    if not generated:
        return {"pass_rate": None, "molecules": 0, "conformers": 0}

    by_smiles, summary, pass_rate = bust_full_gens(
        smiles_to_confs=generated,
        num_workers=num_workers,
        config=config,
        full_report=False,
    )
    payload: Dict[str, object] = {
        "pass_rate": None if pass_rate is None else float(pass_rate),
        "molecules": len(generated),
        "conformers": int(sum(len(v) for v in generated.values())),
        "config": config,
    }
    try:
        payload["per_check"] = {
            str(k): float(v) for k, v in summary.iloc[0].items()
            if isinstance(v, (int, float))
        }
    except Exception:  # noqa: BLE001 - summary shape is advisory only
        payload["per_check"] = {}
    payload["_by_smiles"] = by_smiles
    return payload


def posebusters_sharded(
    generated: Dict[str, List[Chem.Mol]],
    out_dir: str | Path,
    num_workers: int = 8,
    config: str = "mol",
    molecules_per_shard: int = 25,
    shard_indices: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """PoseBusters over molecule shards, persisting each shard as it completes.

    Results are written incrementally rather than at the end, so a walltime kill
    loses at most one shard instead of the whole run, and a resubmission resumes
    from what is already on disk. ``pass_rate`` is a conformer-level mean
    upstream, so shards merge weighted by conformer count.
    """
    from molgen3D.evaluation.posebusters_check import bust_full_gens

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(generated)
    shards = [
        keys[i : i + molecules_per_shard]
        for i in range(0, len(keys), molecules_per_shard)
    ]

    wanted = None if shard_indices is None else set(int(i) for i in shard_indices)
    completed: List[Dict[str, object]] = []
    for index, shard_keys in enumerate(shards):
        path = out_dir / f"shard_{index:04d}.json"
        if path.exists():
            existing = json.loads(path.read_text())
            completed.append(existing)
            _write_partial(out_dir, completed, len(shards))
            continue
        if wanted is not None and index not in wanted:
            continue  # another array task owns this shard

        subset = {key: generated[key] for key in shard_keys if generated.get(key)}
        if not subset:
            continue
        # bust_full_gens splits work into task_chunk_size-conformer chunks, so a
        # shard must contain enough chunks to fill every worker. Its 600 default
        # against a ~4,600-conformer shard yields only ~8 chunks, which caps
        # parallelism at 8 regardless of num_workers. Size chunks to give each
        # worker several, so the pool is saturated and load-balanced.
        shard_conformers = sum(len(v) for v in subset.values())
        chunk = max(1, shard_conformers // max(1, num_workers * 4))
        by_smiles, _summary, pass_rate = bust_full_gens(
            smiles_to_confs=subset,
            num_workers=num_workers,
            config=config,
            full_report=False,
            log_progress=False,
            task_chunk_size=chunk,
        )
        record = {
            "shard": index,
            "molecules": len(subset),
            "conformers": int(sum(len(v) for v in subset.values())),
            "pass_rate": None if pass_rate is None else float(pass_rate),
        }
        # Write the shard before anything else can fail.
        path.write_text(json.dumps(record, indent=1))
        try:
            by_smiles.to_csv(out_dir / f"shard_{index:04d}_by_smiles.csv", index=False)
        except Exception:  # noqa: BLE001 - the JSON is the durable record
            pass
        completed.append(record)
        _write_partial(out_dir, completed, len(shards))
        print(
            f"[posebusters] shard {index + 1}/{len(shards)} "
            f"pass_rate={record['pass_rate']} "
            f"conformers={record['conformers']}",
            flush=True,
        )

    # Re-read from disk so an array task's aggregate includes shards its
    # siblings finished while it was working.
    on_disk = [
        json.loads(p.read_text()) for p in sorted(out_dir.glob("shard_*.json"))
    ]
    merged = _merge_shards(on_disk, len(shards))
    _write_partial(out_dir, on_disk, len(shards))
    return merged


def _merge_shards(records: Sequence[Dict[str, object]], total_shards: int) -> Dict[str, object]:
    conformers = sum(int(r["conformers"]) for r in records)
    molecules = sum(int(r["molecules"]) for r in records)
    weighted = sum(
        float(r["pass_rate"]) * int(r["conformers"])
        for r in records
        if r.get("pass_rate") is not None
    )
    counted = sum(
        int(r["conformers"]) for r in records if r.get("pass_rate") is not None
    )
    return {
        "pass_rate": (weighted / counted) if counted else None,
        "molecules": molecules,
        "conformers": conformers,
        "shards_done": len(records),
        "shards_total": total_shards,
        "complete": len(records) == total_shards,
    }


def _write_partial(out_dir: Path, records: Sequence[Dict[str, object]], total: int) -> None:
    """Keep a running aggregate on disk so partial results are always usable."""
    (out_dir / "partial.json").write_text(
        json.dumps(_merge_shards(records, total), indent=1)
    )


def stratify(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Break the per-molecule table down by the four requested covariates."""
    definitions = {
        "heavy_atoms": [(0, 20), (20, 25), (25, 30), (30, 10_000)],
        "rotatable_bonds": [(0, 3), (3, 6), (6, 9), (9, 10_000)],
        "n_references": [(0, 20), (20, 50), (50, 100), (100, 100_000)],
        "rings": [(0, 2), (2, 3), (3, 4), (4, 10_000)],
    }
    output: Dict[str, object] = {}
    for field, edges in definitions.items():
        buckets = []
        for low, high in edges:
            subset = [
                row
                for row in rows
                if row.get(field) is not None and low <= float(row[field]) < high
            ]
            if not subset:
                continue
            buckets.append(
                {
                    "range": [low, high],
                    "molecules": len(subset),
                    **{
                        metric: float(
                            np.nanmean([float(row[metric]) for row in subset])
                        )
                        for metric in ("cov_r", "amr_r", "cov_p", "amr_p")
                    },
                }
            )
        output[field] = buckets
    return output


def paired_bootstrap(
    left_rows: Sequence[Dict[str, object]],
    right_rows: Sequence[Dict[str, object]],
    metric: str = "cov_r",
    resamples: int = 10_000,
    seed: int = 17,
) -> Dict[str, object]:
    """Paired bootstrap over the test molecules shared by both arms."""
    left = {str(row["key"]): float(row[metric]) for row in left_rows}
    right = {str(row["key"]): float(row[metric]) for row in right_rows}
    shared = sorted(set(left) & set(right))
    if not shared:
        return {"metric": metric, "paired_molecules": 0}

    differences = np.asarray([right[key] - left[key] for key in shared])
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, differences.size, size=(resamples, differences.size))
    means = differences[draws].mean(axis=1)
    return {
        "metric": metric,
        "paired_molecules": len(shared),
        "mean_difference": float(differences.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "resamples": resamples,
        "direction": "right minus left",
    }


def calibrate(archived_predictions: str | Path, expected: Dict[str, float]) -> Dict[str, object]:
    """Re-score an archived model's conformers and compare with known values.

    The project's evaluator has already been shown to reproduce four published
    models on their released conformers; this replays one of them so the
    calibration is recorded inside this experiment's artifacts too.
    """
    import pickle

    with open(archived_predictions, "rb") as handle:
        payload = pickle.load(handle)
    result = evaluate(payload)
    observed = {
        "cov_r_mean": result["cov_r"]["mean"],
        "amr_r_mean": result["amr_r"]["mean"],
        "cov_p_mean": result["cov_p"]["mean"],
        "amr_p_mean": result["amr_p"]["mean"],
    }
    return {
        "archive": str(archived_predictions),
        "expected": expected,
        "observed": observed,
        "absolute_difference": {
            key: abs(observed[key] - value)
            for key, value in expected.items()
            if key in observed
        },
    }
