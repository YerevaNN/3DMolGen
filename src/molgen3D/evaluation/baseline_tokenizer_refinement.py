#!/usr/bin/env python3
"""One tokenizer encode/decode pass over an external baseline's conformers.

Section 4 applied a single CoordToken pass to docked poses and to Platinum
Diverse ensembles. This does the same for other GEOM conformer generators —
Torsional Diffusion, MCF, ETFlow — so their validity can be compared before and
after on the same footing as our own model.

Input is a pickle mapping a molecule key to a list of single-conformer ``Mol``
objects, which is the shape all four baselines ship. Two conditions are stored
per molecule:

``before_with_h``  the conformers exactly as the baseline shipped them
``before``         the same conformers with hydrogens removed
``after``          the hydrogen-free conformers after one encode/decode pass

The tokenizer works on heavy atoms only, so ``after`` can never carry
hydrogens, and ``before`` is the matched basis for a paired comparison against
it. ``before_with_h`` is kept as well so the published baseline numbers, which
were computed on the shipped conformers, can be reproduced exactly rather than
approximately. On a 406-conformer sample the two conventions differ by 0.25
points of PB-valid.

Records are written in the same shape the Setup 3 PoseBusters script consumes,
so validity is scored with ``scripts/setup3_posebusters.py`` unchanged.

Four phases, because only one of them needs an H100: ``prepare`` writes the two
``before`` conditions, ``refine`` adds ``after``, ``gate`` adds ``gated`` from
per-conformer PoseBusters flags, and ``evaluate`` scores COV/MAT against the
GEOM references. Splitting them lets the baseline validity be scored while the
GPU half is still queued.

``gated`` keeps a conformer's original coordinates where the original already
passed PoseBusters and takes the refined ones otherwise. COV/MAT is not computed
for ``before_with_h``: the RMSD path removes hydrogens anyway, so it would be
identical to ``before``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

import numpy as np

from molgen3D.evaluation.conformer_interpolation import (
    _atomic_json_dump,
    _atomic_pickle_dump,
    _extract_conformers,
    _rmsd_matrix,
)
from molgen3D.evaluation.setup3_tokenizer_refinement import (
    gate_interpolants,
    refine_molecules,
)
from molgen3D.evaluation.utils import DEFAULT_THRESHOLDS, covmat_metrics

METRIC_CONDITIONS = ("before", "after", "gated")
MAIN_THRESHOLD = 0.75

RDLogger.DisableLog("rdApp.*")


def _record_name(molecule_key: str) -> str:
    return hashlib.sha256(molecule_key.encode()).hexdigest() + ".pickle"


def _strip_hydrogens(molecules) -> list[Chem.Mol]:
    out = []
    for molecule in molecules:
        try:
            out.append(Chem.RemoveHs(molecule))
        except Exception:
            out.append(molecule)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    (output_dir / "molecules").mkdir(parents=True, exist_ok=True)

    # Only the phases that touch the source data need to read it; `gate` and
    # `evaluate` work from the per-molecule records alone.
    needs_input = args.phase in ("prepare", "refine")
    data: dict = {}
    keys: list = []
    if needs_input:
        with Path(args.input).open("rb") as handle:
            data = pickle.load(handle)
        if not isinstance(data, dict):
            raise TypeError(f"Expected a dict of key -> [Mol], got {type(data)!r}")
        keys = list(data)
        if args.max_molecules is not None:
            keys = keys[: args.max_molecules]

    config = {
        "method": args.method,
        "input": str(Path(args.input).expanduser().resolve()) if args.input else None,
        "fsq_checkpoint": str(args.fsq_checkpoint),
        "max_tokens": args.max_tokens,
        "molecules": len(keys) if needs_input else None,
        "conformers": sum(len(data[key]) for key in keys) if needs_input else None,
        "conditions": {
            "before_with_h": "conformers as shipped by the baseline",
            "before": "the same conformers with hydrogens removed",
            "after": "one FSQ encode/decode pass over the hydrogen-free conformers",
        },
    }
    if needs_input:
        _atomic_json_dump(config, output_dir / "config.json")
        print(
            f"{args.method}: {config['molecules']} molecules, "
            f"{config['conformers']:,} conformers, phase {args.phase}",
            flush=True,
        )
    else:
        print(f"{args.method}: phase {args.phase}", flush=True)

    if args.phase == "gate":
        touched = add_gated(
            output_dir,
            Path(args.pb_before_dir).expanduser().resolve(),
            Path(args.pb_after_dir).expanduser().resolve(),
            args.gate_basis,
        )
        print(f"gated added for {touched} molecules", flush=True)
        return {"config": config, "phase": "gate", "molecules": touched}

    if args.phase == "evaluate":
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with Path(args.ground_truth).open("rb") as handle:
            ground_truth = pickle.load(handle)
        jobs = []
        for path in sorted((output_dir / "molecules").glob("*.pickle")):
            with path.open("rb") as handle:
                record = pickle.load(handle)
            key = record.get("molecule_key")
            if key not in ground_truth:
                continue
            if all(
                c not in record["conformers"] or c in record.get("metrics", {})
                for c in METRIC_CONDITIONS
            ):
                continue
            jobs.append(
                (str(path), _extract_conformers(ground_truth[key]), METRIC_CONDITIONS)
            )
        print(f"evaluating {len(jobs)} molecules with {args.num_workers} workers", flush=True)
        if jobs:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(_evaluate_one, job) for job in jobs]
                for position, future in enumerate(as_completed(futures), start=1):
                    future.result()
                    if position % 50 == 0 or position == len(futures):
                        print(f"evaluated {position}/{len(futures)}", flush=True)
        summary = summarize(output_dir)
        print(json.dumps(summary["conditions"], indent=2), flush=True)
        return summary

    if args.phase == "prepare":
        for position, key in enumerate(keys, start=1):
            path = output_dir / "molecules" / _record_name(key)
            if args.resume and path.exists():
                continue
            original = list(data[key])
            _atomic_pickle_dump(
                {
                    "molecule_key": key,
                    "status": "complete",
                    "method": args.method,
                    "n_conformers": len(original),
                    "conformers": {
                        "before_with_h": original,
                        "before": _strip_hydrogens(original),
                    },
                },
                path,
            )
            if position % args.log_every == 0 or position == len(keys):
                print(f"prepared {position}/{len(keys)}", flush=True)
        return {"config": config, "phase": "prepare"}

    from molgen3D.data_processing.smiles_encoder_decoder_fsq import configure_fsq_encoder

    model = configure_fsq_encoder(
        ckpt_path=args.fsq_checkpoint, device=args.device, max_tokens=args.max_tokens
    )

    totals = {"n": 0, "encode_failures": 0, "decode_failures": 0}
    for position, key in enumerate(keys, start=1):
        path = output_dir / "molecules" / _record_name(key)
        if not path.exists():
            raise RuntimeError(f"Run --phase prepare first; missing {path}")
        with path.open("rb") as handle:
            record = pickle.load(handle)
        if args.resume and "after" in record["conformers"]:
            for name in totals:
                totals[name] += int(record.get("statistics", {}).get(name, 0))
            continue
        before = record["conformers"]["before"]
        after, stats = refine_molecules(model, before, batch_size=args.model_batch_size)
        record["conformers"]["after"] = after
        record["statistics"] = stats
        for name in totals:
            totals[name] += int(stats.get(name, 0))
        _atomic_pickle_dump(record, path)
        if position % args.log_every == 0 or position == len(keys):
            print(f"refined {position}/{len(keys)}", flush=True)

    summary = {"config": config, "refine_statistics": totals}
    _atomic_json_dump(summary, output_dir / "refine_summary.json")
    print(json.dumps(totals, indent=2), flush=True)
    return summary


def _condition_metrics(references, candidates) -> dict[str, Any] | None:
    if not candidates:
        return None
    matrix = _rmsd_matrix(references, candidates)
    if np.isnan(matrix).all():
        return None
    cov_r, mat_r, cov_p, mat_p = covmat_metrics(matrix, DEFAULT_THRESHOLDS)
    return {
        "n_true": len(references),
        "n_model": len(candidates),
        "coverage_recall": cov_r,
        "coverage_precision": cov_p,
        "matching_recall": float(mat_r),
        "matching_precision": float(mat_p),
    }


def _evaluate_one(args_tuple) -> str:
    path_text, references, conditions = args_tuple
    path = Path(path_text)
    with path.open("rb") as handle:
        record = pickle.load(handle)
    metrics = record.setdefault("metrics", {})
    pending = [c for c in conditions if c in record["conformers"] and c not in metrics]
    if not pending:
        return path_text
    for condition in pending:
        metrics[condition] = _condition_metrics(references, record["conformers"][condition])
    _atomic_pickle_dump(record, path)
    return path_text


def _flags(directory: Path, name: str, condition: str) -> str | None:
    path = directory / name
    if not path.exists():
        return None
    entry = json.loads(path.read_text()).get("conditions", {}).get(condition, {})
    return entry.get("flags", {}).get("__all__")


def add_gated(output_dir: Path, pb_before_dir: Path, pb_after_dir: Path, basis: str) -> int:
    """Assemble the `gated` condition from per-conformer PoseBusters flags."""
    touched = 0
    for path in sorted((output_dir / "molecules").glob("*.pickle")):
        with path.open("rb") as handle:
            record = pickle.load(handle)
        if "gated" in record["conformers"]:
            continue
        name = path.stem + ".json"
        before_flags = _flags(pb_before_dir, name, basis)
        after_flags = _flags(pb_after_dir, name, "after")
        before = record["conformers"].get("before")
        after = record["conformers"].get("after")
        if not before or not after or not before_flags or not after_flags:
            continue
        if not len(before) == len(after) == len(before_flags) == len(after_flags):
            continue
        record["conformers"]["gated"] = gate_interpolants(before, after, before_flags)
        record["gate_statistics"] = {
            "n": len(before_flags),
            "refined": before_flags.count("0"),
            "kept": before_flags.count("1"),
        }
        _atomic_pickle_dump(record, path)
        touched += 1
    return touched


def summarize(output_dir: Path, conditions=METRIC_CONDITIONS) -> dict[str, Any]:
    index = int(np.argmin(np.abs(np.asarray(DEFAULT_THRESHOLDS) - MAIN_THRESHOLD)))
    rows: dict[str, list] = {c: [] for c in conditions}
    for path in sorted((output_dir / "molecules").glob("*.pickle")):
        with path.open("rb") as handle:
            record = pickle.load(handle)
        for condition in conditions:
            metric = record.get("metrics", {}).get(condition)
            if metric is not None:
                rows[condition].append(metric)
    summary: dict[str, Any] = {"threshold": MAIN_THRESHOLD, "conditions": {}}
    for condition, values in rows.items():
        if not values:
            continue
        summary["conditions"][condition] = {
            "molecule_count": len(values),
            "COV-R": float(np.nanmean(np.vstack([v["coverage_recall"] for v in values])[:, index])),
            "COV-P": float(np.nanmean(np.vstack([v["coverage_precision"] for v in values])[:, index])),
            "MAT-R": float(np.nanmean([v["matching_recall"] for v in values])),
            "MAT-P": float(np.nanmean([v["matching_precision"] for v in values])),
            "mean_n_model": float(np.mean([v["n_model"] for v in values])),
        }
    _atomic_json_dump(summary, output_dir / "covmat_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="required for --phase prepare and refine")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--fsq-checkpoint", required=True)
    parser.add_argument(
        "--phase", choices=("prepare", "refine", "gate", "evaluate"), default="refine"
    )
    parser.add_argument("--pb-before-dir")
    parser.add_argument("--pb-after-dir")
    parser.add_argument("--gate-basis", default="before")
    parser.add_argument(
        "--ground-truth", default="/mnt/weka/fgeikyan/tokenizer/distinct_smi_filtered_13.pickle"
    )
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=218)
    parser.add_argument("--max-molecules", type=int)
    parser.add_argument("--model-batch-size", type=int, default=128)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=50)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
