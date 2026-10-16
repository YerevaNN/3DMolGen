#!/usr/bin/env python3
"""Setup 3 addendum: one tokenizer encode/decode pass over the interpolated ensembles.

Setup 3 leaves the interpolants off-manifold — bond lengths pass 61% for
``latent`` and 41% for ``cartesian`` against 99.7% for the model's own output.
Section 4 showed that a single CoordToken encode/decode pass pulls broken
geometry back onto the tokenizer's manifold, and that it helps most exactly
where the input is worst. This applies that pass to Setup 3's own output.

Nothing is regenerated: the conformers already stored per molecule by
``generated_conformer_interpolation.py`` are re-encoded and re-decoded once.

For every source condition two policies come out of the same pass:

``<cond>_rt_all``      refine every conformer, anchors included
``<cond>_rt_interp``   keep the original anchors, refine only the interpolants

``anchors_rt`` is the control the Setup 2 notes asked for and never got: a
round trip on the anchors with no interpolation anywhere, so its gap from the
raw anchors *is* the tokenizer's reconstruction error.

Conformers whose encode or decode fails keep their original coordinates, so a
tokenizer failure is charged to refinement rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rdkit import Chem

from molgen3D.data_processing.smiles_encoder_decoder import encode_cartesian_v2
from molgen3D.evaluation.conformer_interpolation import (
    _atomic_json_dump,
    _atomic_pickle_dump,
    _decode_enriched_batch,
    _decode_in_chunks,
    _extract_conformers,
    _record_path,
    _rmsd_matrix,
)
from molgen3D.evaluation.utils import DEFAULT_THRESHOLDS, covmat_metrics

SOURCE_CONDITIONS = ("latent", "random_token", "cartesian")
BASELINE_CONDITIONS = ("generated_all", "anchors")
MAIN_THRESHOLD = 0.75


def refined_condition_names(source_conditions: Sequence[str]) -> tuple[str, ...]:
    names = ["anchors_rt"]
    for condition in source_conditions:
        names.append(f"{condition}_rt_all")
        names.append(f"{condition}_rt_interp")
    return tuple(names)


def gated_condition_names(source_conditions: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"{condition}_gated" for condition in source_conditions)


def gate_interpolants(
    before: Sequence[Chem.Mol],
    after: Sequence[Chem.Mol],
    flags: str,
) -> list[Chem.Mol]:
    """Keep the original conformer where it already passed, else take the refined one.

    ``flags`` is one character per conformer, "1" where the original passed
    PoseBusters, in the same order as both lists.
    """
    if not len(before) == len(after) == len(flags):
        raise ValueError(
            f"Gate inputs disagree: {len(before)} before, {len(after)} after, "
            f"{len(flags)} flags"
        )
    return [
        original if flag == "1" else refined
        for original, refined, flag in zip(before, after, flags)
    ]


def refine_molecules(
    model,
    molecules: Sequence[Chem.Mol],
    *,
    batch_size: int,
) -> tuple[list[Chem.Mol], dict[str, int]]:
    """One FSQ encode/decode pass per conformer, keeping failures unchanged.

    Conformers are grouped by their re-encoded token sequence before decoding,
    because the decoder takes one token list for a whole batch. In practice
    every conformer of a molecule shares one sequence, but a broken interpolant
    can canonicalize differently and must not drag the batch with it.
    """
    refined = list(molecules)
    stats = {
        "n": len(molecules),
        "encode_failures": 0,
        "decode_failures": 0,
        "token_groups": 0,
    }
    if not molecules:
        return refined, stats

    texts: list[tuple[int, str]] = []
    for index, molecule in enumerate(molecules):
        try:
            enriched, _ = encode_cartesian_v2(molecule)
        except Exception:
            stats["encode_failures"] += 1
            continue
        texts.append((index, enriched))

    representations: dict[int, Any] = {}
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        try:
            encoded = model.encode_representations([text for _, text in chunk])
            if len(encoded) != len(chunk):
                raise ValueError("Encoder returned the wrong number of representations.")
            for (index, _), representation in zip(chunk, encoded):
                representations[index] = representation
            continue
        except Exception:
            pass
        # Isolate malformed or overlength inputs if a whole batch fails.
        for index, text in chunk:
            try:
                representations[index] = model.encode_representations([text])[0]
            except Exception:
                stats["encode_failures"] += 1

    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, representation in representations.items():
        groups[tuple(representation.tokens)].append(index)
    stats["token_groups"] = len(groups)

    for tokens, indices in groups.items():
        token_list = list(tokens)
        codes = np.stack([representations[index].codes for index in indices])
        decoded_text = _decode_in_chunks(
            codes,
            batch_size=batch_size,
            decode=lambda batch: model.decode_tokens_with_code_batch(token_list, batch),
        )
        decoded_mols, successful = _decode_enriched_batch(decoded_text)
        for position, molecule in zip(successful, decoded_mols):
            refined[indices[position]] = molecule
        stats["decode_failures"] += len(indices) - len(successful)

    return refined, stats


def process_record(
    model,
    record: dict[str, Any],
    *,
    source_conditions: Sequence[str],
    batch_size: int,
) -> dict[str, Any]:
    """Refine one molecule's stored Setup 3 ensembles."""
    n_anchor = int(record["n_anchor"])
    anchors = list(record["conformers"]["anchors"])
    if len(anchors) != n_anchor:
        raise ValueError(
            f"{record['molecule_key']}: expected {n_anchor} anchors, "
            f"found {len(anchors)}"
        )

    refined_anchors, anchor_stats = refine_molecules(model, anchors, batch_size=batch_size)
    conformers: dict[str, list[Chem.Mol]] = {"anchors_rt": refined_anchors}
    statistics: dict[str, dict[str, int]] = {"anchors_rt": anchor_stats}

    for condition in source_conditions:
        stored = record["conformers"].get(condition)
        if not stored:
            continue
        interpolants = list(stored[n_anchor:])
        refined_interpolants, stats = refine_molecules(
            model, interpolants, batch_size=batch_size
        )
        conformers[f"{condition}_rt_all"] = refined_anchors + refined_interpolants
        conformers[f"{condition}_rt_interp"] = anchors + refined_interpolants
        statistics[condition] = stats

    return {
        "molecule_key": record["molecule_key"],
        "status": "complete",
        "k": record["k"],
        "n_anchor": n_anchor,
        "variant": record["variant"],
        "conformers": conformers,
        "statistics": statistics,
        "source_metrics": record.get("metrics", {}),
    }


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


def _evaluate_record_path(record_path_text: str, references) -> str:
    """Fill in metrics for any condition that does not have them yet."""
    record_path = Path(record_path_text)
    with record_path.open("rb") as handle:
        record = pickle.load(handle)
    if record.get("status") != "complete":
        return record_path_text
    metrics = record.setdefault("metrics", {})
    pending = [
        condition for condition in record["conformers"] if condition not in metrics
    ]
    if not pending:
        return record_path_text
    for condition in pending:
        metrics[condition] = _condition_metrics(references, record["conformers"][condition])
    _atomic_pickle_dump(record, record_path)
    return record_path_text


def _add_gated_conditions(
    records,
    *,
    paired_dir: Path,
    flags_dir: Path,
    source_conditions: Sequence[str],
) -> int:
    """Assemble `<cond>_gated` ensembles from per-conformer PoseBusters flags."""
    touched = 0
    for path, record in records:
        if record.get("status") != "complete":
            continue
        paired_path = paired_dir / "molecules" / path.name
        flags_path = flags_dir / f"{path.stem}.json"
        if not paired_path.exists() or not flags_path.exists():
            continue
        with paired_path.open("rb") as handle:
            paired = pickle.load(handle)
        validity = json.loads(flags_path.read_text())
        if paired.get("status") != "complete" or validity.get("status") != "complete":
            continue
        n_anchor = record["n_anchor"]
        statistics = record.setdefault("gate_statistics", {})
        changed = False
        for condition in source_conditions:
            name = f"{condition}_gated"
            if name in record["conformers"]:
                continue
            before = paired["conformers"].get(f"{condition}_interp")
            after = paired["conformers"].get(f"{condition}_interp_rt")
            flags = (
                validity["conditions"]
                .get(f"{condition}_interp", {})
                .get("flags", {})
                .get("__all__")
            )
            anchors = (record["conformers"].get(f"{condition}_rt_interp") or [])[:n_anchor]
            if not before or not after or not flags or len(anchors) != n_anchor:
                continue
            record["conformers"][name] = list(anchors) + gate_interpolants(
                before, after, flags
            )
            statistics[name] = {
                "n_interpolants": len(flags),
                "refined": flags.count("0"),
                "kept": flags.count("1"),
            }
            changed = True
        if changed:
            _atomic_pickle_dump(record, path)
            touched += 1
    return touched


def _fallback_metrics_path(output_dir: Path, molecule_key: str) -> Path:
    return output_dir / "fallback" / _record_path(output_dir, molecule_key).name


def _compute_fallback(args_tuple) -> str:
    """Metrics for a molecule Setup 3 could not interpolate: plain 2k generation."""
    out_path_text, generated, references = args_tuple
    out_path = Path(out_path_text)
    _atomic_pickle_dump(_condition_metrics(references, generated), out_path)
    return out_path_text


def _mean_at_threshold(rows: Sequence[dict[str, Any]], key: str, index: int) -> float:
    return float(np.nanmean(np.vstack([row[key] for row in rows])[:, index]))


def _aggregate(rows: Sequence[dict[str, Any]], index: int) -> dict[str, Any] | None:
    if not rows:
        return None
    return {
        "molecule_count": len(rows),
        "COV-R": _mean_at_threshold(rows, "coverage_recall", index),
        "COV-P": _mean_at_threshold(rows, "coverage_precision", index),
        "MAT-R": float(np.nanmean([row["matching_recall"] for row in rows])),
        "MAT-P": float(np.nanmean([row["matching_precision"] for row in rows])),
        "mean_n_model": float(np.mean([row["n_model"] for row in rows])),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    (output_dir / "molecules").mkdir(parents=True, exist_ok=True)
    (output_dir / "fallback").mkdir(parents=True, exist_ok=True)

    source_paths = sorted((source_dir / "molecules").glob("*.pickle"))
    if not source_paths:
        raise RuntimeError(f"No Setup 3 molecule records under {source_dir}")
    if args.max_molecules is not None:
        source_paths = source_paths[: args.max_molecules]

    source_conditions = tuple(args.conditions)
    config = {
        "source_dir": str(source_dir),
        "fsq_checkpoint": str(args.fsq_checkpoint),
        "generated": str(args.generated),
        "ground_truth": str(args.ground_truth),
        "source_conditions": list(source_conditions),
        "refined_conditions": list(refined_condition_names(source_conditions)),
        "max_tokens": args.max_tokens,
        "policy": {
            "rt_all": "refine every conformer, anchors included",
            "rt_interp": "keep the original anchors, refine only the interpolants",
            "failures": "conformers that fail encode or decode keep their original coordinates",
        },
    }
    _atomic_json_dump(config, output_dir / "config.json")

    if args.phase in ("all", "refine"):
        from molgen3D.data_processing.smiles_encoder_decoder_fsq import (
            configure_fsq_encoder,
        )

        model = configure_fsq_encoder(
            ckpt_path=args.fsq_checkpoint, device=args.device, max_tokens=args.max_tokens
        )
        for position, source_path in enumerate(source_paths, start=1):
            out_path = output_dir / "molecules" / source_path.name
            if args.resume and out_path.exists():
                continue
            with source_path.open("rb") as handle:
                source_record = pickle.load(handle)
            if source_record.get("status") != "complete":
                _atomic_pickle_dump(
                    {
                        "molecule_key": source_record.get("molecule_key"),
                        "status": source_record.get("status"),
                    },
                    out_path,
                )
                continue
            refined = process_record(
                model,
                source_record,
                source_conditions=source_conditions,
                batch_size=args.model_batch_size,
            )
            _atomic_pickle_dump(refined, out_path)
            if position % args.log_every == 0 or position == len(source_paths):
                print(f"refined {position}/{len(source_paths)}", flush=True)

    with Path(args.ground_truth).open("rb") as handle:
        ground_truth = pickle.load(handle)

    records = []
    for source_path in source_paths:
        out_path = output_dir / "molecules" / source_path.name
        if not out_path.exists():
            continue
        with out_path.open("rb") as handle:
            records.append((out_path, pickle.load(handle)))

    if args.phase in ("all", "gate") and args.gate_dir:
        added = _add_gated_conditions(
            records,
            paired_dir=Path(args.gate_dir).expanduser().resolve(),
            flags_dir=Path(args.gate_pb_dir).expanduser().resolve(),
            source_conditions=source_conditions,
        )
        print(f"gated conditions added for {added} molecules", flush=True)
        records = []
        for source_path in source_paths:
            out_path = output_dir / "molecules" / source_path.name
            if not out_path.exists():
                continue
            with out_path.open("rb") as handle:
                records.append((out_path, pickle.load(handle)))

    # `gate` scores what it just assembled; without this the new conditions
    # would land in the records with no metrics.
    if args.phase in ("all", "evaluate", "gate"):
        pending = [
            (str(path), _extract_conformers(ground_truth[record["molecule_key"]]))
            for path, record in records
            if record.get("status") == "complete"
            and any(
                condition not in record.get("metrics", {})
                for condition in record["conformers"]
            )
        ]
        print(f"evaluating {len(pending)} records with {args.num_workers} workers", flush=True)
        if pending:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [
                    executor.submit(_evaluate_record_path, path, references)
                    for path, references in pending
                ]
                for position, future in enumerate(as_completed(futures), start=1):
                    future.result()
                    if position % 50 == 0 or position == len(futures):
                        print(f"evaluated {position}/{len(futures)}", flush=True)

        # Molecules Setup 3 could not interpolate fall back to plain 2k generation,
        # so every condition covers the same molecule set as `generated_all`.
        with Path(args.generated).open("rb") as handle:
            generated = pickle.load(handle)
        fallback_jobs = []
        for _, record in records:
            key = record.get("molecule_key")
            if record.get("status") == "complete" or key is None:
                continue
            out_path = _fallback_metrics_path(output_dir, key)
            if out_path.exists() or key not in generated or key not in ground_truth:
                continue
            fallback_jobs.append(
                (str(out_path), list(generated[key]), _extract_conformers(ground_truth[key]))
            )
        print(f"fallback metrics for {len(fallback_jobs)} molecules", flush=True)
        if fallback_jobs:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                for future in as_completed(
                    [executor.submit(_compute_fallback, job) for job in fallback_jobs]
                ):
                    future.result()

        records = []
        for source_path in source_paths:
            out_path = output_dir / "molecules" / source_path.name
            if not out_path.exists():
                continue
            with out_path.open("rb") as handle:
                records.append((out_path, pickle.load(handle)))

    status_counts: dict[str, int] = {}
    for _, record in records:
        status = str(record.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1

    fallback_rows = []
    for _, record in records:
        key = record.get("molecule_key")
        if record.get("status") == "complete" or key is None:
            continue
        path = _fallback_metrics_path(output_dir, key)
        if not path.exists():
            continue
        with path.open("rb") as handle:
            metrics = pickle.load(handle)
        if metrics is not None:
            fallback_rows.append(metrics)

    index = int(np.argmin(np.abs(np.asarray(DEFAULT_THRESHOLDS) - MAIN_THRESHOLD)))
    refined = set(refined_condition_names(source_conditions)) | set(
        gated_condition_names(source_conditions)
    )
    conditions = (
        list(BASELINE_CONDITIONS)
        + list(source_conditions)
        + list(refined_condition_names(source_conditions))
        + list(gated_condition_names(source_conditions))
    )
    raw: dict[str, Any] = {}
    with_fallback: dict[str, Any] = {}
    for condition in conditions:
        bucket = "metrics" if condition in refined else "source_metrics"
        rows = []
        for _, record in records:
            if record.get("status") != "complete":
                continue
            metrics = record.get(bucket, {}).get(condition)
            if metrics is not None:
                rows.append(metrics)
        aggregate = _aggregate(rows, index)
        if aggregate is None:
            continue
        raw[condition] = aggregate
        with_fallback[condition] = _aggregate(rows + fallback_rows, index)

    summary = {
        "config": config,
        "status_counts": status_counts,
        "threshold": MAIN_THRESHOLD,
        "fallback_molecules": len(fallback_rows),
        "raw": raw,
        "with_fallback": with_fallback,
        "refine_statistics": _refine_statistics(records),
        "gate_statistics": _gate_statistics(records),
    }
    _atomic_json_dump(summary, output_dir / "summary.json")
    print(
        json.dumps(
            {
                condition: {
                    key: round(value, 4)
                    for key, value in aggregate.items()
                    if key in ("COV-R", "COV-P", "MAT-R", "MAT-P")
                }
                for condition, aggregate in with_fallback.items()
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def _refine_statistics(records) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for _, record in records:
        for name, stats in record.get("statistics", {}).items():
            bucket = totals.setdefault(
                name, {"n": 0, "encode_failures": 0, "decode_failures": 0}
            )
            for key in bucket:
                bucket[key] += int(stats.get(key, 0))
    return totals


def _gate_statistics(records) -> dict[str, dict[str, int]]:
    """How many interpolants the gate actually refined, per condition."""
    totals: dict[str, dict[str, int]] = {}
    for _, record in records:
        for name, stats in record.get("gate_statistics", {}).items():
            bucket = totals.setdefault(name, {"n_interpolants": 0, "refined": 0, "kept": 0})
            for key in bucket:
                bucket[key] += int(stats.get(key, 0))
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Setup 3 run directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fsq-checkpoint", required=True)
    parser.add_argument(
        "--generated", required=True, help="generation_results.pickle, for the fallback"
    )
    parser.add_argument(
        "--ground-truth", default="/mnt/weka/fgeikyan/tokenizer/distinct_smi_filtered_13.pickle"
    )
    parser.add_argument("--conditions", nargs="+", default=list(SOURCE_CONDITIONS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=218)
    parser.add_argument("--max-molecules", type=int)
    parser.add_argument(
        "--phase", choices=("all", "refine", "evaluate", "gate"), default="all"
    )
    parser.add_argument(
        "--gate-dir",
        help="setup3_gate_prep.py output, holding paired original/refined interpolants",
    )
    parser.add_argument(
        "--gate-pb-dir",
        help="PoseBusters output for --gate-dir, carrying per-conformer flags",
    )
    parser.add_argument("--model-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=25)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
