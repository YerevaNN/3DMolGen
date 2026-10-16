#!/usr/bin/env python3
"""Setup 3: interpolate between *model-generated* conformers instead of references.

Setups 1 and 2 interpolated between ground-truth GEOM conformers. Here the
anchors come from a trained model's own output, which is the deployable
question: can a model emit fewer conformers and have interpolation fill the
rest of the ensemble at no extra sampling cost?

Two variants, both ending at the standard ``2k`` candidate budget so every
number is comparable to the model's ordinary ``2k`` generation:

``half``  ``floor(k/2)`` generated anchors + interpolants up to ``2k``
``full``  ``k`` generated anchors + ``k`` interpolants

Conditions scored per molecule against that molecule's full ``k`` references:

``generated_all``   every generated conformer the run produced (the baseline)
``anchors``         the generated anchors alone, no interpolation
``latent``          anchors + latent-midpoint interpolants
``random_token``    anchors + random-balanced-bit interpolants
``cartesian``       anchors + coordinate-midpoint interpolants (no tokenizer)

GPU work is the ``generate`` phase; ``evaluate`` is CPU-only. Records are
written per molecule so either phase resumes.
"""

from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rdkit import Chem

from molgen3D.evaluation.conformer_interpolation import (
    _atomic_json_dump,
    _atomic_pickle_dump,
    _decode_enriched_batch,
    _decode_in_chunks,
    _encode_anchors,
    _extract_conformers,
    _record_path,
    _rmsd_matrix,
    _same_tokens,
    random_balanced_bit_interpolation,
    select_balanced_local_pairs,
    stable_molecule_seed,
)
from molgen3D.evaluation.utils import DEFAULT_THRESHOLDS, covmat_metrics

CONDITIONS = ("generated_all", "anchors", "latent", "random_token", "cartesian")
MAIN_THRESHOLD = 0.75


def cartesian_midpoint(first: Chem.Mol, second: Chem.Mol) -> Chem.Mol | None:
    """Superpose `second` onto `first` and average their coordinates.

    The no-tokenizer control: a straight line in coordinate space, with no
    encode or decode anywhere. Atom correspondence is the identity mapping,
    since both conformers are the same molecule in the same atom order.
    """
    from rdkit.Chem import rdMolAlign

    try:
        probe = Chem.Mol(second)
        rdMolAlign.AlignMol(probe, first)
        start = first.GetConformer().GetPositions()
        end = probe.GetConformer().GetPositions()
        if start.shape != end.shape:
            return None
        midpoint = 0.5 * (start + end)
        if not np.isfinite(midpoint).all():
            return None
        out = Chem.Mol(first)
        conformer = out.GetConformer()
        for index in range(out.GetNumAtoms()):
            conformer.SetAtomPosition(index, midpoint[index].tolist())
        return out
    except Exception:
        return None


def anchor_count(k: int, variant: str) -> int:
    if variant == "half":
        return k // 2
    if variant == "full":
        return k
    raise ValueError(f"Unknown variant: {variant}")


def process_molecule(
    model,
    molecule_key: str,
    references: Sequence[Chem.Mol],
    generated: Sequence[Chem.Mol],
    *,
    variant: str,
    seed: int,
    model_batch_size: int,
) -> dict[str, Any]:
    k = len(references)
    molecule_seed = stable_molecule_seed(seed, molecule_key)
    n_anchor = anchor_count(k, variant)
    target_total = 2 * k
    base: dict[str, Any] = {
        "molecule_key": molecule_key,
        "seed": molecule_seed,
        "k": k,
        "n_generated": len(generated),
        "n_anchor": n_anchor,
        "variant": variant,
    }

    if n_anchor < 2:
        return {**base, "status": "skipped_too_few_anchors"}
    if len(generated) < n_anchor:
        return {**base, "status": "skipped_too_few_generated"}

    anchors = list(generated[:n_anchor])
    representations, encode_failures = _encode_anchors(
        model, anchors, batch_size=model_batch_size
    )
    if encode_failures or len(representations) != len(anchors):
        return {**base, "status": "tokenizer_failure", "encode_failures": encode_failures}
    if not _same_tokens(representations):
        return {**base, "status": "token_sequence_mismatch"}

    n_interpolants = max(0, target_total - n_anchor)
    positions = list(range(n_anchor))
    pairs = select_balanced_local_pairs(
        positions,
        representations,
        target_count=n_interpolants,
        seed=stable_molecule_seed(molecule_seed, "balanced-local-pairs"),
    )
    if not pairs:
        return {**base, "status": "no_pairs"}

    latent_batch = np.stack(
        [
            0.5 * representations[first].latents + 0.5 * representations[second].latents
            for first, second in pairs
        ]
    )
    token_rng = np.random.default_rng(stable_molecule_seed(molecule_seed, "token-bits"))
    random_code_batch = np.stack(
        [
            random_balanced_bit_interpolation(
                representations[first].codes,
                representations[second].codes,
                rng=token_rng,
            )
            for first, second in pairs
        ]
    )
    tokens = representations[0].tokens

    latent_text = _decode_in_chunks(
        latent_batch,
        batch_size=model_batch_size,
        decode=lambda batch: model.decode_tokens_with_latent_batch(tokens, batch),
    )
    random_token_text = _decode_in_chunks(
        random_code_batch,
        batch_size=model_batch_size,
        decode=lambda batch: model.decode_tokens_with_code_batch(tokens, batch),
    )
    latent_mols, _ = _decode_enriched_batch(latent_text)
    random_token_mols, _ = _decode_enriched_batch(random_token_text)

    cartesian_mols = []
    for first, second in pairs:
        molecule = cartesian_midpoint(anchors[first], anchors[second])
        if molecule is not None:
            cartesian_mols.append(molecule)

    statistics = {}
    for condition, molecules in (
        ("latent", latent_mols),
        ("random_token", random_token_mols),
        ("cartesian", cartesian_mols),
    ):
        statistics[condition] = {
            "attempted": len(pairs),
            "decoded": len(molecules),
            "decode_success_rate": len(molecules) / len(pairs),
        }

    return {
        **base,
        "status": "complete",
        "n_pairs": len(pairs),
        "pairs": pairs,
        "conformers": {
            "generated_all": list(generated),
            "anchors": anchors,
            "latent": anchors + latent_mols,
            "random_token": anchors + random_token_mols,
            "cartesian": anchors + cartesian_mols,
        },
        "statistics": statistics,
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
    record_path = Path(record_path_text)
    with record_path.open("rb") as handle:
        record = pickle.load(handle)
    if record.get("status") != "complete" or "metrics" in record:
        return record_path_text
    conformers = record["conformers"]
    record["metrics"] = {
        condition: _condition_metrics(references, conformers[condition])
        for condition in CONDITIONS
    }
    _atomic_pickle_dump(record, record_path)
    return record_path_text


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    (output_dir / "molecules").mkdir(parents=True, exist_ok=True)

    with Path(args.ground_truth).open("rb") as handle:
        ground_truth = pickle.load(handle)
    with Path(args.generated).open("rb") as handle:
        generated = pickle.load(handle)

    keys = [key for key in ground_truth if key in generated]
    if args.max_molecules is not None:
        keys = keys[: args.max_molecules]
    print(f"molecules: {len(keys)}  variant: {args.variant}", flush=True)

    config = {
        "generated": str(args.generated),
        "ground_truth": str(args.ground_truth),
        "fsq_checkpoint": str(args.fsq_checkpoint),
        "variant": args.variant,
        "seed": args.seed,
        "pairing_strategy": "balanced_local",
        "max_tokens": args.max_tokens,
    }
    _atomic_json_dump(config, output_dir / "config.json")

    if args.phase in ("all", "generate"):
        from molgen3D.data_processing.smiles_encoder_decoder_fsq import (
            configure_fsq_encoder,
        )

        model = configure_fsq_encoder(
            ckpt_path=args.fsq_checkpoint, device=args.device, max_tokens=args.max_tokens
        )
        for position, key in enumerate(keys, start=1):
            path = _record_path(output_dir, key)
            if args.resume and path.exists():
                continue
            record = process_molecule(
                model,
                key,
                _extract_conformers(ground_truth[key]),
                generated[key],
                variant=args.variant,
                seed=args.seed,
                model_batch_size=args.model_batch_size,
            )
            _atomic_pickle_dump(record, path)
            if position % args.log_every == 0 or position == len(keys):
                print(f"generated {position}/{len(keys)}", flush=True)

    records = []
    for key in keys:
        path = _record_path(output_dir, key)
        if path.exists():
            with path.open("rb") as handle:
                records.append(pickle.load(handle))

    if args.phase in ("all", "evaluate"):
        pending = [
            (str(_record_path(output_dir, r["molecule_key"])), _extract_conformers(ground_truth[r["molecule_key"]]))
            for r in records
            if r.get("status") == "complete" and "metrics" not in r
        ]
        print(f"evaluating {len(pending)} records with {args.num_workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(_evaluate_record_path, p, refs) for p, refs in pending]
            for position, future in enumerate(as_completed(futures), start=1):
                future.result()
                if position % 50 == 0 or position == len(futures):
                    print(f"evaluated {position}/{len(futures)}", flush=True)
        records = []
        for key in keys:
            path = _record_path(output_dir, key)
            if path.exists():
                with path.open("rb") as handle:
                    records.append(pickle.load(handle))

    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1

    index = int(np.argmin(np.abs(np.asarray(DEFAULT_THRESHOLDS) - MAIN_THRESHOLD)))
    summary: dict[str, Any] = {
        "config": config,
        "status_counts": status_counts,
        "threshold": MAIN_THRESHOLD,
        "conditions": {},
    }
    for condition in CONDITIONS:
        rows = [
            r["metrics"][condition]
            for r in records
            if r.get("metrics", {}).get(condition) is not None
        ]
        if not rows:
            continue
        summary["conditions"][condition] = {
            "molecule_count": len(rows),
            "COV-R": float(np.nanmean(np.vstack([r["coverage_recall"] for r in rows])[:, index])),
            "COV-P": float(np.nanmean(np.vstack([r["coverage_precision"] for r in rows])[:, index])),
            "MAT-R": float(np.nanmean([r["matching_recall"] for r in rows])),
            "MAT-P": float(np.nanmean([r["matching_precision"] for r in rows])),
            "mean_n_model": float(np.mean([r["n_model"] for r in rows])),
        }
    _atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps(summary["conditions"], indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", required=True, help="generation_results.pickle")
    parser.add_argument(
        "--ground-truth", default="/mnt/weka/fgeikyan/tokenizer/distinct_smi_filtered_13.pickle"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fsq-checkpoint", required=True)
    parser.add_argument("--variant", choices=("half", "full"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=218)
    parser.add_argument("--max-molecules", type=int)
    parser.add_argument("--phase", choices=("all", "generate", "evaluate"), default="all")
    parser.add_argument("--model-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=25)
    run(parser.parse_args())


if __name__ == "__main__":
    main()