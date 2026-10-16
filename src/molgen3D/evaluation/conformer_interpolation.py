"""Ground-truth conformer interpolation sanity check for the FSQ tokenizer."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
import pickle
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from rdkit import Chem

from molgen3D.config.paths import get_data_path
from molgen3D.data_processing.smiles_encoder_decoder import (
    decode_cartesian_v2,
    encode_cartesian_v2,
)
from molgen3D.data_processing.smiles_encoder_decoder_fsq import (
    FSQRepresentation,
    configure_fsq_encoder,
)
from molgen3D.evaluation import rdkit_utils
from molgen3D.evaluation.run_eval import summarize_metrics
from molgen3D.evaluation.utils import DEFAULT_THRESHOLDS, covmat_metrics

INTERPOLATION_CONDITIONS = ("latent", "token", "random_token")
CONDITIONS = ("anchors", *INTERPOLATION_CONDITIONS)
MAIN_THRESHOLD = 0.75


def stable_molecule_seed(seed: int, molecule_key: str) -> int:
    """Derive a resume/order-independent NumPy seed for one molecule."""
    digest = hashlib.sha256(f"{seed}\0{molecule_key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def select_anchors_and_pairs(
    k: int,
    *,
    seed: int,
    output_multiplier: int = 2,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Select floor(k/2) anchors and sample distinct endpoint pairs."""
    n_anchors = k // 2
    if n_anchors < 2:
        return [], []

    rng = np.random.default_rng(seed)
    anchors = sorted(int(i) for i in rng.choice(k, size=n_anchors, replace=False))
    pairs = [
        tuple(int(i) for i in rng.choice(anchors, size=2, replace=False))
        for _ in range(output_multiplier * k)
    ]
    return anchors, pairs


def select_balanced_local_pairs(
    anchor_indices: Sequence[int],
    representations: Sequence[FSQRepresentation],
    *,
    target_count: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Pair nearby anchors in latent space, balancing coverage across anchors.

    Unordered pairs are unique until every possible pair has been used.  When
    ``target_count`` exceeds that number, the ranked pair list is cycled because
    repetition is mathematically unavoidable.
    """
    if len(anchor_indices) != len(representations):
        raise ValueError("Anchor indices and representations must have equal length.")
    if len(anchor_indices) < 2 or target_count <= 0:
        return []

    flattened = np.stack(
        [np.asarray(representation.latents, dtype=np.float64).reshape(-1) for representation in representations]
    )
    squared_norms = np.sum(flattened * flattened, axis=1)
    distances = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * flattened @ flattened.T
    )
    distances = np.maximum(distances, 0.0)
    np.fill_diagonal(distances, np.inf)

    rng = np.random.default_rng(seed)
    anchor_order = rng.permutation(len(anchor_indices))
    neighbor_order = np.argsort(distances, axis=1, kind="stable")
    ranked_pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    # Round-robin over neighbor rank prevents one dense region from consuming
    # the pair budget before every anchor has participated.
    for neighbor_rank in range(len(anchor_indices) - 1):
        for first_position in anchor_order:
            second_position = int(neighbor_order[first_position, neighbor_rank])
            unordered = tuple(sorted((int(first_position), second_position)))
            if unordered in seen:
                continue
            seen.add(unordered)
            first = int(anchor_indices[unordered[0]])
            second = int(anchor_indices[unordered[1]])
            if rng.integers(2):
                first, second = second, first
            ranked_pairs.append((first, second))
            if len(ranked_pairs) == target_count:
                return ranked_pairs

    return [
        ranked_pairs[index % len(ranked_pairs)]
        for index in range(target_count)
    ]


def alternating_bit_interpolation(
    first: np.ndarray | Sequence[int],
    second: np.ndarray | Sequence[int],
    *,
    width: int = 12,
) -> np.ndarray:
    """Take endpoint 2 at the 2nd, 4th, ... differing MSB-first bits."""
    first_array = np.asarray(first, dtype=np.int64)
    second_array = np.asarray(second, dtype=np.int64)
    _validate_code_endpoints(first_array, second_array, width)

    result = first_array.copy()
    difference_number = np.zeros(first_array.shape, dtype=np.int16)
    for bit in range(width - 1, -1, -1):
        mask = 1 << bit
        differs = ((first_array ^ second_array) & mask) != 0
        difference_number += differs
        take_second = differs & ((difference_number % 2) == 0)
        result[take_second] = (result[take_second] & ~mask) | (
            second_array[take_second] & mask
        )
    return result


def _validate_code_endpoints(
    first_array: np.ndarray,
    second_array: np.ndarray,
    width: int,
) -> None:
    if first_array.shape != second_array.shape:
        raise ValueError(
            f"Endpoint code shapes differ: {first_array.shape} != {second_array.shape}"
        )
    if width <= 0:
        raise ValueError("Bit width must be positive.")
    upper_bound = 1 << width
    if (
        np.any(first_array < 0)
        or np.any(first_array >= upper_bound)
        or np.any(second_array < 0)
        or np.any(second_array >= upper_bound)
    ):
        raise ValueError(f"FSQ codes must be in [0, {upper_bound}).")


def random_balanced_bit_interpolation(
    first: np.ndarray | Sequence[int],
    second: np.ndarray | Sequence[int],
    *,
    rng: np.random.Generator,
    width: int = 12,
) -> np.ndarray:
    """Randomly take approximately half of each token's differing bits."""
    first_array = np.asarray(first, dtype=np.int64)
    second_array = np.asarray(second, dtype=np.int64)
    _validate_code_endpoints(first_array, second_array, width)

    result = first_array.copy().reshape(-1)
    first_flat = first_array.reshape(-1)
    second_flat = second_array.reshape(-1)
    for index, (first_code, second_code) in enumerate(zip(first_flat, second_flat)):
        differing_bits = [
            bit for bit in range(width) if ((first_code ^ second_code) & (1 << bit))
        ]
        take_count = len(differing_bits) // 2
        if len(differing_bits) % 2:
            take_count += int(rng.integers(0, 2))
        if take_count == 0:
            continue
        selected_bits = rng.choice(differing_bits, size=take_count, replace=False)
        for bit in np.atleast_1d(selected_bits):
            mask = 1 << int(bit)
            result[index] = (result[index] & ~mask) | (second_code & mask)
    return result.reshape(first_array.shape)


def _atomic_pickle_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temp_path = Path(handle.name)
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, path)


def _atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, mode="w", suffix=".tmp", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temp_path, path)


def _record_path(output_dir: Path, molecule_key: str) -> Path:
    digest = hashlib.sha256(molecule_key.encode()).hexdigest()
    return output_dir / "molecules" / f"{digest}.pickle"


def _extract_conformers(value: Any) -> list[Chem.Mol]:
    if isinstance(value, dict):
        value = value.get("confs", [])
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Expected conformer list, got {type(value)!r}")
    return list(value)


def _same_tokens(representations: Iterable[FSQRepresentation]) -> bool:
    iterator = iter(representations)
    try:
        expected = next(iterator).tokens
    except StopIteration:
        return True
    return all(representation.tokens == expected for representation in iterator)


def _decode_enriched_batch(
    enriched_batch: Sequence[str | None],
) -> tuple[list[Chem.Mol], list[int]]:
    molecules: list[Chem.Mol] = []
    successful_indices: list[int] = []
    for index, enriched in enumerate(enriched_batch):
        if enriched is None:
            continue
        try:
            molecule = decode_cartesian_v2(enriched)
            coords = molecule.GetConformer().GetPositions()
            if not np.isfinite(coords).all():
                raise ValueError("Decoded coordinates are not finite.")
            molecules.append(molecule)
            successful_indices.append(index)
        except Exception:
            continue
    return molecules, successful_indices


def _rmsd_matrix(references: Sequence[Chem.Mol], candidates: Sequence[Chem.Mol]) -> np.ndarray:
    matrix = np.full((len(references), len(candidates)), np.nan, dtype=float)
    normalized_refs = [rdkit_utils._normalize_coords(Chem.RemoveHs(mol)) for mol in references]
    candidates_no_h = [Chem.RemoveHs(mol) for mol in candidates]
    for row, reference in enumerate(normalized_refs):
        matrix[row] = [
            rdkit_utils._best_rmsd(candidate, reference, use_alignmol=False)
            for candidate in candidates_no_h
        ]
    return matrix


def _condition_metrics(
    references: Sequence[Chem.Mol], candidates: Sequence[Chem.Mol]
) -> dict[str, Any] | None:
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
        "matching_recall": mat_r,
        "matching_precision": mat_p,
    }


def duplicate_rate(
    conformers: Sequence[Chem.Mol],
    *,
    rmsd_threshold: float,
) -> float:
    """Return the fraction duplicating an earlier conformer by best RMSD."""
    if not conformers:
        return float("nan")
    if rmsd_threshold <= 0:
        raise ValueError("Duplicate RMSD threshold must be positive.")
    unique_by_signature: dict[bytes, list[Chem.Mol]] = {}
    duplicates = 0
    for molecule in conformers:
        candidate = Chem.RemoveHs(molecule)
        coords = candidate.GetConformer().GetPositions()
        centered = coords - coords.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(centered[:, np.newaxis, :] - centered[np.newaxis, :, :], axis=-1)
        signature = np.rint(distances / rmsd_threshold).astype(np.int64).tobytes()
        is_duplicate = any(
            rdkit_utils._best_rmsd(candidate, reference, use_alignmol=False) < rmsd_threshold
            for reference in unique_by_signature.get(signature, [])
        )
        if is_duplicate:
            duplicates += 1
        else:
            unique_by_signature.setdefault(signature, []).append(candidate)
    return duplicates / len(conformers)


def _encode_anchors(
    model,
    anchors: Sequence[Chem.Mol],
    *,
    batch_size: int,
) -> tuple[list[FSQRepresentation], int]:
    enriched: list[str] = []
    encode_failures = 0
    for molecule in anchors:
        try:
            text, _ = encode_cartesian_v2(molecule)
            enriched.append(text)
        except Exception:
            encode_failures += 1
    if encode_failures:
        return [], encode_failures
    representations: list[FSQRepresentation] = []
    for start in range(0, len(enriched), batch_size):
        batch = enriched[start : start + batch_size]
        try:
            representations.extend(model.encode_representations(batch))
            continue
        except Exception:
            pass
        # Isolate malformed/overlength inputs if a whole batch fails.
        for text in batch:
            try:
                representations.extend(model.encode_representations([text]))
            except Exception:
                encode_failures += 1
    if encode_failures:
        return [], encode_failures
    return representations, 0


def _decode_in_chunks(
    values: np.ndarray,
    *,
    batch_size: int,
    decode,
) -> list[str | None]:
    decoded: list[str | None] = [None] * len(values)
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size]
        try:
            batch_decoded = decode(batch)
            if len(batch_decoded) != len(batch):
                raise ValueError(
                    f"Decoder returned {len(batch_decoded)} items for " f"a batch of {len(batch)}."
                )
            decoded[start : start + len(batch)] = batch_decoded
        except Exception:
            continue
    return decoded


def process_molecule(
    *,
    molecule_key: str,
    references: Sequence[Chem.Mol],
    model,
    seed: int,
    duplicate_rmsd_threshold: float,
    model_batch_size: int,
    pairing_strategy: str = "random",
) -> dict[str, Any]:
    k = len(references)
    molecule_seed = stable_molecule_seed(seed, molecule_key)
    token_interpolation_seed = stable_molecule_seed(
        molecule_seed, "token-bit-interpolation"
    )
    anchor_indices, pairs = select_anchors_and_pairs(k, seed=molecule_seed)
    base = {
        "molecule_key": molecule_key,
        "seed": molecule_seed,
        "token_interpolation_seed": token_interpolation_seed,
        "reference_count": k,
        "anchor_indices": anchor_indices,
        "pairs": pairs,
        "pairing_strategy": pairing_strategy,
    }
    if not anchor_indices:
        return {**base, "status": "skipped_too_few_anchors"}

    anchors = [references[index] for index in anchor_indices]
    representations, encode_failures = _encode_anchors(model, anchors, batch_size=model_batch_size)
    if encode_failures or len(representations) != len(anchors):
        return {
            **base,
            "status": "tokenizer_failure",
            "encode_failures": encode_failures,
        }
    if not _same_tokens(representations):
        return {**base, "status": "token_sequence_mismatch"}

    if pairing_strategy == "balanced_local":
        pairs = select_balanced_local_pairs(
            anchor_indices,
            representations,
            target_count=2 * k,
            seed=stable_molecule_seed(molecule_seed, "balanced-local-pairs"),
        )
        base["pairs"] = pairs
    elif pairing_strategy != "random":
        raise ValueError(f"Unknown pairing strategy: {pairing_strategy}")

    anchor_position = {reference_index: pos for pos, reference_index in enumerate(anchor_indices)}
    pair_positions = [(anchor_position[first], anchor_position[second]) for first, second in pairs]
    latent_batch = np.stack(
        [
            0.5 * representations[first].latents + 0.5 * representations[second].latents
            for first, second in pair_positions
        ]
    )
    code_batch = np.stack(
        [
            alternating_bit_interpolation(
                representations[first].codes,
                representations[second].codes,
            )
            for first, second in pair_positions
        ]
    )
    token_rng = np.random.default_rng(token_interpolation_seed)
    random_code_batch = np.stack(
        [
            random_balanced_bit_interpolation(
                representations[first].codes,
                representations[second].codes,
                rng=token_rng,
            )
            for first, second in pair_positions
        ]
    )
    tokens = representations[0].tokens

    latent_text = _decode_in_chunks(
        latent_batch,
        batch_size=model_batch_size,
        decode=lambda batch: model.decode_tokens_with_latent_batch(tokens, batch),
    )
    token_text = _decode_in_chunks(
        code_batch,
        batch_size=model_batch_size,
        decode=lambda batch: model.decode_tokens_with_code_batch(tokens, batch),
    )
    random_token_text = _decode_in_chunks(
        random_code_batch,
        batch_size=model_batch_size,
        decode=lambda batch: model.decode_tokens_with_code_batch(tokens, batch),
    )

    latent_mols, latent_successful_indices = _decode_enriched_batch(latent_text)
    token_mols, token_successful_indices = _decode_enriched_batch(token_text)
    random_token_mols, random_token_successful_indices = _decode_enriched_batch(
        random_token_text
    )
    successful_pair_indices = {
        "latent": latent_successful_indices,
        "token": token_successful_indices,
        "random_token": random_token_successful_indices,
    }

    output_count = len(pairs)
    statistics = {}
    for condition, molecules in (
        ("latent", latent_mols),
        ("token", token_mols),
        ("random_token", random_token_mols),
    ):
        valid_count = len(molecules)
        statistics[condition] = {
            "attempted": output_count,
            "decoded": valid_count,
            "valid": valid_count,
            "decode_failures": output_count - valid_count,
            "decode_success_rate": valid_count / output_count,
            "validity_rate": valid_count / output_count,
            "duplicate_rate": duplicate_rate(molecules, rmsd_threshold=duplicate_rmsd_threshold),
        }

    return {
        **base,
        "status": "complete",
        "representations": [
            {
                "tokens": representation.tokens,
                "latents": representation.latents,
                "codes": representation.codes,
            }
            for representation in representations
        ],
        "conformers": {
            "anchors": anchors,
            "latent": latent_mols,
            "token": token_mols,
            "random_token": random_token_mols,
        },
        "statistics": statistics,
        "successful_pair_indices": successful_pair_indices,
    }


def _evaluate_record_path(
    record_path_text: str,
    references: Sequence[Chem.Mol],
) -> str:
    """Evaluate one persisted molecule record in a CPU worker."""
    record_path = Path(record_path_text)
    with record_path.open("rb") as handle:
        record = pickle.load(handle)
    if record.get("status") != "complete" or "metrics" in record:
        return record_path_text

    conformers = record["conformers"]
    record["metrics"] = {
        condition: _condition_metrics(references, conformers[condition]) for condition in CONDITIONS
    }
    _atomic_pickle_dump(record, record_path)
    return record_path_text


def _evaluate_records(
    records: Sequence[dict[str, Any]],
    ground_truth: dict[str, Any],
    output_dir: Path,
    *,
    num_workers: int,
) -> list[dict[str, Any]]:
    pending = [
        (
            str(_record_path(output_dir, record["molecule_key"])),
            _extract_conformers(ground_truth[record["molecule_key"]]),
        )
        for record in records
        if record.get("status") == "complete" and "metrics" not in record
    ]
    if num_workers <= 1:
        for position, (record_path, references) in enumerate(pending, start=1):
            _evaluate_record_path(record_path, references)
            if position % 10 == 0 or position == len(pending):
                print(f"Evaluated {position}/{len(pending)} molecule records", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_evaluate_record_path, record_path, references)
                for record_path, references in pending
            ]
            for position, future in enumerate(as_completed(futures), start=1):
                future.result()
                if position % 10 == 0 or position == len(futures):
                    print(
                        f"Evaluated {position}/{len(futures)} molecule records",
                        flush=True,
                    )

    evaluated = []
    for record in records:
        record_path = _record_path(output_dir, record["molecule_key"])
        with record_path.open("rb") as handle:
            evaluated.append(pickle.load(handle))
    return evaluated


def _aggregate_condition(
    records: Sequence[dict[str, Any]], condition: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    coverage_recall = []
    coverage_precision = []
    matching_recall = []
    matching_precision = []
    for record in records:
        metric = record.get("metrics", {}).get(condition)
        if metric is not None:
            coverage_recall.append(metric["coverage_recall"])
            coverage_precision.append(metric["coverage_precision"])
            matching_recall.append(metric["matching_recall"])
            matching_precision.append(metric["matching_precision"])
    aggregate: dict[str, np.ndarray] = {
        "thresholds": DEFAULT_THRESHOLDS,
        "CoverageR": (
            np.vstack(coverage_recall)
            if coverage_recall
            else np.empty((0, len(DEFAULT_THRESHOLDS)))
        ),
        "CoverageP": (
            np.vstack(coverage_precision)
            if coverage_precision
            else np.empty((0, len(DEFAULT_THRESHOLDS)))
        ),
        "MatchingR": np.asarray(matching_recall),
        "MatchingP": np.asarray(matching_precision),
    }
    if not coverage_recall:
        return aggregate, {
            "coverage": [],
            "matching": {
                name: None
                for name in (
                    "COV-R_mean",
                    "COV-R_median",
                    "COV-P_mean",
                    "COV-P_median",
                    "MAT-R_mean",
                    "MAT-R_median",
                    "MAT-P_mean",
                    "MAT-P_median",
                )
            },
            "molecule_count": 0,
        }
    coverage_frame, matching = summarize_metrics(aggregate)
    return aggregate, {
        "coverage": coverage_frame.to_dict(orient="records"),
        "matching": matching,
        "molecule_count": len(coverage_recall),
    }


def _summarize_rates(records: Sequence[dict[str, Any]], condition: str) -> dict[str, Any]:
    stats = [
        record["statistics"][condition] for record in records if record.get("status") == "complete"
    ]
    attempted = sum(item["attempted"] for item in stats)
    decoded = sum(item["decoded"] for item in stats)
    valid = sum(item["valid"] for item in stats)
    duplicate_numerator = sum(
        item["duplicate_rate"] * item["valid"]
        for item in stats
        if np.isfinite(item["duplicate_rate"])
    )
    return {
        "attempted": attempted,
        "decoded": decoded,
        "valid": valid,
        "decode_failures": sum(item["decode_failures"] for item in stats),
        "decode_success_rate": decoded / attempted if attempted else None,
        "validity_rate": valid / attempted if attempted else None,
        "duplicate_rate": duplicate_numerator / valid if valid else None,
    }


def _metric_at_threshold(
    aggregate: dict[str, np.ndarray], metric: str, threshold: float
) -> float | None:
    index = int(np.argmin(np.abs(aggregate["thresholds"] - threshold)))
    values = aggregate[metric]
    if values.ndim != 2 or values.shape[0] == 0:
        return None
    return float(np.nanmean(values[:, index]))


def decide_stage_one(
    *,
    aggregates: dict[str, dict[str, np.ndarray]],
    rates: dict[str, dict[str, Any]],
    threshold: float,
    max_precision_drop: float,
    min_validity_rate: float,
    min_decode_success_rate: float,
) -> dict[str, Any]:
    anchor_recall = _metric_at_threshold(aggregates["anchors"], "CoverageR", threshold)
    anchor_precision = _metric_at_threshold(aggregates["anchors"], "CoverageP", threshold)
    if anchor_recall is None or anchor_precision is None:
        raise RuntimeError("No anchor metrics were available for the Stage 1 decision.")
    methods = {}
    for condition in INTERPOLATION_CONDITIONS:
        recall = _metric_at_threshold(aggregates[condition], "CoverageR", threshold)
        precision = _metric_at_threshold(aggregates[condition], "CoverageP", threshold)
        rate = rates[condition]
        checks = {
            "coverage_improved": recall is not None and recall > anchor_recall,
            "precision_acceptable": (
                precision is not None and precision >= anchor_precision - max_precision_drop
            ),
            "validity_acceptable": (
                rate["validity_rate"] is not None and rate["validity_rate"] >= min_validity_rate
            ),
            "decode_success_acceptable": (
                rate["decode_success_rate"] is not None
                and rate["decode_success_rate"] >= min_decode_success_rate
            ),
        }
        methods[condition] = {
            "passed": all(checks.values()),
            "checks": checks,
            "cov_r": recall,
            "cov_r_change": recall - anchor_recall if recall is not None else None,
            "cov_p": precision,
            "cov_p_change": precision - anchor_precision if precision is not None else None,
        }
    return {
        "passed": any(method["passed"] for method in methods.values()),
        "main_metric": f"COV-R@{threshold:g}",
        "baseline": {"cov_r": anchor_recall, "cov_p": anchor_precision},
        "criteria": {
            "max_precision_drop": max_precision_drop,
            "min_validity_rate": min_validity_rate,
            "min_decode_success_rate": min_decode_success_rate,
        },
        "methods": methods,
    }


def _per_molecule_rows(records: Sequence[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    index = int(np.argmin(np.abs(DEFAULT_THRESHOLDS - threshold)))
    rows = []
    for record in records:
        if record.get("status") != "complete":
            continue
        baseline = record["metrics"]["anchors"]
        row: dict[str, Any] = {
            "molecule_key": record["molecule_key"],
            "reference_count": record["reference_count"],
            "anchor_count": len(record["anchor_indices"]),
            "anchor_cov_r": baseline["coverage_recall"][index],
            "anchor_cov_p": baseline["coverage_precision"][index],
            "anchor_mat_r": baseline["matching_recall"],
            "anchor_mat_p": baseline["matching_precision"],
        }
        for condition in INTERPOLATION_CONDITIONS:
            metric = record["metrics"][condition]
            if metric is None:
                for name in ("cov_r", "cov_p", "mat_r", "mat_p"):
                    row[f"{condition}_{name}"] = None
                    row[f"{condition}_{name}_change"] = None
                continue
            values = {
                "cov_r": metric["coverage_recall"][index],
                "cov_p": metric["coverage_precision"][index],
                "mat_r": metric["matching_recall"],
                "mat_p": metric["matching_precision"],
            }
            baseline_values = {
                "cov_r": baseline["coverage_recall"][index],
                "cov_p": baseline["coverage_precision"][index],
                "mat_r": baseline["matching_recall"],
                "mat_p": baseline["matching_precision"],
            }
            for name, value in values.items():
                row[f"{condition}_{name}"] = value
                row[f"{condition}_{name}_change"] = value - baseline_values[name]
        rows.append(row)
    return rows


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_resume_config(existing: dict[str, Any], current: dict[str, Any]) -> None:
    immutable_keys = (
        "input",
        "fsq_checkpoint",
        "max_tokens",
        "seed",
        "alpha",
        "output_multiplier",
        "pairing_strategy",
        "bit_width",
        "token_bit_rule",
        "random_token_bit_rule",
        "duplicate_rmsd_threshold",
        "source_hashes",
    )
    mismatches = [key for key in immutable_keys if existing.get(key) != current.get(key)]
    if mismatches:
        raise ValueError(
            "Refusing to resume with changed generation settings: " + ", ".join(mismatches)
        )


def _generation_config(
    args: argparse.Namespace,
    *,
    input_path: Path,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    return {
        "stage": 1,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "fsq_checkpoint": str(Path(args.fsq_checkpoint).expanduser().resolve()),
        "device": device,
        "max_tokens": args.max_tokens,
        "model_batch_size": args.model_batch_size,
        "seed": args.seed,
        "alpha": 0.5,
        "output_multiplier": 2,
        "pairing_strategy": getattr(args, "pairing_strategy", "random"),
        "bit_width": 12,
        "token_bit_rule": (
            "MSB-first; take endpoint 2 at the 2nd, 4th, ... differing bit"
        ),
        "random_token_bit_rule": (
            "For each token, randomly take floor(d/2) or ceil(d/2) of its d "
            "differing bits from endpoint 2; fixed seeded RNG per molecule"
        ),
        "duplicate_rmsd_threshold": args.duplicate_rmsd_threshold,
        "gate_threshold": args.gate_threshold,
        "max_precision_drop": args.max_precision_drop,
        "min_validity_rate": args.min_validity_rate,
        "min_decode_success_rate": args.min_decode_success_rate,
        "git_commit": _git_commit(),
        "source_hashes": {
            "conformer_interpolation.py": _sha256_file(Path(__file__)),
            "smiles_encoder_decoder_fsq.py": _sha256_file(
                Path(__file__).parents[1] / "data_processing" / "smiles_encoder_decoder_fsq.py"
            ),
        },
    }


def _load_records(
    selected_items: Sequence[tuple[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for molecule_key, _ in selected_items:
        record_path = _record_path(output_dir, molecule_key)
        if not record_path.exists():
            missing.append(molecule_key)
            continue
        with record_path.open("rb") as handle:
            record = pickle.load(handle)
        if record.get("molecule_key") != molecule_key:
            raise ValueError(f"Mismatched molecule record: {record_path}")
        records.append(record)
    if missing:
        raise RuntimeError(
            f"Generation is incomplete: {len(missing)} molecule records are missing."
        )
    return records


def _status_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        status = record["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def run_stage_one(args: argparse.Namespace) -> dict[str, Any]:
    if args.model_batch_size <= 0:
        raise ValueError("--model-batch-size must be positive.")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive.")
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with input_path.open("rb") as handle:
        ground_truth = pickle.load(handle)
    if not isinstance(ground_truth, dict):
        raise TypeError(f"Expected dictionary ground truth, got {type(ground_truth)!r}")

    selected_items = list(ground_truth.items())
    if args.max_molecules is not None:
        selected_items = selected_items[: args.max_molecules]

    config_path = output_dir / "config.json"
    config = _generation_config(
        args,
        input_path=input_path,
        output_dir=output_dir,
        device=args.device,
    )
    if config_path.exists():
        with config_path.open() as handle:
            existing_config = json.load(handle)
        _validate_resume_config(existing_config, config)
        config = existing_config
    elif args.phase == "evaluate":
        raise RuntimeError(f"Generation config does not exist: {config_path}")

    if args.phase in ("all", "generate"):
        model = configure_fsq_encoder(
            ckpt_path=args.fsq_checkpoint,
            device=args.device,
            max_tokens=args.max_tokens,
        )
        config["device"] = str(model.device)
        _atomic_json_dump(config, config_path)
        records: list[dict[str, Any]] = []
        for position, (molecule_key, value) in enumerate(selected_items, start=1):
            record_path = _record_path(output_dir, molecule_key)
            if args.resume and record_path.exists():
                with record_path.open("rb") as handle:
                    record = pickle.load(handle)
                if record.get("molecule_key") == molecule_key:
                    records.append(record)
                    continue

            references = _extract_conformers(value)
            record = process_molecule(
                molecule_key=molecule_key,
                references=references,
                model=model,
                seed=args.seed,
                duplicate_rmsd_threshold=args.duplicate_rmsd_threshold,
                model_batch_size=args.model_batch_size,
                pairing_strategy=getattr(args, "pairing_strategy", "random"),
            )
            _atomic_pickle_dump(record, record_path)
            records.append(record)
            if position % args.log_every == 0 or position == len(selected_items):
                complete = sum(record["status"] == "complete" for record in records)
                print(
                    f"Processed {position}/{len(selected_items)} molecules "
                    f"({complete} complete)",
                    flush=True,
                )
        if args.phase == "generate":
            generation_summary = {
                "phase": "generate",
                "config": config,
                "status_counts": _status_counts(records),
                "generation_complete": len(records) == len(selected_items),
            }
            _atomic_json_dump(
                generation_summary, output_dir / "generation_summary.json"
            )
            return generation_summary
    else:
        records = _load_records(selected_items, output_dir)

    records = _evaluate_records(records, ground_truth, output_dir, num_workers=args.num_workers)
    complete_records = [record for record in records if record["status"] == "complete"]
    aggregates = {}
    summaries = {}
    for condition in CONDITIONS:
        aggregates[condition], summaries[condition] = _aggregate_condition(
            complete_records, condition
        )
    rates = {
        condition: _summarize_rates(complete_records, condition)
        for condition in INTERPOLATION_CONDITIONS
    }
    decision = decide_stage_one(
        aggregates=aggregates,
        rates=rates,
        threshold=args.gate_threshold,
        max_precision_drop=args.max_precision_drop,
        min_validity_rate=args.min_validity_rate,
        min_decode_success_rate=args.min_decode_success_rate,
    )
    status_counts = _status_counts(records)
    eligible_count = len(records) - status_counts.get("skipped_too_few_anchors", 0)
    tokenizer_failure_count = status_counts.get("tokenizer_failure", 0) + status_counts.get(
        "token_sequence_mismatch", 0
    )

    serializable_summary = {
        "config": config,
        "status_counts": status_counts,
        "tokenizer_failure_rate": (
            tokenizer_failure_count / eligible_count if eligible_count else None
        ),
        "conditions": summaries,
        "rates": rates,
        "stage_one_decision": decision,
        "stage_two_authorized": decision["passed"],
    }
    _atomic_pickle_dump(
        {"summary": serializable_summary, "aggregates": aggregates},
        output_dir / "aggregate_metrics.pickle",
    )
    _atomic_json_dump(serializable_summary, output_dir / "summary.json")
    _write_csv(
        _per_molecule_rows(complete_records, args.gate_threshold),
        output_dir / "per_molecule_metrics.csv",
    )
    return serializable_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 1 of the FSQ conformer interpolation experiment."
    )
    parser.add_argument(
        "--input",
        default=str(get_data_path("distinct_smi")),
        help="GEOM ground-truth pickle (dict values contain a 'confs' list).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fsq-checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=218)
    parser.add_argument("--max-molecules", type=int)
    parser.add_argument(
        "--phase",
        choices=("all", "generate", "evaluate"),
        default="all",
        help="Split GPU generation from CPU metric evaluation when requested.",
    )
    parser.add_argument("--model-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--pairing-strategy",
        choices=("random", "balanced_local"),
        default="random",
        help=(
            "Choose endpoint pairs randomly or prioritize balanced nearest "
            "neighbors in tokenizer latent space."
        ),
    )
    parser.add_argument("--duplicate-rmsd-threshold", type=float, default=1e-3)
    parser.add_argument("--gate-threshold", type=float, default=MAIN_THRESHOLD)
    parser.add_argument("--max-precision-drop", type=float, default=0.05)
    parser.add_argument("--min-validity-rate", type=float, default=0.90)
    parser.add_argument("--min-decode-success-rate", type=float, default=0.90)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_stage_one(args)
    if args.phase == "generate":
        print(json.dumps(summary, indent=2))
        print("Generation phase complete. Run the evaluation phase next.")
        return
    decision = summary["stage_one_decision"]
    print(json.dumps(decision, indent=2))
    if decision["passed"]:
        print("Stage 1 passed. Stage 2 is authorized.")
    else:
        print("Stage 1 did not pass. Stage 2 must not run.")


if __name__ == "__main__":
    main()
