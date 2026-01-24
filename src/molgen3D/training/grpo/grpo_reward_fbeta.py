"""GRPO Reward Function - F-beta coverage with multi-conformer completions."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger
import wandb
from rdkit import Chem

from molgen3D.data_processing.smiles_encoder_decoder import (
    decode_cartesian_binned,
    decode_cartesian_v2,
    get_bins_for_coords,
    strip_smiles,
)
from molgen3D.evaluation.utils import extract_between, same_molecular_graph
from molgen3D.training.grpo.grpo_reward_v3 import group_by_prompt
from .reward_utils import (
    RewardProfiler,
    apply_posebusters_gate,
    compute_distance_matrix,
    compute_rmsd_safe,
    get_cached_ground_truths,
    make_reward_rng,
    normalize_posebusters_config,
    profile_section,
)

DEFAULT_TARGET_K = 8
DEFAULT_MIN_VALID_TO_SCORE = 0
DEFAULT_DROP_IF_VALID_LT = 3
DEFAULT_BETA = 1.5
DEFAULT_GAMMA = 1.5
DEFAULT_DELTA = 0.75
DEFAULT_DUP_RMSD_TAU = 0.3
DEFAULT_RECALL_REF_SAMPLE_M = 12
DEFAULT_BINNED_RANGES = [(-13.0, 13.0), (-13.0, 13.0), (-13.0, 13.0)]
DEFAULT_BIN_SIZE = 0.104
DEFAULT_WARMUP_LAMBDA = 0.1
DEFAULT_WARMUP_SIGMA = 0.75

_CONFORMER_BLOCK_RE = re.compile(r"\[CONFORMER\](.*?)\[/CONFORMER\]", re.DOTALL)
_BIN_LOGGED = False


@dataclass
class CompletionMetrics:
    n_blocks_found: int
    n_blocks_parsed: int
    t_valid: int
    cov_r: float
    cov_p: float
    fbeta: float
    completion_factor: float
    uniq_frac: float
    reward: float
    min_rmsd_gen: np.ndarray
    min_rmsd_ref: np.ndarray
    sample_completion: Optional[str] = None
    sample_generated_smiles: Optional[str] = None
    sample_stats: Optional[Dict[str, object]] = None


def _get_config_value(config, name: str, default):
    return getattr(getattr(config, "grpo", config), name, default)


def _extract_conformer_blocks(completion: str, max_blocks: int) -> Tuple[List[str], int]:
    if not completion:
        return [], 0
    blocks = _CONFORMER_BLOCK_RE.findall(completion)
    if max_blocks > 0:
        blocks = blocks[:max_blocks]
    return blocks, completion.count("[CONFORMER]")


def _decode_conformer_block(block: str, binned: bool, bins) -> Optional[Chem.Mol]:
    if not block:
        return None
    if binned:
        return decode_cartesian_binned(block, bins)
    return decode_cartesian_v2(block)


def _f_beta(cov_r: float, cov_p: float, beta: float) -> float:
    beta_sq = beta * beta
    denom = beta_sq * cov_p + cov_r + 1e-8
    if denom <= 0.0:
        return 0.0
    return float((1.0 + beta_sq) * cov_p * cov_r / denom)


def _count_unique_conformers(mols: Sequence[Chem.Mol], tau: float) -> int:
    if not mols:
        return 0
    reps: List[Chem.Mol] = []
    for mol in mols:
        if mol is None:
            continue
        is_dup = False
        for rep in reps:
            if compute_rmsd_safe(mol, rep) < tau:
                is_dup = True
                break
        if not is_dup:
            reps.append(mol)
    return len(reps)


def _sample_references(
    references: Sequence[Chem.Mol],
    max_samples: int,
    rng: np.random.Generator,
) -> List[Chem.Mol]:
    if max_samples <= 0 or len(references) <= max_samples:
        return list(references)
    indices = rng.choice(len(references), size=max_samples, replace=False)
    return [references[i] for i in indices]


def _compute_cov_metrics(
    valid_mols: List[Chem.Mol],
    references: List[Chem.Mol],
    recall_refs: List[Chem.Mol],
    delta: float,
    rmsd_workers: int,
    ref_cache_key: str,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    if not valid_mols or not references:
        return 0.0, 0.0, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    validity = np.ones(len(valid_mols), dtype=np.int32)
    D_prec = compute_distance_matrix(
        valid_mols,
        references,
        validity,
        rmsd_workers=rmsd_workers,
        ref_cache_key=ref_cache_key,
    )
    min_gen = np.min(D_prec, axis=1)
    cov_p = float(np.mean(min_gen < delta)) if min_gen.size > 0 else 0.0

    D_rec = compute_distance_matrix(
        valid_mols,
        recall_refs,
        validity,
        rmsd_workers=rmsd_workers,
        ref_cache_key=f"{ref_cache_key}:recall:{len(recall_refs)}",
    )
    min_ref = np.min(D_rec, axis=0)
    cov_r = float(np.mean(min_ref < delta)) if min_ref.size > 0 else 0.0

    return cov_r, cov_p, min_gen.astype(np.float32, copy=False), min_ref.astype(np.float32, copy=False)


def _soft_precision(min_gen: np.ndarray, sigma: float) -> float:
    if min_gen.size == 0:
        return 0.0
    sigma = max(float(sigma), 1e-8)
    values = np.exp(-((min_gen.astype(np.float32) ** 2) / (sigma ** 2)))
    return float(np.mean(values)) if values.size > 0 else 0.0


def _compute_completion_reward(
    canonical_smiles: str,
    completion: str,
    references: List[Chem.Mol],
    config,
    stats,
    rng: np.random.Generator,
    profiler: Optional[RewardProfiler],
    bins,
    binned: bool,
    target_k: int,
) -> Tuple[float, CompletionMetrics, Dict[str, int]]:
    delta = float(_get_config_value(config, "fbeta_delta", DEFAULT_DELTA))
    beta = float(_get_config_value(config, "fbeta_beta", DEFAULT_BETA))
    gamma = float(_get_config_value(config, "fbeta_gamma", DEFAULT_GAMMA))
    dup_tau = float(_get_config_value(config, "fbeta_dup_rmsd_tau", DEFAULT_DUP_RMSD_TAU))
    min_valid = int(_get_config_value(config, "fbeta_min_valid_to_score", DEFAULT_MIN_VALID_TO_SCORE))
    drop_if_valid_lt = int(_get_config_value(config, "fbeta_drop_if_valid_lt", DEFAULT_DROP_IF_VALID_LT))
    recall_sample = int(_get_config_value(config, "fbeta_recall_ref_sample", DEFAULT_RECALL_REF_SAMPLE_M))
    rmsd_workers = int(_get_config_value(config, "rmsd_workers", 0) or 0)
    warmup_lambda = float(_get_config_value(config, "fbeta_warmup_lambda", DEFAULT_WARMUP_LAMBDA))
    warmup_sigma = float(_get_config_value(config, "fbeta_warmup_sigma", DEFAULT_WARMUP_SIGMA))

    blocks, blocks_found = _extract_conformer_blocks(completion, target_k)
    parsed_blocks = len(blocks)

    failure_counts = {
        "decode_fail": 0,
        "rdkit_fail": 0,
        "smiles_mismatch": 0,
        "posebusters_fail": 0,
    }

    if not references:
        stats.failed_ground_truth += 1
        metrics = CompletionMetrics(
            n_blocks_found=blocks_found,
            n_blocks_parsed=parsed_blocks,
            t_valid=0,
            cov_r=0.0,
            cov_p=0.0,
            fbeta=0.0,
            completion_factor=0.0,
            uniq_frac=1.0,
            reward=0.0,
            min_rmsd_gen=np.array([], dtype=np.float32),
            min_rmsd_ref=np.array([], dtype=np.float32),
        )
        return 0.0, metrics, failure_counts

    rollout_mols: List[Optional[Chem.Mol]] = []
    base_valid_flags: List[bool] = []
    with profile_section(profiler, "reward_parse"):
        for block in blocks:
            mol = None
            try:
                mol = _decode_conformer_block(block, binned=binned, bins=bins)
            except Exception:
                failure_counts["decode_fail"] += 1
                stats.failed_conformer_generation += 1
                rollout_mols.append(None)
                base_valid_flags.append(False)
                continue

            if mol is None or mol.GetNumConformers() == 0:
                failure_counts["rdkit_fail"] += 1
                stats.failed_conformer_generation += 1
                rollout_mols.append(None)
                base_valid_flags.append(False)
                continue

            generated_smiles = strip_smiles(block)
            if not same_molecular_graph(canonical_smiles, generated_smiles):
                failure_counts["smiles_mismatch"] += 1
                stats.failed_matching_smiles += 1
                rollout_mols.append(None)
                base_valid_flags.append(False)
                continue

            rollout_mols.append(mol)
            base_valid_flags.append(True)

    base_valid_mask = np.array(base_valid_flags, dtype=bool)
    pose_cfg = normalize_posebusters_config(getattr(config.grpo, "posebusters", None))
    with profile_section(profiler, "reward_posebusters"):
        pose_mask, pose_summary = apply_posebusters_gate(
            rollout_mols,
            base_valid_mask.astype(bool, copy=False),
            pose_cfg,
        )

    pose_checked = int(pose_summary["checked"])
    pose_passed = int(pose_summary["passed"])
    pose_failed = int(pose_summary["failed"])
    pose_errors = int(pose_summary["errors"])
    stats.posebusters_checked += pose_checked
    stats.posebusters_failed += pose_failed
    stats.posebusters_errors += pose_errors
    stats.posebusters_time_ms += pose_summary["time_ms"]
    stats.posebusters_successes += pose_passed
    stats.posebusters_failures += pose_failed + pose_errors

    if base_valid_mask.any():
        pose_fail_mask = base_valid_mask & (~pose_mask)
        failure_counts["posebusters_fail"] += int(np.count_nonzero(pose_fail_mask))

    valid_mols = [mol for mol, flag in zip(rollout_mols, pose_mask) if flag and mol is not None]
    t_valid = len(valid_mols)
    sample_completion = None
    sample_generated_smiles = None
    sample_stats: Optional[Dict[str, object]] = None
    if t_valid > 0:
        sample_completion = completion
        first_valid_idx = int(np.where(pose_mask)[0][0])
        if 0 <= first_valid_idx < len(blocks):
            sample_generated_smiles = strip_smiles(blocks[first_valid_idx])
        sample_stats = {
            "blocks_found": blocks_found,
            "blocks_parsed": parsed_blocks,
            "valid": t_valid,
            "decode_fail": failure_counts["decode_fail"],
            "rdkit_fail": failure_counts["rdkit_fail"],
            "smiles_mismatch": failure_counts["smiles_mismatch"],
            "posebusters_fail": failure_counts["posebusters_fail"],
        }

    completion_factor = (t_valid / float(max(target_k, 1))) ** gamma if t_valid > 0 else 0.0
    uniq_count = _count_unique_conformers(valid_mols, dup_tau)
    uniq_frac = float(uniq_count) / float(t_valid) if t_valid > 0 else 1.0

    ref_cache_key = f"{canonical_smiles}:{len(references)}"
    recall_refs = _sample_references(references, recall_sample, rng)
    with profile_section(profiler, "reward_rmsd"):
        cov_r, cov_p, min_gen, min_ref = _compute_cov_metrics(
            valid_mols,
            references,
            recall_refs,
            delta,
            rmsd_workers,
            ref_cache_key,
        )
    fbeta = _f_beta(cov_r, cov_p, beta)
    soft_p = _soft_precision(min_gen, warmup_sigma)
    reward = (fbeta + warmup_lambda * soft_p) * completion_factor * uniq_frac

    if t_valid < min_valid:
        reward = 0.0
    if drop_if_valid_lt > 0 and t_valid < drop_if_valid_lt:
        reward = 0.0

    if t_valid > 0:
        stats.successful_generations += 1
        for value in min_gen:
            if np.isfinite(value):
                stats.add_rmsd(float(value))
    elif parsed_blocks == 0:
        stats.failed_conformer_generation += 1

    metrics = CompletionMetrics(
        n_blocks_found=blocks_found,
        n_blocks_parsed=parsed_blocks,
        t_valid=t_valid,
        cov_r=cov_r,
        cov_p=cov_p,
        fbeta=fbeta,
        completion_factor=completion_factor,
        uniq_frac=uniq_frac,
        reward=reward,
        min_rmsd_gen=min_gen,
        min_rmsd_ref=min_ref,
        sample_completion=sample_completion,
        sample_generated_smiles=sample_generated_smiles,
        sample_stats=sample_stats,
    )
    return reward, metrics, failure_counts


def _summarize_metrics(
    metrics_list: Sequence[CompletionMetrics],
    failure_totals: Dict[str, int],
    target_k: int,
) -> Dict[str, float]:
    if not metrics_list:
        return {}

    n = len(metrics_list)
    n_blocks_found = np.array([m.n_blocks_found for m in metrics_list], dtype=np.float32)
    n_blocks_parsed = np.array([m.n_blocks_parsed for m in metrics_list], dtype=np.float32)
    t_valid = np.array([m.t_valid for m in metrics_list], dtype=np.float32)
    cov_r = np.array([m.cov_r for m in metrics_list], dtype=np.float32)
    cov_p = np.array([m.cov_p for m in metrics_list], dtype=np.float32)
    fbeta = np.array([m.fbeta for m in metrics_list], dtype=np.float32)
    completion = np.array([m.completion_factor for m in metrics_list], dtype=np.float32)
    uniq = np.array([m.uniq_frac for m in metrics_list], dtype=np.float32)
    reward = np.array([m.reward for m in metrics_list], dtype=np.float32)

    all_min_gen = np.concatenate([m.min_rmsd_gen for m in metrics_list if m.min_rmsd_gen.size > 0]) \
        if any(m.min_rmsd_gen.size > 0 for m in metrics_list) else np.array([], dtype=np.float32)
    all_min_ref = np.concatenate([m.min_rmsd_ref for m in metrics_list if m.min_rmsd_ref.size > 0]) \
        if any(m.min_rmsd_ref.size > 0 for m in metrics_list) else np.array([], dtype=np.float32)

    total_blocks_parsed = int(n_blocks_parsed.sum())
    total_blocks_found = int(n_blocks_found.sum())

    def _safe_mean(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if arr.size > 0 else 0.0

    def _safe_std(arr: np.ndarray) -> float:
        return float(np.std(arr)) if arr.size > 0 else 0.0

    result = {
        "reward/final_mean": _safe_mean(reward),
        "reward/final_std": _safe_std(reward),
        "reward/fbeta_mean": _safe_mean(fbeta),
        "reward/cov_r_mean": _safe_mean(cov_r),
        "reward/cov_p_mean": _safe_mean(cov_p),
        "reward/completion_mean": _safe_mean(completion),
        "reward/uniq_frac_mean": _safe_mean(uniq),
        "reward/t_valid_mean": _safe_mean(t_valid),
        "parse/n_blocks_found_mean": _safe_mean(n_blocks_found),
        "parse/n_blocks_parsed_mean": _safe_mean(n_blocks_parsed),
        "parse/fraction_with_ltK_blocks": float(np.mean(n_blocks_parsed < target_k)),
        "parse/fraction_empty_completion": float(np.mean(n_blocks_found == 0)),
        "gate/decode_fail_rate": float(failure_totals["decode_fail"] / max(total_blocks_parsed, 1)),
        "gate/rdkit_fail_rate": float(failure_totals["rdkit_fail"] / max(total_blocks_parsed, 1)),
        "gate/smiles_mismatch_rate": float(failure_totals["smiles_mismatch"] / max(total_blocks_parsed, 1)),
        "gate/posebusters_fail_rate": float(failure_totals["posebusters_fail"] / max(total_blocks_parsed, 1)),
        "gate/valid_rate": float(np.mean(t_valid > 0)),
        "rmsd/gen_min_mean": _safe_mean(all_min_gen),
        "rmsd/gen_min_p50": float(np.percentile(all_min_gen, 50)) if all_min_gen.size > 0 else 0.0,
        "rmsd/gen_min_p90": float(np.percentile(all_min_gen, 90)) if all_min_gen.size > 0 else 0.0,
        "rmsd/ref_min_mean": _safe_mean(all_min_ref),
        "rmsd/ref_min_p50": float(np.percentile(all_min_ref, 50)) if all_min_ref.size > 0 else 0.0,
        "rmsd/ref_min_p90": float(np.percentile(all_min_ref, 90)) if all_min_ref.size > 0 else 0.0,
        "collapse/uniq_frac_mean": _safe_mean(uniq),
        "collapse/percent_rollouts_all_same": float(np.mean((t_valid > 1) & (uniq <= 1.0 / np.maximum(t_valid, 1.0)))),
        "reward/nonzero_frac": float(np.mean(reward > 0.0)),
        "reward/t_valid_p50": float(np.percentile(t_valid, 50)) if t_valid.size > 0 else 0.0,
        "reward/t_valid_p90": float(np.percentile(t_valid, 90)) if t_valid.size > 0 else 0.0,
    }
    result["parse/n_blocks_found_total"] = float(total_blocks_found)
    result["parse/n_blocks_parsed_total"] = float(total_blocks_parsed)
    return result


def reward_function(
    prompts: List[str],
    completions: List[str],
    stats,
    tokenizer,
    config,
    completion_entropies: Optional[List[Optional[float]]] = None,
    completion_lengths: Optional[List[Optional[float]]] = None,
) -> List[float]:
    """Main GRPO reward function (TRL-compatible)."""
    del tokenizer, completion_entropies, completion_lengths  # unused

    target_k = int(_get_config_value(config, "target_conformers", DEFAULT_TARGET_K))
    binned = bool(_get_config_value(config, "fbeta_use_binned_decoder", False))
    bin_ranges = _get_config_value(config, "fbeta_binned_ranges", DEFAULT_BINNED_RANGES)
    bin_size = float(_get_config_value(config, "fbeta_binned_bin_size", DEFAULT_BIN_SIZE))
    rmsd_workers = int(_get_config_value(config, "rmsd_workers", 0) or 0)
    profile_enabled = bool(_get_config_value(config, "profile_rewards", False))
    log_every_steps = max(int(_get_config_value(config, "log_every_steps", 1)), 1)
    log_success_every = max(int(_get_config_value(config, "log_success_sample_every", 0) or 0), 0)
    log_success_chars = max(int(_get_config_value(config, "log_success_sample_chars", 400) or 0), 0)
    log_debug_every = max(int(_get_config_value(config, "log_debug_sample_every", 0) or 0), 0)
    log_debug_chars = max(int(_get_config_value(config, "log_debug_sample_chars", 400) or 0), 0)
    log_full_every = max(int(_get_config_value(config, "log_full_sample_every", 0) or 0), 0)

    bins = None
    if binned:
        if not bin_ranges:
            bin_ranges = DEFAULT_BINNED_RANGES
        bins = get_bins_for_coords(bin_ranges, bin_size=bin_size)
        global _BIN_LOGGED
        if not _BIN_LOGGED:
            bin_sizes = [len(axis) for axis in bins] if bins is not None else []
            logger.info(
                "[reward_fbeta] binned decode enabled: ranges={}, bin_size={}, bin_counts={}",
                bin_ranges,
                bin_size,
                bin_sizes,
            )
            _BIN_LOGGED = True

    profiler = RewardProfiler(enabled=profile_enabled)
    total_start = time.perf_counter() if profile_enabled else None
    reward_rng = make_reward_rng(config, stats)

    expected_k = int(_get_config_value(config, "num_generations", 1))
    initial_processed = getattr(stats, "processed_prompts", 0)
    denom = max(int(_get_config_value(config, "num_generations", 1)), 1)
    step_index = getattr(stats, "global_step", None)
    if step_index is None:
        step_index = initial_processed // denom

    groups = group_by_prompt(prompts, completions, expected_k)
    final_rewards = [0.0] * len(completions)
    metrics_list: List[CompletionMetrics] = []
    failure_totals = {
        "decode_fail": 0,
        "rdkit_fail": 0,
        "smiles_mismatch": 0,
        "posebusters_fail": 0,
    }

    for group in groups:
        stats.processed_prompts += len(group["completions"])
        stats.distinct_prompts += 1

        canonical_smiles = group["canonical_smiles"]
        references = get_cached_ground_truths(
            canonical_smiles,
            num_gt=_get_config_value(config, "max_ground_truths", 0),
        )

        for completion, global_idx in zip(group["completions"], group["indices"]):
            reward, metrics, failure_counts = _compute_completion_reward(
                canonical_smiles=canonical_smiles,
                completion=completion,
                references=references,
                config=config,
                stats=stats,
                rng=reward_rng,
                profiler=profiler if profile_enabled else None,
                bins=bins,
                binned=binned,
                target_k=target_k,
            )
            final_rewards[global_idx] = float(reward)
            metrics_list.append(metrics)
            for key in failure_totals:
                failure_totals[key] += failure_counts.get(key, 0)

    with profile_section(profiler, "reward_logging"):
        metrics = _summarize_metrics(metrics_list, failure_totals, target_k)
        should_log_metrics = wandb.run is not None and (step_index % log_every_steps == 0)
        if should_log_metrics and metrics:
            wandb.log(metrics)

        if metrics:
            logger.info(
                "[reward_fbeta] cov_r={:.3f}, cov_p={:.3f}, fbeta={:.3f}, reward_mean={:.3f}, t_valid_mean={:.2f}",
                metrics.get("reward/cov_r_mean", 0.0),
                metrics.get("reward/cov_p_mean", 0.0),
                metrics.get("reward/fbeta_mean", 0.0),
                metrics.get("reward/final_mean", 0.0),
                metrics.get("reward/t_valid_mean", 0.0),
            )

        if profile_enabled and total_start is not None:
            profiling_metrics = {
                "profiling/reward_total_s": time.perf_counter() - total_start,
                "profiling/reward_parse_s": profiler.sections.get("reward_parse", 0.0),
                "profiling/reward_posebusters_s": profiler.sections.get("reward_posebusters", 0.0),
                "profiling/reward_rmsd_s": profiler.sections.get("reward_rmsd", 0.0),
                "profiling/reward_logging_s": profiler.sections.get("reward_logging", 0.0),
            }
            logger.info(
                "[reward_fbeta] profiler totals (s): {}",
                ", ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in profiling_metrics.items()),
            )

    return final_rewards


def _run_fbeta_reward_sanity_checks() -> None:
    """Minimal sanity checks for parsing and reward math."""
    blocks = "[CONFORMER]AAA[/CONFORMER][CONFORMER]BBB[/CONFORMER]"
    parsed, found = _extract_conformer_blocks(blocks, 8)
    assert found == 2
    assert len(parsed) == 2
    assert _f_beta(0.0, 0.5, 1.5) == 0.0
    assert _f_beta(0.5, 0.5, 1.5) > 0.0
    print("[reward_fbeta] sanity checks passed.")


if __name__ == "__main__":
    _run_fbeta_reward_sanity_checks()
