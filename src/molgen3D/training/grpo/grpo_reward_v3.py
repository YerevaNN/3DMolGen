"""GRPO Reward Function - GEOM-Drugs aligned entrypoint."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
import wandb

from molgen3D.evaluation.utils import extract_between

from .reward_utils import (
    EMPTY_FLOAT32,
    RewardProfiler,
    apply_posebusters_gate,
    combine_rewards,
    compute_distance_matrix,
    compute_matching_reward,
    compute_pairwise_rollout_distances,
    compute_quality_reward,
    compute_smooth_coverage_reward,
    get_cached_ground_truths,
    make_reward_rng,
    normalize_posebusters_config,
    parse_rollout_group,
    profile_section,
    sample_array,
    summarize_batch_metrics,
)

DEFAULT_ENABLE_HARD_RMSD_GATE = True


@dataclass
class GroupMetrics:
    K: int
    M: int
    graph_match_rate: float
    finite_rmsd_rate: float
    validity_rate: float
    d_min_mean: float
    d_min_p50: float
    d_min_p90: float
    num_matched: int
    refs_hit: int
    max_possible_matches: int
    match_efficiency: float
    r_qual_mean: float
    r_smcov_mean: float
    r_match_mean: float
    soft_cov_mean: float
    pct_gt_0_5: float
    fraction_under_delta: float
    matched_dists_sample: np.ndarray
    eligible_dists_sample: np.ndarray
    d_min_sample: np.ndarray
    soft_cov_sample: np.ndarray
    pairwise_sample: np.ndarray
    valid_count: int
    posebusters_checked: int
    posebusters_passed: int
    posebusters_failed: int
    posebusters_errors: int
    posebusters_time_ms: float
    sampled_percentiles: bool = False


def group_by_prompt(
    prompts: List[str],
    completions: List[str],
    expected_k: int,
    rollout_entropies: Optional[List[Optional[float]]] = None,
    completion_lengths: Optional[List[Optional[float]]] = None,
) -> List[Dict]:
    """Group flat batch into prompt groups."""
    groups: List[Dict[str, object]] = []
    active_groups: Dict[Tuple[str, str], Dict[str, object]] = {}

    for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
        canonical_smiles = extract_between(prompt, "[SMILES]", "[/SMILES]")
        if canonical_smiles is None:
            logger.warning("Prompt missing SMILES tags")
            canonical_smiles = ""

        key = (prompt, canonical_smiles)
        if key not in active_groups:
            group = {
                "prompt": prompt,
                "canonical_smiles": canonical_smiles,
                "completions": [],
                "indices": [],
                "entropy_values": [],
                "completion_lengths": [],
            }
            groups.append(group)
            active_groups[key] = group

        group = active_groups[key]
        group["completions"].append(completion)
        group["indices"].append(idx)
        entropy_val = (
            rollout_entropies[idx] if rollout_entropies is not None and idx < len(rollout_entropies) else None
        )
        group["entropy_values"].append(entropy_val)
        length_val = (
            completion_lengths[idx]
            if completion_lengths is not None and idx < len(completion_lengths)
            else None
        )
        group["completion_lengths"].append(length_val)

    if expected_k and groups:
        for group in groups:
            if len(group["completions"]) != expected_k:
                logger.debug(
                    "[reward_v3] group '%s' has %d completions, expected %d",
                    group["canonical_smiles"],
                    len(group["completions"]),
                    expected_k,
                )

    return groups


def compute_group_reward(
    canonical_smiles: str,
    completions: List[str],
    config,
    stats,
    rollout_entropies: Optional[List[Optional[float]]] = None,
    completion_lengths: Optional[List[Optional[float]]] = None,
    profiler: Optional[RewardProfiler] = None,
    distance_sample_limit: int = 0,
    enable_pairwise_logging: bool = False,
    pairwise_sample_limit: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, GroupMetrics]:
    """Compute rewards for a single prompt group (K rollouts)."""
    del rollout_entropies, completion_lengths  # kept for API parity

    K = len(completions)
    grpo_cfg = config.grpo
    delta = float(getattr(grpo_cfg, "delta", 0.75))
    sigma = float(getattr(grpo_cfg, "sigma", 0.25))
    rho = float(getattr(grpo_cfg, "rho", 0.75))
    lambda_qual = float(getattr(grpo_cfg, "lambda_qual", 1.0))
    lambda_smcov = float(getattr(grpo_cfg, "lambda_smcov", 1.0))
    lambda_match = float(getattr(grpo_cfg, "lambda_match", 1.0))
    r_floor = float(getattr(grpo_cfg, "r_floor", -1.0))
    hard_rmsd_gate = bool(getattr(grpo_cfg, "hard_rmsd_gate", DEFAULT_ENABLE_HARD_RMSD_GATE))
    rmsd_workers = int(getattr(grpo_cfg, "rmsd_workers", 0) or 0)

    reference_mols = get_cached_ground_truths(
        canonical_smiles,
        num_gt=grpo_cfg.max_ground_truths,
    )
    empty_rewards = np.full(K, r_floor, dtype=np.float32)

    def _build_metrics(**overrides) -> GroupMetrics:
        values = dict(
            K=K,
            M=len(reference_mols),
            graph_match_rate=overrides.get("graph_match_rate", 0.0),
            finite_rmsd_rate=overrides.get("finite_rmsd_rate", 0.0),
            validity_rate=overrides.get("validity_rate", 0.0),
            d_min_mean=overrides.get("d_min_mean", float("nan")),
            d_min_p50=overrides.get("d_min_p50", float("nan")),
            d_min_p90=overrides.get("d_min_p90", float("nan")),
            num_matched=overrides.get("num_matched", 0),
            refs_hit=overrides.get("refs_hit", 0),
            max_possible_matches=overrides.get("max_possible_matches", 0),
            match_efficiency=overrides.get("match_efficiency", 0.0),
            r_qual_mean=overrides.get("r_qual_mean", 0.0),
            r_smcov_mean=overrides.get("r_smcov_mean", 0.0),
            r_match_mean=overrides.get("r_match_mean", 0.0),
            soft_cov_mean=overrides.get("soft_cov_mean", float("nan")),
            pct_gt_0_5=overrides.get("pct_gt_0_5", float("nan")),
            fraction_under_delta=overrides.get("fraction_under_delta", 0.0),
            matched_dists_sample=overrides.get("matched_dists_sample", EMPTY_FLOAT32),
            eligible_dists_sample=overrides.get("eligible_dists_sample", EMPTY_FLOAT32),
            d_min_sample=overrides.get("d_min_sample", EMPTY_FLOAT32),
            soft_cov_sample=overrides.get("soft_cov_sample", EMPTY_FLOAT32),
            pairwise_sample=overrides.get("pairwise_sample", EMPTY_FLOAT32),
            valid_count=overrides.get("valid_count", 0),
            posebusters_checked=overrides.get("posebusters_checked", 0),
            posebusters_passed=overrides.get("posebusters_passed", 0),
            posebusters_failed=overrides.get("posebusters_failed", 0),
            posebusters_errors=overrides.get("posebusters_errors", 0),
            posebusters_time_ms=overrides.get("posebusters_time_ms", 0.0),
            sampled_percentiles=overrides.get("sampled_percentiles", False),
        )
        return GroupMetrics(**values)

    if not reference_mols:
        stats.failed_ground_truth += K
        return empty_rewards, _build_metrics()

    rollout_mols, graph_mask, parsed_mask = parse_rollout_group(
        canonical_smiles,
        completions,
        stats,
        profiler,
    )
    graph_match_rate = float(np.mean(graph_mask)) if K > 0 else 0.0

    if not np.any(graph_mask):
        return empty_rewards, _build_metrics(graph_match_rate=graph_match_rate)

    base_mask = graph_mask & parsed_mask
    pose_cfg = normalize_posebusters_config(getattr(grpo_cfg, "posebusters", None))
    with profile_section(profiler, "reward_posebusters"):
        pose_mask, pose_summary = apply_posebusters_gate(
            rollout_mols,
            base_mask.astype(bool, copy=False),
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

    valid_mask = pose_mask.astype(bool, copy=False)
    mask_for_distance = valid_mask.astype(np.int32, copy=False)
    ref_cache_key = f"{canonical_smiles}:{len(reference_mols)}"
    with profile_section(profiler, "reward_rmsd"):
        distance_matrix = compute_distance_matrix(
            rollout_mols,
            reference_mols,
            mask_for_distance,
            rmsd_workers=rmsd_workers,
            ref_cache_key=ref_cache_key,
        )

    min_distances = np.min(distance_matrix, axis=1)
    finite_mask = np.isfinite(min_distances)
    problematic_mask = mask_for_distance.astype(bool) & (~finite_mask)
    num_problematic = int(np.count_nonzero(problematic_mask))
    if num_problematic > 0:
        stats.failed_rmsd += num_problematic
        message = (
            f"[reward_v3] {num_problematic} PoseBusters-valid rollouts lacked finite RMSD; "
            "quality/coverage rewards will be zero for them."
        )
        logger.warning(message if hard_rmsd_gate else message.replace("will be zero", "applying reward floor"))
        valid_mask &= finite_mask
        mask_for_distance = valid_mask.astype(np.int32, copy=False)

    with profile_section(profiler, "reward_smcov"):
        r_smcov, (soft_cov_mean, soft_cov_pcts, soft_cov_values) = compute_smooth_coverage_reward(
            distance_matrix,
            mask_for_distance,
            rho,
            return_details=True,
        )

    with profile_section(profiler, "reward_qual"):
        r_qual = compute_quality_reward(distance_matrix, mask_for_distance, sigma)

    eligible_matrix = valid_mask[:, None] & (distance_matrix < delta)
    refs_hit = int(np.count_nonzero(eligible_matrix.any(axis=0)))
    num_valid = int(np.count_nonzero(valid_mask))
    max_possible_matches = min(num_valid, refs_hit)

    with profile_section(profiler, "reward_match"):
        r_match, num_matched, _num_eligible_edges, matched_pairs = compute_matching_reward(
            distance_matrix,
            mask_for_distance,
            delta,
            eligible_matrix=eligible_matrix,
        )

    rewards = combine_rewards(
        r_qual,
        r_smcov,
        r_match,
        mask_for_distance,
        lambda_qual,
        lambda_smcov,
        lambda_match,
        r_floor,
    )

    finite_rmsd_rate = float(np.mean(np.isfinite(min_distances))) if K > 0 else 0.0
    validity_rate = float(np.mean(valid_mask)) if K > 0 else 0.0
    finite_d_valid = min_distances[valid_mask]
    finite_d_valid = finite_d_valid[np.isfinite(finite_d_valid)]
    under_threshold = float(np.mean(finite_d_valid < delta)) if finite_d_valid.size > 0 else 0.0

    def _sample(values: np.ndarray) -> Tuple[np.ndarray, bool]:
        if values.size == 0:
            return EMPTY_FLOAT32, False
        limit = distance_sample_limit if distance_sample_limit > 0 else values.size
        if limit <= 0:
            return EMPTY_FLOAT32, False
        return sample_array(values, limit, rng=rng)

    matched_dists = (
        np.array([distance_matrix[i, j] for (i, j) in matched_pairs], dtype=np.float32)
        if matched_pairs
        else EMPTY_FLOAT32
    )
    eligible_dists = (
        distance_matrix[eligible_matrix].astype(np.float32) if np.any(eligible_matrix) else EMPTY_FLOAT32
    )

    matched_sample, matched_sampled = _sample(matched_dists)
    eligible_sample, eligible_sampled = _sample(eligible_dists)
    d_min_sample, d_min_sampled = _sample(finite_d_valid.astype(np.float32, copy=False))
    soft_cov_sample, _ = _sample(soft_cov_values.astype(np.float32, copy=False))

    pairwise_sample_cap = (
        pairwise_sample_limit if pairwise_sample_limit is not None else distance_sample_limit
    )
    if enable_pairwise_logging:
        max_pairs = max(pairwise_sample_cap or 0, 0)
        pairwise_dists = compute_pairwise_rollout_distances(
            rollout_mols,
            valid_mask,
            max_samples=max_pairs,
        )
    else:
        pairwise_dists = EMPTY_FLOAT32

    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count > 0:
        r_qual_mean = float(np.mean(r_qual[valid_mask]))
        r_smcov_mean = float(np.mean(r_smcov[valid_mask]))
        r_match_mean = float(np.mean(r_match[valid_mask]))
    else:
        r_qual_mean = r_smcov_mean = r_match_mean = 0.0

    for i in range(K):
        if valid_mask[i] and np.isfinite(min_distances[i]):
            stats.add_rmsd(float(min_distances[i]))

    sampling_flag = matched_sampled or eligible_sampled or d_min_sampled
    match_efficiency = float(num_matched) / max_possible_matches if max_possible_matches > 0 else 0.0

    metrics = _build_metrics(
        graph_match_rate=graph_match_rate,
        finite_rmsd_rate=finite_rmsd_rate,
        validity_rate=validity_rate,
        d_min_mean=float(np.mean(finite_d_valid)) if finite_d_valid.size > 0 else float("nan"),
        d_min_p50=float(np.percentile(finite_d_valid, 50)) if finite_d_valid.size > 0 else float("nan"),
        d_min_p90=float(np.percentile(finite_d_valid, 90)) if finite_d_valid.size > 0 else float("nan"),
        num_matched=num_matched,
        refs_hit=refs_hit,
        max_possible_matches=max_possible_matches,
        match_efficiency=match_efficiency,
        r_qual_mean=r_qual_mean,
        r_smcov_mean=r_smcov_mean,
        r_match_mean=r_match_mean,
        soft_cov_mean=soft_cov_mean,
        pct_gt_0_5=float(soft_cov_pcts[2]) if len(soft_cov_pcts) >= 3 else float("nan"),
        fraction_under_delta=under_threshold,
        matched_dists_sample=matched_sample,
        eligible_dists_sample=eligible_sample,
        d_min_sample=d_min_sample,
        soft_cov_sample=soft_cov_sample,
        pairwise_sample=pairwise_dists,
        valid_count=valid_count,
        posebusters_checked=pose_checked,
        posebusters_passed=pose_passed,
        posebusters_failed=pose_failed,
        posebusters_errors=pose_errors,
        posebusters_time_ms=pose_summary["time_ms"],
        sampled_percentiles=sampling_flag,
    )

    return rewards, metrics


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
    del tokenizer  # unused

    expected_k = config.grpo.num_generations
    lambda_qual = float(getattr(config.grpo, "lambda_qual", 1.0))
    lambda_smcov = float(getattr(config.grpo, "lambda_smcov", 1.0))
    lambda_match = float(getattr(config.grpo, "lambda_match", 1.0))
    distance_sample_limit = int(getattr(config.grpo, "log_distance_samples_per_group", 0))
    profile_enabled = bool(getattr(config.grpo, "profile_rewards", False))
    profiler = RewardProfiler(enabled=profile_enabled)
    total_start = time.perf_counter() if profile_enabled else None
    reward_rng = make_reward_rng(config, stats)
    log_every_steps = max(int(getattr(config.grpo, "log_every_steps", 1)), 1)
    pairwise_freq = max(int(getattr(config.grpo, "pairwise_rmsd_log_every", 50)), 1)
    pairwise_flag = bool(getattr(config.grpo, "enable_pairwise_rmsd_logging", False))
    initial_processed = getattr(stats, "processed_prompts", 0)
    denom = max(int(getattr(config.grpo, "num_generations", 1)), 1)
    step_index = getattr(stats, "global_step", None)
    if step_index is None:
        step_index = initial_processed // denom
    should_log_pairwise = pairwise_flag and (step_index % pairwise_freq == 0)

    if completion_lengths is not None:
        completion_lengths = [
            None if length is None else int(length) for length in completion_lengths
        ]

    groups = group_by_prompt(
        prompts,
        completions,
        expected_k,
        completion_entropies,
        completion_lengths,
    )

    final_rewards = [0.0] * len(completions)
    metrics_list: List[GroupMetrics] = []

    for group in groups:
        stats.processed_prompts += len(group["completions"])
        stats.distinct_prompts += 1

        rewards, group_metrics = compute_group_reward(
            canonical_smiles=group["canonical_smiles"],
            completions=group["completions"],
            config=config,
            stats=stats,
            profiler=profiler if profile_enabled else None,
            distance_sample_limit=distance_sample_limit,
            enable_pairwise_logging=should_log_pairwise,
            pairwise_sample_limit=distance_sample_limit if distance_sample_limit > 0 else None,
            rng=reward_rng,
        )

        for local_idx, global_idx in enumerate(group["indices"]):
            final_rewards[global_idx] = float(rewards[local_idx])

        metrics_list.append(group_metrics)

    with profile_section(profiler, "reward_logging"):
        essential_metrics = summarize_batch_metrics(metrics_list, lambda_qual, lambda_smcov, lambda_match)
        should_log_metrics = wandb.run is not None and (step_index % log_every_steps == 0)
        if should_log_metrics:
            wandb.log(essential_metrics, step=step_index)

        gm = essential_metrics.get("graph_match_rate", 0.0)
        fr = essential_metrics.get("finite_rmsd_rate", 0.0)
        vr = essential_metrics.get("validity_rate", 0.0)
        matched_total = int(essential_metrics.get("match/matched_total", 0.0))
        max_possible_total = int(essential_metrics.get("match/max_possible_total", 0.0))
        match_eff = essential_metrics.get("match/match_efficiency", 0.0)
        rq = essential_metrics.get("reward/component_quality", 0.0)
        rs = essential_metrics.get("reward/component_smcov", 0.0)
        rm = essential_metrics.get("reward/component_match", 0.0)
        pose_checked = int(essential_metrics.get("posebusters/checked_total", 0.0))
        pose_pass_rate = essential_metrics.get("posebusters/pass_rate", 0.0)

        logger.info(
            "[reward_v3] validity: graph={:.3f}, finite_rmsd={:.3f}, final={:.3f}; "
            "matching: max_possible={}, matched={}, eff={:.3f}; "
            "rewards: r_qual={:.3f}, r_smcov={:.3f}, r_match={:.3f}; "
            "posebusters: checked={}, pass_rate={:.3f}",
            gm,
            fr,
            vr,
            max_possible_total,
            matched_total,
            match_eff,
            rq,
            rs,
            rm,
            pose_checked,
            pose_pass_rate,
        )

        if profile_enabled and total_start is not None:
            profiling_metrics = {
                "profiling/reward_total_s": time.perf_counter() - total_start,
                "profiling/reward_parse_s": profiler.sections.get("reward_parse", 0.0),
                "profiling/reward_posebusters_s": profiler.sections.get("reward_posebusters", 0.0),
                "profiling/reward_rmsd_s": profiler.sections.get("reward_rmsd", 0.0),
                "profiling/reward_smcov_s": profiler.sections.get("reward_smcov", 0.0),
                "profiling/reward_match_s": profiler.sections.get("reward_match", 0.0),
                "profiling/reward_logging_s": profiler.sections.get("reward_logging", 0.0),
            }
            if wandb.run is not None:
                wandb.log(profiling_metrics, step=step_index)
            logger.info(
                "[reward_v3] profiler totals (s): {}",
                ", ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in profiling_metrics.items()),
            )

    return final_rewards
