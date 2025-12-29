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
    graph_match_count: int
    rdkit_parse_rate: float
    rdkit_parse_count: int
    base_valid_rate: float
    base_valid_count: int
    final_valid_rate: float
    valid_rollouts: int
    d_min_mean: float
    frac_under_delta: float
    frac_under_2delta: float
    refs_hit: int
    cov_ratio: float
    unique_nearest_refs: int
    nearest_collision_rate: float
    num_matched: int
    max_possible_matches: int
    match_efficiency: float
    eligible_edges: int
    eligible_edge_density: float
    soft_cov_mean: float
    pct_cov_gt_0p1: float
    pct_cov_gt_0p5: float
    comp_qual_sum: float
    comp_smcov_sum: float
    comp_match_sum: float
    pose_checked: int
    pose_passed: int
    pose_errors: int
    d_min_values: np.ndarray
    matched_dists: np.ndarray
    pairwise_dists: np.ndarray
    reward_total_values: np.ndarray


METRIC_KEYS: List[str] = [
    "gate/graph_match_rate",
    "gate/rdkit_parse_rate",
    "gate/base_valid_rate",
    "gate/final_valid_rate",
    "pose/checked_rate",
    "pose/pass_rate",
    "pose/error_rate",
    "rmsd/d_min_mean",
    "rmsd/d_min_p50",
    "rmsd/d_min_p90",
    "rmsd/frac_under_delta",
    "rmsd/frac_under_2delta",
    "cov/refs_hit_mean",
    "cov/refs_hit_p50",
    "cov/cov_ratio_mean",
    "cov/unique_nearest_refs_mean",
    "cov/nearest_collision_rate_mean",
    "cov/valid_rollouts_mean",
    "match/num_matched_mean",
    "match/max_possible_mean",
    "match/efficiency_mean",
    "match/matched_dist_p50",
    "match/matched_dist_p90",
    "match/eligible_edge_density",
    "smcov/soft_cov_mean",
    "smcov/pct_gt_cov_gt_0p1",
    "smcov/pct_gt_cov_gt_0p5",
    "smcov/corr_with_refs_hit",
    "reward/total_mean",
    "reward/total_std",
    "reward/comp_qual_mean",
    "reward/comp_smcov_mean",
    "reward/comp_match_mean",
    "reward/comp_smcov_frac",
    "div/pairwise_rmsd_p50",
]


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
    empty_rewards = np.full(K, r_floor, dtype=np.float32)

    rollout_mols, graph_mask, parsed_mask = parse_rollout_group(
        canonical_smiles,
        completions,
        stats,
        profiler,
    )
    graph_match_count = int(np.count_nonzero(graph_mask))
    graph_match_rate = float(graph_match_count) / K if K > 0 else 0.0
    rdkit_parse_count = int(np.count_nonzero(parsed_mask))
    rdkit_parse_rate = float(rdkit_parse_count) / K if K > 0 else 0.0
    base_mask = graph_mask & parsed_mask
    base_valid_count = int(np.count_nonzero(base_mask))
    base_valid_rate = float(base_valid_count) / K if K > 0 else 0.0

    reference_mols = get_cached_ground_truths(
        canonical_smiles,
        num_gt=grpo_cfg.max_ground_truths,
    )

    def _build_metrics(**overrides) -> GroupMetrics:
        values = dict(
            K=K,
            M=len(reference_mols),
            graph_match_rate=overrides.get("graph_match_rate", 0.0),
            graph_match_count=overrides.get("graph_match_count", 0),
            rdkit_parse_rate=overrides.get("rdkit_parse_rate", 0.0),
            rdkit_parse_count=overrides.get("rdkit_parse_count", 0),
            base_valid_rate=overrides.get("base_valid_rate", 0.0),
            base_valid_count=overrides.get("base_valid_count", 0),
            final_valid_rate=overrides.get("final_valid_rate", 0.0),
            valid_rollouts=overrides.get("valid_rollouts", 0),
            d_min_mean=overrides.get("d_min_mean", float("nan")),
            frac_under_delta=overrides.get("frac_under_delta", float("nan")),
            frac_under_2delta=overrides.get("frac_under_2delta", float("nan")),
            refs_hit=overrides.get("refs_hit", 0),
            cov_ratio=overrides.get("cov_ratio", 0.0),
            unique_nearest_refs=overrides.get("unique_nearest_refs", 0),
            nearest_collision_rate=overrides.get("nearest_collision_rate", 0.0),
            num_matched=overrides.get("num_matched", 0),
            max_possible_matches=overrides.get("max_possible_matches", 0),
            match_efficiency=overrides.get("match_efficiency", 0.0),
            eligible_edges=overrides.get("eligible_edges", 0),
            eligible_edge_density=overrides.get("eligible_edge_density", 0.0),
            soft_cov_mean=overrides.get("soft_cov_mean", float("nan")),
            pct_cov_gt_0p1=overrides.get("pct_cov_gt_0p1", float("nan")),
            pct_cov_gt_0p5=overrides.get("pct_cov_gt_0p5", float("nan")),
            comp_qual_sum=overrides.get("comp_qual_sum", 0.0),
            comp_smcov_sum=overrides.get("comp_smcov_sum", 0.0),
            comp_match_sum=overrides.get("comp_match_sum", 0.0),
            pose_checked=overrides.get("pose_checked", 0),
            pose_passed=overrides.get("pose_passed", 0),
            pose_errors=overrides.get("pose_errors", 0),
            d_min_values=overrides.get("d_min_values", EMPTY_FLOAT32),
            matched_dists=overrides.get("matched_dists", EMPTY_FLOAT32),
            pairwise_dists=overrides.get("pairwise_dists", EMPTY_FLOAT32),
            reward_total_values=overrides.get("reward_total_values", EMPTY_FLOAT32),
        )
        return GroupMetrics(**values)

    if not reference_mols:
        stats.failed_ground_truth += K
        return empty_rewards, _build_metrics(
            graph_match_rate=graph_match_rate,
            graph_match_count=graph_match_count,
            rdkit_parse_rate=rdkit_parse_rate,
            rdkit_parse_count=rdkit_parse_count,
            base_valid_rate=base_valid_rate,
            base_valid_count=base_valid_count,
        )

    if not np.any(graph_mask):
        return empty_rewards, _build_metrics(
            graph_match_rate=graph_match_rate,
            graph_match_count=graph_match_count,
            rdkit_parse_rate=rdkit_parse_rate,
            rdkit_parse_count=rdkit_parse_count,
            base_valid_rate=base_valid_rate,
            base_valid_count=base_valid_count,
        )

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
        r_smcov, (soft_cov_mean, _soft_cov_pcts, soft_cov_values) = compute_smooth_coverage_reward(
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

    matched_dists = (
        np.array([distance_matrix[i, j] for (i, j) in matched_pairs], dtype=np.float32)
        if matched_pairs
        else EMPTY_FLOAT32
    )

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

    valid_rollouts = int(np.count_nonzero(valid_mask))
    final_valid_rate = float(valid_rollouts) / K if K > 0 else 0.0
    finite_d_valid = min_distances[valid_mask]
    finite_d_valid = finite_d_valid[np.isfinite(finite_d_valid)]
    d_min_values = (
        finite_d_valid.astype(np.float32, copy=False) if finite_d_valid.size > 0 else EMPTY_FLOAT32
    )
    d_min_mean = float(np.mean(finite_d_valid)) if finite_d_valid.size > 0 else float("nan")
    frac_under_delta = float(np.mean(finite_d_valid < delta)) if finite_d_valid.size > 0 else float("nan")
    frac_under_two_delta = (
        float(np.mean(finite_d_valid < (2 * delta))) if finite_d_valid.size > 0 else float("nan")
    )

    unique_nearest_refs = 0
    nearest_collision_rate = 0.0
    if valid_rollouts > 0 and reference_mols:
        valid_indices = np.where(valid_mask)[0]
        nearest_rows = distance_matrix[valid_indices]
        if nearest_rows.size > 0:
            nearest_refs = np.argmin(nearest_rows, axis=1)
            nearest_values = nearest_rows[np.arange(nearest_refs.size), nearest_refs]
            finite_nearest = np.isfinite(nearest_values)
            if np.any(finite_nearest):
                nearest_refs = nearest_refs[finite_nearest]
                unique_nearest_refs = int(np.unique(nearest_refs).size)
                total_nearest = int(nearest_refs.size)
                if total_nearest > 0:
                    nearest_collision_rate = float(max(0.0, 1.0 - unique_nearest_refs / total_nearest))

    soft_cov_arr = soft_cov_values.astype(np.float32, copy=False)
    pct_cov_gt_0p1 = float(np.mean(soft_cov_arr > 0.1)) if soft_cov_arr.size > 0 else float("nan")
    pct_cov_gt_0p5 = float(np.mean(soft_cov_arr > 0.5)) if soft_cov_arr.size > 0 else float("nan")

    reward_total_values = (
        rewards[valid_mask].astype(np.float32, copy=False) if valid_rollouts > 0 else EMPTY_FLOAT32
    )
    comp_qual_sum = float(np.sum((lambda_qual * r_qual)[valid_mask])) if valid_rollouts > 0 else 0.0
    comp_smcov_sum = float(np.sum((lambda_smcov * r_smcov)[valid_mask])) if valid_rollouts > 0 else 0.0
    comp_match_sum = float(np.sum((lambda_match * r_match)[valid_mask])) if valid_rollouts > 0 else 0.0

    eligible_edges = int(np.count_nonzero(eligible_matrix))
    total_edges = K * len(reference_mols)
    eligible_edge_density = float(eligible_edges) / float(total_edges) if total_edges > 0 else float("nan")
    cov_ratio = (float(refs_hit) / len(reference_mols)) if reference_mols else float("nan")

    for i in range(K):
        if valid_mask[i] and np.isfinite(min_distances[i]):
            stats.add_rmsd(float(min_distances[i]))

    match_efficiency = float(num_matched) / max_possible_matches if max_possible_matches > 0 else 0.0

    metrics = _build_metrics(
        graph_match_rate=graph_match_rate,
        graph_match_count=graph_match_count,
        rdkit_parse_rate=rdkit_parse_rate,
        rdkit_parse_count=rdkit_parse_count,
        base_valid_rate=base_valid_rate,
        base_valid_count=base_valid_count,
        final_valid_rate=final_valid_rate,
        valid_rollouts=valid_rollouts,
        d_min_mean=d_min_mean,
        frac_under_delta=frac_under_delta,
        frac_under_2delta=frac_under_two_delta,
        refs_hit=refs_hit,
        cov_ratio=cov_ratio,
        unique_nearest_refs=unique_nearest_refs,
        nearest_collision_rate=nearest_collision_rate,
        num_matched=num_matched,
        max_possible_matches=max_possible_matches,
        match_efficiency=match_efficiency,
        eligible_edges=eligible_edges,
        eligible_edge_density=eligible_edge_density,
        soft_cov_mean=soft_cov_mean,
        pct_cov_gt_0p1=pct_cov_gt_0p1,
        pct_cov_gt_0p5=pct_cov_gt_0p5,
        comp_qual_sum=comp_qual_sum,
        comp_smcov_sum=comp_smcov_sum,
        comp_match_sum=comp_match_sum,
        pose_checked=pose_checked,
        pose_passed=pose_passed,
        pose_errors=pose_errors,
        d_min_values=d_min_values,
        matched_dists=matched_dists,
        pairwise_dists=pairwise_dists,
        reward_total_values=reward_total_values,
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
    delta = float(getattr(config.grpo, "delta", 0.75))
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
        raw_metrics = summarize_batch_metrics(metrics_list, delta=delta)
        metrics = {key: float("nan") for key in METRIC_KEYS}
        for key, value in raw_metrics.items():
            if key in metrics:
                metrics[key] = float(value)
        assert set(metrics.keys()) == set(METRIC_KEYS)

        should_log_metrics = wandb.run is not None and (step_index % log_every_steps == 0)
        if should_log_metrics:
            wandb.log(metrics)

        gm = metrics["gate/graph_match_rate"]
        rd = metrics["gate/rdkit_parse_rate"]
        base_rate = metrics["gate/base_valid_rate"]
        final_rate = metrics["gate/final_valid_rate"]
        pose_checked_rate = metrics["pose/checked_rate"]
        pose_pass_rate = metrics["pose/pass_rate"]
        pose_error_rate = metrics["pose/error_rate"]
        rmsd_mean = metrics["rmsd/d_min_mean"]
        match_eff = metrics["match/efficiency_mean"]
        reward_mean = metrics["reward/total_mean"]
        div_p50 = metrics["div/pairwise_rmsd_p50"]

        logger.info(
            "[reward_v3] gates: graph={:.3f}, parse={:.3f}, base={:.3f}, final={:.3f}; "
            "pose: checked_rate={:.3f}, pass={:.3f}, error={:.3f}; "
            "rmsd_mean={:.3f}, match_eff={:.3f}, reward_mean={:.3f}, div_p50={:.3f}",
            gm,
            rd,
            base_rate,
            final_rate,
            pose_checked_rate,
            pose_pass_rate,
            pose_error_rate,
            rmsd_mean,
            match_eff,
            reward_mean,
            div_p50,
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
            logger.info(
                "[reward_v3] profiler totals (s): {}",
                ", ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in profiling_metrics.items()),
            )

    return final_rewards
