"""GRPO Reward Function - GEOM-Drugs Aligned Implementation.

This implements the reward specification from docs/grpo_reward.md with:
- Hard graph-match validity gate
- Dense quality term (AMR-P proxy)
- Smooth marginal coverage (group-aware, pre-threshold signal)
- Hard unique-coverage matching bonus (max-cardinality under δ)

Design targets all four GEOM-Drugs metrics under small-K constraints.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple, Dict, Union
from collections import OrderedDict

import numpy as np
from loguru import logger
from rdkit import Chem
import wandb

from molgen3D.data_processing.smiles_encoder_decoder import decode_cartesian_v2, strip_smiles
from molgen3D.evaluation.utils import extract_between, same_molecular_graph
from molgen3D.training.grpo.utils import load_ground_truths
from molgen3D.utils.utils import get_best_rmsd


# Toggle for the optional RMSD hard validity gate (overridable via config).
# When enabled, rollouts that pass the graph-match gate but lack finite RMSD are dropped.
DEFAULT_ENABLE_HARD_RMSD_GATE = True


# Small LRU cache for reference conformers to avoid repeated disk hits.
GROUND_TRUTH_CACHE_SIZE = 256
_GROUND_TRUTH_CACHE: "OrderedDict[Tuple[str, int], List[Chem.Mol]]" = OrderedDict()


# ============================================================================
# Step A: Graph Validity Gate
# ============================================================================

def compute_validity(
    graph_matches: Optional[np.ndarray],
    num_rollouts: int,
) -> np.ndarray:
    """Compute validity indicators using only graph-matching information."""
    if graph_matches is None:
        return np.ones(num_rollouts, dtype=np.int32)

    graph_matches = np.asarray(graph_matches, dtype=bool)
    if graph_matches.shape[0] != num_rollouts:
        raise ValueError("graph_matches length must equal number of rollouts")

    return graph_matches.astype(np.int32)

# ============================================================================
# Parsing
# ============================================================================

def parse_conformer(completion: str) -> Optional[Chem.Mol]:
    """Parse conformer from model output.

    Returns None if parsing fails (treated as invalid).
    """
    conformer_text = extract_between(completion, "[CONFORMER]", "[/CONFORMER]")
    if not conformer_text:
        return None

    try:
        mol = decode_cartesian_v2(conformer_text)
        return mol
    except Exception as exc:
        logger.debug(f"Conformer parsing failed: {exc}")
        return None


# ============================================================================
# Ground Truth Reference Cache
# ============================================================================

def get_cached_ground_truths(
    canonical_smiles: str,
    num_gt: int,
) -> List[Chem.Mol]:
    """Load reference conformers with a small in-memory LRU cache."""
    if num_gt <= 0:
        return []

    key = (canonical_smiles, num_gt)
    cached = _GROUND_TRUTH_CACHE.get(key)
    if cached is not None:
        _GROUND_TRUTH_CACHE.move_to_end(key)
        return cached

    references = load_ground_truths(canonical_smiles, num_gt=num_gt) or []
    if references:
        _GROUND_TRUTH_CACHE[key] = references
        if len(_GROUND_TRUTH_CACHE) > GROUND_TRUTH_CACHE_SIZE:
            _GROUND_TRUTH_CACHE.popitem(last=False)
    return references

# ============================================================================
# Step B: RMSD Matrix
# ============================================================================

def compute_rmsd_safe(probe: Chem.Mol, ref: Chem.Mol) -> float:
    """Compute RMSD with error handling.

    Returns inf on failure.
    """
    if probe is None or ref is None:
        return float('inf')

    try:
        rmsd = get_best_rmsd(probe, ref, use_alignmol=True)
        if rmsd is None or np.isnan(rmsd):
            return float('inf')
        return float(rmsd)
    except Exception as exc:
        logger.debug(f"RMSD computation failed: {exc}")
        return float('inf')


def compute_distance_matrix(
    rollout_mols: List[Optional[Chem.Mol]],
    reference_mols: List[Chem.Mol],
    validity: np.ndarray
) -> np.ndarray:
    """Compute RMSD distance matrix D[i,j] = d(y_i, g_j).

    For invalid rollouts (v_i=0), distances are set to inf.

    Args:
        rollout_mols: K generated conformers
        reference_mols: M reference conformers
        validity: K-length array of validity indicators

    Returns:
        Array of shape (K, M) with RMSD distances
    """
    K = len(rollout_mols)
    M = len(reference_mols)

    if K == 0 or M == 0:
        return np.full((K, M), float('inf'), dtype=np.float32)

    D = np.full((K, M), float('inf'), dtype=np.float32)
    

    for i in range(K):
        if validity[i] == 0:
            continue  # Leave as inf

        mol = rollout_mols[i]
        if mol is None:
            continue

        for j, ref_mol in enumerate(reference_mols):
            D[i, j] = compute_rmsd_safe(mol, ref_mol)

    return D


# ============================================================================
# Step C: Term 1 - Dense Quality r^qual
# ============================================================================

def compute_quality_reward(D: np.ndarray, validity: np.ndarray, sigma: float) -> np.ndarray:
    """Compute dense per-sample quality reward.

    r_i^qual = exp(-d_i / sigma)
    where d_i = min_j D[i,j]

    Args:
        D: Distance matrix (K, M)
        validity: Validity indicators (K,)
        sigma: Quality scale parameter

    Returns:
        Array of shape (K,) with quality rewards
    """
    K, M = D.shape

    if M == 0:
        return np.zeros(K, dtype=np.float32)

    d_i = np.min(D, axis=1)  # Shape (K,)
    r_qual = np.zeros(K, dtype=np.float32)

    sigma = max(float(sigma), 1e-8)
    valid_mask = (validity == 1) & np.isfinite(d_i)
    if np.any(valid_mask):
        r_qual[valid_mask] = np.exp(-d_i[valid_mask] / sigma).astype(np.float32)

    return r_qual


# ============================================================================
# Step D: Term 2 - Smooth Marginal Coverage r^smcov
# ============================================================================

def compute_smooth_coverage_reward(
    D: np.ndarray,
    validity: np.ndarray,
    rho: float,
    return_details: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[float, List[float], np.ndarray]]]:
    """Compute smooth group marginal coverage reward (optionally returning coverage stats)."""
    K, M = D.shape

    if M == 0:
        empty = np.zeros(K, dtype=np.float32)
        if return_details:
            return empty, (float('nan'), [float('nan')] * 3, np.array([], dtype=np.float32))
        return empty

    rho = max(float(rho), 1e-8)

    valid_rows = validity == 1
    K_matrix = np.zeros((K, M), dtype=np.float64)
    if np.any(valid_rows):
        D_valid = D[valid_rows].astype(np.float64)
        D_valid[~np.isfinite(D_valid)] = np.inf
        K_valid = np.exp(-((D_valid / rho) ** 2))
        K_valid[~np.isfinite(K_valid)] = 0.0
        K_matrix[valid_rows, :] = K_valid

    one_minus = np.clip(1.0 - K_matrix, 0.0, 1.0)
    zero_mask = one_minus <= 1e-12
    one_minus_safe = np.where(zero_mask, 1.0, one_minus)
    log_one_minus = np.log(one_minus_safe)
    log_prod_safe = np.sum(log_one_minus, axis=0, keepdims=True)  # (1, M)
    zero_counts = zero_mask.sum(axis=0, keepdims=True)

    prod_excl = np.exp(log_prod_safe - log_one_minus)
    has_zero = zero_counts > 0
    prod_excl = np.where(has_zero, 0.0, prod_excl)
    only_zero = (zero_counts == 1) & zero_mask
    if np.any(only_zero):
        prod_excl = np.where(only_zero, np.exp(log_prod_safe), prod_excl)

    Delta = K_matrix * prod_excl
    r_smcov = (Delta.sum(axis=1) / M).astype(np.float32)
    r_smcov[~valid_rows] = 0.0

    if not return_details:
        return r_smcov

    soft_cov = 1.0 - np.prod(one_minus, axis=0)
    soft_cov = np.clip(soft_cov, 0.0, 1.0)
    soft_mean = float(np.mean(soft_cov)) if soft_cov.size > 0 else float('nan')
    soft_pcts = [
        float(np.mean(soft_cov > thresh)) if soft_cov.size > 0 else float('nan')
        for thresh in (0.1, 0.3, 0.5)
    ]
    return r_smcov, (soft_mean, soft_pcts, soft_cov.astype(np.float32))


def compute_pairwise_rollout_distances(
    rollout_mols: List[Optional[Chem.Mol]],
    validity: np.ndarray,
) -> np.ndarray:
    """Compute pairwise RMSD distances between valid rollout conformers."""
    valid_indices = [
        idx for idx, flag in enumerate(validity) if flag == 1 and rollout_mols[idx] is not None
    ]
    if len(valid_indices) < 2:
        return np.array([], dtype=np.float32)

    distances: List[float] = []
    for i_pos, i in enumerate(valid_indices):
        mol_i = rollout_mols[i]
        if mol_i is None:
            continue
        for j in valid_indices[i_pos + 1 :]:
            mol_j = rollout_mols[j]
            if mol_j is None:
                continue
            d_val = compute_rmsd_safe(mol_i, mol_j)
            if np.isfinite(d_val):
                distances.append(float(d_val))

    return np.array(distances, dtype=np.float32)


# ============================================================================
# Step E: Term 3 - Hard Matching Bonus r^match
# ============================================================================

def compute_matching_reward(
    D: np.ndarray,
    validity: np.ndarray,
    delta: float,
    eligible_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, int]:
    """Compute hard unique-coverage matching bonus via max-cardinality matching.

    This implements COV-R under δ proxy with one-to-one assignment.

    Algorithm:
    1. Build eligible edges: (i,j) where v_i=1 and D[i,j] < delta
    2. Find max-cardinality matching with min-cost tie-break
    3. Assign shaped reward: r_match[i] = 1 - D[i,j]/delta for matched pairs

    Args:
        D: Distance matrix (K, M)
        validity: Validity indicators (K,)
        delta: RMSD threshold

    Returns:
        Tuple of (r_match, num_matched, num_eligible_edges)
        - r_match: Array of shape (K,) with matching rewards
        - num_matched: Number of matched pairs
        - num_eligible_edges: Count of eligible edges under δ
    """
    K, M = D.shape

    r_match = np.zeros(K, dtype=np.float32)

    if M == 0 or K == 0:
        return r_match, 0, 0

    valid_mask = validity.astype(bool)
    eligible = (
        eligible_matrix if eligible_matrix is not None
        else valid_mask[:, None] & (D < delta)
    )
    num_eligible_edges = int(np.count_nonzero(eligible))

    if num_eligible_edges == 0:
        return r_match, 0, 0

    # Step E2: Max-cardinality min-cost matching
    try:
        from scipy.optimize import linear_sum_assignment

        # Build cost matrix for Hungarian algorithm
        # Use a large penalty for ineligible edges
        BIG = 1e6
        C = np.where(eligible, D, BIG).astype(np.float64)

        # Pad to square if needed
        if K != M:
            max_dim = max(K, M)
            C_square = np.full((max_dim, max_dim), BIG, dtype=np.float64)
            C_square[:K, :M] = C
            C = C_square
            padded = True
        else:
            padded = False

        # Run Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(C)

        # Extract valid matches
        matched_pairs = []
        for i, j in zip(row_ind, col_ind):
            if not padded:
                # No padding, check eligibility
                if i < K and j < M and eligible[i, j]:
                    matched_pairs.append((i, j))
            else:
                # With padding, check bounds and eligibility
                if i < K and j < M and eligible[i, j]:
                    matched_pairs.append((i, j))

        # Step E3: Compute per-rollout match reward
        for i, j in matched_pairs:
            # Depth shaping: 1 - D[i,j]/delta
            r_match[i] = max(0.0, 1.0 - D[i, j] / delta)

        num_matched = len(matched_pairs)
        if num_matched == 0 and num_eligible_edges > 0:
            logger.warning(
                f"[reward_v3] Matching found 0 pairs despite {num_eligible_edges} eligible edges "
                f"(K={K}, M={M}). Check BIG value / padding logic."
            )

    except Exception as exc:
        logger.warning(f"Matching solver failed: {exc}. Falling back to r_match=0")
        r_match = np.zeros(K, dtype=np.float32)
        num_matched = 0

    return r_match, num_matched, num_eligible_edges


# ============================================================================
# Step F: Combine Rewards
# ============================================================================

def combine_rewards(
    r_qual: np.ndarray,
    r_smcov: np.ndarray,
    r_match: np.ndarray,
    validity: np.ndarray,
    lambda_qual: float,
    lambda_smcov: float,
    lambda_match: float,
    r_floor: float
) -> np.ndarray:
    """Combine reward components and apply hard validity gate.

    For invalid samples: r[i] = r_floor
    For valid samples: r[i] = λ_qual*r_qual + λ_smcov*r_smcov + λ_match*r_match

    Args:
        r_qual: Quality rewards (K,)
        r_smcov: Smooth coverage rewards (K,)
        r_match: Matching rewards (K,)
        validity: Validity indicators (K,)
        lambda_qual: Weight for quality term
        lambda_smcov: Weight for smooth coverage term
        lambda_match: Weight for matching term
        r_floor: Reward for invalid samples

    Returns:
        Array of shape (K,) with final rewards
    """
    K = len(validity)
    r = np.zeros(K, dtype=np.float32)

    for i in range(K):
        if validity[i] == 0:
            r[i] = r_floor
        else:
            r[i] = (
                lambda_qual * r_qual[i] +
                lambda_smcov * r_smcov[i] +
                lambda_match * r_match[i]
            )

    return r


# ============================================================================
# Group Reward Computation
# ============================================================================

def compute_group_reward(
    canonical_smiles: str,
    completions: List[str],
    config,
    stats,
    rollout_entropies: Optional[List[Optional[float]]] = None,
    completion_lengths: Optional[List[Optional[float]]] = None,
) -> Tuple[np.ndarray, Dict]:
    """Compute rewards for a single prompt group (K rollouts).

    Args:
        canonical_smiles: Canonical SMILES for the molecule
        completions: List of K completion strings
        config: GRPO config
        stats: RunStatistics object

    Returns:
        Tuple of (rewards, debug_info)
        - rewards: Array of shape (K,) with final rewards
        - debug_info: Dict with diagnostic information
    """
    K = len(completions)
    length_array = (
        np.array(
            [
                float(length) if length is not None else np.nan
                for length in (completion_lengths or [])
            ],
            dtype=np.float32,
        )
        if completion_lengths is not None and len(completion_lengths) == K
        else np.full(K, np.nan, dtype=np.float32)
    )

    # Hyperparameters
    delta = getattr(config.grpo, 'delta', 0.75)
    sigma = getattr(config.grpo, 'sigma', 0.25)
    rho = getattr(config.grpo, 'rho', 0.75)
    lambda_qual = getattr(config.grpo, 'lambda_qual', 1.0)
    lambda_smcov = getattr(config.grpo, 'lambda_smcov', 1.0)
    lambda_match = getattr(config.grpo, 'lambda_match', 1.0)
    r_floor = getattr(config.grpo, 'r_floor', -1.0)
    hard_rmsd_gate = getattr(
        config.grpo,
        'hard_rmsd_gate',
        DEFAULT_ENABLE_HARD_RMSD_GATE,
    )

    # Load references
    reference_mols = get_cached_ground_truths(
        canonical_smiles,
        num_gt=config.grpo.max_ground_truths
    )

    if not reference_mols:
        stats.failed_ground_truth += K
        return np.full(K, r_floor, dtype=np.float32), {
            'M': 0,
            'K': K,
            'graph_match_rate': 0.0,
            'finite_rmsd_rate': 0.0,
            'validity_rate': 0.0,
            'mean_d_i': float('nan'),
            'min_d_i': float('nan'),
            'max_d_i': float('nan'),
            'mean_r_qual': 0.0,
            'mean_r_smcov': 0.0,
            'mean_r_match': 0.0,
            'num_matched': 0,
            'num_eligible_edges': 0,
            'fraction_under_delta': 0.0,
            'max_possible_matches': 0,
            'match_efficiency': 0.0,
            'valid_reward_values': np.array([], dtype=np.float32),
            'valid_r_qual_values': np.array([], dtype=np.float32),
            'valid_r_smcov_values': np.array([], dtype=np.float32),
            'valid_r_match_values': np.array([], dtype=np.float32),
            'valid_advantage_values': np.array([], dtype=np.float32),
            'mean_entropy': float('nan'),
            'entropy_list': [],
        }

    M = len(reference_mols)

    # Parse rollouts
    rollout_mols: List[Optional[Chem.Mol]] = []
    num_graph_matches = 0
    num_graph_checked = 0
    num_missing_conformer = 0
    num_empty_stripped = 0
    num_parsed_success = 0
    graph_match_flags: List[bool] = []

    for idx, completion in enumerate(completions):
        conformer_text = extract_between(completion, "[CONFORMER]", "[/CONFORMER]")
        generated_smiles = strip_smiles(conformer_text) if conformer_text else ""

        graph_match_flag = False

        if not conformer_text:
            num_missing_conformer += 1
        elif not generated_smiles:
            num_empty_stripped += 1
        else:
            graph_match = (
                bool(canonical_smiles)
                and same_molecular_graph(canonical_smiles, generated_smiles)
            )
            if graph_match:
                num_graph_matches += 1
                num_graph_checked += 1
                graph_match_flag = True
            else:
                stats.failed_matching_smiles += 1
                num_graph_checked += 1

        graph_match_flags.append(graph_match_flag)

        mol = parse_conformer(completion)
        rollout_mols.append(mol)

        if mol is None:
            stats.failed_conformer_generation += 1
        else:
            num_parsed_success += 1

    # Step A: Compute validity
    graph_match_array = np.array(graph_match_flags, dtype=bool)
    validity = compute_validity(graph_match_array, K)
    validity_before_rmsd_gate = validity.copy()
    graph_match_rate = float(np.mean(validity_before_rmsd_gate)) if K > 0 else 0.0

    # Early exit if all invalid before RMSD computation
    if np.sum(validity) == 0:
        return np.full(K, r_floor, dtype=np.float32), {
            'M': M,
            'K': K,
            'graph_match_rate': graph_match_rate,
            'finite_rmsd_rate': 0.0,
            'validity_rate': 0.0,
            'mean_d_i': float('nan'),
            'min_d_i': float('nan'),
            'max_d_i': float('nan'),
            'mean_r_qual': 0.0,
            'mean_r_smcov': 0.0,
            'mean_r_match': 0.0,
            'num_matched': 0,
            'num_eligible_edges': 0,
            'fraction_under_delta': 0.0,
            'max_possible_matches': 0,
            'match_efficiency': 0.0,
            'valid_reward_values': np.array([], dtype=np.float32),
            'valid_r_qual_values': np.array([], dtype=np.float32),
            'valid_r_smcov_values': np.array([], dtype=np.float32),
            'valid_r_match_values': np.array([], dtype=np.float32),
            'valid_advantage_values': np.array([], dtype=np.float32),
        }

    # Step B: Compute RMSD matrix
    D = compute_distance_matrix(rollout_mols, reference_mols, validity)

    # HARD CONSISTENCY: treat graph-valid rollouts with no finite RMSD as invalid
    d_i_all = np.min(D, axis=1) if M > 0 else np.full(K, np.inf, dtype=np.float32)
    finite_mask = np.isfinite(d_i_all)

    if hard_rmsd_gate:
        validity = (validity.astype(bool) & finite_mask).astype(np.int32)

        num_valid_before = int(np.sum(validity_before_rmsd_gate))
        num_valid_after = int(np.sum(validity))
        num_dropped = num_valid_before - num_valid_after
        if num_dropped > 0:
            logger.warning(
                f"[reward_v3] RMSD-gate dropped {num_dropped}/{num_valid_before} graph-valid rollouts "
                f"(no finite RMSD)."
            )
            stats.failed_rmsd += num_dropped

    if np.sum(validity) == 0:
        finite_rmsd_rate = float(np.mean(finite_mask)) if K > 0 else 0.0
        return np.full(K, r_floor, dtype=np.float32), {
                'M': M,
                'K': K,
                'graph_match_rate': graph_match_rate,
                'finite_rmsd_rate': finite_rmsd_rate,
                'validity_rate': 0.0,
                'mean_d_i': float('nan'),
                'min_d_i': float('nan'),
                'max_d_i': float('nan'),
                'mean_r_qual': 0.0,
                'mean_r_smcov': 0.0,
                'mean_r_match': 0.0,
                'num_matched': 0,
                'num_eligible_edges': 0,
                'fraction_under_delta': 0.0,
                'max_possible_matches': 0,
                'match_efficiency': 0.0,
                'valid_reward_values': np.array([], dtype=np.float32),
                'valid_r_qual_values': np.array([], dtype=np.float32),
                'valid_r_smcov_values': np.array([], dtype=np.float32),
                'valid_r_match_values': np.array([], dtype=np.float32),
                'valid_advantage_values': np.array([], dtype=np.float32),
            'mean_entropy': float('nan'),
            'entropy_list': [],
            }
    else:
        num_problematic = int(np.sum((validity_before_rmsd_gate == 1) & (~finite_mask)))
        if num_problematic > 0:
            logger.warning(
                f"[reward_v3] RMSD-gate disabled — {num_problematic} graph-valid rollouts have no finite RMSD."
            )

    # Step C: Quality reward
    r_qual = compute_quality_reward(D, validity, sigma)

    # Step D: Smooth coverage reward
    smcov_result = compute_smooth_coverage_reward(D, validity, rho, return_details=True)
    if isinstance(smcov_result, tuple):
        r_smcov, (soft_cov_mean, soft_cov_pcts, soft_cov_values) = smcov_result
    else:
        r_smcov = smcov_result
        soft_cov_mean = float('nan')
        soft_cov_pcts = [float('nan'), float('nan'), float('nan')]
        soft_cov_values = np.array([], dtype=np.float32)

    valid_mask = validity.astype(bool)
    eligible_matrix = valid_mask[:, None] & (D < delta)
    refs_hit = int(np.count_nonzero(eligible_matrix.any(axis=0)))
    num_valid = int(np.count_nonzero(valid_mask))
    max_possible_matches = min(num_valid, refs_hit)

    # Step E: Matching reward
    r_match, num_matched, num_eligible_edges = compute_matching_reward(
        D, validity, delta, eligible_matrix=eligible_matrix
    )

    match_efficiency = (
        float(num_matched) / max_possible_matches if max_possible_matches > 0 else 0.0
    )
    matched_dists = np.array([], dtype=np.float32)
    if num_matched > 0 and D.size > 0:
        matched_mask = eligible_matrix & (D < delta)
        if np.any(matched_mask):
            matched_dists = D[matched_mask]
    matched_quantiles = (
        np.percentile(matched_dists, [10, 50, 90]).tolist() if matched_dists.size > 0 else [np.nan] * 3
    )
    matched_shaped = (
        np.clip(1.0 - matched_dists / max(delta, 1e-8), 0.0, 1.0) if matched_dists.size > 0 else np.array([], dtype=np.float32)
    )

    pairwise_dists = compute_pairwise_rollout_distances(rollout_mols, validity)
    pairwise_mean = float(np.mean(pairwise_dists)) if pairwise_dists.size > 0 else float('nan')
    pairwise_min = float(np.min(pairwise_dists)) if pairwise_dists.size > 0 else float('nan')
    pairwise_quantiles = (
        np.percentile(pairwise_dists, [10, 50, 90]).tolist() if pairwise_dists.size > 0 else [np.nan] * 3
    )

    # Step F: Combine rewards
    rewards = combine_rewards(
        r_qual, r_smcov, r_match, validity,
        lambda_qual, lambda_smcov, lambda_match, r_floor
    )

    if rewards.size > 0:
        group_advantages = rewards - float(np.mean(rewards))
    else:
        group_advantages = np.zeros_like(rewards)

    finite_length_mask = np.isfinite(length_array)
    length_values = length_array[finite_length_mask]
    length_mean = float(np.mean(length_values)) if length_values.size > 0 else float('nan')
    length_quantiles = (
        np.percentile(length_values, [10, 50, 90]).tolist() if length_values.size > 0 else [np.nan, np.nan, np.nan]
    )

    entropy_inputs = rollout_entropies if rollout_entropies is not None else [None] * K
    entropy_values: List[float] = []
    for val in entropy_inputs:
        if val is None:
            entropy_values.append(float('nan'))
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            num = float('nan')
        entropy_values.append(num)
    entropy_array = np.array(entropy_values, dtype=np.float32)
    has_finite_entropy = np.isfinite(entropy_array).any()
    mean_entropy_group = float(np.nanmean(entropy_array)) if has_finite_entropy else float('nan')
    entropy_list = [
        float(val) if np.isfinite(val) else float('nan')
        for val in entropy_array
    ]
    finite_entropy_sorted = np.sort(entropy_array[np.isfinite(entropy_array)])
    entropy_quantiles = (
        np.percentile(finite_entropy_sorted, [10, 50, 90]).tolist()
        if finite_entropy_sorted.size > 0
        else [np.nan, np.nan, np.nan]
    )

    # Diagnostics
    finite_rmsd_rate = float(np.mean(finite_mask)) if K > 0 else 0.0
    validity_rate = float(np.mean(validity)) if K > 0 else 0.0

    valid_idx = np.where(validity == 1)[0]
    d_valid = d_i_all[valid_idx] if valid_idx.size > 0 else np.array([], dtype=np.float32)
    finite_d_valid = d_valid[np.isfinite(d_valid)]
    under_threshold = float(np.mean(finite_d_valid < delta)) if finite_d_valid.size > 0 else 0.0
    under_threshold_count = (
        int(np.count_nonzero(finite_d_valid < delta)) if finite_d_valid.size > 0 else 0
    )
    under_threshold_total = int(finite_d_valid.size)
    d_quantiles = (
        np.percentile(finite_d_valid, [10, 50, 90]).tolist() if finite_d_valid.size > 0 else [np.nan, np.nan, np.nan]
    )

    mean_r_match_group = float(np.mean(r_match[valid_idx])) if valid_idx.size > 0 else 0.0
    mean_r_qual_group = float(np.mean(r_qual[valid_idx])) if valid_idx.size > 0 else 0.0
    mean_r_smcov_group = float(np.mean(r_smcov[valid_idx])) if valid_idx.size > 0 else 0.0

    pre_rmsd_valid_count = int(np.sum(validity_before_rmsd_gate))
    final_valid_count = int(np.sum(validity))
    graph_match_pct = (
        100.0 * num_graph_matches / num_graph_checked if num_graph_checked > 0 else 0.0
    )
    mean_reward = float(np.mean(rewards)) if rewards.size > 0 else 0.0
    baseline_std = float(np.std(rewards)) if rewards.size > 0 else float('nan')
    rewards_list = [float(val) for val in rewards] if rewards.size > 0 else []
    min_rmsds_list = [float(val) for val in d_i_all.tolist()] if d_i_all.size > 0 else []
    advantages_list = (
        [float(val) for val in group_advantages.tolist()] if group_advantages.size > 0 else []
    )

    if valid_idx.size > 0:
        valid_adv_values = group_advantages[valid_idx].astype(np.float32)
        adv_mean_group = float(np.mean(valid_adv_values))
        adv_std_group = float(np.std(valid_adv_values))
    else:
        valid_adv_values = np.array([], dtype=np.float32)
        adv_mean_group = float('nan')
        adv_std_group = float('nan')

    debug_info = {
        'smiles': canonical_smiles if canonical_smiles else "<missing>",
        'M': M,
        'K': K,
        'graph_match_rate': graph_match_rate,
        'finite_rmsd_rate': finite_rmsd_rate,
        'validity_rate': validity_rate,
        'mean_d_i': float(np.mean(finite_d_valid)) if finite_d_valid.size > 0 else float('nan'),
        'min_d_i': float(np.min(finite_d_valid)) if finite_d_valid.size > 0 else float('nan'),
        'max_d_i': float(np.max(finite_d_valid)) if finite_d_valid.size > 0 else float('nan'),
        'num_matched': int(num_matched),
        'num_eligible_edges': int(num_eligible_edges),
        'refs_hit': refs_hit,
        'max_possible_matches': int(max_possible_matches),
        'match_efficiency': match_efficiency,
        'fraction_under_delta': under_threshold,
        'valid_reward_values': rewards[valid_idx].astype(np.float32) if valid_idx.size > 0 else np.array([], dtype=np.float32),
        'valid_r_qual_values': r_qual[valid_idx].astype(np.float32) if valid_idx.size > 0 else np.array([], dtype=np.float32),
        'valid_r_smcov_values': r_smcov[valid_idx].astype(np.float32) if valid_idx.size > 0 else np.array([], dtype=np.float32),
        'valid_r_match_values': r_match[valid_idx].astype(np.float32) if valid_idx.size > 0 else np.array([], dtype=np.float32),
        'valid_advantage_values': valid_adv_values,
        'advantage_mean': adv_mean_group,
        'advantage_std': adv_std_group,
        'mean_entropy': mean_entropy_group,
        'entropy_list': entropy_list,
        'entropy_quantiles': entropy_quantiles,
        'd_quantiles': d_quantiles,
        'd_values': finite_d_valid.tolist(),
        'matched_distances': matched_dists.tolist() if matched_dists.size > 0 else [],
        'matched_quantiles': matched_quantiles,
        'matched_shaped_distances': matched_shaped.tolist() if matched_shaped.size > 0 else [],
        'soft_coverage_mean': soft_cov_mean,
        'soft_coverage_percentiles': soft_cov_pcts,
        'pairwise_distances': pairwise_dists.tolist(),
        'pairwise_mean': pairwise_mean,
        'pairwise_min': pairwise_min,
        'pairwise_quantiles': pairwise_quantiles,
        'completion_lengths': length_array.tolist(),
        'length_mean': length_mean,
        'length_quantiles': length_quantiles,
        'prompt_log_data': {
            'smiles': canonical_smiles if canonical_smiles else "<missing>",
            'rollouts': K,
            'parsed': num_parsed_success,
            'pre_rmsd_valid': pre_rmsd_valid_count,
            'final_valid': final_valid_count,
            'graph_match': num_graph_matches,
            'graph_checked': num_graph_checked,
            'graph_pct': graph_match_pct,
            'missing_conformer': num_missing_conformer,
            'empty_strip': num_empty_stripped,
            'mean_r_match_group': mean_r_match_group if np.isfinite(mean_r_match_group) else 0.0,
            'mean_r_qual_group': mean_r_qual_group if np.isfinite(mean_r_qual_group) else 0.0,
            'mean_r_smcov_group': mean_r_smcov_group if np.isfinite(mean_r_smcov_group) else 0.0,
            'mean_reward': mean_reward,
            'min_d_i': float(np.min(finite_d_valid)) if finite_d_valid.size > 0 else float('nan'),
            'fraction_under_delta': under_threshold,
            'fraction_under_delta_numer': under_threshold_count,
            'fraction_under_delta_denom': under_threshold_total,
            'advantage_mean': adv_mean_group,
            'advantage_std': adv_std_group,
            'rewards_list': rewards_list,
            'advantage_baseline': mean_reward,
            'advantage_baseline_std': baseline_std,
            'min_rmsds_list': min_rmsds_list,
            'advantages_list': advantages_list,
            'refs_hit': refs_hit,
            'refs_total': M,
            'mean_token_entropy_group': mean_entropy_group,
            'entropy_list': entropy_list,
            'entropy_quantiles': entropy_quantiles,
            'd_quantiles': d_quantiles,
            'matched_quantiles': matched_quantiles,
            'soft_cov_mean': soft_cov_mean,
            'soft_cov_pcts': soft_cov_pcts,
            'pairwise_mean': pairwise_mean,
            'pairwise_min': pairwise_min,
            'pairwise_quantiles': pairwise_quantiles,
            'length_mean': length_mean,
            'length_quantiles': length_quantiles,
        },
    }

    # Update stats for finite RMSDs
    for i in range(K):
        if validity[i] == 1 and np.isfinite(d_i_all[i]):
            stats.add_rmsd(d_i_all[i])

    return rewards, debug_info


# ============================================================================
# Main Reward Function (TRL-compatible)
# ============================================================================

def group_by_prompt(
    prompts: List[str],
    completions: List[str],
    expected_k: int,
    rollout_entropies: Optional[List[Optional[float]]] = None,
    completion_lengths: Optional[List[Optional[float]]] = None,
) -> List[Dict]:
    """Group flat batch into prompt groups."""
    groups = []
    active_groups = {}

    for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
        canonical_smiles = extract_between(prompt, "[SMILES]", "[/SMILES]")
        if canonical_smiles is None:
            logger.warning("Prompt missing SMILES tags")
            canonical_smiles = ""

        key = (prompt, canonical_smiles)

        if key not in active_groups:
            group = {
                'prompt': prompt,
                'canonical_smiles': canonical_smiles,
                'completions': [],
                'indices': []
            }
            groups.append(group)
            active_groups[key] = group

        group = active_groups[key]
        group['completions'].append(completion)
        group['indices'].append(idx)
        entropy_val = None
        if rollout_entropies is not None and idx < len(rollout_entropies):
            entropy_val = rollout_entropies[idx]
        group.setdefault('entropy_values', []).append(entropy_val)

        length_val = None
        if completion_lengths is not None and idx < len(completion_lengths):
            length_val = completion_lengths[idx]
        group.setdefault('completion_lengths', []).append(length_val)

    return groups


def reward_function(
    prompts: List[str],
    completions: List[str],
    stats,
    tokenizer,
    config,
    completion_entropies: Optional[List[Optional[float]]] = None,
    completion_lengths: Optional[List[Optional[float]]] = None,
) -> List[float]:
    """Main GRPO reward function (TRL-compatible).

    Implements GEOM-Drugs-aligned reward with:
    - Hard graph-match gate
    - Dense quality (AMR-P proxy)
    - Smooth marginal coverage (group-aware)
    - Hard matching bonus (COV-R under δ)

    Args:
        prompts: List of prompt strings (B*K with repeats)
        completions: List of completion strings (B*K)
        stats: RunStatistics object
        tokenizer: Tokenizer instance
        config: GRPO config

    Returns:
        List of reward scalars (B*K)
    """
    expected_k = config.grpo.num_generations
    lambda_qual = float(getattr(config.grpo, 'lambda_qual', 1.0))
    lambda_smcov = float(getattr(config.grpo, 'lambda_smcov', 1.0))
    lambda_match = float(getattr(config.grpo, 'lambda_match', 1.0))
    delta = float(getattr(config.grpo, 'delta', 0.75))

    if completion_lengths is not None:
        completion_lengths = [
            None if length is None else int(length)
            for length in completion_lengths
        ]
    groups = group_by_prompt(prompts, completions, expected_k, completion_entropies, completion_lengths)

    final_rewards = [0.0] * len(completions)

    # Aggregate diagnostics
    all_validity_rates = []
    all_graph_match_rates = []
    all_finite_rmsd_rates = []
    all_mean_d_i = []
    all_num_matched = []
    all_num_eligible_edges = []
    all_max_possible_matches = []
    all_fraction_under_delta = []
    all_mean_rewards = []
    valid_reward_arrays = []
    valid_r_qual_arrays = []
    valid_r_smcov_arrays = []
    valid_r_match_arrays = []
    valid_advantage_arrays = []
    all_mean_entropies = []
    all_refs_hit = []
    prompt_log_data = []
    group_sizes = []
    total_M = 0
    total_K = 0
    total_max_possible = 0
    all_d_quantiles = []
    all_matched_dists = []
    all_soft_cov = []
    all_soft_cov_pcts = []
    all_d_values = []
    all_pairwise_dists = []
    all_length_means = []
    all_length_values = []
    all_matched_shaped = []

    for group in groups:
        stats.processed_prompts += len(group['completions'])
        stats.distinct_prompts += 1

        rewards, debug_info = compute_group_reward(
            canonical_smiles=group['canonical_smiles'],
            completions=group['completions'],
            config=config,
            stats=stats,
            rollout_entropies=group.get('entropy_values'),
            completion_lengths=group.get('completion_lengths'),
        )

        # Assign back to flat batch
        for local_idx, global_idx in enumerate(group['indices']):
            final_rewards[global_idx] = float(rewards[local_idx])

        # Collect diagnostics
        all_validity_rates.append(debug_info['validity_rate'])
        all_graph_match_rates.append(debug_info['graph_match_rate'])
        all_finite_rmsd_rates.append(debug_info['finite_rmsd_rate'])
        all_mean_d_i.append(debug_info['mean_d_i'])
        all_num_matched.append(debug_info['num_matched'])
        all_num_eligible_edges.append(debug_info['num_eligible_edges'])
        all_max_possible_matches.append(debug_info['max_possible_matches'])
        all_fraction_under_delta.append(debug_info['fraction_under_delta'])
        all_mean_rewards.append(float(np.mean(rewards)) if rewards.size > 0 else 0.0)
        valid_reward_arrays.append(debug_info['valid_reward_values'])
        valid_r_qual_arrays.append(debug_info['valid_r_qual_values'])
        valid_r_smcov_arrays.append(debug_info['valid_r_smcov_values'])
        valid_r_match_arrays.append(debug_info['valid_r_match_values'])
        valid_advantage_arrays.append(debug_info['valid_advantage_values'])
        group_sizes.append(debug_info['K'])
        total_M += debug_info['M']
        total_K += debug_info['K']
        total_max_possible += debug_info['max_possible_matches']
        prompt_log_data.append(debug_info.get('prompt_log_data'))
        all_mean_entropies.append(debug_info.get('mean_entropy', float('nan')))
        all_d_quantiles.append(debug_info.get('d_quantiles', [np.nan, np.nan, np.nan]))
        all_matched_dists.extend(debug_info.get('matched_distances', []))
        all_soft_cov.append(debug_info.get('soft_coverage_mean', float('nan')))
        all_soft_cov_pcts.append(debug_info.get('soft_coverage_percentiles', [np.nan, np.nan, np.nan]))
        all_refs_hit.append(debug_info.get('refs_hit', float('nan')))
        all_d_values.extend(debug_info.get('d_values', []))
        all_pairwise_dists.extend(debug_info.get('pairwise_distances', []))
        all_matched_shaped.extend(debug_info.get('matched_shaped_distances', []))
        all_length_means.append(debug_info.get('length_mean', float('nan')))
        for length_val in debug_info.get('completion_lengths', []):
            if length_val is not None and np.isfinite(length_val):
                all_length_values.append(length_val)

    # Step G: Logging
    mean_validity = float(np.nanmean(all_validity_rates)) if all_validity_rates else 0.0
    mean_graph_match = float(np.nanmean(all_graph_match_rates)) if all_graph_match_rates else 0.0
    mean_finite_rmsd = float(np.nanmean(all_finite_rmsd_rates)) if all_finite_rmsd_rates else 0.0
    mean_d_i = float(np.nanmean(all_mean_d_i)) if all_mean_d_i else float('nan')
    mean_fraction_under_delta = float(np.nanmean(all_fraction_under_delta)) if all_fraction_under_delta else 0.0
    total_matched = int(np.sum(all_num_matched)) if all_num_matched else 0
    total_eligible_edges = int(np.sum(all_num_eligible_edges)) if all_num_eligible_edges else 0
    total_max_possible = int(total_max_possible)
    total_valid_final = float(np.nansum([vr * k for vr, k in zip(all_validity_rates, group_sizes)])) if group_sizes else 0.0
    valid_denominator = total_valid_final if np.isfinite(total_valid_final) and total_valid_final > 0 else 1.0
    mean_reward_overall = float(np.nanmean(all_mean_rewards)) if all_mean_rewards else 0.0
    match_efficiency_total = (
        float(total_matched) / total_max_possible if total_max_possible > 0 else 0.0
    )
    mean_unique_refs_hit = float(np.nanmean(all_refs_hit)) if all_refs_hit else 0.0

    def _concat(values_list: List[np.ndarray]) -> np.ndarray:
        filtered = [arr for arr in values_list if arr.size > 0]
        return (
            np.concatenate(filtered)
            if filtered else np.array([], dtype=np.float32)
        )

    batch_valid_rewards = _concat(valid_reward_arrays)
    batch_r_qual = _concat(valid_r_qual_arrays)
    batch_r_smcov = _concat(valid_r_smcov_arrays)
    batch_r_match = _concat(valid_r_match_arrays)
    batch_advantages = _concat(valid_advantage_arrays)

    def _summary_stats(values: np.ndarray) -> Tuple[float, float]:
        if values.size == 0:
            return 0.0, 0.0
        return (
            float(np.mean(values)),
            float(np.std(values)),
        )

    r_total_mean, r_total_std = _summary_stats(batch_valid_rewards)
    r_qual_mean, r_qual_std = _summary_stats(batch_r_qual)
    r_smcov_mean, r_smcov_std = _summary_stats(batch_r_smcov)
    r_match_mean, r_match_std = _summary_stats(batch_r_match)
    adv_mean, adv_std = _summary_stats(batch_advantages)

    adv_baseline_mean = mean_reward_overall
    adv_baseline_std = r_total_std
    fraction_positive_adv = (
        float(np.mean(batch_advantages > 0)) if batch_advantages.size > 0 else 0.0
    )
    absolute_mean_adv = (
        float(np.mean(np.abs(batch_advantages))) if batch_advantages.size > 0 else 0.0
    )

    batch_unique_prompts = len(groups)
    batch_total_prompts = total_K

    finite_entropy_vals = [
        val for val in all_mean_entropies if val is not None and np.isfinite(val)
    ]
    mean_entropy_overall = (
        float(np.mean(finite_entropy_vals)) if finite_entropy_vals else float('nan')
    )

    def _finite_array(values: List[float]) -> Optional[np.ndarray]:
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float32)
        mask = np.isfinite(arr)
        if not np.any(mask):
            return None
        return arr[mask]

    geometry_metrics: Dict[str, float] = {}
    d_arr = _finite_array(all_d_values)
    if d_arr is not None and d_arr.size > 0:
        geometry_metrics = {
            "geom/d_min_p10": float(np.percentile(d_arr, 10)),
            "geom/d_min_p50": float(np.percentile(d_arr, 50)),
            "geom/d_min_p90": float(np.percentile(d_arr, 90)),
            "geom/d_min_mean": float(np.mean(d_arr)),
            "geom/d_min_median": float(np.median(d_arr)),
        }

    match_dist_metrics: Dict[str, float] = {}
    matched_arr = _finite_array(all_matched_dists)
    if matched_arr is not None and matched_arr.size > 0:
        match_dist_metrics.update(
            {
                "match/dist_p10": float(np.percentile(matched_arr, 10)),
                "match/dist_p50": float(np.percentile(matched_arr, 50)),
                "match/dist_p90": float(np.percentile(matched_arr, 90)),
            }
        )
    matched_shaped_arr = _finite_array(all_matched_shaped)
    if matched_shaped_arr is not None and matched_shaped_arr.size > 0:
        match_dist_metrics["match/shaped_p50"] = float(np.percentile(matched_shaped_arr, 50))

    coverage_metrics: Dict[str, float] = {}
    soft_arr = _finite_array(all_soft_cov)
    if soft_arr is not None and soft_arr.size > 0:
        coverage_metrics.update(
            {
                "coverage/soft_mean": float(np.mean(soft_arr)),
                "coverage/soft_p10": float(np.percentile(soft_arr, 10)),
                "coverage/soft_p50": float(np.percentile(soft_arr, 50)),
                "coverage/soft_p90": float(np.percentile(soft_arr, 90)),
            }
        )
    if all_soft_cov_pcts:
        pct_array = np.asarray(all_soft_cov_pcts, dtype=np.float32)
        coverage_metrics.update(
            {
                "coverage/pct_gt_0.1": float(np.nanmean(pct_array[:, 0])),
                "coverage/pct_gt_0.3": float(np.nanmean(pct_array[:, 1])),
                "coverage/pct_gt_0.5": float(np.nanmean(pct_array[:, 2])),
            }
        )

    match_count_metrics: Dict[str, float] = {}
    def _count_stats(prefix: str, data: List[float]) -> None:
        arr = _finite_array(data)
        if arr is None or arr.size == 0:
            return
        match_count_metrics.update(
            {
                f"{prefix}_p10": float(np.percentile(arr, 10)),
                f"{prefix}_p50": float(np.percentile(arr, 50)),
                f"{prefix}_p90": float(np.percentile(arr, 90)),
            }
        )

    _count_stats("match/matched_count", all_num_matched)
    _count_stats("match/eligible_count", all_num_eligible_edges)
    _count_stats("match/max_possible_count", all_max_possible_matches)
    _count_stats("match/refs_hit", all_refs_hit)

    diversity_metrics: Dict[str, float] = {}
    pair_arr = _finite_array(all_pairwise_dists)
    if pair_arr is not None and pair_arr.size > 0:
        diversity_metrics = {
            "diversity/pairwise_mean": float(np.mean(pair_arr)),
            "diversity/pairwise_min": float(np.min(pair_arr)),
            "diversity/pairwise_p10": float(np.percentile(pair_arr, 10)),
            "diversity/pairwise_p50": float(np.percentile(pair_arr, 50)),
            "diversity/pairwise_p90": float(np.percentile(pair_arr, 90)),
        }

    length_metrics: Dict[str, float] = {}
    len_arr = _finite_array(all_length_values)
    if len_arr is not None and len_arr.size > 0:
        length_metrics.update(
            {
                "len/completion_mean": float(np.mean(len_arr)),
                "len/completion_p10": float(np.percentile(len_arr, 10)),
                "len/completion_p50": float(np.percentile(len_arr, 50)),
                "len/completion_p90": float(np.percentile(len_arr, 90)),
            }
        )

    component_metrics = {
        "reward/component_quality": lambda_qual * r_qual_mean,
        "reward/component_smcov": lambda_smcov * r_smcov_mean,
        "reward/component_match": lambda_match * r_match_mean,
    }

    if wandb.run is not None:
        reward_metrics = {
            "reward/mean_quality": r_qual_mean,
            "reward/mean_smcov": r_smcov_mean,
            "reward/mean_match": r_match_mean,
            "reward/matched_total": total_matched,
            "reward/max_possible_matches": total_max_possible,
            "reward/avg_unique_refs_hit": mean_unique_refs_hit,
        }

        valid_metrics = {
            "valid/final_rate": mean_validity,
            "valid/graph_match_rate": mean_graph_match,
            "valid/match_efficiency": match_efficiency_total,
        }

        rmsd_metrics = {
            "rmsd/finite_rate": mean_finite_rmsd,
            "rmsd/fraction_under_delta": mean_fraction_under_delta,
            "rmsd/mean_d_i": mean_d_i,
        }

        entropy_metrics = {
            "entropy/mean": mean_entropy_overall,
        }

        adv_metrics = {
            "adv/baseline_mean": adv_baseline_mean,
            "adv/baseline_std": adv_baseline_std,
            "adv/advantage_mean": adv_mean,
            "adv/advantage_std": adv_std,
            "adv/fraction_positive": fraction_positive_adv,
            "adv/absolute_mean": absolute_mean_adv,
        }

        gen_metrics = {
            "gen/unique_prompts": stats.distinct_prompts,
            "gen/total_rollouts": stats.processed_prompts,
            "gen/avg_M": total_M / max(len(groups), 1),
            "gen/avg_K": total_K / max(len(groups), 1),
        }

        wandb.log(
            {
                **reward_metrics,
                **valid_metrics,
                **rmsd_metrics,
                **entropy_metrics,
                **adv_metrics,
                **gen_metrics,
                **component_metrics,
                **geometry_metrics,
                **match_dist_metrics,
                **coverage_metrics,
                **match_count_metrics,
                **diversity_metrics,
                **length_metrics,
            }
        )

    total_valid_final_int = int(round(total_valid_final)) if np.isfinite(total_valid_final) and total_valid_final > 0 else 1

    batch_log = (
        f"[PID {os.getpid()}] [reward_v3] Batch summary\n"
        f"  validity: graph_match={mean_graph_match:.3f}, finite_rmsd={mean_finite_rmsd:.3f}, "
        f"final={mean_validity:.3f}\n"
        f"  prompts: unique={batch_unique_prompts}, total={batch_total_prompts}\n"
        f"  coverage: mean_d_i={mean_d_i:.3f}, fraction_under_delta={mean_fraction_under_delta:.3f}\n"
        f"  entropy: mean_token={mean_entropy_overall:.3f}\n"
        f"  rewards: r_total_mean={r_total_mean:.3f}, r_qual={r_qual_mean:.3f}, "
        f"r_smcov={r_smcov_mean:.3f}, r_match={r_match_mean:.3f}\n"
        f"  advantages: mean={adv_mean:.3f}, std={adv_std:.3f}\n"
        f"  matching: eligible_edges={total_eligible_edges}, max_possible={total_max_possible}, "
        f"matched={total_matched}/{total_valid_final_int} (valid), match_eff={match_efficiency_total:.3f}"
    )
    if prompt_log_data:
        prompts_lines = []
        for log_data in prompt_log_data:
            if log_data is None:
                continue
            min_d_val = log_data.get('min_d_i')
            fraction_val = log_data.get('fraction_under_delta')
            adv_mean_val = log_data.get('advantage_mean')
            adv_std_val = log_data.get('advantage_std')
            rewards_vals = log_data.get('rewards_list', [])

            def _fmt(val):
                if isinstance(val, str):
                    return val
                if val is None:
                    return "nan"
                try:
                    finite = np.isfinite(val)
                except TypeError:
                    return str(val)
                if not finite:
                    return "inf" if np.isinf(val) else "nan"
                return f"{float(val):.3f}"

            def _fmt_list(values):
                return ", ".join(_fmt(val) for val in values) if values else ""

            rewards_str = _fmt_list(rewards_vals)
            min_rmsds_vals = log_data.get('min_rmsds_list', [])
            min_rmsds_str = _fmt_list(min_rmsds_vals)
            advantages_vals = log_data.get('advantages_list', [])
            advantages_str = _fmt_list(advantages_vals)
            entropy_mean_val = log_data.get('mean_token_entropy_group')
            entropy_vals = log_data.get('entropy_list', [])
            entropy_str = _fmt_list(entropy_vals)
            fraction_numer = log_data.get('fraction_under_delta_numer')
            fraction_denom = log_data.get('fraction_under_delta_denom')
            fraction_display = (
                f"{fraction_numer}/{fraction_denom}"
                if fraction_numer is not None and fraction_denom is not None
                else _fmt(fraction_val)
            )
            adv_baseline_val = log_data.get('advantage_baseline')
            adv_baseline_std_val = log_data.get('advantage_baseline_std')

            prompts_lines.append(
                "    SMILES: {smiles}\n"
                "      min_d_i={min_d}, fraction_under_delta={fraction}, "
                "entropy_mean={entropy_mean}, adv_baseline={adv_baseline}, baseline_std={baseline_std}, "
                "advantage_mean={adv_mean}, advantage_std={adv_std}\n"
                "      rewards=[{rewards_str}]\n"
                "      min_rmsds=[{min_rmsds_str}]\n"
                "      advantages=[{advantages_str}]\n"
                "      token_entropy=[{entropy_str}]\n"
                "      reward_components: r_qual={mean_r_qual_group:.3f}, "
                "r_smcov={mean_r_smcov_group:.3f}, r_match={mean_r_match_group:.3f}, "
                "total={mean_reward:.3f}\n"
                "      rollouts={rollouts}, parsed={parsed}, "
                "pre_rmsd_valid={pre_rmsd_valid}, final_valid={final_valid}, "
                "graph_match={graph_match}/{graph_checked} ({graph_pct:.2f}%), "
                "unique_refs_hit={refs_hit}/{refs_total}, "
                "missing_conformer={missing_conformer}, empty_strip={empty_strip}".format(
                    min_d=_fmt(min_d_val),
                    fraction=fraction_display,
                    entropy_mean=_fmt(entropy_mean_val),
                    adv_baseline=_fmt(adv_baseline_val),
                    baseline_std=_fmt(adv_baseline_std_val),
                    adv_mean=_fmt(adv_mean_val),
                    adv_std=_fmt(adv_std_val),
                    rewards_str=rewards_str,
                    min_rmsds_str=min_rmsds_str,
                    advantages_str=advantages_str,
                    entropy_str=entropy_str,
                    **log_data
                )
            )
        prompts_block = "\n".join(prompts_lines)
        logger.info(f"{batch_log}\n  prompts_detail:\n{prompts_block}")
    else:
        logger.info(batch_log)

    return final_rewards
