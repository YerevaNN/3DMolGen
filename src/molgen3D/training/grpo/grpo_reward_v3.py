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
) -> Tuple[np.ndarray, int, int, List[Tuple[int, int]]]:
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
    matched_pairs: List[Tuple[int, int]] = []

    if M == 0 or K == 0:
        return r_match, 0, 0, matched_pairs

    valid_mask = validity.astype(bool)
    eligible = (
        eligible_matrix if eligible_matrix is not None
        else valid_mask[:, None] & (D < delta)
    )
    num_eligible_edges = int(np.count_nonzero(eligible))

    if num_eligible_edges == 0:
        return r_match, 0, 0, matched_pairs

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
        matched_pairs = []

    return r_match, num_matched, num_eligible_edges, matched_pairs


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
    r_match, num_matched, num_eligible_edges, matched_pairs = compute_matching_reward(
        D, validity, delta, eligible_matrix=eligible_matrix
    )

    match_efficiency = (
        float(num_matched) / max_possible_matches if max_possible_matches > 0 else 0.0
    )
    matched_dists = (
        np.array([D[i, j] for (i, j) in matched_pairs], dtype=np.float32)
        if matched_pairs else np.array([], dtype=np.float32)
    )
    eligible_dists = (
        D[eligible_matrix].astype(np.float32)
        if np.any(eligible_matrix) else np.array([], dtype=np.float32)
    )

    pairwise_dists = compute_pairwise_rollout_distances(rollout_mols, validity)

    # Step F: Combine rewards
    rewards = combine_rewards(
        r_qual, r_smcov, r_match, validity,
        lambda_qual, lambda_smcov, lambda_match, r_floor
    )

    if rewards.size > 0:
        group_advantages = rewards - float(np.mean(rewards))
    else:
        group_advantages = np.zeros_like(rewards)


    # Diagnostics
    finite_rmsd_rate = float(np.mean(finite_mask)) if K > 0 else 0.0
    validity_rate = float(np.mean(validity)) if K > 0 else 0.0

    valid_idx = np.where(validity == 1)[0]
    d_valid = d_i_all[valid_idx] if valid_idx.size > 0 else np.array([], dtype=np.float32)
    finite_d_valid = d_valid[np.isfinite(d_valid)]
    under_threshold = float(np.mean(finite_d_valid < delta)) if finite_d_valid.size > 0 else 0.0

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
        'd_values': finite_d_valid.tolist(),
        'matched_distances': matched_dists.tolist() if matched_dists.size > 0 else [],
        'eligible_distances': eligible_dists.tolist() if eligible_dists.size > 0 else [],
        'soft_coverage_mean': soft_cov_mean,
        'soft_coverage_percentiles': soft_cov_pcts,
        'pairwise_distances': pairwise_dists.tolist(),
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

    # Aggregate diagnostics (only essential metrics)
    all_validity_rates = []
    all_graph_match_rates = []
    all_finite_rmsd_rates = []
    all_num_matched = []
    all_fraction_under_delta = []
    all_refs_hit = []
    group_sizes = []
    total_max_possible = 0
    all_matched_dists = []
    all_eligible_dists = []
    all_soft_cov = []
    all_soft_cov_pcts = []
    all_d_values = []
    all_pairwise_dists = []
    valid_r_qual_arrays = []
    valid_r_smcov_arrays = []
    valid_r_match_arrays = []

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

        # Collect diagnostics (only essential metrics)
        all_validity_rates.append(debug_info['validity_rate'])
        all_graph_match_rates.append(debug_info['graph_match_rate'])
        all_finite_rmsd_rates.append(debug_info['finite_rmsd_rate'])
        all_num_matched.append(debug_info['num_matched'])
        all_fraction_under_delta.append(debug_info['fraction_under_delta'])
        valid_r_qual_arrays.append(debug_info['valid_r_qual_values'])
        valid_r_smcov_arrays.append(debug_info['valid_r_smcov_values'])
        valid_r_match_arrays.append(debug_info['valid_r_match_values'])
        group_sizes.append(debug_info['K'])
        total_max_possible += debug_info['max_possible_matches']
        all_matched_dists.extend(debug_info.get('matched_distances', []))
        all_eligible_dists.extend(debug_info.get('eligible_distances', []))
        all_soft_cov.append(debug_info.get('soft_coverage_mean', float('nan')))
        all_soft_cov_pcts.append(debug_info.get('soft_coverage_percentiles', [np.nan, np.nan, np.nan]))
        all_refs_hit.append(debug_info.get('refs_hit', float('nan')))
        all_d_values.extend(debug_info.get('d_values', []))
        all_pairwise_dists.extend(debug_info.get('pairwise_distances', []))

    # Step G: Logging (only essential metrics)
    mean_validity = float(np.nanmean(all_validity_rates)) if all_validity_rates else 0.0
    mean_graph_match = float(np.nanmean(all_graph_match_rates)) if all_graph_match_rates else 0.0
    mean_finite_rmsd = float(np.nanmean(all_finite_rmsd_rates)) if all_finite_rmsd_rates else 0.0
    mean_fraction_under_delta = float(np.nanmean(all_fraction_under_delta)) if all_fraction_under_delta else 0.0
    total_matched = int(np.sum(all_num_matched)) if all_num_matched else 0
    total_max_possible = int(total_max_possible)
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

    batch_r_qual = _concat(valid_r_qual_arrays)
    batch_r_smcov = _concat(valid_r_smcov_arrays)
    batch_r_match = _concat(valid_r_match_arrays)

    def _mean(values: np.ndarray) -> float:
        return float(np.mean(values)) if values.size > 0 else 0.0

    r_qual_mean = _mean(batch_r_qual)
    r_smcov_mean = _mean(batch_r_smcov)
    r_match_mean = _mean(batch_r_match)

    def _finite_array(values: List[float]) -> Optional[np.ndarray]:
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float32)
        mask = np.isfinite(arr)
        if not np.any(mask):
            return None
        return arr[mask]

    # Compute only essential metrics
    essential_metrics: Dict[str, float] = {}
    
    # Validity metrics (3)
    essential_metrics["validity_rate"] = mean_validity
    essential_metrics["graph_match_rate"] = mean_graph_match
    essential_metrics["finite_rmsd_rate"] = mean_finite_rmsd
    
    # Geometry metrics (3)
    d_arr = _finite_array(all_d_values)
    if d_arr is not None and d_arr.size > 0:
        essential_metrics["geom/d_min_p50"] = float(np.percentile(d_arr, 50))
        essential_metrics["geom/d_min_p90"] = float(np.percentile(d_arr, 90))
        essential_metrics["geom/d_min_mean"] = float(np.mean(d_arr))
    
    # Matching/Coverage metrics (5)
    essential_metrics["match/match_efficiency"] = float(match_efficiency_total)
    essential_metrics["match/num_matched"] = float(np.mean(all_num_matched)) if all_num_matched else 0.0
    matched_arr = _finite_array(all_matched_dists)
    if matched_arr is not None and matched_arr.size > 0:
        essential_metrics["match/dist_p50"] = float(np.percentile(matched_arr, 50))
        essential_metrics["match/dist_p90"] = float(np.percentile(matched_arr, 90))
    essential_metrics["match/refs_hit"] = float(mean_unique_refs_hit)
    
    # Reward components (3)
    essential_metrics["reward/component_quality"] = float(lambda_qual * r_qual_mean)
    essential_metrics["reward/component_smcov"] = float(lambda_smcov * r_smcov_mean)
    essential_metrics["reward/component_match"] = float(lambda_match * r_match_mean)
    
    # Coverage metrics (2)
    soft_arr = _finite_array(all_soft_cov)
    if soft_arr is not None and soft_arr.size > 0:
        essential_metrics["coverage/soft_mean"] = float(np.mean(soft_arr))
    if all_soft_cov_pcts:
        pct_array = np.asarray(all_soft_cov_pcts, dtype=np.float32)
        essential_metrics["coverage/pct_gt_0.5"] = float(np.nanmean(pct_array[:, 2]))
    
    # Diversity metric (1)
    pair_arr = _finite_array(all_pairwise_dists) 
    if pair_arr is not None and pair_arr.size > 0:
        essential_metrics["diversity/pairwise_mean"] = float(np.mean(pair_arr))
    
    # Additional useful metrics (3)
    essential_metrics["fraction_under_delta"] = mean_fraction_under_delta
    eligible_arr = _finite_array(all_eligible_dists)
    if eligible_arr is not None and eligible_arr.size > 0:
        essential_metrics["match/eligible_dist_p50"] = float(np.percentile(eligible_arr, 50))
    essential_metrics["reward/matched_total"] = float(total_matched)

    if wandb.run is not None:
        wandb.log(essential_metrics)

    batch_log = (
        f"[PID {os.getpid()}] [reward_v3] Batch summary\n"
        f"  validity: graph_match={mean_graph_match:.3f}, finite_rmsd={mean_finite_rmsd:.3f}, "
        f"final={mean_validity:.3f}\n"
        f"  matching: max_possible={total_max_possible}, "
        f"matched={total_matched}, match_eff={match_efficiency_total:.3f}\n"
        f"  rewards: r_qual={r_qual_mean:.3f}, r_smcov={r_smcov_mean:.3f}, r_match={r_match_mean:.3f}"
    )
    logger.info(batch_log)

    return final_rewards
