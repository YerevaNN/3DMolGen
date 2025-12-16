"""GRPO Reward Function - GEOM-Drugs Aligned Implementation.

This implements the reward specification from docs/grpo_reward.md with:
- Hard PoseBusters validity gate
- Dense quality term (AMR-P proxy)
- Smooth marginal coverage (group-aware, pre-threshold signal)
- Hard unique-coverage matching bonus (max-cardinality under δ)

Design targets all four GEOM-Drugs metrics under small-K constraints.
"""

from __future__ import annotations

import os
import math
from typing import List, Optional, Tuple, Dict

import numpy as np
from loguru import logger
from rdkit import Chem
import wandb

from molgen3D.data_processing.smiles_encoder_decoder import decode_cartesian_v2, strip_smiles
from molgen3D.evaluation.utils import extract_between, same_molecular_graph
from molgen3D.training.grpo.utils import load_ground_truths
from molgen3D.utils.utils import get_best_rmsd


# ============================================================================
# PoseBusters Integration
# ============================================================================

_POSEBUSTERS_INSTANCE = None
_POSEBUSTERS_AVAILABLE = None

# Toggle for the optional RMSD hard validity gate.
# Set to True to drop PoseBusters-valid rollouts that still lack finite RMSD.
ENABLE_HARD_RMSD_GATE = False


def get_posebusters_checker():
    """Lazy-load PoseBusters instance with per-process caching."""
    global _POSEBUSTERS_INSTANCE, _POSEBUSTERS_AVAILABLE

    if _POSEBUSTERS_AVAILABLE is False:
        return lambda mol: False

    if _POSEBUSTERS_INSTANCE is None:
        try:
            from posebusters import PoseBusters
            _POSEBUSTERS_INSTANCE = PoseBusters(config="mol")
            _POSEBUSTERS_AVAILABLE = True
            logger.info("PoseBusters initialized successfully")
        except Exception as exc:
            logger.warning(f"PoseBusters unavailable, validity disabled: {exc}")
            _POSEBUSTERS_AVAILABLE = False
            _POSEBUSTERS_INSTANCE = None

    if _POSEBUSTERS_INSTANCE is None:
        return lambda mol: False

    def checker(mol: Chem.Mol) -> bool:
        if mol is None:
            return False
        try:
            df = _POSEBUSTERS_INSTANCE.bust([mol], None, None, full_report=False)
            return bool(df.all(axis=1).iloc[0])
        except Exception as exc:
            logger.debug(f"PoseBusters check failed: {exc}")
            return False

    return checker


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
# Step A: PoseBusters Gate
# ============================================================================

def compute_validity(
    rollout_mols: List[Optional[Chem.Mol]],
    posebusters_checker,
    enable_posebusters: bool
) -> np.ndarray:
    """Compute validity indicators v_i ∈ {0,1} for each rollout.

    Args:
        rollout_mols: K generated conformers (may contain None)
        posebusters_checker: PoseBusters checker callable
        enable_posebusters: Whether PoseBusters is enabled

    Returns:
        Array of shape (K,) with validity indicators
    """
    K = len(rollout_mols)
    validity = np.zeros(K, dtype=np.int32)

    if not enable_posebusters:
        # If PoseBusters disabled, only check for None
        for i, mol in enumerate(rollout_mols):
            validity[i] = 1 if mol is not None else 0
        return validity

    for i, mol in enumerate(rollout_mols):
        if mol is None:
            validity[i] = 0
        else:
            try:
                passed = posebusters_checker(mol)
                validity[i] = 1 if passed else 0
            except Exception as exc:
                logger.debug(f"PoseBusters check exception for rollout {i}: {exc}")
                validity[i] = 0

    return validity


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

    # Compute nearest reference distance for each rollout
    d_i = np.min(D, axis=1)  # Shape (K,)

    # Compute quality reward
    r_qual = np.zeros(K, dtype=np.float32)

    for i in range(K):
        if validity[i] == 0:
            r_qual[i] = 0.0
        elif np.isfinite(d_i[i]):
            r_qual[i] = float(np.exp(-d_i[i] / max(sigma, 1e-8)))
        else:
            r_qual[i] = 0.0

    return r_qual


# ============================================================================
# Step D: Term 2 - Smooth Marginal Coverage r^smcov
# ============================================================================

def compute_smooth_coverage_reward(
    D: np.ndarray,
    validity: np.ndarray,
    rho: float
) -> np.ndarray:
    """Compute smooth group marginal coverage reward.

    This is the differentiable surrogate for "add new reference coverage".

    Algorithm:
    1. Compute kernel K[i,j] = exp(-(D[i,j]/rho)^2)
    2. For invalid rollouts, set K[i,:] = 0
    3. Compute marginal contribution Δ[i,j] = K[i,j] * prod_{ℓ≠i}(1 - K[ℓ,j])
    4. Sum over references: r_i^smcov = (1/M) * sum_j Δ[i,j]

    Args:
        D: Distance matrix (K, M)
        validity: Validity indicators (K,)
        rho: Softness scale for kernel

    Returns:
        Array of shape (K,) with smooth coverage rewards
    """
    K, M = D.shape

    if M == 0:
        return np.zeros(K, dtype=np.float32)

    rho = max(float(rho), 1e-8)

    # Step D1: Compute kernel matrix in float64 for numerical stability
    K_matrix = np.zeros((K, M), dtype=np.float64)
    valid_rows = np.where(validity == 1)[0]
    if valid_rows.size > 0:
        D_valid = D[valid_rows].astype(np.float64)
        D_valid[~np.isfinite(D_valid)] = np.inf
        K_valid = np.exp(-((D_valid / rho) ** 2))
        K_valid[~np.isfinite(K_valid)] = 0.0
        K_matrix[valid_rows, :] = K_valid

    # Step D2: Compute per-reference "soft uncovered product"
    A = 1.0 - K_matrix  # Shape (K, M)
    np.clip(A, 0.0, 1.0, out=A)

    # P_j = prod_i A[i,j] for each reference j
    P_j = np.prod(A, axis=0)  # Shape (M,)

    # Step D3: Compute marginal contribution per rollout
    r_smcov = np.zeros(K, dtype=np.float32)

    for i in range(K):
        if validity[i] == 0:
            r_smcov[i] = 0.0
            continue

        delta_sum = 0.0

        for j in range(M):
            # Compute prod_{ℓ≠i}(1 - K[ℓ,j])
            denom = A[i, j]
            if denom < 1e-12:
                # Numerical risk: A[i,j] ≈ 0, recompute product excluding i
                prod_exclude_i = 1.0
                for ell in range(K):
                    if ell != i:
                        prod_exclude_i *= A[ell, j]
            else:
                # Stable computation: P_j / A[i,j]
                prod_exclude_i = P_j[j] / denom

            # Marginal contribution
            delta_ij = K_matrix[i, j] * prod_exclude_i
            delta_sum += delta_ij

        # Average over references
        r_smcov[i] = delta_sum / M

    return r_smcov.astype(np.float32)


# ============================================================================
# Step E: Term 3 - Hard Matching Bonus r^match
# ============================================================================

def compute_matching_reward(
    D: np.ndarray,
    validity: np.ndarray,
    delta: float
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

    # Step E1: Build eligible edges
    eligible = np.zeros((K, M), dtype=bool)

    for i in range(K):
        if validity[i] == 1:
            for j in range(M):
                if D[i, j] < delta:
                    eligible[i, j] = True

    # Count eligible edges
    num_eligible_edges = int(np.sum(eligible))

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
    posebusters_checker
) -> Tuple[np.ndarray, Dict]:
    """Compute rewards for a single prompt group (K rollouts).

    Args:
        canonical_smiles: Canonical SMILES for the molecule
        completions: List of K completion strings
        config: GRPO config
        stats: RunStatistics object
        posebusters_checker: PoseBusters checker

    Returns:
        Tuple of (rewards, debug_info)
        - rewards: Array of shape (K,) with final rewards
        - debug_info: Dict with diagnostic information
    """
    K = len(completions)

    # Hyperparameters
    delta = getattr(config.grpo, 'delta', 0.75)
    sigma = getattr(config.grpo, 'sigma', 0.25)
    rho = getattr(config.grpo, 'rho', 0.75)
    lambda_qual = getattr(config.grpo, 'lambda_qual', 1.0)
    lambda_smcov = getattr(config.grpo, 'lambda_smcov', 1.0)
    lambda_match = getattr(config.grpo, 'lambda_match', 1.0)
    r_floor = getattr(config.grpo, 'r_floor', -1.0)
    enable_posebusters = getattr(config.grpo, 'enable_posebusters', False)

    # Load references
    reference_mols = load_ground_truths(
        canonical_smiles,
        num_gt=config.grpo.max_ground_truths
    )

    if not reference_mols:
        stats.failed_ground_truth += K
        return np.full(K, r_floor, dtype=np.float32), {
            'M': 0,
            'K': K,
            'validity_rate_posebusters': 0.0,
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
        }

    M = len(reference_mols)

    # Parse rollouts
    rollout_mols: List[Optional[Chem.Mol]] = []
    num_graph_matches = 0
    num_graph_checked = 0
    num_missing_conformer = 0
    num_empty_stripped = 0
    num_parsed_success = 0

    for idx, completion in enumerate(completions):
        conformer_text = extract_between(completion, "[CONFORMER]", "[/CONFORMER]")
        generated_smiles = strip_smiles(conformer_text) if conformer_text else ""

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
            else:
                stats.failed_matching_smiles += 1
                num_graph_checked += 1

        mol = parse_conformer(completion)
        rollout_mols.append(mol)

        if mol is None:
            stats.failed_conformer_generation += 1
        else:
            num_parsed_success += 1

    # Step A: Compute validity
    validity = compute_validity(rollout_mols, posebusters_checker, enable_posebusters)
    validity_before_rmsd_gate = validity.copy()

    # Update stats
    for i, mol in enumerate(rollout_mols):
        if mol is None:
            stats.failed_conformer_generation += 1

    # Early exit if all invalid before RMSD computation
    if np.sum(validity) == 0:
        stats.failed_rmsd += K
        validity_rate_posebusters = float(np.mean(validity_before_rmsd_gate)) if K > 0 else 0.0
        return np.full(K, r_floor, dtype=np.float32), {
            'M': M,
            'K': K,
            'validity_rate_posebusters': validity_rate_posebusters,
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
        }

    # Step B: Compute RMSD matrix
    D = compute_distance_matrix(rollout_mols, reference_mols, validity)

    # HARD CONSISTENCY: treat PoseBusters-valid rollouts with no finite RMSD as invalid
    d_i_all = np.min(D, axis=1) if M > 0 else np.full(K, np.inf, dtype=np.float32)
    finite_mask = np.isfinite(d_i_all)

    if ENABLE_HARD_RMSD_GATE:
        validity = (validity.astype(bool) & finite_mask).astype(np.int32)

        num_valid_before = int(np.sum(validity_before_rmsd_gate))
        num_valid_after = int(np.sum(validity))
        num_dropped = num_valid_before - num_valid_after
        if num_dropped > 0:
            logger.warning(
                f"[reward_v3] RMSD-gate dropped {num_dropped}/{num_valid_before} PoseBusters-valid rollouts "
                f"(no finite RMSD)."
            )
            stats.failed_rmsd += num_dropped

        if np.sum(validity) == 0:
            finite_rmsd_rate = float(np.mean(finite_mask)) if K > 0 else 0.0
            stats.failed_rmsd += K
            return np.full(K, r_floor, dtype=np.float32), {
                'M': M,
                'K': K,
                'validity_rate_posebusters': float(np.mean(validity_before_rmsd_gate)) if K > 0 else 0.0,
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
            }
    else:
        num_problematic = int(np.sum((validity_before_rmsd_gate == 1) & (~finite_mask)))
        if num_problematic > 0:
            logger.warning(
                f"[reward_v3] RMSD-gate disabled — {num_problematic} PoseBusters-valid rollouts have no finite RMSD."
            )
            stats.failed_rmsd += num_problematic

    # Step C: Quality reward
    r_qual = compute_quality_reward(D, validity, sigma)

    # Step D: Smooth coverage reward
    r_smcov = compute_smooth_coverage_reward(D, validity, rho)

    # Step E: Matching reward
    r_match, num_matched, num_eligible_edges = compute_matching_reward(D, validity, delta)

    # Step F: Combine rewards
    rewards = combine_rewards(
        r_qual, r_smcov, r_match, validity,
        lambda_qual, lambda_smcov, lambda_match, r_floor
    )

    # Diagnostics
    validity_rate_posebusters = float(np.mean(validity_before_rmsd_gate)) if K > 0 else 0.0
    finite_rmsd_rate = float(np.mean(finite_mask)) if K > 0 else 0.0
    validity_rate = float(np.mean(validity)) if K > 0 else 0.0

    valid_idx = np.where(validity == 1)[0]
    d_valid = d_i_all[valid_idx] if valid_idx.size > 0 else np.array([], dtype=np.float32)
    finite_d_valid = d_valid[np.isfinite(d_valid)]
    under_threshold = float(np.mean(finite_d_valid < delta)) if finite_d_valid.size > 0 else 0.0

    debug_info = {
        'M': M,
        'K': K,
        'validity_rate_posebusters': validity_rate_posebusters,
        'finite_rmsd_rate': finite_rmsd_rate,
        'validity_rate': validity_rate,
        'mean_d_i': float(np.mean(finite_d_valid)) if finite_d_valid.size > 0 else float('nan'),
        'min_d_i': float(np.min(finite_d_valid)) if finite_d_valid.size > 0 else float('nan'),
        'max_d_i': float(np.max(finite_d_valid)) if finite_d_valid.size > 0 else float('nan'),
        'mean_r_qual': float(np.mean(r_qual[valid_idx])) if valid_idx.size > 0 else 0.0,
        'mean_r_smcov': float(np.mean(r_smcov[valid_idx])) if valid_idx.size > 0 else 0.0,
        'mean_r_match': float(np.mean(r_match[valid_idx])) if valid_idx.size > 0 else 0.0,
        'num_matched': int(num_matched),
        'num_eligible_edges': int(num_eligible_edges),
        'fraction_under_delta': under_threshold,
    }

    posebusters_valid_count = int(np.sum(validity_before_rmsd_gate))
    final_valid_count = int(np.sum(validity))
    graph_match_pct = (
        100.0 * num_graph_matches / num_graph_checked if num_graph_checked > 0 else 0.0
    )
    mean_reward = float(np.mean(rewards)) if rewards.size > 0 else 0.0

    logger.info(
        "[PID {pid}] [reward_v3] Prompt SMILES: {smiles}\n"
        "[PID {pid}] [reward_v3] Stats | rollouts={rollouts}, parsed={parsed}, "
        "posebusters_valid={posebusters_valid}, final_valid={final_valid}, "
        "graph_match={graph_match}/{graph_checked} ({graph_pct:.2f}%), "
        "missing_conformer={missing}, empty_strip={empty}, "
        "mean_r_qual={r_qual:.3f}, mean_r_smcov={r_smcov:.3f}, mean_r_match={r_match:.3f}, "
        "mean_reward={mean_reward:.3f}",
        pid=os.getpid(),
        smiles=canonical_smiles if canonical_smiles else "<missing>",
        rollouts=K,
        parsed=num_parsed_success,
        posebusters_valid=posebusters_valid_count,
        final_valid=final_valid_count,
        graph_match=num_graph_matches,
        graph_checked=num_graph_checked,
        graph_pct=graph_match_pct,
        missing=num_missing_conformer,
        empty=num_empty_stripped,
        r_qual=debug_info['mean_r_qual'] if np.isfinite(debug_info['mean_r_qual']) else 0.0,
        r_smcov=debug_info['mean_r_smcov'] if np.isfinite(debug_info['mean_r_smcov']) else 0.0,
        r_match=debug_info['mean_r_match'] if np.isfinite(debug_info['mean_r_match']) else 0.0,
        mean_reward=mean_reward,
    )

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
    expected_k: int
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

        if key not in active_groups or len(active_groups[key]['completions']) >= expected_k:
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

        if len(group['completions']) >= expected_k:
            active_groups.pop(key, None)

    return groups


def reward_function(
    prompts: List[str],
    completions: List[str],
    stats,
    tokenizer,
    config
) -> List[float]:
    """Main GRPO reward function (TRL-compatible).

    Implements GEOM-Drugs-aligned reward with:
    - Hard PoseBusters gate
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
    groups = group_by_prompt(prompts, completions, expected_k)

    final_rewards = [0.0] * len(completions)
    posebusters_checker = get_posebusters_checker()

    # Aggregate diagnostics
    all_validity_rates = []
    all_validity_rates_posebusters = []
    all_finite_rmsd_rates = []
    all_mean_d_i = []
    all_mean_r_qual = []
    all_mean_r_smcov = []
    all_mean_r_match = []
    all_num_matched = []
    all_num_eligible_edges = []
    all_fraction_under_delta = []
    all_mean_rewards = []
    group_sizes = []
    total_M = 0
    total_K = 0

    for group in groups:
        stats.processed_prompts += len(group['completions'])

        rewards, debug_info = compute_group_reward(
            canonical_smiles=group['canonical_smiles'],
            completions=group['completions'],
            config=config,
            stats=stats,
            posebusters_checker=posebusters_checker
        )

        # Assign back to flat batch
        for local_idx, global_idx in enumerate(group['indices']):
            final_rewards[global_idx] = float(rewards[local_idx])

        # Collect diagnostics
        all_validity_rates.append(debug_info['validity_rate'])
        all_validity_rates_posebusters.append(debug_info['validity_rate_posebusters'])
        all_finite_rmsd_rates.append(debug_info['finite_rmsd_rate'])
        all_mean_d_i.append(debug_info['mean_d_i'])
        all_mean_r_qual.append(debug_info['mean_r_qual'])
        all_mean_r_smcov.append(debug_info['mean_r_smcov'])
        all_mean_r_match.append(debug_info['mean_r_match'])
        all_num_matched.append(debug_info['num_matched'])
        all_num_eligible_edges.append(debug_info['num_eligible_edges'])
        all_fraction_under_delta.append(debug_info['fraction_under_delta'])
        all_mean_rewards.append(float(np.mean(rewards)) if rewards.size > 0 else 0.0)
        group_sizes.append(debug_info['K'])
        total_M += debug_info['M']
        total_K += debug_info['K']

    # Step G: Logging
    mean_validity = float(np.nanmean(all_validity_rates)) if all_validity_rates else 0.0
    mean_posebusters = float(np.nanmean(all_validity_rates_posebusters)) if all_validity_rates_posebusters else 0.0
    mean_finite_rmsd = float(np.nanmean(all_finite_rmsd_rates)) if all_finite_rmsd_rates else 0.0
    mean_d_i = float(np.nanmean(all_mean_d_i)) if all_mean_d_i else float('nan')
    mean_r_qual = float(np.nanmean(all_mean_r_qual)) if all_mean_r_qual else 0.0
    mean_r_smcov = float(np.nanmean(all_mean_r_smcov)) if all_mean_r_smcov else 0.0
    mean_r_match = float(np.nanmean(all_mean_r_match)) if all_mean_r_match else 0.0
    mean_fraction_under_delta = float(np.nanmean(all_fraction_under_delta)) if all_fraction_under_delta else 0.0
    total_matched = int(np.sum(all_num_matched)) if all_num_matched else 0
    total_eligible_edges = int(np.sum(all_num_eligible_edges)) if all_num_eligible_edges else 0
    total_valid_final = float(np.nansum([vr * k for vr, k in zip(all_validity_rates, group_sizes)])) if group_sizes else 0.0
    valid_denominator = total_valid_final if np.isfinite(total_valid_final) and total_valid_final > 0 else 1.0
    mean_reward_overall = float(np.nanmean(all_mean_rewards)) if all_mean_rewards else 0.0

    if wandb.run is not None:
        log_data = {
            "reward_v3/validity_rate": mean_validity,
            "reward_v3/validity_rate_posebusters": mean_posebusters,
            "reward_v3/finite_rmsd_rate": mean_finite_rmsd,
            "reward_v3/mean_d_i": mean_d_i,
            "reward_v3/mean_r_qual": mean_r_qual,
            "reward_v3/mean_r_smcov": mean_r_smcov,
            "reward_v3/mean_r_match": mean_r_match,
            "reward_v3/total_matched": total_matched,
            "reward_v3/eligible_edges": total_eligible_edges,
            "reward_v3/matched_per_valid": float(total_matched) / valid_denominator,
            "reward_v3/fraction_under_delta": mean_fraction_under_delta,
            "reward_v3/mean_reward": mean_reward_overall,
            "reward_v3/avg_M": total_M / max(len(groups), 1),
            "reward_v3/avg_K": total_K / max(len(groups), 1),
        }
        wandb.log(log_data)

    total_valid_final_int = int(round(total_valid_final)) if np.isfinite(total_valid_final) and total_valid_final > 0 else 1
    logger.info(
        f"[PID {os.getpid()}] Batch: "
        f"valid_posebusters={mean_posebusters:.3f}, "
        f"finite_rmsd={mean_finite_rmsd:.3f}, "
        f"valid_final={mean_validity:.3f}, "
        f"d_i={mean_d_i:.3f}, "
        f"r_qual={mean_r_qual:.3f}, "
        f"r_smcov={mean_r_smcov:.3f}, "
        f"r_match={mean_r_match:.3f}, "
        f"eligible_edges={total_eligible_edges}, "
        f"matched={total_matched}/{total_valid_final_int}(valid)"
    )

    return final_rewards
