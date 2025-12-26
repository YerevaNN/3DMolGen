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
import time
from dataclasses import dataclass, field
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

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - exercised when SciPy is missing
    linear_sum_assignment = None  # type: ignore[assignment]


# Toggle for the optional RMSD hard validity gate (overridable via config).
# When enabled, rollouts that pass the graph-match gate but lack finite RMSD are dropped.
DEFAULT_ENABLE_HARD_RMSD_GATE = True


# Small LRU cache for reference conformers to avoid repeated disk hits.
GROUND_TRUTH_CACHE_SIZE = 256
_GROUND_TRUTH_CACHE: "OrderedDict[Tuple[str, int], List[Chem.Mol]]" = OrderedDict()
_LINEAR_SUM_WARNING_EMITTED = False


class _NoOpSection:
    """Context manager that does nothing when profiling is disabled."""

    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class _ProfileSection:
    __slots__ = ("profiler", "key", "start")

    def __init__(self, profiler: "RewardProfiler", key: str):
        self.profiler = profiler
        self.key = key
        self.start = time.perf_counter()

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter()
        self.profiler.add_time(self.key, end - self.start)
        return False


@dataclass
class RewardProfiler:
    """Aggregates timing information for key reward sections."""

    enabled: bool = False
    sections: Dict[str, float] = field(default_factory=dict)

    def section(self, key: str):
        if not self.enabled:
            return _NoOpSection()
        return _ProfileSection(self, key)

    def add_time(self, key: str, duration: float) -> None:
        if not self.enabled:
            return
        self.sections[key] = self.sections.get(key, 0.0) + duration


_NO_OP_CONTEXT = _NoOpSection()


def profile_section(profiler: Optional[RewardProfiler], key: str):
    if profiler is None:
        return _NO_OP_CONTEXT
    return profiler.section(key)


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


def sample_array(values: np.ndarray, max_samples: int) -> np.ndarray:
    """Return at most `max_samples` entries as float32 without copying unnecessarily."""
    if values.size == 0 or max_samples <= 0:
        return np.array([], dtype=np.float32)
    if values.size <= max_samples:
        return values.astype(np.float32, copy=False)
    return values[:max_samples].astype(np.float32, copy=False)


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

def extract_conformer_text(completion: str) -> Optional[str]:
    """Extract conformer text between tags, returning None if missing."""
    conformer_text = extract_between(completion, "[CONFORMER]", "[/CONFORMER]")
    return conformer_text if conformer_text else None


def parse_conformer_text(conformer_text: Optional[str]) -> Optional[Chem.Mol]:
    """Parse pre-extracted conformer text into an RDKit molecule."""
    if not conformer_text:
        return None

    try:
        return decode_cartesian_v2(conformer_text)
    except Exception as exc:  # pragma: no cover - depends on RDKit decode errors
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
    validity_mask: np.ndarray,
    max_samples: int = 0,
) -> np.ndarray:
    """Compute pairwise RMSD distances between valid rollout conformers."""
    valid_indices = [
        idx
        for idx, flag in enumerate(validity_mask)
        if flag and rollout_mols[idx] is not None
    ]
    if len(valid_indices) < 2:
        return np.array([], dtype=np.float32)

    if max_samples > 1 and len(valid_indices) > max_samples:
        valid_indices = valid_indices[:max_samples]

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

    if linear_sum_assignment is None:
        global _LINEAR_SUM_WARNING_EMITTED
        if not _LINEAR_SUM_WARNING_EMITTED:
            logger.warning("SciPy not available; matching reward disabled.")
            _LINEAR_SUM_WARNING_EMITTED = True
        return r_match, 0, num_eligible_edges, matched_pairs

    # Step E2: Max-cardinality min-cost matching
    try:
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

        row_ind, col_ind = linear_sum_assignment(C)

        for i, j in zip(row_ind, col_ind):
            if i < K and j < M and eligible[i, j]:
                matched_pairs.append((i, j))

        for i, j in matched_pairs:
            r_match[i] = max(0.0, 1.0 - D[i, j] / delta)

        num_matched = len(matched_pairs)
        if num_matched == 0 and num_eligible_edges > 0:
            logger.warning(
                f"[reward_v3] Matching found 0 pairs despite {num_eligible_edges} eligible edges "
                f"(K={K}, M={M}). Check BIG value / padding logic."
            )

    except Exception as exc:  # pragma: no cover - requires SciPy failure
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
    combined = (
        lambda_qual * r_qual +
        lambda_smcov * r_smcov +
        lambda_match * r_match
    ).astype(np.float32, copy=False)
    valid_mask = validity.astype(bool)
    if combined.shape != valid_mask.shape:
        raise ValueError("Validity mask must match reward shapes")
    return np.where(valid_mask, combined, r_floor).astype(np.float32, copy=False)


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
    profiler: Optional[RewardProfiler] = None,
    distance_sample_limit: int = 0,
    enable_pairwise_logging: bool = False,
    pairwise_sample_limit: Optional[int] = None,
) -> Tuple[np.ndarray, GroupMetrics]:
    """Compute rewards for a single prompt group (K rollouts)."""
    del rollout_entropies, completion_lengths  # Unused but kept for API parity

    K = len(completions)

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

    reference_mols = get_cached_ground_truths(
        canonical_smiles,
        num_gt=config.grpo.max_ground_truths
    )

    empty_rewards = np.full(K, r_floor, dtype=np.float32)
    empty_sample = np.array([], dtype=np.float32)

    if not reference_mols:
        stats.failed_ground_truth += K
        metrics = GroupMetrics(
            K=K,
            M=0,
            graph_match_rate=0.0,
            finite_rmsd_rate=0.0,
            validity_rate=0.0,
            d_min_mean=float('nan'),
            d_min_p50=float('nan'),
            d_min_p90=float('nan'),
            num_matched=0,
            refs_hit=0,
            max_possible_matches=0,
            match_efficiency=0.0,
            r_qual_mean=0.0,
            r_smcov_mean=0.0,
            r_match_mean=0.0,
            soft_cov_mean=float('nan'),
            pct_gt_0_5=float('nan'),
            fraction_under_delta=0.0,
            matched_dists_sample=empty_sample,
            eligible_dists_sample=empty_sample,
            d_min_sample=empty_sample,
            soft_cov_sample=empty_sample,
            pairwise_sample=empty_sample,
            valid_count=0,
        )
        return empty_rewards, metrics

    M = len(reference_mols)

    rollout_mols: List[Optional[Chem.Mol]] = []
    graph_match_flags: List[bool] = []
    parsed_flags: List[bool] = []

    with profile_section(profiler, "reward_parse"):
        for completion in completions:
            conformer_text = extract_conformer_text(completion)
            generated_smiles = strip_smiles(conformer_text) if conformer_text else ""

            graph_match_flag = False
            if conformer_text is not None and generated_smiles:
                graph_match = (
                    bool(canonical_smiles)
                    and same_molecular_graph(canonical_smiles, generated_smiles)
                )
                if graph_match:
                    graph_match_flag = True
                else:
                    stats.failed_matching_smiles += 1

            mol = parse_conformer_text(conformer_text)
            if mol is None:
                stats.failed_conformer_generation += 1
            rollout_mols.append(mol)
            parsed_flags.append(mol is not None)
            graph_match_flags.append(graph_match_flag)

    mask_graph = np.array(graph_match_flags, dtype=bool)
    mask_parsed = np.array(parsed_flags, dtype=bool)
    graph_match_rate = float(np.mean(mask_graph)) if K > 0 else 0.0

    if not np.any(mask_graph):
        metrics = GroupMetrics(
            K=K,
            M=M,
            graph_match_rate=graph_match_rate,
            finite_rmsd_rate=0.0,
            validity_rate=0.0,
            d_min_mean=float('nan'),
            d_min_p50=float('nan'),
            d_min_p90=float('nan'),
            num_matched=0,
            refs_hit=0,
            max_possible_matches=0,
            match_efficiency=0.0,
            r_qual_mean=0.0,
            r_smcov_mean=0.0,
            r_match_mean=0.0,
            soft_cov_mean=float('nan'),
            pct_gt_0_5=float('nan'),
            fraction_under_delta=0.0,
            matched_dists_sample=empty_sample,
            eligible_dists_sample=empty_sample,
            d_min_sample=empty_sample,
            soft_cov_sample=empty_sample,
            pairwise_sample=empty_sample,
            valid_count=0,
        )
        return empty_rewards, metrics

    mask_graph_parsed = mask_graph & mask_parsed
    validity_for_distance = mask_graph_parsed.astype(np.int32)

    with profile_section(profiler, "reward_rmsd"):
        D = compute_distance_matrix(rollout_mols, reference_mols, validity_for_distance)

    d_i_all = np.min(D, axis=1) if M > 0 else np.full(K, np.inf, dtype=np.float32)
    mask_finite = np.isfinite(d_i_all)
    mask_valid_final = mask_graph_parsed & mask_finite

    problematic_mask = mask_graph_parsed & (~mask_finite)
    num_problematic = int(np.count_nonzero(problematic_mask))
    if hard_rmsd_gate and num_problematic > 0:
        logger.warning(
            f"[reward_v3] RMSD-gate dropped {num_problematic}/"
            f"{int(np.count_nonzero(mask_graph_parsed))} graph-valid rollouts (no finite RMSD)."
        )
        stats.failed_rmsd += num_problematic
    elif not hard_rmsd_gate and num_problematic > 0:
        logger.warning(
            f"[reward_v3] RMSD unavailable for {num_problematic} graph-valid rollouts; "
            "treating rewards as invalid (r_floor)."
        )

    validity_final = mask_valid_final.astype(np.int32)

    with profile_section(profiler, "reward_smcov"):
        smcov_result = compute_smooth_coverage_reward(D, validity_final, rho, return_details=True)
    if isinstance(smcov_result, tuple):
        r_smcov, (soft_cov_mean, soft_cov_pcts, soft_cov_values) = smcov_result
    else:
        r_smcov = smcov_result
        soft_cov_mean = float('nan')
        soft_cov_pcts = [float('nan'), float('nan'), float('nan')]
        soft_cov_values = np.array([], dtype=np.float32)

    with profile_section(profiler, "reward_qual"):
        r_qual = compute_quality_reward(D, validity_final, sigma)

    valid_mask = mask_valid_final
    eligible_matrix = valid_mask[:, None] & (D < delta)
    refs_hit = int(np.count_nonzero(eligible_matrix.any(axis=0)))
    num_valid = int(np.count_nonzero(valid_mask))
    max_possible_matches = min(num_valid, refs_hit)

    with profile_section(profiler, "reward_match"):
        r_match, num_matched, _num_eligible_edges, matched_pairs = compute_matching_reward(
            D, validity_final, delta, eligible_matrix=eligible_matrix
        )

    match_efficiency = (
        float(num_matched) / max_possible_matches if max_possible_matches > 0 else 0.0
    )
    matched_dists = (
        np.array([D[i, j] for (i, j) in matched_pairs], dtype=np.float32)
        if matched_pairs else empty_sample
    )
    eligible_dists = (
        D[eligible_matrix].astype(np.float32)
        if np.any(eligible_matrix) else empty_sample
    )

    rewards = combine_rewards(
        r_qual, r_smcov, r_match, validity_final,
        lambda_qual, lambda_smcov, lambda_match, r_floor
    )

    finite_rmsd_rate = float(np.mean(mask_finite)) if K > 0 else 0.0
    validity_rate = float(np.mean(valid_mask)) if K > 0 else 0.0

    valid_indices = np.where(valid_mask)[0]
    finite_d_valid = d_i_all[valid_indices] if valid_indices.size > 0 else np.array([], dtype=np.float32)
    finite_d_valid = finite_d_valid[np.isfinite(finite_d_valid)]
    under_threshold = float(np.mean(finite_d_valid < delta)) if finite_d_valid.size > 0 else 0.0

    def _sample(values: np.ndarray) -> np.ndarray:
        max_samples = distance_sample_limit if distance_sample_limit > 0 else values.size
        return sample_array(values, max_samples)

    matched_sample = _sample(matched_dists)
    eligible_sample = _sample(eligible_dists)
    d_min_sample = _sample(finite_d_valid.astype(np.float32, copy=False))
    soft_cov_sample = _sample(soft_cov_values.astype(np.float32, copy=False))

    pairwise_sample_cap = pairwise_sample_limit if pairwise_sample_limit is not None else distance_sample_limit
    if enable_pairwise_logging:
        pairwise_dists = compute_pairwise_rollout_distances(
            rollout_mols,
            valid_mask,
            max_samples=max(pairwise_sample_cap, 0),
        )
    else:
        pairwise_dists = empty_sample

    valid_count = int(valid_indices.size)
    if valid_count > 0:
        r_qual_mean = float(np.mean(r_qual[valid_mask]))
        r_smcov_mean = float(np.mean(r_smcov[valid_mask]))
        r_match_mean = float(np.mean(r_match[valid_mask]))
    else:
        r_qual_mean = r_smcov_mean = r_match_mean = 0.0

    for i in range(K):
        if valid_mask[i] and np.isfinite(d_i_all[i]):
            stats.add_rmsd(float(d_i_all[i]))

    metrics = GroupMetrics(
        K=K,
        M=M,
        graph_match_rate=graph_match_rate,
        finite_rmsd_rate=finite_rmsd_rate,
        validity_rate=validity_rate,
        d_min_mean=float(np.mean(finite_d_valid)) if finite_d_valid.size > 0 else float('nan'),
        d_min_p50=float(np.percentile(finite_d_valid, 50)) if finite_d_valid.size > 0 else float('nan'),
        d_min_p90=float(np.percentile(finite_d_valid, 90)) if finite_d_valid.size > 0 else float('nan'),
        num_matched=int(num_matched),
        refs_hit=refs_hit,
        max_possible_matches=int(max_possible_matches),
        match_efficiency=match_efficiency,
        r_qual_mean=r_qual_mean,
        r_smcov_mean=r_smcov_mean,
        r_match_mean=r_match_mean,
        soft_cov_mean=soft_cov_mean,
        pct_gt_0_5=float(soft_cov_pcts[2]) if len(soft_cov_pcts) >= 3 else float('nan'),
        fraction_under_delta=under_threshold,
        matched_dists_sample=matched_sample,
        eligible_dists_sample=eligible_sample,
        d_min_sample=d_min_sample,
        soft_cov_sample=soft_cov_sample,
        pairwise_sample=pairwise_dists,
        valid_count=valid_count,
    )

    return rewards, metrics


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
    """Main GRPO reward function (TRL-compatible)."""
    expected_k = config.grpo.num_generations
    lambda_qual = float(getattr(config.grpo, 'lambda_qual', 1.0))
    lambda_smcov = float(getattr(config.grpo, 'lambda_smcov', 1.0))
    lambda_match = float(getattr(config.grpo, 'lambda_match', 1.0))
    delta = float(getattr(config.grpo, 'delta', 0.75))
    distance_sample_limit = int(getattr(config.grpo, 'log_distance_samples_per_group', 0))
    profile_enabled = bool(getattr(config.grpo, 'profile_rewards', False))
    profiler = RewardProfiler(enabled=profile_enabled)
    total_start = time.perf_counter() if profile_enabled else None
    log_every_steps = max(int(getattr(config.grpo, 'log_every_steps', 1)), 1)
    pairwise_freq = max(int(getattr(config.grpo, 'pairwise_rmsd_log_every', 50)), 1)
    pairwise_flag = bool(getattr(config.grpo, 'enable_pairwise_rmsd_logging', False))
    initial_processed = getattr(stats, "processed_prompts", 0)
    denom = max(int(getattr(config.grpo, 'num_generations', 1)), 1)
    step_index = getattr(stats, "global_step", None)
    if step_index is None:
        step_index = initial_processed // denom
    should_log_pairwise = pairwise_flag and (step_index % pairwise_freq == 0)

    if completion_lengths is not None:
        completion_lengths = [
            None if length is None else int(length)
            for length in completion_lengths
        ]
    groups = group_by_prompt(prompts, completions, expected_k, completion_entropies, completion_lengths)

    final_rewards = [0.0] * len(completions)
    metrics_list: List[GroupMetrics] = []

    for group in groups:
        stats.processed_prompts += len(group['completions'])
        stats.distinct_prompts += 1

        rewards, group_metrics = compute_group_reward(
            canonical_smiles=group['canonical_smiles'],
            completions=group['completions'],
            config=config,
            stats=stats,
            rollout_entropies=group.get('entropy_values'),
            completion_lengths=group.get('completion_lengths'),
            profiler=profiler if profile_enabled else None,
            distance_sample_limit=distance_sample_limit,
            enable_pairwise_logging=should_log_pairwise,
            pairwise_sample_limit=distance_sample_limit if distance_sample_limit > 0 else None,
        )

        # Assign back to flat batch
        for local_idx, global_idx in enumerate(group['indices']):
            final_rewards[global_idx] = float(rewards[local_idx])

        metrics_list.append(group_metrics)

    with profile_section(profiler, "reward_logging"):
        essential_metrics: Dict[str, float] = {}

        def _nanmean(values: List[float]) -> float:
            if not values:
                return 0.0
            arr = np.asarray(values, dtype=np.float32)
            if arr.size == 0:
                return 0.0
            return float(np.nanmean(arr))

        def _concat(samples: List[np.ndarray]) -> np.ndarray:
            arrays = [arr for arr in samples if arr.size > 0]
            return np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)

        validity_rates = [m.validity_rate for m in metrics_list]
        graph_match_rates = [m.graph_match_rate for m in metrics_list]
        finite_rmsd_rates = [m.finite_rmsd_rate for m in metrics_list]
        fraction_under_delta = [m.fraction_under_delta for m in metrics_list]
        refs_hit = [float(m.refs_hit) for m in metrics_list]
        num_matched_list = [float(m.num_matched) for m in metrics_list]
        soft_cov_means = [m.soft_cov_mean for m in metrics_list]
        pct_over_half = [m.pct_gt_0_5 for m in metrics_list]

        d_samples = _concat([m.d_min_sample for m in metrics_list])
        matched_samples = _concat([m.matched_dists_sample for m in metrics_list])
        eligible_samples = _concat([m.eligible_dists_sample for m in metrics_list])
        pairwise_samples = _concat([m.pairwise_sample for m in metrics_list])

        total_matched = int(sum(m.num_matched for m in metrics_list))
        total_max_possible = int(sum(m.max_possible_matches for m in metrics_list))
        match_efficiency_total = (
            float(total_matched) / total_max_possible if total_max_possible > 0 else 0.0
        )
        mean_unique_refs_hit = _nanmean(refs_hit)

        total_valid = sum(m.valid_count for m in metrics_list)
        if total_valid > 0:
            r_qual_mean = sum(m.r_qual_mean * m.valid_count for m in metrics_list) / total_valid
            r_smcov_mean = sum(m.r_smcov_mean * m.valid_count for m in metrics_list) / total_valid
            r_match_mean = sum(m.r_match_mean * m.valid_count for m in metrics_list) / total_valid
        else:
            r_qual_mean = r_smcov_mean = r_match_mean = 0.0

        essential_metrics["validity_rate"] = _nanmean(validity_rates)
        essential_metrics["graph_match_rate"] = _nanmean(graph_match_rates)
        essential_metrics["finite_rmsd_rate"] = _nanmean(finite_rmsd_rates)

        if d_samples.size > 0:
            essential_metrics["geom/d_min_p50"] = float(np.percentile(d_samples, 50))
            essential_metrics["geom/d_min_p90"] = float(np.percentile(d_samples, 90))
            essential_metrics["geom/d_min_mean"] = float(np.mean(d_samples))

        essential_metrics["match/match_efficiency"] = match_efficiency_total
        essential_metrics["match/num_matched"] = _nanmean(num_matched_list)
        if matched_samples.size > 0:
            essential_metrics["match/dist_p50"] = float(np.percentile(matched_samples, 50))
            essential_metrics["match/dist_p90"] = float(np.percentile(matched_samples, 90))
        essential_metrics["match/refs_hit"] = mean_unique_refs_hit

        essential_metrics["reward/component_quality"] = float(lambda_qual * r_qual_mean)
        essential_metrics["reward/component_smcov"] = float(lambda_smcov * r_smcov_mean)
        essential_metrics["reward/component_match"] = float(lambda_match * r_match_mean)

        if soft_cov_means:
            essential_metrics["coverage/soft_mean"] = _nanmean(soft_cov_means)
        if pct_over_half:
            essential_metrics["coverage/pct_gt_0.5"] = _nanmean(pct_over_half)

        if pairwise_samples.size > 0:
            essential_metrics["diversity/pairwise_mean"] = float(np.mean(pairwise_samples))

        essential_metrics["fraction_under_delta"] = _nanmean(fraction_under_delta)
        if eligible_samples.size > 0:
            essential_metrics["match/eligible_dist_p50"] = float(np.percentile(eligible_samples, 50))
        essential_metrics["reward/matched_total"] = float(total_matched)

        should_log_metrics = wandb.run is not None and (step_index % log_every_steps == 0)
        if should_log_metrics:
            wandb.log(essential_metrics)

        batch_log = (
            f"[PID {os.getpid()}] [reward_v3] Batch summary\n"
            f"  validity: graph_match={essential_metrics['graph_match_rate']:.3f}, "
            f"finite_rmsd={essential_metrics['finite_rmsd_rate']:.3f}, "
            f"final={essential_metrics['validity_rate']:.3f}\n"
            f"  matching: max_possible={total_max_possible}, "
            f"matched={total_matched}, match_eff={match_efficiency_total:.3f}\n"
            f"  rewards: r_qual={r_qual_mean:.3f}, r_smcov={r_smcov_mean:.3f}, "
            f"r_match={r_match_mean:.3f}"
        )
        logger.info(batch_log)

        if profile_enabled and wandb.run is not None and total_start is not None:
            profiling_metrics = {
                "profiling/reward_total_s": time.perf_counter() - total_start,
                "profiling/reward_parse_s": profiler.sections.get("reward_parse", 0.0),
                "profiling/reward_rmsd_s": profiler.sections.get("reward_rmsd", 0.0),
                "profiling/reward_smcov_s": profiler.sections.get("reward_smcov", 0.0),
                "profiling/reward_match_s": profiler.sections.get("reward_match", 0.0),
                "profiling/reward_logging_s": profiler.sections.get("reward_logging", 0.0),
            }
            wandb.log(profiling_metrics)

    return final_rewards
