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
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import multiprocessing as mp

import numpy as np
import pandas as pd
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

try:  # pragma: no cover - optional dependency used only when PoseBusters gating is enabled
    from posebusters import PoseBusters  # type: ignore[import]
except Exception:  # pragma: no cover - PoseBusters may be unavailable in some environments
    PoseBusters = None  # type: ignore[assignment]


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
    posebusters_checked: int
    posebusters_passed: int
    posebusters_failed: int
    posebusters_errors: int
    posebusters_time_ms: float
    sampled_percentiles: bool = False


@dataclass
class PoseBustersGateSummary:
    checked: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class PoseBustersRuntimeConfig:
    mode: str = "off"
    max_workers: int = 0
    chunk_size: int = 100
    energy_num_threads: int = 1


@dataclass
class ParsedRolloutBatch:
    """Container with parsed rollout artifacts for a single prompt group."""

    rollout_mols: List[Optional[Chem.Mol]]
    graph_mask: np.ndarray
    parsed_mask: np.ndarray
    conformer_texts: List[Optional[str]]


@dataclass
class GroupValidity:
    """Tracks gating masks applied to a rollout batch."""

    graph_mask: np.ndarray
    parsed_mask: np.ndarray
    base_mask: np.ndarray
    pose_mask: np.ndarray
    reward_mask: np.ndarray


@dataclass
class DistanceArtifacts:
    """Pre-computed RMSD artifacts needed for reward terms."""

    distance_matrix: np.ndarray
    min_distances: np.ndarray
    finite_mask: np.ndarray


@dataclass
class RewardComponents:
    """Holds decomposed reward terms for reporting."""

    quality: np.ndarray
    smooth_coverage: np.ndarray
    matching: np.ndarray
    combined: np.ndarray


@dataclass
class MatchingDiagnostics:
    """Details about eligible edges and solved assignments."""

    eligible_matrix: np.ndarray
    matched_pairs: List[Tuple[int, int]]
    refs_hit: int
    num_matched: int


@dataclass
class SamplingSummary:
    """Indicates whether logged percentiles/metrics are sample-based."""

    percentiles_sampled: bool = False


_POSEBUSTERS_BASE_CONFIG: Dict[str, Any] = {
    "modules": [
        {
            "name": "Loading",
            "function": "loading",
            "chosen_binary_test_output": ["mol_pred_loaded"],
            "rename_outputs": {"mol_pred_loaded": "MOL_PRED loaded"},
        },
        {
            "name": "Chemistry",
            "function": "rdkit_sanity",
            "chosen_binary_test_output": ["passes_rdkit_sanity_checks"],
            "rename_outputs": {"passes_rdkit_sanity_checks": "Sanitization"},
        },
        {
            "name": "Chemistry",
            "function": "inchi_convertible",
            "chosen_binary_test_output": ["inchi_convertible"],
            "rename_outputs": {"inchi_convertible": "InChI convertible"},
        },
        {
            "name": "Chemistry",
            "function": "atoms_connected",
            "chosen_binary_test_output": ["all_atoms_connected"],
            "rename_outputs": {"all_atoms_connected": "All atoms connected"},
        },
        {
            "name": "Geometry",
            "function": "distance_geometry",
            "parameters": {
                "bound_matrix_params": {
                    "set15bounds": True,
                    "scaleVDW": True,
                    "doTriangleSmoothing": True,
                    "useMacrocycle14config": False,
                },
                "threshold_bad_bond_length": 0.25,
                "threshold_bad_angle": 0.25,
                "threshold_clash": 0.3,
                "ignore_hydrogens": True,
                "sanitize": True,
            },
            "chosen_binary_test_output": [
                "bond_lengths_within_bounds",
                "bond_angles_within_bounds",
                "no_internal_clash",
            ],
            "rename_outputs": {
                "bond_lengths_within_bounds": "Bond lengths",
                "bond_angles_within_bounds": "Bond angles",
                "no_internal_clash": "Internal steric clash",
            },
        },
        {
            "name": "Ring flatness",
            "function": "flatness",
            "parameters": {
                "flat_systems": {
                    "aromatic_5_membered_rings_sp2": "[ar5^2]1[ar5^2][ar5^2][ar5^2][ar5^2]1",
                    "aromatic_6_membered_rings_sp2": "[ar6^2]1[ar6^2][ar6^2][ar6^2][ar6^2][ar6^2]1",
                },
                "threshold_flatness": 0.25,
            },
            "chosen_binary_test_output": ["flatness_passes"],
            "rename_outputs": {
                "flatness_passes": "Aromatic ring flatness",
                "num_systems_checked": "number_aromatic_rings_checked",
                "num_systems_passed": "number_aromatic_rings_pass",
                "max_distance": "aromatic_ring_maximum_distance_from_plane",
            },
        },
        {
            "name": "Ring non-flatness",
            "function": "flatness",
            "parameters": {
                "check_nonflat": True,
                "flat_systems": {
                    "non-aromatic_6_membered_rings": "[C,O,S,N;R1]~1[C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1]1",
                    "non-aromatic_6_membered_rings_db03_0": "[C;R1]~1[C;R1][C,O,S,N;R1]~[C,O,S,N;R1][C;R1][C;R1]1",
                    "non-aromatic_6_membered_rings_db03_1": "[C;R1]~1[C;R1][C;R1]~[C;R1][C,O,S,N;R1][C;R1]1",
                    "non-aromatic_6_membered_rings_db02_0": "[C;R1]~1[C;R1][C;R1][C,O,S,N;R1]~[C,O,S,N;R1][C;R1]1",
                    "non-aromatic_6_membered_rings_db02_1": "[C;R1]~1[C;R1][C,O,S,N;R1][C;R1]~[C;R1][C;R1]1",
                },
                "threshold_flatness": 0.05,
            },
            "chosen_binary_test_output": ["flatness_passes"],
            "rename_outputs": {
                "flatness_passes": "Non-aromatic ring non-flatness",
                "num_systems_checked": "number_non-aromatic_rings_checked",
                "num_systems_passed": "number_non-aromatic_rings_pass",
                "max_distance": "non-aromatic_ring_maximum_distance_from_plane",
            },
        },
        {
            "name": "Double bond flatness",
            "function": "flatness",
            "parameters": {
                "flat_systems": {
                    "trigonal_planar_double_bonds": "[C;X3;^2](*)(*)=[C;X3;^2](*)(*)",
                },
                "threshold_flatness": 0.25,
            },
            "chosen_binary_test_output": ["flatness_passes"],
            "rename_outputs": {
                "flatness_passes": "Double bond flatness",
                "num_systems_checked": "number_double_bonds_checked",
                "num_systems_passed": "number_double_bonds_pass",
                "max_distance": "double_bond_maximum_distance_from_plane",
            },
        },
        {
            "name": "Energy ratio",
            "function": "energy_ratio",
            "parameters": {
                "threshold_energy_ratio": 100.0,
                "ensemble_number_conformations": 50,
                "inchi_strict": False,
            },
            "chosen_binary_test_output": ["energy_ratio_passes"],
            "rename_outputs": {"energy_ratio_passes": "Internal energy"},
        },
    ],
    "loading": {
        "mol_pred": {
            "cleanup": False,
            "sanitize": False,
            "add_hs": False,
            "assign_stereo": False,
            "load_all": True,
        },
        "mol_true": {
            "cleanup": False,
            "sanitize": False,
            "add_hs": False,
            "assign_stereo": False,
            "load_all": True,
        },
        "mol_cond": {
            "cleanup": False,
            "sanitize": False,
            "add_hs": False,
            "assign_stereo": False,
            "proximityBonding": False,
        },
    },
}

_POSEBUSTERS_STAGE_COUNTS: Dict[str, int] = {
    "basic": 4,
    "geometry": 5,
    "full": len(_POSEBUSTERS_BASE_CONFIG["modules"]),
}

_POSEBUSTERS_RUNNER_CACHE: Dict[Tuple[str, int, int, int], PoseBusters] = {}
_POSEBUSTERS_ENERGY_WARNING_EMITTED = False


_EMPTY_FLOAT32 = np.array([], dtype=np.float32)
_RMSD_POOL: Optional[mp.pool.Pool] = None
_RMSD_POOL_SIZE = 0
_RMSD_WORKER_STATE: Dict[str, Any] = {"ref_key": None, "ref_mols": []}


def _normalize_posebusters_config(raw_cfg: Any) -> PoseBustersRuntimeConfig:
    """Normalize user-provided config objects/dicts into a runtime config."""
    if isinstance(raw_cfg, PoseBustersRuntimeConfig):
        return raw_cfg
    if raw_cfg is None:
        return PoseBustersRuntimeConfig()
    if isinstance(raw_cfg, str):
        data: Dict[str, Any] = {"mode": raw_cfg}
    elif isinstance(raw_cfg, dict):
        data = raw_cfg
    else:
        data = {
            "mode": getattr(raw_cfg, "mode", None),
            "max_workers": getattr(raw_cfg, "max_workers", None),
            "chunk_size": getattr(raw_cfg, "chunk_size", None),
            "energy_num_threads": getattr(raw_cfg, "energy_num_threads", None),
        }

    mode = str(data.get("mode", "off") or "off").lower()
    if mode not in {"off", "basic", "geometry", "full"}:
        logger.warning(f"Unknown PoseBusters mode '{mode}'. Falling back to 'off'.")
        mode = "off"

    def _to_int(value: Any, default: int) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    max_workers = _to_int(data.get("max_workers", 0), 0)
    chunk_size = _to_int(data.get("chunk_size", 100), 100)
    energy_threads = _to_int(data.get("energy_num_threads", 1), 1)

    global _POSEBUSTERS_ENERGY_WARNING_EMITTED
    effective_energy_threads = energy_threads
    if mode == "full" and max_workers > 0 and energy_threads != 1:
        if not _POSEBUSTERS_ENERGY_WARNING_EMITTED:
            logger.info(
                "[reward_v3] Clamping PoseBusters energy_num_threads to 1 when using multiprocessing "
                "to avoid CPU oversubscription."
            )
            _POSEBUSTERS_ENERGY_WARNING_EMITTED = True
        effective_energy_threads = 1

    return PoseBustersRuntimeConfig(
        mode=mode,
        max_workers=max_workers,
        chunk_size=chunk_size,
        energy_num_threads=effective_energy_threads,
    )


def _build_posebusters_config(mode: str, energy_threads: int) -> Dict[str, Any]:
    """Return a PoseBusters config dictionary tailored to the requested mode."""
    if mode not in _POSEBUSTERS_STAGE_COUNTS:
        raise ValueError(f"Unsupported PoseBusters mode '{mode}'.")

    module_count = _POSEBUSTERS_STAGE_COUNTS[mode]
    modules = deepcopy(_POSEBUSTERS_BASE_CONFIG["modules"][:module_count])
    if mode == "full" and modules:
        energy_module = modules[-1]
        if energy_module.get("function") == "energy_ratio":
            params = energy_module.setdefault("parameters", {})
            params["num_threads"] = energy_threads
    return {
        "modules": modules,
        "loading": deepcopy(_POSEBUSTERS_BASE_CONFIG["loading"]),
    }


def _get_posebusters_runner(settings: PoseBustersRuntimeConfig) -> PoseBusters:
    """Return (and cache) a PoseBusters runner for the given settings."""
    if PoseBusters is None:
        raise RuntimeError("PoseBusters package is unavailable, but posebusters mode is enabled.")
    cache_key = (
        settings.mode,
        settings.max_workers,
        settings.chunk_size,
        settings.energy_num_threads,
    )
    runner = _POSEBUSTERS_RUNNER_CACHE.get(cache_key)
    if runner is None:
        config_dict = _build_posebusters_config(settings.mode, settings.energy_num_threads)
        runner = PoseBusters(
            config=config_dict,
            max_workers=settings.max_workers,
            chunk_size=(settings.chunk_size if settings.chunk_size > 0 else None),
        )
        _POSEBUSTERS_RUNNER_CACHE[cache_key] = runner
    return runner


def _extract_posebusters_pass_vector(table: pd.DataFrame, expected_len: int) -> List[bool]:
    """Extract per-row pass/fail booleans from a PoseBusters result table."""
    if table is None or table.empty:
        return [False] * expected_len
    bool_df = table.select_dtypes(include=["bool"])
    if bool_df.empty:
        return [False] * expected_len
    passes_raw = bool_df.all(axis=1)
    passes: List[bool] = []
    for value in passes_raw.tolist():
        if value is pd.NA or value is None:
            passes.append(False)
        else:
            passes.append(bool(value))
    if len(passes) < expected_len:
        passes.extend([False] * (expected_len - len(passes)))
    elif len(passes) > expected_len:
        passes = passes[:expected_len]
    return passes


def sample_array(
    values: np.ndarray,
    max_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, bool]:
    """Return at most `max_samples` entries as float32, with unbiased sampling.

    Returns:
        (sampled_values, used_sampling_flag)
    """
    if values.size == 0 or max_samples <= 0:
        return _EMPTY_FLOAT32, False
    if values.size <= max_samples:
        return values.astype(np.float32, copy=False), False
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.choice(values.size, size=max_samples, replace=False)
    return values[idx].astype(np.float32, copy=False), True


def make_reward_rng(config, stats) -> np.random.Generator:
    """Create a deterministic RNG seeded from config + training state."""

    def _maybe_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    seed_candidates = [
        _maybe_int(getattr(config, "seed", None)),
        _maybe_int(getattr(getattr(config, "training", None), "seed", None)),
        _maybe_int(getattr(config.grpo, "reward_seed", None)),
        _maybe_int(getattr(stats, "global_step", None)),
        os.getpid(),
    ]
    combined = 0
    for value in seed_candidates:
        if value is None:
            continue
        combined = (combined * 6364136223846793005 + value) & 0xFFFFFFFFFFFFFFFF
    if combined == 0:
        combined = int(time.time() * 1e6) & 0xFFFFFFFFFFFFFFFF
    return np.random.default_rng(combined)


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


def _close_rmsd_pool() -> None:
    global _RMSD_POOL, _RMSD_POOL_SIZE
    if _RMSD_POOL is not None:
        _RMSD_POOL.close()
        _RMSD_POOL.join()
    _RMSD_POOL = None
    _RMSD_POOL_SIZE = 0


def _get_rmsd_pool(worker_count: int) -> Optional[mp.pool.Pool]:
    if worker_count <= 1:
        return None
    global _RMSD_POOL, _RMSD_POOL_SIZE
    if _RMSD_POOL is not None and _RMSD_POOL_SIZE == worker_count:
        return _RMSD_POOL
    _close_rmsd_pool()
    try:
        ctx = mp.get_context("spawn")
        _RMSD_POOL = ctx.Pool(processes=worker_count)
        _RMSD_POOL_SIZE = worker_count
        return _RMSD_POOL
    except Exception as exc:
        logger.warning("[reward_v3] Failed to create RMSD pool (%s); falling back to serial.", exc)
        _RMSD_POOL = None
        _RMSD_POOL_SIZE = 0
        return None


def _mol_to_block(mol: Optional[Chem.Mol]) -> Optional[str]:
    if mol is None:
        return None
    try:
        return Chem.MolToMolBlock(mol, includeStereo=False)
    except Exception:
        return None


def _rmsd_parallel_worker(
    payload: Tuple[int, Optional[str], Optional[str], Optional[List[str]]]
) -> Tuple[int, Optional[List[float]]]:
    idx, rollout_block, ref_key, ref_blocks = payload
    if rollout_block is None or ref_blocks is None or ref_key is None:
        return idx, None

    state = _RMSD_WORKER_STATE
    if state["ref_key"] != ref_key:
        ref_mols: List[Chem.Mol] = []
        for block in ref_blocks:
            try:
                mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
            except Exception:
                mol = None
            if mol is not None:
                ref_mols.append(mol)
        state["ref_key"] = ref_key
        state["ref_mols"] = ref_mols

    try:
        rollout_mol = Chem.MolFromMolBlock(rollout_block, sanitize=False, removeHs=False)
    except Exception:
        rollout_mol = None

    if rollout_mol is None or not state["ref_mols"]:
        return idx, None

    distances = [compute_rmsd_safe(rollout_mol, ref) for ref in state["ref_mols"]]
    return idx, distances

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


def _compute_distance_matrix_serial(
    rollout_mols: List[Optional[Chem.Mol]],
    reference_mols: List[Chem.Mol],
    validity: np.ndarray,
) -> np.ndarray:
    K = len(rollout_mols)
    M = len(reference_mols)
    D = np.full((K, M), float("inf"), dtype=np.float32)

    for i in range(K):
        if not validity[i]:
            continue
        mol = rollout_mols[i]
        if mol is None:
            continue
        for j, ref_mol in enumerate(reference_mols):
            D[i, j] = compute_rmsd_safe(mol, ref_mol)
    return D

def _compute_distance_matrix_parallel(
    rollout_mols: List[Optional[Chem.Mol]],
    reference_mols: List[Chem.Mol],
    validity: np.ndarray,
    worker_count: int,
    ref_cache_key: str,
) -> Optional[np.ndarray]:
    K = len(rollout_mols)
    M = len(reference_mols)
    pool = _get_rmsd_pool(worker_count)
    if pool is None:
        return None

    valid_indices = [
        idx for idx, flag in enumerate(validity) if flag and rollout_mols[idx] is not None
    ]
    if not valid_indices:
        return np.full((K, M), float("inf"), dtype=np.float32)

    ref_blocks = [_mol_to_block(mol) for mol in reference_mols]
    if any(block is None for block in ref_blocks):
        logger.debug("[reward_v3] Unable to serialize reference mols for parallel RMSD.")
        return None

    tasks: List[Tuple[int, Optional[str], Optional[str], Optional[List[str]]]] = []
    for idx in valid_indices:
        tasks.append((idx, _mol_to_block(rollout_mols[idx]), ref_cache_key, ref_blocks))

    try:
        results = pool.map(_rmsd_parallel_worker, tasks)
    except Exception as exc:
        logger.warning("[reward_v3] Parallel RMSD failed (%s); falling back to serial.", exc)
        return None

    D = np.full((K, M), float("inf"), dtype=np.float32)
    for idx, distances in results:
        if distances is None:
            continue
        D[idx, :] = np.asarray(distances, dtype=np.float32)
    return D


def compute_distance_matrix(
    rollout_mols: List[Optional[Chem.Mol]],
    reference_mols: List[Chem.Mol],
    validity: np.ndarray,
    rmsd_workers: int = 0,
    ref_cache_key: Optional[str] = None,
) -> np.ndarray:
    """Compute RMSD distance matrix D[i,j] = d(y_i, g_j)."""
    K = len(rollout_mols)
    M = len(reference_mols)

    if K == 0 or M == 0:
        return np.full((K, M), float("inf"), dtype=np.float32)

    if rmsd_workers and rmsd_workers > 1 and ref_cache_key:
        parallel = _compute_distance_matrix_parallel(
            rollout_mols,
            reference_mols,
            validity,
            rmsd_workers,
            ref_cache_key,
        )
        if parallel is not None:
            return parallel

    return _compute_distance_matrix_serial(rollout_mols, reference_mols, validity)


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
    K_matrix = np.zeros((K, M), dtype=np.float32)
    if np.any(valid_rows):
        D_valid = D[valid_rows].astype(np.float32, copy=False)
        D_valid = np.where(np.isfinite(D_valid), D_valid, np.inf).astype(np.float32, copy=False)
        exponent = -((D_valid / rho) ** 2)
        K_valid = np.exp(exponent).astype(np.float32, copy=False)
        K_valid = np.where(np.isfinite(K_valid), K_valid, 0.0).astype(np.float32, copy=False)
        K_matrix[valid_rows, :] = K_valid

    one_minus = np.clip(1.0 - K_matrix, 0.0, 1.0).astype(np.float32, copy=False)
    zero_mask = one_minus <= 1e-12
    one_minus_safe = np.where(zero_mask, 1.0, one_minus).astype(np.float32, copy=False)
    one_minus_f64 = one_minus_safe.astype(np.float64, copy=False)
    log_one_minus = np.log(one_minus_f64)
    log_prod_safe = np.sum(log_one_minus, axis=0, keepdims=True)
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


def apply_posebusters_gate(
    rollout_mols: List[Optional[Chem.Mol]],
    base_valid_mask: np.ndarray,
    settings: PoseBustersRuntimeConfig,
) -> Tuple[np.ndarray, PoseBustersGateSummary]:
    """Apply PoseBusters gating atop the base validity mask."""
    summary = PoseBustersGateSummary()
    if settings.mode == "off":
        return base_valid_mask, summary

    valid_indices = [
        idx
        for idx, flag in enumerate(base_valid_mask)
        if flag and rollout_mols[idx] is not None
    ]
    if not valid_indices:
        return base_valid_mask, summary

    summary.checked = len(valid_indices)
    try:
        runner = _get_posebusters_runner(settings)
    except Exception as exc:  # pragma: no cover - depends on runtime environments
        logger.warning(
            "[reward_v3] PoseBusters unavailable (%s); marking %d samples invalid.",
            exc,
            summary.checked,
        )
        updated = base_valid_mask.copy()
        updated[valid_indices] = False
        summary.errors = summary.checked
        return updated, summary

    mol_batch = [rollout_mols[i] for i in valid_indices]
    start = time.perf_counter()
    try:
        report = runner.bust(mol_pred=mol_batch, full_report=False)
        if not isinstance(report, pd.DataFrame):
            raise TypeError("PoseBusters returned an unexpected result.")
    except Exception as exc:  # pragma: no cover - depends on PoseBusters runtime
        logger.warning("[reward_v3] PoseBusters execution failed: %s", exc)
        updated = base_valid_mask.copy()
        updated[valid_indices] = False
        summary.errors = summary.checked
        summary.elapsed_ms = (time.perf_counter() - start) * 1000.0
        return updated, summary

    summary.elapsed_ms = (time.perf_counter() - start) * 1000.0
    passes = _extract_posebusters_pass_vector(report, len(valid_indices))
    summary.passed = int(sum(passes))
    summary.failed = summary.checked - summary.passed

    updated_mask = base_valid_mask.copy()
    for idx, passed in zip(valid_indices, passes):
        updated_mask[idx] = bool(passed)

    return updated_mask, summary


def parse_rollout_group(
    canonical_smiles: str,
    completions: List[str],
    stats,
    profiler: Optional[RewardProfiler],
) -> ParsedRolloutBatch:
    rollout_mols: List[Optional[Chem.Mol]] = []
    graph_flags: List[bool] = []
    parsed_flags: List[bool] = []
    conformer_texts: List[Optional[str]] = []

    with profile_section(profiler, "reward_parse"):
        for completion in completions:
            conformer_text = extract_conformer_text(completion)
            conformer_texts.append(conformer_text)
            graph_match_flag = False

            if conformer_text:
                generated_smiles = strip_smiles(conformer_text)
                if generated_smiles:
                    if canonical_smiles and same_molecular_graph(canonical_smiles, generated_smiles):
                        graph_match_flag = True
                    else:
                        stats.failed_matching_smiles += 1

            mol = parse_conformer_text(conformer_text)
            if mol is None:
                stats.failed_conformer_generation += 1

            rollout_mols.append(mol)
            parsed_flags.append(mol is not None)
            graph_flags.append(graph_match_flag)

    mask_graph = np.array(graph_flags, dtype=bool)
    mask_parsed = np.array(parsed_flags, dtype=bool)

    return ParsedRolloutBatch(
        rollout_mols=rollout_mols,
        graph_mask=mask_graph,
        parsed_mask=mask_parsed,
        conformer_texts=conformer_texts,
    )


def build_group_validity(
    parsed: ParsedRolloutBatch,
    pose_cfg: PoseBustersRuntimeConfig,
    profiler: Optional[RewardProfiler],
) -> Tuple[GroupValidity, PoseBustersGateSummary]:
    graph_mask = parsed.graph_mask.astype(bool, copy=False)
    parsed_mask = parsed.parsed_mask.astype(bool, copy=False)
    base_mask = graph_mask & parsed_mask
    pose_mask = base_mask.copy()
    pose_summary = PoseBustersGateSummary()

    if np.any(base_mask):
        with profile_section(profiler, "reward_posebusters"):
            pose_mask, pose_summary = apply_posebusters_gate(
                parsed.rollout_mols,
                base_mask.astype(bool, copy=False),
                pose_cfg,
            )

    validity = GroupValidity(
        graph_mask=graph_mask,
        parsed_mask=parsed_mask,
        base_mask=base_mask,
        pose_mask=pose_mask.astype(bool, copy=False),
        reward_mask=pose_mask.astype(bool, copy=False),
    )
    return validity, pose_summary


def compute_distance_artifacts(
    parsed: ParsedRolloutBatch,
    reference_mols: List[Chem.Mol],
    validity: GroupValidity,
    hard_rmsd_gate: bool,
    profiler: Optional[RewardProfiler],
    stats,
    rmsd_workers: int,
    ref_cache_key: str,
) -> DistanceArtifacts:
    mask_for_distance = validity.reward_mask.astype(np.int32, copy=False)
    with profile_section(profiler, "reward_rmsd"):
        D = compute_distance_matrix(
            parsed.rollout_mols,
            reference_mols,
            mask_for_distance,
            rmsd_workers=rmsd_workers,
            ref_cache_key=ref_cache_key,
        )

    if reference_mols:
        d_i_all = np.min(D, axis=1)
    else:
        d_i_all = np.full(len(parsed.rollout_mols), np.inf, dtype=np.float32)

    mask_finite = np.isfinite(d_i_all)
    problematic_mask = validity.reward_mask & (~mask_finite)
    num_problematic = int(np.count_nonzero(problematic_mask))
    if num_problematic > 0:
        stats.failed_rmsd += num_problematic
        if hard_rmsd_gate:
            logger.warning(
                f"[reward_v3] {num_problematic} PoseBusters-valid rollouts "
                "lacked finite RMSD; quality/coverage rewards will be zero for them."
            )
        else:
            logger.warning(
                f"[reward_v3] {num_problematic} PoseBusters-valid rollouts "
                "lacked finite RMSD; applying reward floor for those entries."
            )
        validity.reward_mask &= mask_finite

    return DistanceArtifacts(
        distance_matrix=D,
        min_distances=d_i_all,
        finite_mask=mask_finite,
    )


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
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, GroupMetrics]:
    """Compute rewards for a single prompt group (K rollouts)."""
    del rollout_entropies, completion_lengths  # Unused but kept for API parity

    K = len(completions)
    delta = float(getattr(config.grpo, "delta", 0.75))
    sigma = float(getattr(config.grpo, "sigma", 0.25))
    rho = float(getattr(config.grpo, "rho", 0.75))
    lambda_qual = float(getattr(config.grpo, "lambda_qual", 1.0))
    lambda_smcov = float(getattr(config.grpo, "lambda_smcov", 1.0))
    lambda_match = float(getattr(config.grpo, "lambda_match", 1.0))
    r_floor = float(getattr(config.grpo, "r_floor", -1.0))
    hard_rmsd_gate = bool(
        getattr(config.grpo, "hard_rmsd_gate", DEFAULT_ENABLE_HARD_RMSD_GATE)
    )
    rmsd_workers = int(getattr(config.grpo, "rmsd_workers", 0) or 0)

    reference_mols = get_cached_ground_truths(
        canonical_smiles,
        num_gt=config.grpo.max_ground_truths,
    )

    empty_rewards = np.full(K, r_floor, dtype=np.float32)

    if not reference_mols:
        stats.failed_ground_truth += K
        metrics = GroupMetrics(
            K=K,
            M=0,
            graph_match_rate=0.0,
            finite_rmsd_rate=0.0,
            validity_rate=0.0,
            d_min_mean=float("nan"),
            d_min_p50=float("nan"),
            d_min_p90=float("nan"),
            num_matched=0,
            refs_hit=0,
            max_possible_matches=0,
            match_efficiency=0.0,
            r_qual_mean=0.0,
            r_smcov_mean=0.0,
            r_match_mean=0.0,
            soft_cov_mean=float("nan"),
            pct_gt_0_5=float("nan"),
            fraction_under_delta=0.0,
            matched_dists_sample=_EMPTY_FLOAT32,
            eligible_dists_sample=_EMPTY_FLOAT32,
            d_min_sample=_EMPTY_FLOAT32,
            soft_cov_sample=_EMPTY_FLOAT32,
            pairwise_sample=_EMPTY_FLOAT32,
            valid_count=0,
            posebusters_checked=0,
            posebusters_passed=0,
            posebusters_failed=0,
            posebusters_errors=0,
            posebusters_time_ms=0.0,
            sampled_percentiles=False,
        )
        return empty_rewards, metrics

    parsed = parse_rollout_group(canonical_smiles, completions, stats, profiler)
    graph_match_rate = float(np.mean(parsed.graph_mask)) if K > 0 else 0.0

    if not np.any(parsed.graph_mask):
        metrics = GroupMetrics(
            K=K,
            M=len(reference_mols),
            graph_match_rate=graph_match_rate,
            finite_rmsd_rate=0.0,
            validity_rate=0.0,
            d_min_mean=float("nan"),
            d_min_p50=float("nan"),
            d_min_p90=float("nan"),
            num_matched=0,
            refs_hit=0,
            max_possible_matches=0,
            match_efficiency=0.0,
            r_qual_mean=0.0,
            r_smcov_mean=0.0,
            r_match_mean=0.0,
            soft_cov_mean=float("nan"),
            pct_gt_0_5=float("nan"),
            fraction_under_delta=0.0,
            matched_dists_sample=_EMPTY_FLOAT32,
            eligible_dists_sample=_EMPTY_FLOAT32,
            d_min_sample=_EMPTY_FLOAT32,
            soft_cov_sample=_EMPTY_FLOAT32,
            pairwise_sample=_EMPTY_FLOAT32,
            valid_count=0,
            posebusters_checked=0,
            posebusters_passed=0,
            posebusters_failed=0,
            posebusters_errors=0,
            posebusters_time_ms=0.0,
            sampled_percentiles=False,
        )
        return empty_rewards, metrics

    pose_cfg = _normalize_posebusters_config(getattr(config.grpo, "posebusters", None))
    validity, pose_summary = build_group_validity(parsed, pose_cfg, profiler)

    stats.posebusters_checked += pose_summary.checked
    stats.posebusters_failed += pose_summary.failed
    stats.posebusters_errors += pose_summary.errors
    stats.posebusters_time_ms += pose_summary.elapsed_ms
    stats.posebusters_successes += pose_summary.passed
    stats.posebusters_failures += pose_summary.failed + pose_summary.errors

    ref_cache_key = f"{canonical_smiles}:{len(reference_mols)}"
    distance_artifacts = compute_distance_artifacts(
        parsed=parsed,
        reference_mols=reference_mols,
        validity=validity,
        hard_rmsd_gate=hard_rmsd_gate,
        profiler=profiler,
        stats=stats,
        rmsd_workers=rmsd_workers,
        ref_cache_key=ref_cache_key,
    )

    validity_final = validity.reward_mask.astype(np.int32, copy=False)

    with profile_section(profiler, "reward_smcov"):
        smcov_result = compute_smooth_coverage_reward(
            distance_artifacts.distance_matrix,
            validity_final,
            rho,
            return_details=True,
        )
    if isinstance(smcov_result, tuple):
        r_smcov, (soft_cov_mean, soft_cov_pcts, soft_cov_values) = smcov_result
    else:
        r_smcov = smcov_result
        soft_cov_mean = float("nan")
        soft_cov_pcts = [float("nan"), float("nan"), float("nan")]
        soft_cov_values = _EMPTY_FLOAT32

    with profile_section(profiler, "reward_qual"):
        r_qual = compute_quality_reward(distance_artifacts.distance_matrix, validity_final, sigma)

    valid_mask = validity.reward_mask
    eligible_matrix = valid_mask[:, None] & (distance_artifacts.distance_matrix < delta)
    refs_hit = int(np.count_nonzero(eligible_matrix.any(axis=0)))
    num_valid = int(np.count_nonzero(valid_mask))
    max_possible_matches = min(num_valid, refs_hit)

    with profile_section(profiler, "reward_match"):
        r_match, num_matched, _num_eligible_edges, matched_pairs = compute_matching_reward(
            distance_artifacts.distance_matrix,
            validity_final,
            delta,
            eligible_matrix=eligible_matrix,
        )

    matching_diag = MatchingDiagnostics(
        eligible_matrix=eligible_matrix,
        matched_pairs=matched_pairs,
        refs_hit=refs_hit,
        num_matched=num_matched,
    )

    rewards = combine_rewards(
        r_qual,
        r_smcov,
        r_match,
        validity_final,
        lambda_qual,
        lambda_smcov,
        lambda_match,
        r_floor,
    )

    finite_rmsd_rate = float(np.mean(distance_artifacts.finite_mask)) if K > 0 else 0.0
    validity_rate = float(np.mean(valid_mask)) if K > 0 else 0.0
    finite_d_valid = distance_artifacts.min_distances[valid_mask]
    finite_d_valid = finite_d_valid[np.isfinite(finite_d_valid)]
    under_threshold = float(np.mean(finite_d_valid < delta)) if finite_d_valid.size > 0 else 0.0

    def _sample(values: np.ndarray) -> Tuple[np.ndarray, bool]:
        limit = distance_sample_limit if distance_sample_limit > 0 else values.size
        if limit <= 0:
            return _EMPTY_FLOAT32, False
        return sample_array(values, limit, rng=rng)

    reward_components = RewardComponents(
        quality=r_qual,
        smooth_coverage=r_smcov,
        matching=r_match,
        combined=rewards,
    )

    match_efficiency = (
        float(matching_diag.num_matched) / max_possible_matches if max_possible_matches > 0 else 0.0
    )

    matched_dists = (
        np.array(
            [distance_artifacts.distance_matrix[i, j] for (i, j) in matching_diag.matched_pairs],
            dtype=np.float32,
        )
        if matching_diag.matched_pairs
        else _EMPTY_FLOAT32
    )
    eligible_dists = (
        distance_artifacts.distance_matrix[eligible_matrix].astype(np.float32)
        if np.any(eligible_matrix)
        else _EMPTY_FLOAT32
    )

    matched_sample, matched_sampled = _sample(matched_dists)
    eligible_sample, eligible_sampled = _sample(eligible_dists)
    d_min_sample, d_min_sampled = _sample(finite_d_valid.astype(np.float32, copy=False))
    soft_cov_sample, _ = _sample(soft_cov_values.astype(np.float32, copy=False))

    pairwise_sample_cap = (
        pairwise_sample_limit if pairwise_sample_limit is not None else distance_sample_limit
    )
    if enable_pairwise_logging:
        pairwise_dists = compute_pairwise_rollout_distances(
            parsed.rollout_mols,
            valid_mask,
            max_samples=max(pairwise_sample_cap, 0),
        )
    else:
        pairwise_dists = _EMPTY_FLOAT32

    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count > 0:
        r_qual_mean = float(np.mean(reward_components.quality[valid_mask]))
        r_smcov_mean = float(np.mean(reward_components.smooth_coverage[valid_mask]))
        r_match_mean = float(np.mean(reward_components.matching[valid_mask]))
    else:
        r_qual_mean = r_smcov_mean = r_match_mean = 0.0

    for i in range(K):
        if valid_mask[i] and np.isfinite(distance_artifacts.min_distances[i]):
            stats.add_rmsd(float(distance_artifacts.min_distances[i]))

    sampling_summary = SamplingSummary(
        percentiles_sampled=matched_sampled or eligible_sampled or d_min_sampled,
    )

    metrics = GroupMetrics(
        K=K,
        M=len(reference_mols),
        graph_match_rate=graph_match_rate,
        finite_rmsd_rate=finite_rmsd_rate,
        validity_rate=validity_rate,
        d_min_mean=float(np.mean(finite_d_valid)) if finite_d_valid.size > 0 else float("nan"),
        d_min_p50=float(np.percentile(finite_d_valid, 50)) if finite_d_valid.size > 0 else float("nan"),
        d_min_p90=float(np.percentile(finite_d_valid, 90)) if finite_d_valid.size > 0 else float("nan"),
        num_matched=int(matching_diag.num_matched),
        refs_hit=matching_diag.refs_hit,
        max_possible_matches=int(max_possible_matches),
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
        posebusters_checked=pose_summary.checked,
        posebusters_passed=pose_summary.passed,
        posebusters_failed=pose_summary.failed,
        posebusters_errors=pose_summary.errors,
        posebusters_time_ms=pose_summary.elapsed_ms,
        sampled_percentiles=sampling_summary.percentiles_sampled,
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
    reward_rng = make_reward_rng(config, stats)
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
            rng=reward_rng,
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
        posebusters_checks = [float(m.posebusters_checked) for m in metrics_list]
        posebusters_passes = [float(m.posebusters_passed) for m in metrics_list]
        posebusters_failures = [float(m.posebusters_failed) for m in metrics_list]
        posebusters_errors = [float(m.posebusters_errors) for m in metrics_list]
        posebusters_times = [m.posebusters_time_ms for m in metrics_list]
        sampling_flags = [m.sampled_percentiles for m in metrics_list]

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

        total_posebusters_checked = sum(posebusters_checks)
        total_posebusters_passed = sum(posebusters_passes)
        total_posebusters_failed = sum(posebusters_failures)
        total_posebusters_errors = sum(posebusters_errors)
        posebusters_pass_rate = (
            total_posebusters_passed / total_posebusters_checked
            if total_posebusters_checked > 0
            else 0.0
        )

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
        essential_metrics["sampling/approx_percentiles"] = 1.0 if any(sampling_flags) else 0.0
        essential_metrics["posebusters/pass_rate"] = posebusters_pass_rate
        essential_metrics["posebusters/fail_rate"] = (
            total_posebusters_failed / total_posebusters_checked if total_posebusters_checked > 0 else 0.0
        )
        essential_metrics["posebusters/error_rate"] = (
            total_posebusters_errors / total_posebusters_checked if total_posebusters_checked > 0 else 0.0
        )
        if posebusters_times:
            essential_metrics["posebusters/time_ms_per_group"] = float(_nanmean(posebusters_times))

        should_log_metrics = wandb.run is not None and (step_index % log_every_steps == 0)
        if should_log_metrics:
            wandb.log(essential_metrics, step=step_index)

        batch_log = (
            f"[PID {os.getpid()}] [reward_v3] Batch summary\n"
            f"  validity: graph_match={essential_metrics['graph_match_rate']:.3f}, "
            f"finite_rmsd={essential_metrics['finite_rmsd_rate']:.3f}, "
            f"final={essential_metrics['validity_rate']:.3f}\n"
            f"  matching: max_possible={total_max_possible}, "
            f"matched={total_matched}, match_eff={match_efficiency_total:.3f}\n"
            f"  rewards: r_qual={r_qual_mean:.3f}, r_smcov={r_smcov_mean:.3f}, "
            f"r_match={r_match_mean:.3f}\n"
            f"  posebusters: checked={int(total_posebusters_checked)}, "
            f"passed={int(total_posebusters_passed)}, failed={int(total_posebusters_failed)}, "
            f"errors={int(total_posebusters_errors)}, pass_rate={posebusters_pass_rate:.3f}"
        )
        logger.info(batch_log)

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
