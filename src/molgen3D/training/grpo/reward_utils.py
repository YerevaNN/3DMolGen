"""Utility helpers shared by the GRPO reward implementation."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger
from rdkit import Chem

from molgen3D.data_processing.smiles_encoder_decoder import decode_cartesian_v2, strip_smiles
from molgen3D.evaluation.utils import extract_between, same_molecular_graph
from molgen3D.training.grpo.utils import load_ground_truths
from molgen3D.utils.utils import get_best_rmsd

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - SciPy is optional
    linear_sum_assignment = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency used when PoseBusters is enabled
    from posebusters import PoseBusters  # type: ignore[import]
except Exception:  # pragma: no cover - PoseBusters may be unavailable
    PoseBusters = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from molgen3D.training.grpo.grpo_reward_v3 import GroupMetrics

__all__ = [
    "RewardProfiler",
    "profile_section",
    "PoseBustersRuntimeConfig",
    "normalize_posebusters_config",
    "apply_posebusters_gate",
    "get_cached_ground_truths",
    "parse_rollout_group",
    "compute_distance_matrix",
    "compute_quality_reward",
    "compute_smooth_coverage_reward",
    "compute_matching_reward",
    "combine_rewards",
    "compute_pairwise_rollout_distances",
    "sample_array",
    "make_reward_rng",
    "summarize_batch_metrics",
    "_GROUND_TRUTH_CACHE",
    "_build_posebusters_config",
    "_get_posebusters_runner",
]

GROUND_TRUTH_CACHE_SIZE = 256
_GROUND_TRUTH_CACHE: "OrderedDict[Tuple[str, int], List[Chem.Mol]]" = OrderedDict()
_LINEAR_SUM_WARNING_EMITTED = False

EMPTY_FLOAT32 = np.array([], dtype=np.float32)
_RMSD_POOL: Optional[mp.pool.Pool] = None
_RMSD_POOL_SIZE = 0
_RMSD_WORKER_STATE: Dict[str, Any] = {"ref_key": None, "ref_mols": []}


class _NoOpSection:
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


class PoseBustersRuntimeConfig(NamedTuple):
    mode: str = "off"
    max_workers: int = 0
    chunk_size: int = 100
    energy_num_threads: int = 1


def sample_array(values: np.ndarray, max_samples: int, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, bool]:
    if values.size == 0 or max_samples <= 0:
        return EMPTY_FLOAT32, False
    if values.size <= max_samples:
        return values.astype(np.float32, copy=False), False
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.choice(values.size, size=max_samples, replace=False)
    return values[idx].astype(np.float32, copy=False), True


def make_reward_rng(config, stats) -> np.random.Generator:
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


def extract_conformer_text(completion: str) -> Optional[str]:
    conformer_text = extract_between(completion, "[CONFORMER]", "[/CONFORMER]")
    return conformer_text if conformer_text else None


def parse_conformer_text(conformer_text: Optional[str]) -> Optional[Chem.Mol]:
    if not conformer_text:
        return None
    try:
        return decode_cartesian_v2(conformer_text)
    except Exception as exc:  # pragma: no cover - RDKit errors depend on inputs
        logger.debug(f"Conformer parsing failed: {exc}")
        return None


def parse_rollout_group(
    canonical_smiles: str,
    completions: List[str],
    stats,
    profiler: Optional[RewardProfiler],
) -> Tuple[List[Optional[Chem.Mol]], np.ndarray, np.ndarray]:
    rollout_mols: List[Optional[Chem.Mol]] = []
    graph_flags: List[bool] = []
    parsed_flags: List[bool] = []

    with profile_section(profiler, "reward_parse"):
        for completion in completions:
            conformer_text = extract_conformer_text(completion)
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

    graph_mask = np.array(graph_flags, dtype=bool)
    parsed_mask = np.array(parsed_flags, dtype=bool)
    return rollout_mols, graph_mask, parsed_mask


def get_cached_ground_truths(canonical_smiles: str, num_gt: int) -> List[Chem.Mol]:
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


def _rmsd_parallel_worker(payload: Tuple[int, Optional[str], Optional[str], Optional[List[str]]]) -> Tuple[int, Optional[List[float]]]:
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


def compute_rmsd_safe(probe: Optional[Chem.Mol], ref: Optional[Chem.Mol]) -> float:
    if probe is None or ref is None:
        return float("inf")
    try:
        rmsd = get_best_rmsd(probe, ref, use_alignmol=True)
        if rmsd is None or np.isnan(rmsd):
            return float("inf")
        return float(rmsd)
    except Exception as exc:
        logger.debug(f"RMSD computation failed: {exc}")
        return float("inf")


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

    valid_indices = [idx for idx, flag in enumerate(validity) if flag and rollout_mols[idx] is not None]
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
    K = len(rollout_mols)
    M = len(reference_mols)

    if K == 0 or M == 0:
        return np.full((K, M), float("inf"), dtype=np.float32)

    if rmsd_workers > 1 and ref_cache_key:
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
        "mol_pred": {"cleanup": False, "sanitize": False, "add_hs": False, "assign_stereo": False, "load_all": True},
        "mol_true": {"cleanup": False, "sanitize": False, "add_hs": False, "assign_stereo": False, "load_all": True},
        "mol_cond": {"cleanup": False, "sanitize": False, "add_hs": False, "assign_stereo": False, "proximityBonding": False},
    },
}

_POSEBUSTERS_STAGE_COUNTS: Dict[str, int] = {
    "basic": 4,
    "geometry": 5,
    "full": len(_POSEBUSTERS_BASE_CONFIG["modules"]),
}

_POSEBUSTERS_RUNNER_CACHE: Dict[Tuple[str, int, int, int], PoseBusters] = {}
_POSEBUSTERS_ENERGY_WARNING_EMITTED = False


def normalize_posebusters_config(raw_cfg: Any) -> PoseBustersRuntimeConfig:
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
                "[reward_v3] Clamping PoseBusters energy_num_threads to 1 when using multiprocessing to avoid CPU oversubscription."
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
    if mode not in _POSEBUSTERS_STAGE_COUNTS:
        raise ValueError(f"Unsupported PoseBusters mode '{mode}'.")

    module_count = _POSEBUSTERS_STAGE_COUNTS[mode]
    modules = deepcopy(_POSEBUSTERS_BASE_CONFIG["modules"][:module_count])
    if mode == "full" and modules:
        energy_module = modules[-1]
        if energy_module.get("function") == "energy_ratio":
            params = energy_module.setdefault("parameters", {})
            params["num_threads"] = energy_threads
    return {"modules": modules, "loading": deepcopy(_POSEBUSTERS_BASE_CONFIG["loading"])}


def _get_posebusters_runner(settings: PoseBustersRuntimeConfig) -> PoseBusters:
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
    if table is None or table.empty:
        return [False] * expected_len
    bool_df = table.select_dtypes(include=["bool"])
    if bool_df.empty:
        return [False] * expected_len
    passes_raw = bool_df.all(axis=1)
    passes: List[bool] = []
    for value in passes_raw.tolist():
        passes.append(bool(value) if value is not pd.NA else False)
    if len(passes) < expected_len:
        passes.extend([False] * (expected_len - len(passes)))
    elif len(passes) > expected_len:
        passes = passes[:expected_len]
    return passes


def apply_posebusters_gate(
    rollout_mols: List[Optional[Chem.Mol]],
    base_valid_mask: np.ndarray,
    settings: PoseBustersRuntimeConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    summary = {"checked": 0.0, "passed": 0.0, "failed": 0.0, "errors": 0.0, "time_ms": 0.0}
    if settings.mode == "off":
        return base_valid_mask.astype(bool, copy=False), summary

    valid_indices = [idx for idx, flag in enumerate(base_valid_mask) if flag and rollout_mols[idx] is not None]
    if not valid_indices:
        return base_valid_mask.astype(bool, copy=False), summary

    summary["checked"] = float(len(valid_indices))
    try:
        runner = _get_posebusters_runner(settings)
    except Exception as exc:  # pragma: no cover - depends on runtime env
        logger.warning(
            "[reward_v3] PoseBusters unavailable (%s); marking %d samples invalid.",
            exc,
            int(summary["checked"]),
        )
        updated = base_valid_mask.copy()
        updated[valid_indices] = False
        summary["errors"] = summary["checked"]
        return updated.astype(bool, copy=False), summary

    mol_batch = [rollout_mols[i] for i in valid_indices]
    start = time.perf_counter()
    try:
        report = runner.bust(mol_pred=mol_batch, full_report=False)
        if not isinstance(report, pd.DataFrame):
            raise TypeError("PoseBusters returned an unexpected result.")
    except Exception as exc:  # pragma: no cover - PoseBusters runtime varies
        logger.warning("[reward_v3] PoseBusters execution failed: %s", exc)
        updated = base_valid_mask.copy()
        updated[valid_indices] = False
        summary["errors"] = summary["checked"]
        summary["time_ms"] = (time.perf_counter() - start) * 1000.0
        return updated.astype(bool, copy=False), summary

    summary["time_ms"] = (time.perf_counter() - start) * 1000.0
    passes = _extract_posebusters_pass_vector(report, len(valid_indices))
    summary["passed"] = float(sum(passes))
    summary["failed"] = summary["checked"] - summary["passed"]

    updated_mask = base_valid_mask.copy().astype(bool, copy=False)
    for idx, passed in zip(valid_indices, passes):
        updated_mask[idx] = bool(passed)

    return updated_mask, summary


def compute_quality_reward(D: np.ndarray, validity: np.ndarray, sigma: float) -> np.ndarray:
    K, M = D.shape
    if M == 0:
        return np.zeros(K, dtype=np.float32)

    d_i = np.min(D, axis=1)
    r_qual = np.zeros(K, dtype=np.float32)

    sigma = max(float(sigma), 1e-8)
    valid_mask = (validity == 1) & np.isfinite(d_i)
    if np.any(valid_mask):
        r_qual[valid_mask] = np.exp(-d_i[valid_mask] / sigma).astype(np.float32)
    return r_qual

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return 1.0 / (1.0 + np.exp(-x))

def compute_smooth_coverage_reward(
    D: np.ndarray,
    validity: np.ndarray,
    rho: float,
    return_details: bool = False,
    delta: float = 0.75,
    precision_weight: float = 0.10,
    unique_quality_weight: float = 0.20,
) -> Tuple[np.ndarray, Tuple[float, List[float], np.ndarray]]:
    """
    Recall-primary smooth coverage reward aligned with GEOM-Drugs COV-R.

    Uses marginal contributions to dense soft coverage so Σ_i reward_i equals the
    mean soft coverage (up to fp error). Defaults act as fallbacks only—training
    runs should pass explicit weights through configuration for reproducibility.
    """
    K, M = D.shape
    rewards = np.zeros(K, dtype=np.float32)
    default_details = (float("nan"), [float("nan")] * 3, EMPTY_FLOAT32)
    if M == 0:
        return (rewards, default_details) if return_details else rewards

    rho = max(float(rho), 1e-8)
    delta = max(float(delta), 1e-8)
    precision_weight = float(precision_weight)
    unique_quality_weight = float(unique_quality_weight)

    valid_rows = (validity == 1)
    if not np.any(valid_rows):
        soft_cov = np.zeros(M, dtype=np.float32)
        details = (0.0, [0.0, 0.0, 0.0], soft_cov)
        return (rewards, details) if return_details else rewards

    Dv = D.astype(np.float32, copy=True)
    Dv = np.where(np.isfinite(Dv), Dv, np.float32(np.inf))
    Dv[~valid_rows, :] = np.float32(np.inf)

    kernel = _sigmoid((delta - Dv) / rho)
    kernel = np.where(np.isfinite(Dv), kernel, 0.0).astype(np.float32, copy=False)
    kernel[~valid_rows, :] = 0.0

    one_minus = np.clip(1.0 - kernel, 0.0, 1.0)
    zero_mask = one_minus <= 1e-12
    one_minus_safe = np.where(zero_mask, 1.0, one_minus)
    log_one_minus = np.log(one_minus_safe.astype(np.float64, copy=False))
    log_prod_safe = np.sum(log_one_minus, axis=0, keepdims=True)
    zero_counts = zero_mask.sum(axis=0, keepdims=True)

    prod_all = np.exp(log_prod_safe)
    prod_all = np.where(zero_counts > 0, 0.0, prod_all).astype(np.float32, copy=False)

    prod_excl = np.exp(log_prod_safe - log_one_minus)
    prod_excl = prod_excl.astype(np.float32, copy=False)
    has_zero = zero_counts > 0
    prod_excl = np.where(has_zero, 0.0, prod_excl)
    only_zero = (zero_counts == 1) & zero_mask
    if np.any(only_zero):
        prod_excl = np.where(only_zero, np.exp(log_prod_safe).astype(np.float32, copy=False), prod_excl)

    delta_matrix = kernel * prod_excl
    if unique_quality_weight > 0.0:
        depth = 1.0 - (Dv / delta)
        depth = np.clip(depth, 0.0, 1.0)
        depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32, copy=False)
        delta_matrix *= (1.0 + unique_quality_weight * depth)

    rewards = delta_matrix.sum(axis=1).astype(np.float32, copy=False) / float(M)

    if precision_weight > 0.0:
        rollout_best = np.min(Dv, axis=1)
        precision_scores = _sigmoid((delta - rollout_best) / rho).astype(np.float32, copy=False)
        precision_scores[~np.isfinite(rollout_best)] = 0.0
        precision_scores[~valid_rows] = 0.0
        rewards += precision_weight * precision_scores

    rewards[~valid_rows] = 0.0

    soft_cov = (1.0 - prod_all.reshape(-1)).astype(np.float32, copy=False)
    soft_mean = float(np.mean(soft_cov)) if soft_cov.size > 0 else float("nan")
    soft_pcts = [
        float(np.mean(soft_cov > thresh)) if soft_cov.size > 0 else float("nan")
        for thresh in (0.1, 0.3, 0.5)
    ]
    details = (soft_mean, soft_pcts, soft_cov)

    return (rewards, details) if return_details else rewards

# def compute_smooth_coverage_reward(
#     D: np.ndarray,
#     validity: np.ndarray,
#     rho: float,
#     return_details: bool = False,
# ) -> Tuple[np.ndarray, Tuple[float, List[float], np.ndarray]]:
#     K, M = D.shape
#     if M == 0:
#         empty = np.zeros(K, dtype=np.float32)
#         default = (float("nan"), [float("nan")] * 3, EMPTY_FLOAT32)
#         return empty, default if return_details else (empty, default)[0]

#     rho = max(float(rho), 1e-8)
#     valid_rows = validity == 1
#     K_matrix = np.zeros((K, M), dtype=np.float32)
#     if np.any(valid_rows):
#         D_valid = D[valid_rows].astype(np.float32, copy=False)
#         D_valid = np.where(np.isfinite(D_valid), D_valid, np.inf)
#         exponent = -((D_valid / rho) ** 2)
#         K_valid = np.exp(exponent)
#         K_valid = np.where(np.isfinite(K_valid), K_valid, 0.0)
#         K_matrix[valid_rows, :] = K_valid

#     one_minus = np.clip(1.0 - K_matrix, 0.0, 1.0)
#     zero_mask = one_minus <= 1e-12
#     one_minus_safe = np.where(zero_mask, 1.0, one_minus)
#     log_one_minus = np.log(one_minus_safe.astype(np.float64))
#     log_prod_safe = np.sum(log_one_minus, axis=0, keepdims=True)
#     zero_counts = zero_mask.sum(axis=0, keepdims=True)

#     prod_excl = np.exp(log_prod_safe - log_one_minus)
#     has_zero = zero_counts > 0
#     prod_excl = np.where(has_zero, 0.0, prod_excl)
#     only_zero = (zero_counts == 1) & zero_mask
#     if np.any(only_zero):
#         prod_excl = np.where(only_zero, np.exp(log_prod_safe), prod_excl)

#     Delta = K_matrix * prod_excl
#     r_smcov = (Delta.sum(axis=1) / M).astype(np.float32)
#     r_smcov[~valid_rows] = 0.0

#     soft_cov = 1.0 - np.prod(one_minus, axis=0)
#     soft_cov = np.clip(soft_cov, 0.0, 1.0)
#     soft_mean = float(np.mean(soft_cov)) if soft_cov.size > 0 else float("nan")
#     soft_pcts = [
#         float(np.mean(soft_cov > thresh)) if soft_cov.size > 0 else float("nan")
#         for thresh in (0.1, 0.3, 0.5)
#     ]
#     details = (soft_mean, soft_pcts, soft_cov.astype(np.float32))
#     return r_smcov, details if return_details else (r_smcov, details)[0]


def compute_matching_reward(
    D: np.ndarray,
    validity: np.ndarray,
    delta: float,
    eligible_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, int, List[Tuple[int, int]]]:
    K, M = D.shape
    r_match = np.zeros(K, dtype=np.float32)
    matched_pairs: List[Tuple[int, int]] = []

    if M == 0 or K == 0:
        return r_match, 0, 0, matched_pairs

    valid_mask = validity.astype(bool)
    eligible = eligible_matrix if eligible_matrix is not None else valid_mask[:, None] & (D < delta)
    num_eligible_edges = int(np.count_nonzero(eligible))
    if num_eligible_edges == 0:
        return r_match, 0, 0, matched_pairs

    if linear_sum_assignment is None:
        global _LINEAR_SUM_WARNING_EMITTED
        if not _LINEAR_SUM_WARNING_EMITTED:
            logger.warning("SciPy not available; matching reward disabled.")
            _LINEAR_SUM_WARNING_EMITTED = True
        return r_match, 0, num_eligible_edges, matched_pairs

    try:
        BIG = 1e6
        C = np.where(eligible, D, BIG).astype(np.float64)
        if K != M:
            max_dim = max(K, M)
            C_square = np.full((max_dim, max_dim), BIG, dtype=np.float64)
            C_square[:K, :M] = C
            C = C_square

        row_ind, col_ind = linear_sum_assignment(C)
        for i, j in zip(row_ind, col_ind):
            if i < K and j < M and eligible[i, j]:
                matched_pairs.append((i, j))
        for i, j in matched_pairs:
            r_match[i] = max(0.0, 1.0 - D[i, j] / delta)
        num_matched = len(matched_pairs)
        if num_matched == 0 and num_eligible_edges > 0:
            logger.warning(
                f"[reward_v3] Matching found 0 pairs despite {num_eligible_edges} eligible edges (K={K}, M={M})."
            )
    except Exception as exc:
        logger.warning(f"Matching solver failed: {exc}. Falling back to r_match=0")
        r_match = np.zeros(K, dtype=np.float32)
        num_matched = 0
        matched_pairs = []

    return r_match, num_matched, num_eligible_edges, matched_pairs


def combine_rewards(
    r_qual: np.ndarray,
    r_smcov: np.ndarray,
    r_match: np.ndarray,
    validity: np.ndarray,
    lambda_qual: float,
    lambda_smcov: float,
    lambda_match: float,
    r_floor: float,
) -> np.ndarray:
    combined = (
        lambda_qual * r_qual + lambda_smcov * r_smcov + lambda_match * r_match
    ).astype(np.float32, copy=False)
    valid_mask = validity.astype(bool)
    if combined.shape != valid_mask.shape:
        raise ValueError("Validity mask must match reward shapes")
    return np.where(valid_mask, combined, r_floor).astype(np.float32, copy=False)


def compute_pairwise_rollout_distances(
    rollout_mols: List[Optional[Chem.Mol]],
    validity_mask: np.ndarray,
    max_samples: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    valid_indices = [idx for idx, flag in enumerate(validity_mask) if flag and rollout_mols[idx] is not None]
    if len(valid_indices) < 2:
        return EMPTY_FLOAT32
    if max_samples > 0 and len(valid_indices) > max_samples:
        if rng is None:
            rng = np.random.default_rng()
        idx_arr = np.asarray(valid_indices, dtype=np.int64)
        sample = rng.choice(idx_arr, size=max_samples, replace=False)
        valid_indices = sample.tolist()

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
    return np.array(distances, dtype=np.float32) if distances else EMPTY_FLOAT32


def _run_recall_reward_sanity_checks() -> None:
    """Quick manual tests for the recall-primary coverage reward."""
    delta = 0.75
    rho = 0.5

    # Case 1: both rollouts valid, dense credit sums to soft coverage.
    D = np.array([[0.2, 0.9, 0.3], [0.8, 0.4, 0.9]], dtype=np.float32)
    validity = np.array([1, 1], dtype=np.int32)
    rewards, details = compute_smooth_coverage_reward(
        D, validity, rho=rho, return_details=True, delta=delta, precision_weight=0.0, unique_quality_weight=0.0
    )
    soft_cov = details[2]
    assert np.all(rewards >= 0.0)
    assert np.isclose(rewards.sum(), float(np.mean(soft_cov)), atol=1e-5)

    # Case 2: rollout0 invalid, rollout1 only covers GT1.
    validity = np.array([0, 1], dtype=np.int32)
    rewards, _ = compute_smooth_coverage_reward(
        D, validity, rho=rho, return_details=True, delta=delta, precision_weight=0.0, unique_quality_weight=0.0
    )
    assert rewards[0] == 0.0
    assert rewards[1] > 0.0

    # Case 3: non-finite distances should yield zero credit.
    D_inf = np.array([[np.inf, np.inf], [1.2, np.inf]], dtype=np.float32)
    validity = np.array([1, 1], dtype=np.int32)
    rewards, _ = compute_smooth_coverage_reward(
        D_inf, validity, rho=rho, return_details=True, delta=delta, precision_weight=0.0, unique_quality_weight=0.0
    )
    assert np.all(rewards == 0.0)

    print("[reward_utils] recall reward sanity checks passed.")


def summarize_batch_metrics(
    metrics_list: Sequence["GroupMetrics"],
    delta: float,
) -> Dict[str, float]:
    def _safe_ratio(num: float, denom: float) -> float:
        return float(num / denom) if denom > 0 else 0.0

    def _safe_mean(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        arr = np.asarray(values, dtype=np.float32)
        mask = np.isfinite(arr)
        if not np.any(mask):
            return 0.0
        return float(np.mean(arr[mask]))

    def _safe_percentile(values: Sequence[float], pct: float) -> float:
        if not values:
            return 0.0
        arr = np.asarray(values, dtype=np.float32)
        mask = np.isfinite(arr)
        if not np.any(mask):
            return 0.0
        return float(np.percentile(arr[mask], pct))

    def _concat(arrays: Sequence[np.ndarray]) -> np.ndarray:
        filtered = [arr for arr in arrays if arr.size > 0]
        return np.concatenate(filtered) if filtered else EMPTY_FLOAT32

    if not metrics_list:
        return {}

    result: Dict[str, float] = {}

    total_rollouts = sum(m.K for m in metrics_list)
    total_graph = sum(m.graph_match_count for m in metrics_list)
    total_parsed = sum(m.rdkit_parse_count for m in metrics_list)
    total_base_valid = sum(m.base_valid_count for m in metrics_list)
    total_final_valid = sum(m.valid_rollouts for m in metrics_list)

    pose_checked_total = sum(m.pose_checked for m in metrics_list)
    pose_passed_total = sum(m.pose_passed for m in metrics_list)
    pose_errors_total = sum(m.pose_errors for m in metrics_list)

    refs_hit_values = [float(m.refs_hit) for m in metrics_list]
    cov_ratio_values = [m.cov_ratio for m in metrics_list]
    unique_refs_values = [float(m.unique_nearest_refs) for m in metrics_list]
    collision_rates = [m.nearest_collision_rate for m in metrics_list]
    covdiff_cover_ratio_values = [m.covdiff_cover_ratio for m in metrics_list]
    covdiff_unique_cover_ratio_values = [m.covdiff_unique_cover_ratio for m in metrics_list]
    covered_ratio_values = [m.covered_ratio for m in metrics_list]
    unique_covered_ratio_values = [m.unique_covered_ratio for m in metrics_list]
    valid_rollout_values = [float(m.valid_rollouts) for m in metrics_list]

    num_matched_values = [float(m.num_matched) for m in metrics_list]
    max_possible_values = [float(m.max_possible_matches) for m in metrics_list]
    efficiency_values = [m.match_efficiency for m in metrics_list]

    soft_cov_means = [m.soft_cov_mean for m in metrics_list]
    pct_cov_gt_0p1_values = [m.pct_cov_gt_0p1 for m in metrics_list]
    pct_cov_gt_0p5_values = [m.pct_cov_gt_0p5 for m in metrics_list]

    all_d_min = _concat([m.d_min_values for m in metrics_list])
    all_matched = _concat([m.matched_dists for m in metrics_list])
    all_pairwise = _concat([m.pairwise_dists for m in metrics_list])
    all_rewards = _concat([m.reward_total_values for m in metrics_list])
    all_comp_qual = _concat([m.comp_qual_values for m in metrics_list])
    all_comp_smcov = _concat([m.comp_smcov_values for m in metrics_list])
    all_comp_match = _concat([m.comp_match_values for m in metrics_list])
    qual_group_stds = [m.comp_qual_group_std for m in metrics_list]
    smcov_group_stds = [m.comp_smcov_group_std for m in metrics_list]
    match_group_stds = [m.comp_match_group_std for m in metrics_list]
    reward_group_std_values = [m.reward_group_std for m in metrics_list]
    bestcov_rank_corr_values = [m.bestcov_reward_corr for m in metrics_list]

    total_valid_rollouts = sum(m.valid_rollouts for m in metrics_list)
    total_comp_qual = sum(m.comp_qual_sum for m in metrics_list)
    total_comp_smcov = sum(m.comp_smcov_sum for m in metrics_list)
    total_comp_match = sum(m.comp_match_sum for m in metrics_list)

    eligible_edges_total = sum(m.eligible_edges for m in metrics_list)
    possible_edges_total = sum(m.K * m.M for m in metrics_list)

    result["gate/graph_match_rate"] = _safe_ratio(total_graph, total_rollouts)
    result["gate/rdkit_parse_rate"] = _safe_ratio(total_parsed, total_rollouts)
    result["gate/base_valid_rate"] = _safe_ratio(total_base_valid, total_rollouts)
    result["gate/final_valid_rate"] = _safe_ratio(total_final_valid, total_rollouts)

    result["pose/checked_rate"] = _safe_ratio(pose_checked_total, total_base_valid)
    result["pose/pass_rate"] = _safe_ratio(pose_passed_total, pose_checked_total)
    result["pose/error_rate"] = _safe_ratio(pose_errors_total, pose_checked_total)

    result["rmsd/d_min_mean"] = float(np.mean(all_d_min)) if all_d_min.size > 0 else 0.0
    result["rmsd/d_min_p50"] = float(np.percentile(all_d_min, 50)) if all_d_min.size > 0 else 0.0
    result["rmsd/d_min_p90"] = float(np.percentile(all_d_min, 90)) if all_d_min.size > 0 else 0.0
    result["rmsd/frac_under_delta"] = (
        float(np.mean(all_d_min < delta)) if all_d_min.size > 0 else 0.0
    )
    result["rmsd/frac_under_2delta"] = (
        float(np.mean(all_d_min < (2 * delta))) if all_d_min.size > 0 else 0.0
    )

    result["cov/refs_hit_mean"] = _safe_mean(refs_hit_values)
    result["cov/refs_hit_p50"] = _safe_percentile(refs_hit_values, 50.0)
    result["cov/cov_ratio_mean"] = _safe_mean(cov_ratio_values)
    result["cov/unique_nearest_refs_mean"] = _safe_mean(unique_refs_values)
    result["cov/nearest_collision_rate_mean"] = _safe_mean(collision_rates)
    result["covdiff/cover_ratio_mean"] = _safe_mean(covdiff_cover_ratio_values)
    result["covdiff/unique_cover_ratio_mean"] = _safe_mean(covdiff_unique_cover_ratio_values)
    result["covdiff/covered_ratio_mean"] = _safe_mean(covered_ratio_values)
    result["covdiff/unique_covered_ratio_mean"] = _safe_mean(unique_covered_ratio_values)
    result["cov/valid_rollouts_mean"] = _safe_mean(valid_rollout_values)

    result["match/num_matched_mean"] = _safe_mean(num_matched_values)
    result["match/max_possible_mean"] = _safe_mean(max_possible_values)
    result["match/efficiency_mean"] = _safe_mean(efficiency_values)
    result["match/matched_dist_p50"] = (
        float(np.percentile(all_matched, 50)) if all_matched.size > 0 else float("nan")
    )
    result["match/matched_dist_p90"] = (
        float(np.percentile(all_matched, 90)) if all_matched.size > 0 else float("nan")
    )
    result["match/eligible_edge_density"] = _safe_ratio(eligible_edges_total, possible_edges_total)

    result["bestcov/soft_cov_mean"] = _safe_mean(soft_cov_means)
    result["bestcov/pct_gt_cov_gt_0p1"] = _safe_mean(pct_cov_gt_0p1_values)
    result["bestcov/pct_gt_cov_gt_0p5"] = _safe_mean(pct_cov_gt_0p5_values)

    refs_arr = np.asarray(refs_hit_values, dtype=np.float64)
    soft_cov_arr = np.asarray(soft_cov_means, dtype=np.float64)
    valid_mask = np.isfinite(refs_arr) & np.isfinite(soft_cov_arr)
    if np.count_nonzero(valid_mask) >= 2:
        result["bestcov/corr_with_refs_hit"] = float(
            np.corrcoef(refs_arr[valid_mask], soft_cov_arr[valid_mask])[0, 1]
        )
    else:
        result["bestcov/corr_with_refs_hit"] = 0.0

    reward_total_mean = float(np.mean(all_rewards)) if all_rewards.size > 0 else 0.0
    reward_total_std = float(np.std(all_rewards)) if all_rewards.size > 0 else 0.0
    result["reward/total_mean"] = reward_total_mean
    result["reward/total_std"] = reward_total_std

    comp_qual_mean = _safe_ratio(total_comp_qual, total_valid_rollouts)
    comp_smcov_mean = _safe_ratio(total_comp_smcov, total_valid_rollouts)
    comp_match_mean = _safe_ratio(total_comp_match, total_valid_rollouts)
    result["reward/comp_qual_mean"] = comp_qual_mean
    result["reward/comp_qual_std"] = float(np.std(all_comp_qual)) if all_comp_qual.size > 0 else 0.0
    result["reward/comp_smcov_mean"] = comp_smcov_mean
    result["reward/comp_smcov_std"] = float(np.std(all_comp_smcov)) if all_comp_smcov.size > 0 else 0.0
    result["reward/comp_match_mean"] = comp_match_mean
    result["reward/comp_match_std"] = float(np.std(all_comp_match)) if all_comp_match.size > 0 else 0.0
    result["reward/qual_group_std_mean"] = _safe_mean(qual_group_stds)
    result["reward/smcov_group_std_mean"] = _safe_mean(smcov_group_stds)
    result["reward/match_group_std_mean"] = _safe_mean(match_group_stds)
    result["reward/group_std_mean"] = _safe_mean(reward_group_std_values)
    result["reward/bestcov_rank_corr_mean"] = _safe_mean(bestcov_rank_corr_values)
    if np.isfinite(comp_smcov_mean) and np.isfinite(reward_total_mean) and reward_total_mean != 0.0:
        result["reward/comp_smcov_frac"] = comp_smcov_mean / (reward_total_mean + 1e-8)
    else:
        result["reward/comp_smcov_frac"] = 0.0

    result["div/pairwise_rmsd_p50"] = (
        float(np.percentile(all_pairwise, 50)) if all_pairwise.size > 0 else float("nan")
    )

    return result


if __name__ == "__main__":
    _run_recall_reward_sanity_checks()
