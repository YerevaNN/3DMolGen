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


def compute_smooth_coverage_reward(
    D: np.ndarray,
    validity: np.ndarray,
    rho: float,
    return_details: bool = False,
) -> Tuple[np.ndarray, Tuple[float, List[float], np.ndarray]]:
    K, M = D.shape
    if M == 0:
        empty = np.zeros(K, dtype=np.float32)
        default = (float("nan"), [float("nan")] * 3, EMPTY_FLOAT32)
        return empty, default if return_details else (empty, default)[0]

    rho = max(float(rho), 1e-8)
    valid_rows = validity == 1
    K_matrix = np.zeros((K, M), dtype=np.float32)
    if np.any(valid_rows):
        D_valid = D[valid_rows].astype(np.float32, copy=False)
        D_valid = np.where(np.isfinite(D_valid), D_valid, np.inf)
        exponent = -((D_valid / rho) ** 2)
        K_valid = np.exp(exponent)
        K_valid = np.where(np.isfinite(K_valid), K_valid, 0.0)
        K_matrix[valid_rows, :] = K_valid

    one_minus = np.clip(1.0 - K_matrix, 0.0, 1.0)
    zero_mask = one_minus <= 1e-12
    one_minus_safe = np.where(zero_mask, 1.0, one_minus)
    log_one_minus = np.log(one_minus_safe.astype(np.float64))
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

    soft_cov = 1.0 - np.prod(one_minus, axis=0)
    soft_cov = np.clip(soft_cov, 0.0, 1.0)
    soft_mean = float(np.mean(soft_cov)) if soft_cov.size > 0 else float("nan")
    soft_pcts = [
        float(np.mean(soft_cov > thresh)) if soft_cov.size > 0 else float("nan")
        for thresh in (0.1, 0.3, 0.5)
    ]
    details = (soft_mean, soft_pcts, soft_cov.astype(np.float32))
    return r_smcov, details if return_details else (r_smcov, details)[0]


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
) -> np.ndarray:
    valid_indices = [idx for idx, flag in enumerate(validity_mask) if flag and rollout_mols[idx] is not None]
    if len(valid_indices) < 2:
        return EMPTY_FLOAT32
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
    return np.array(distances, dtype=np.float32) if distances else EMPTY_FLOAT32


def summarize_batch_metrics(
    metrics_list: Sequence["GroupMetrics"],
    lambda_qual: float,
    lambda_smcov: float,
    lambda_match: float,
) -> Dict[str, float]:
    def _nanmean(values: List[float]) -> float:
        if not values:
            return 0.0
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return 0.0
        if np.isnan(arr).all():
            return 0.0
        return float(np.nanmean(arr))

    def _concat(samples: List[np.ndarray]) -> np.ndarray:
        arrays = [arr for arr in samples if arr.size > 0]
        return np.concatenate(arrays) if arrays else EMPTY_FLOAT32

    essential_metrics: Dict[str, float] = {}
    if not metrics_list:
        essential_metrics.update(
            {
                "validity_rate": 0.0,
                "graph_match_rate": 0.0,
                "finite_rmsd_rate": 0.0,
                "match/match_efficiency": 0.0,
                "posebusters/pass_rate": 0.0,
                "posebusters/fail_rate": 0.0,
                "posebusters/error_rate": 0.0,
                "reward/component_quality": 0.0,
                "reward/component_smcov": 0.0,
                "reward/component_match": 0.0,
                "sampling/approx_percentiles": 0.0,
                "posebusters/checked_total": 0.0,
                "posebusters/passed_total": 0.0,
                "posebusters/failed_total": 0.0,
                "posebusters/errors_total": 0.0,
                "posebusters/time_ms_per_group": 0.0,
                "match/matched_total": 0.0,
                "match/max_possible_total": 0.0,
            }
        )
        return essential_metrics

    validity_rates = [m.validity_rate for m in metrics_list]
    graph_match_rates = [m.graph_match_rate for m in metrics_list]
    finite_rmsd_rates = [m.finite_rmsd_rate for m in metrics_list]
    fraction_under_delta = [m.fraction_under_delta for m in metrics_list]
    refs_hit = [float(m.refs_hit) for m in metrics_list]
    num_matched_list = [float(m.num_matched) for m in metrics_list]
    d_min_means = [m.d_min_mean for m in metrics_list]
    d_min_p50s = [m.d_min_p50 for m in metrics_list]
    d_min_p90s = [m.d_min_p90 for m in metrics_list]
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
    soft_cov_samples = _concat([m.soft_cov_sample for m in metrics_list])
    pairwise_samples = _concat([m.pairwise_sample for m in metrics_list])

    total_matched = int(sum(m.num_matched for m in metrics_list))
    total_max_possible = int(sum(m.max_possible_matches for m in metrics_list))
    match_efficiency_total = float(total_matched) / total_max_possible if total_max_possible > 0 else 0.0
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
        total_posebusters_passed / total_posebusters_checked if total_posebusters_checked > 0 else 0.0
    )

    essential_metrics["validity_rate"] = _nanmean(validity_rates)
    essential_metrics["graph_match_rate"] = _nanmean(graph_match_rates)
    essential_metrics["finite_rmsd_rate"] = _nanmean(finite_rmsd_rates)

    essential_metrics["geom/d_min_mean"] = _nanmean(d_min_means)
    essential_metrics["geom/d_min_p50"] = _nanmean(d_min_p50s)
    essential_metrics["geom/d_min_p90"] = _nanmean(d_min_p90s)
    if d_samples.size > 0:
        essential_metrics["geom/d_min_sampled_p50"] = float(np.percentile(d_samples, 50))
        essential_metrics["geom/d_min_sampled_p90"] = float(np.percentile(d_samples, 90))
        essential_metrics["geom/d_min_sampled_mean"] = float(np.mean(d_samples))
        essential_metrics["geom/d_min_sampled_size"] = float(d_samples.size)

    essential_metrics["match/match_efficiency"] = match_efficiency_total
    essential_metrics["match/num_matched"] = _nanmean(num_matched_list)
    if matched_samples.size > 0:
        essential_metrics["match/dist_p50"] = float(np.percentile(matched_samples, 50))
        essential_metrics["match/dist_p90"] = float(np.percentile(matched_samples, 90))
        essential_metrics["match/dist_mean"] = float(np.mean(matched_samples))
        essential_metrics["match/dist_count"] = float(matched_samples.size)
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
        essential_metrics["diversity/pairwise_p50"] = float(np.percentile(pairwise_samples, 50))
        essential_metrics["diversity/pairwise_p90"] = float(np.percentile(pairwise_samples, 90))
        essential_metrics["diversity/pairwise_count"] = float(pairwise_samples.size)

    essential_metrics["fraction_under_delta"] = _nanmean(fraction_under_delta)
    if eligible_samples.size > 0:
        essential_metrics["match/eligible_dist_p50"] = float(np.percentile(eligible_samples, 50))
        essential_metrics["match/eligible_dist_p90"] = float(np.percentile(eligible_samples, 90))
        essential_metrics["match/eligible_dist_mean"] = float(np.mean(eligible_samples))
        essential_metrics["match/eligible_count"] = float(eligible_samples.size)
    if soft_cov_samples.size > 0:
        essential_metrics["coverage/soft_sample_mean"] = float(np.mean(soft_cov_samples))
        essential_metrics["coverage/soft_sample_p50"] = float(np.percentile(soft_cov_samples, 50))
        essential_metrics["coverage/soft_sample_p90"] = float(np.percentile(soft_cov_samples, 90))
        essential_metrics["coverage/soft_sample_count"] = float(soft_cov_samples.size)

    essential_metrics["reward/matched_total"] = float(total_matched)
    essential_metrics["match/matched_total"] = float(total_matched)
    essential_metrics["match/max_possible_total"] = float(total_max_possible)
    essential_metrics["sampling/approx_percentiles"] = 1.0 if any(sampling_flags) else 0.0

    essential_metrics["posebusters/pass_rate"] = posebusters_pass_rate
    essential_metrics["posebusters/fail_rate"] = (
        total_posebusters_failed / total_posebusters_checked if total_posebusters_checked > 0 else 0.0
    )
    essential_metrics["posebusters/error_rate"] = (
        total_posebusters_errors / total_posebusters_checked if total_posebusters_checked > 0 else 0.0
    )
    essential_metrics["posebusters/checked_total"] = float(total_posebusters_checked)
    essential_metrics["posebusters/passed_total"] = float(total_posebusters_passed)
    essential_metrics["posebusters/failed_total"] = float(total_posebusters_failed)
    essential_metrics["posebusters/errors_total"] = float(total_posebusters_errors)
    if posebusters_times:
        essential_metrics["posebusters/time_ms_per_group"] = float(_nanmean(posebusters_times))

    wandb_mod = None
    try:  # pragma: no cover - wandb may be unavailable
        import wandb as wandb_mod  # type: ignore
    except Exception:  # pragma: no cover
        wandb_mod = None

    if wandb_mod is not None:
        if d_samples.size > 0:
            essential_metrics["geom/d_min_hist"] = wandb_mod.Histogram(d_samples)
        if matched_samples.size > 0:
            essential_metrics["match/dist_hist"] = wandb_mod.Histogram(matched_samples)
        if eligible_samples.size > 0:
            essential_metrics["match/eligible_hist"] = wandb_mod.Histogram(eligible_samples)
        if soft_cov_samples.size > 0:
            essential_metrics["coverage/soft_hist"] = wandb_mod.Histogram(soft_cov_samples)
        if pairwise_samples.size > 0:
            essential_metrics["diversity/pairwise_hist"] = wandb_mod.Histogram(pairwise_samples)

    return essential_metrics
