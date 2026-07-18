#!/usr/bin/env python3
"""Fast CASF geometric analysis for RDKit vs torsion generation outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem

from molgen3D.evaluation.conformer_benchmark import (
    CASF_GEOMETRIC_CLUSTER_THRESHOLDS,
    best_aligned_rmsd,
    energy_stats,
    forcefield_energy,
    greedy_cluster_metrics,
    mean_torsion_std_deg,
    pairwise_rmsd_stats,
    safe_mean,
    safe_median,
    safe_sum,
    threshold_tag,
)
from molgen3D.pharmacophore.chembl3d_conformer_loader import (
    load_chembl3d_conformers,
    load_topology_mol,
)
from molgen3D.pharmacophore.conformer_sets_layout import (
    DEFAULT_CASF16_DATA,
    DEFAULT_CASF_LIGAND_DIR,
    DEFAULT_CHEMBL_DATASET_ROOT,
    DEFAULT_CHEMBL_MAP_CSV,
    DEFAULT_CORE_INTERSECTION_ROOT,
    METHODS,
    GeometricAnalysisPaths,
    resolve_geometric_paths,
)
from molgen3D.pharmacophore.generate_casf_smiles_conformer_sets import (
    PoseBustersResult,
    get_dg_bounds,
    has_clash,
    posebusters_geometry_passes,
    summarize_posebusters_checks,
)

CLASH_CUTOFF = 0.7
CASF_HIT_THRESHOLDS = (0.25, 0.5, 0.75, 2.0)
REFERENCE_SOURCES = ("casf_crystal", "casf_opt", "chembl3d_sdf", "chembl3d_gt", "chembl3d_gt_pb")
FILTER_REFERENCE_SOURCES = REFERENCE_SOURCES
CHEMBL3D_GT_SOURCES = ("chembl3d_gt", "chembl3d_gt_pb")
CLUSTER_COMPARE_SOURCES = tuple(METHODS) + CHEMBL3D_GT_SOURCES
CASF_HIT_SOURCES = tuple(METHODS) + CHEMBL3D_GT_SOURCES
METHOD_ORDER = {method: index for index, method in enumerate(METHODS)}
REFERENCE_ORDER = {source: len(METHODS) + index for index, source in enumerate(REFERENCE_SOURCES)}
DEFAULT_CASF_OPT_LIGAND_DIR = DEFAULT_CASF16_DATA / "ligands_opt"


@dataclass(frozen=True)
class LigandAnalysisTask:
    mol_id: str
    generation_dir: str
    casf_ligand_dir: str
    casf_opt_ligand_dir: str
    source_input: str
    parts_dir: str
    topology_root: str
    zarr_root: str
    input_smiles: str
    rotatable_bonds: int
    posebusters_workers: int
    posebusters_energy_threads: int
    chembl_row: dict[str, object] | None


def _available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, NotImplementedError, OSError):
        return os.cpu_count() or 1


def _limit_worker_threads() -> None:
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RDKIT_NUM_THREADS",
    ):
        os.environ[var] = "1"


def _write_pickle_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def parse_method_metadata(method: str) -> tuple[str, str, str]:
    if method.startswith("rdkit_random_"):
        family = "rdkit_random"
        rest = method[len("rdkit_random_") :]
    elif method.startswith("torsion_"):
        family = "torsion"
        rest = method[len("torsion_") :]
    else:
        return method, "reference", "reference"
    for tier in ("chembl_count", "dynamic", "fixed"):
        suffix = f"_{tier}"
        if rest.endswith(suffix):
            return family, rest[: -len(suffix)], tier
    return family, rest, "unknown"


def source_order_index(source: object) -> int:
    source_str = str(source)
    if source_str in METHOD_ORDER:
        return METHOD_ORDER[source_str]
    if source_str in REFERENCE_ORDER:
        return REFERENCE_ORDER[source_str]
    return len(METHODS) + len(REFERENCE_SOURCES)


def sort_table_by_source(df: pd.DataFrame, extra_cols: Sequence[str] = ()) -> pd.DataFrame:
    if df.empty or "source" not in df.columns:
        return df
    view = df.copy()
    sort_cols = [col for col in extra_cols if col in view.columns]
    view["__source_order"] = view["source"].map(source_order_index)
    sort_cols.extend(["__source_order", "source"])
    view = view.sort_values(sort_cols, kind="stable").drop(columns=["__source_order"])
    return view.reset_index(drop=True)


def finalize_table_types(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for column in out.columns:
        if not _is_count_column(str(column)):
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        finite = values.dropna()
        if finite.empty or np.all(np.isclose(finite.to_numpy(dtype=float), np.round(finite.to_numpy(dtype=float)))):
            out[column] = values.round().astype("Int64")
    return out


def sort_and_finalize_table(df: pd.DataFrame, extra_cols: Sequence[str] = ()) -> pd.DataFrame:
    return finalize_table_types(sort_table_by_source(df, extra_cols=extra_cols))


def load_sdf(path: Path, required: bool = False) -> list[Chem.Mol]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required SDF output is missing: {path}")
        return []
    if path.stat().st_size == 0:
        return []
    mols: list[Chem.Mol] = []
    failed_indices: list[int] = []
    for index, mol in enumerate(Chem.SDMolSupplier(str(path), removeHs=False)):
        if mol is None:
            failed_indices.append(index)
        else:
            mols.append(mol)
    if failed_indices:
        preview = ", ".join(str(index) for index in failed_indices[:5])
        suffix = f" (+{len(failed_indices) - 5} more)" if len(failed_indices) > 5 else ""
        raise ValueError(f"Failed to parse conformer(s) {preview}{suffix} in {path}")
    return mols


def discover_mol_ids(generation_dir: Path, limit: int | None = None) -> list[str]:
    mol_ids: set[str] = set()
    for method in METHODS:
        method_dir = generation_dir / method
        if method_dir.exists():
            mol_ids.update(path.stem for path in method_dir.glob("*.sdf"))
    ordered = sorted(mol_ids)
    return ordered[:limit] if limit is not None else ordered


def _generation_sdf_is_valid(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def select_analysis_mol_ids(
    chembl_map: pd.DataFrame,
    generation_dir: Path,
    manifest_df: pd.DataFrame,
) -> tuple[list[str], dict[str, list[str]]]:
    excluded: dict[str, list[str]] = {
        "not_in_manifest": [],
        "zero_rotatable_bonds": [],
        "missing_generation_sdf": [],
    }
    if not chembl_map.empty and "ligand_id" in chembl_map.columns:
        candidate_ids = [str(mol_id) for mol_id in chembl_map["ligand_id"].dropna().tolist()]
    else:
        candidate_ids = discover_mol_ids(generation_dir)

    manifest_ids = {str(mol_id) for mol_id in manifest_df["mol_id"].dropna().tolist()}
    rot_lookup = (
        manifest_df.drop_duplicates("mol_id")[["mol_id", "rotatable_bonds"]]
        .set_index("mol_id")["rotatable_bonds"]
        .to_dict()
    )

    eligible: list[str] = []
    for mol_id in candidate_ids:
        if mol_id not in manifest_ids:
            excluded["not_in_manifest"].append(mol_id)
            continue
        if int(rot_lookup.get(mol_id, 0)) <= 0:
            excluded["zero_rotatable_bonds"].append(mol_id)
            continue
        missing_methods = [
            method
            for method in METHODS
            if not _generation_sdf_is_valid(generation_dir / method / f"{mol_id}.sdf")
        ]
        if missing_methods:
            excluded["missing_generation_sdf"].append(mol_id)
            continue
        eligible.append(mol_id)
    return eligible, excluded


def _log_excluded_ligands(excluded: dict[str, list[str]]) -> None:
    for reason, mol_ids in excluded.items():
        if not mol_ids:
            continue
        preview = ", ".join(sorted(mol_ids)[:5])
        suffix = f" (+{len(mol_ids) - 5} more)" if len(mol_ids) > 5 else ""
        print(f"Excluded {len(mol_ids)} ligand(s) [{reason}]: {preview}{suffix}", flush=True)


def load_casf_ligand(
    mol_id: str,
    ligand_dir: Path,
    source_input: str = "",
    *,
    restrict_to_ligand_dir: bool = False,
) -> Chem.Mol:
    candidates: list[Path] = []
    if restrict_to_ligand_dir:
        if source_input:
            candidates.append(ligand_dir / Path(source_input).name)
        candidates.append(ligand_dir / f"{mol_id}.mol2")
    else:
        if source_input:
            source_path = Path(source_input)
            candidates.append(source_path)
            candidates.append(ligand_dir / source_path.name)
            candidates.append(DEFAULT_CASF16_DATA / "core_chembl3d_exact_intersection_ligands" / source_path.name)
            candidates.append(DEFAULT_CASF_LIGAND_DIR / source_path.name)
        candidates.extend(
            [
                ligand_dir / f"{mol_id}.mol2",
                DEFAULT_CASF16_DATA / "core_chembl3d_exact_intersection_ligands" / f"{mol_id}.mol2",
                DEFAULT_CASF_LIGAND_DIR / f"{mol_id}.mol2",
            ]
        )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"CASF ligand {mol_id} not found. Searched: {searched}")
    return load_ligand_mol_strict(path)


def load_ligand_mol_strict(path: Path) -> Chem.Mol:
    mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=True)
    if mol is None:
        mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError(f"Failed to load ligand mol2: {path}")
    return mol


def try_load_casf_ligand(
    mol_id: str,
    ligand_dir: Path,
    source_input: str = "",
    *,
    restrict_to_ligand_dir: bool = False,
) -> Chem.Mol | None:
    try:
        return load_casf_ligand(
            mol_id,
            ligand_dir,
            source_input,
            restrict_to_ligand_dir=restrict_to_ligand_dir,
        )
    except FileNotFoundError:
        return None


def load_manifest(paths: GeometricAnalysisPaths) -> pd.DataFrame:
    manifest_path = paths.manifest_path
    if manifest_path.exists():
        return pd.read_csv(manifest_path, sep="\t")
    part_paths = sorted(paths.manifest_parts_dir.glob("*.tsv"))
    if not part_paths:
        raise FileNotFoundError(f"No manifest at {manifest_path} and no parts in {paths.manifest_parts_dir}")
    frames = [pd.read_csv(part, sep="\t") for part in part_paths]
    manifest_df = pd.concat(frames, ignore_index=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_path, sep="\t", index=False)
    return manifest_df


def load_chembl_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "ligand_id" not in df.columns:
        raise ValueError(f"Expected ligand_id column in {path}")
    duplicated = df["ligand_id"].duplicated(keep=False)
    if duplicated.any():
        ligand_ids = sorted(str(value) for value in df.loc[duplicated, "ligand_id"].dropna().unique())
        preview = ", ".join(ligand_ids[:5])
        suffix = f" (+{len(ligand_ids) - 5} more)" if len(ligand_ids) > 5 else ""
        raise ValueError(f"Duplicate ligand_id row(s) in {path}: {preview}{suffix}")
    return df


def _analysis_parts_dir(paths: GeometricAnalysisPaths) -> Path:
    return paths.ligand_metrics_parts_dir


def get_chembl_mols(
    chembl_row: pd.Series,
    topology_root: Path,
    zarr_root: Path,
) -> list[Chem.Mol]:
    group = str(chembl_row["chembl3d_group"]).zfill(3)
    chembl_mol_id = str(chembl_row["chembl3d_mol_id"])
    return load_chembl3d_conformers(group, chembl_mol_id, topology_root, zarr_root)


def _finite_values(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]


def clash_passes(mols: Sequence[Chem.Mol], cutoff: float = CLASH_CUTOFF) -> list[bool]:
    if not mols:
        return []
    try:
        dg_bounds = get_dg_bounds(mols[0])
    except Exception:
        return [False] * len(mols)
    passes = []
    for index, mol in enumerate(mols):
        try:
            passes.append(not has_clash(mol, *dg_bounds, cutoff=cutoff))
        except Exception:
            passes.append(False)
    return passes


def filter_by_passes(mols: Sequence[Chem.Mol], passes: Sequence[bool]) -> list[Chem.Mol]:
    if len(mols) != len(passes):
        raise ValueError(f"Pass mask length {len(passes)} does not match conformer count {len(mols)}")
    return [mol for mol, passed in zip(mols, passes) if passed]


def posebusters_result(
    mols: Sequence[Chem.Mol],
    reference_mol: Chem.Mol | None,
    max_workers: int = 1,
    energy_num_threads: int = 1,
) -> PoseBustersResult:
    if not mols:
        return PoseBustersResult([], [])
    if reference_mol is None:
        raise ValueError("PoseBusters reference molecule is required for non-empty conformer checks.")
    return posebusters_geometry_passes(
        list(mols),
        reference_mol,
        max_workers=max_workers,
        energy_num_threads=energy_num_threads,
    )


def source_metadata(source: str) -> tuple[str, str, str]:
    if source in METHODS:
        return parse_method_metadata(source)
    return source, "reference", "reference"


def reference_rmsd_metrics(
    mols: Sequence[Chem.Mol],
    reference_mol: Chem.Mol | None,
    prefix: str,
    label: str,
) -> dict[str, float]:
    empty = {
        f"{prefix}_best_rmsd": math.nan,
        f"{prefix}_median_rmsd": math.nan,
        **{f"{prefix}_hit_{threshold_tag(threshold)}": math.nan for threshold in CASF_HIT_THRESHOLDS},
    }
    if reference_mol is None or not mols:
        return empty
    values = []
    for index, mol in enumerate(mols):
        value = best_aligned_rmsd(mol, reference_mol)
        if not math.isfinite(value):
            try:
                gen_atoms = Chem.RemoveHs(mol).GetNumAtoms()
                ref_atoms = Chem.RemoveHs(reference_mol).GetNumAtoms()
            except Exception:
                gen_atoms = mol.GetNumAtoms()
                ref_atoms = reference_mol.GetNumAtoms()
            raise ValueError(
                f"Failed {label} heavy-atom aligned RMSD for conformer {index}: "
                f"generated heavy atoms={gen_atoms}, reference heavy atoms={ref_atoms}"
            )
        values.append(value)
    finite = _finite_values(values)
    if not finite:
        return empty
    best = min(finite)
    out = {
        f"{prefix}_best_rmsd": best,
        f"{prefix}_median_rmsd": float(np.median(np.asarray(finite, dtype=float))),
    }
    for threshold in CASF_HIT_THRESHOLDS:
        out[f"{prefix}_hit_{threshold_tag(threshold)}"] = float(best <= threshold)
    return out


def casf_rmsd_metrics(mols: Sequence[Chem.Mol], casf_bound: Chem.Mol | None) -> dict[str, float]:
    return reference_rmsd_metrics(mols, casf_bound, "casf", "CASF")


def analyze_source(
    mol_id: str,
    source: str,
    mols: list[Chem.Mol],
    casf_bound: Chem.Mol | None,
    casf_opt_bound: Chem.Mol | None,
    rotatable_bonds: int,
    run_reference_posebusters: bool,
    posebusters_workers: int = 1,
    posebusters_energy_threads: int = 1,
    pb_result_override: PoseBustersResult | None = None,
) -> dict[str, object]:
    family, stage, tier = source_metadata(source)
    clash = clash_passes(mols)
    if pb_result_override is not None:
        pb_result = pb_result_override
    elif run_reference_posebusters and mols:
        try:
            pb_result = posebusters_result(
                mols,
                casf_bound,
                max_workers=posebusters_workers,
                energy_num_threads=posebusters_energy_threads,
            )
        except Exception:
            pb_result = PoseBustersResult([False] * len(mols), [])
    elif run_reference_posebusters:
        pb_result = PoseBustersResult([], [])
    else:
        pb_result = PoseBustersResult([True] * len(mols), [])
    pb = pb_result.passes
    if len(pb) != len(mols):
        raise ValueError(f"PoseBusters returned {len(pb)} result(s) for {len(mols)} conformer(s).")
    post_pb_mols = filter_by_passes(mols, pb)
    pb_fail_counts_json, pb_fail_rates_json = summarize_posebusters_checks(pb_result.check_rows)

    row: dict[str, object] = {
        "mol_id": mol_id,
        "source": source,
        "family": family,
        "stage": stage,
        "tier": tier,
        "rotatable_bonds": rotatable_bonds,
        "conformer_count": len(mols),
        "clash_cutoff": CLASH_CUTOFF,
        "clash_input_confs": len(mols),
        "clash_pass_confs": int(sum(clash)),
        "clash_fail_confs": int(len(clash) - sum(clash)),
        "clash_pass_rate": float(sum(clash) / len(clash)) if clash else math.nan,
        "pb_input_confs": len(pb),
        "pb_pass_confs": int(sum(pb)),
        "pb_fail_confs": int(len(pb) - sum(pb)),
        "pb_pass_rate": float(sum(pb) / len(pb)) if pb else math.nan,
        "pb_check_fail_counts_json": pb_fail_counts_json,
        "pb_check_fail_rates_json": pb_fail_rates_json,
        "post_pb_confs": len(post_pb_mols),
        "mean_torsion_std_deg": mean_torsion_std_deg(mols),
        **pairwise_rmsd_stats(mols),
        **greedy_cluster_metrics(mols),
        **energy_stats(post_pb_mols),
    }
    if source in CASF_HIT_SOURCES:
        row.update(casf_rmsd_metrics(mols, casf_bound))
        row.update(reference_rmsd_metrics(mols, casf_opt_bound, "casf_opt", "CASF optimized ligand"))
    return row


def energy_records(
    mol_id: str,
    source: str,
    mols: Sequence[Chem.Mol],
) -> list[dict[str, object]]:
    family, stage, tier = source_metadata(source)
    rows = []
    for conf_index, mol in enumerate(mols):
        energy = forcefield_energy(mol)
        if not math.isfinite(energy):
            continue
        rows.append(
            {
                "mol_id": mol_id,
                "source": source,
                "family": family,
                "stage": stage,
                "tier": tier,
                "conformer_index": conf_index,
                "energy": energy,
            }
        )
    return rows


def analyze_ligand(task: LigandAnalysisTask) -> dict[str, list[dict[str, object]]]:
    generation_dir = Path(task.generation_dir)
    generated = {
        method: load_sdf(generation_dir / method / f"{task.mol_id}.sdf", required=True)
        for method in METHODS
    }
    casf_bound = load_casf_ligand(task.mol_id, Path(task.casf_ligand_dir), task.source_input)
    casf_opt_bound = try_load_casf_ligand(
        task.mol_id,
        Path(task.casf_opt_ligand_dir),
        task.source_input,
        restrict_to_ligand_dir=True,
    )
    chembl_mols: list[Chem.Mol] = []
    chembl_sdf_mol: Chem.Mol | None = None
    chembl_status = "not_mapped"
    if task.chembl_row is not None:
        group = str(task.chembl_row["chembl3d_group"]).zfill(3)
        chembl_mol_id = str(task.chembl_row["chembl3d_mol_id"])
        chembl_sdf_mol = load_topology_mol(group, chembl_mol_id, Path(task.topology_root))
        chembl_mols = get_chembl_mols(
            pd.Series(task.chembl_row),
            Path(task.topology_root),
            Path(task.zarr_root),
        )
        if not chembl_mols:
            raise RuntimeError(
                f"No ChEMBL3D conformers loaded for mapped ligand {task.mol_id} "
                f"({group}/{chembl_mol_id})"
            )
        chembl_status = "ok" if chembl_mols else "empty"

    rows: list[dict[str, object]] = []
    energy_rows: list[dict[str, object]] = []
    for method in METHODS:
        method_mols = generated[method]
        method_pb = posebusters_result(
            method_mols,
            casf_bound,
            max_workers=task.posebusters_workers,
            energy_num_threads=task.posebusters_energy_threads,
        )
        rows.append(
            analyze_source(
                task.mol_id,
                method,
                method_mols,
                casf_bound,
                casf_opt_bound,
                task.rotatable_bonds,
                run_reference_posebusters=False,
                pb_result_override=method_pb,
            )
        )
        energy_rows.extend(
            energy_records(
                task.mol_id,
                method,
                filter_by_passes(method_mols, method_pb.passes),
            )
        )
    rows.append(
        analyze_source(
            task.mol_id,
            "casf_crystal",
            [casf_bound],
            casf_bound,
            casf_opt_bound,
            task.rotatable_bonds,
            run_reference_posebusters=True,
            posebusters_workers=task.posebusters_workers,
            posebusters_energy_threads=task.posebusters_energy_threads,
        )
    )
    if casf_opt_bound is not None:
        rows.append(
            analyze_source(
                task.mol_id,
                "casf_opt",
                [casf_opt_bound],
                casf_opt_bound,
                casf_opt_bound,
                task.rotatable_bonds,
                run_reference_posebusters=True,
                posebusters_workers=task.posebusters_workers,
                posebusters_energy_threads=task.posebusters_energy_threads,
            )
        )
    rows.append(
        analyze_source(
            task.mol_id,
            "chembl3d_sdf",
            [chembl_sdf_mol] if chembl_sdf_mol is not None else [],
            casf_bound,
            casf_opt_bound,
            task.rotatable_bonds,
            run_reference_posebusters=True,
            posebusters_workers=task.posebusters_workers,
            posebusters_energy_threads=task.posebusters_energy_threads,
        )
    )
    chembl_pb = posebusters_result(
        chembl_mols,
        casf_bound,
        max_workers=task.posebusters_workers,
        energy_num_threads=task.posebusters_energy_threads,
    )
    chembl_row = analyze_source(
        task.mol_id,
        "chembl3d_gt",
        chembl_mols,
        casf_bound,
        casf_opt_bound,
        task.rotatable_bonds,
        run_reference_posebusters=True,
        posebusters_workers=task.posebusters_workers,
        posebusters_energy_threads=task.posebusters_energy_threads,
        pb_result_override=chembl_pb,
    )
    chembl_row["chembl_load_status"] = chembl_status
    rows.append(chembl_row)
    chembl_pb_mols = filter_by_passes(chembl_mols, chembl_pb.passes)
    chembl_pb_row = analyze_source(
        task.mol_id,
        "chembl3d_gt_pb",
        chembl_pb_mols,
        casf_bound,
        casf_opt_bound,
        task.rotatable_bonds,
        run_reference_posebusters=False,
    )
    chembl_pb_row["chembl_load_status"] = chembl_status
    rows.append(chembl_pb_row)
    energy_rows.extend(energy_records(task.mol_id, "chembl3d_gt", chembl_mols))
    energy_rows.extend(energy_records(task.mol_id, "chembl3d_gt_pb", chembl_pb_mols))
    return {"metrics": rows, "energy_values": energy_rows}


def _process_ligand_task(task: LigandAnalysisTask) -> tuple[str, str | None]:
    _limit_worker_threads()
    try:
        payload = analyze_ligand(task)
        _write_pickle_atomic(Path(task.parts_dir) / f"{task.mol_id}.pkl", payload)
        return task.mol_id, None
    except Exception as exc:  # noqa: BLE001 - propagate worker context
        return task.mol_id, f"{type(exc).__name__}: {exc}"


def _resolve_worker_count(workers: int | None, pending_count: int) -> int:
    if pending_count <= 0:
        return 1
    cpu_count = _available_cpu_count()
    if workers is None or workers <= 0:
        return max(1, min(cpu_count, int(pending_count)))
    return max(1, min(int(workers), int(pending_count)))


def _resolve_posebusters_parallelism(
    posebusters_workers: int | None,
    posebusters_energy_threads: int | None,
    ligand_worker_count: int,
) -> tuple[int, int]:
    cpu_count = _available_cpu_count()
    ligand_worker_count = max(1, int(ligand_worker_count))
    slots_per_ligand = max(1, cpu_count // ligand_worker_count)

    if posebusters_workers is not None and posebusters_workers > 0:
        pb_workers = min(int(posebusters_workers), slots_per_ligand)
    elif ligand_worker_count >= cpu_count:
        pb_workers = 1
    else:
        pb_workers = slots_per_ligand

    if pb_workers > 1:
        pb_energy_threads = 1
    elif posebusters_energy_threads is not None and posebusters_energy_threads > 0:
        pb_energy_threads = min(int(posebusters_energy_threads), slots_per_ligand)
    elif ligand_worker_count > 1:
        pb_energy_threads = slots_per_ligand
    else:
        pb_energy_threads = cpu_count
    return max(1, pb_workers), max(1, pb_energy_threads)


def _build_ligand_tasks(
    mol_ids: Sequence[str],
    paths: GeometricAnalysisPaths,
    casf_opt_ligand_dir: Path,
    manifest_df: pd.DataFrame,
    chembl_map: pd.DataFrame,
    topology_root: Path,
    zarr_root: Path,
    posebusters_workers: int,
    posebusters_energy_threads: int,
) -> list[LigandAnalysisTask]:
    rot_lookup = (
        manifest_df.drop_duplicates("mol_id")[["mol_id", "rotatable_bonds", "input_smiles", "source_input"]]
        .set_index("mol_id")
    )
    chembl_lookup = chembl_map.set_index("ligand_id") if not chembl_map.empty else None
    tasks = []
    for mol_id in mol_ids:
        meta = rot_lookup.loc[mol_id] if mol_id in rot_lookup.index else pd.Series({})
        chembl_row = None
        if chembl_lookup is not None and mol_id in chembl_lookup.index:
            chembl_row = chembl_lookup.loc[mol_id].to_dict()
        tasks.append(
            LigandAnalysisTask(
                mol_id=mol_id,
                generation_dir=str(paths.generation_dir),
                casf_ligand_dir=str(paths.casf_ligand_dir),
                casf_opt_ligand_dir=str(casf_opt_ligand_dir),
                source_input=str(meta.get("source_input", "")),
                parts_dir=str(_analysis_parts_dir(paths)),
                topology_root=str(topology_root),
                zarr_root=str(zarr_root),
                input_smiles=str(meta.get("input_smiles", "")),
                rotatable_bonds=int(meta.get("rotatable_bonds", 0)),
                posebusters_workers=posebusters_workers,
                posebusters_energy_threads=posebusters_energy_threads,
                chembl_row=chembl_row,
            )
        )
    return tasks


def load_or_compute_ligand_metrics(
    paths: GeometricAnalysisPaths,
    casf_opt_ligand_dir: Path,
    mol_ids: list[str],
    manifest_df: pd.DataFrame,
    chembl_map: pd.DataFrame,
    topology_root: Path,
    zarr_root: Path,
    report_only: bool,
    workers: int | None,
    posebusters_workers: int | None,
    posebusters_energy_threads: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts_dir = _analysis_parts_dir(paths)
    parts_dir.mkdir(parents=True, exist_ok=True)

    pending = [] if report_only else list(mol_ids)
    worker_count = _resolve_worker_count(workers, len(pending) or len(mol_ids))
    pb_workers, pb_energy_threads = _resolve_posebusters_parallelism(
        posebusters_workers,
        posebusters_energy_threads,
        worker_count if pending else 1,
    )
    if report_only:
        print(
            f"CASF analysis: report-only mode, loading {len(mol_ids)} ligand part(s)",
            flush=True,
        )
    elif pending:
        print(
            f"CASF analysis: recomputing {len(pending)} ligand(s) from scratch with "
            f"{worker_count} ligand worker(s), "
            f"{pb_workers} PoseBusters worker(s), {pb_energy_threads} PB energy thread(s), "
            f"{_available_cpu_count()} available CPU(s)",
            flush=True,
        )
    tasks = _build_ligand_tasks(
        pending,
        paths,
        casf_opt_ligand_dir,
        manifest_df,
        chembl_map,
        topology_root,
        zarr_root,
        pb_workers,
        pb_energy_threads,
    )

    if worker_count == 1:
        for index, task in enumerate(tasks, start=1):
            mol_id, error = _process_ligand_task(task)
            if error is not None:
                raise RuntimeError(f"CASF analysis failed for {mol_id}: {error}")
            if index == 1 or index % 5 == 0 or index == len(tasks):
                print(f"CASF analysis: {index}/{len(pending)} ({mol_id})", flush=True)
    elif tasks:
        failures: list[str] = []
        completed = 0
        executor_kwargs: dict[str, object] = {
            "max_workers": worker_count,
            "initializer": _limit_worker_threads,
        }
        if sys.version_info >= (3, 11):
            executor_kwargs["max_tasks_per_child"] = 8
        with ProcessPoolExecutor(**executor_kwargs) as pool:
            futures = {pool.submit(_process_ligand_task, task): task.mol_id for task in tasks}
            for future in as_completed(futures):
                mol_id, error = future.result()
                completed += 1
                if error is not None:
                    failures.append(f"{mol_id}: {error}")
                if completed == 1 or completed % 5 == 0 or completed == len(tasks):
                    print(
                        f"CASF analysis: {completed}/{len(pending)} ({mol_id})",
                        flush=True,
                    )
        if failures:
            preview = "; ".join(failures[:5])
            suffix = f" (+{len(failures) - 5} more)" if len(failures) > 5 else ""
            raise RuntimeError(f"{len(failures)} ligand job(s) failed: {preview}{suffix}")

    part_paths = [parts_dir / f"{mol_id}.pkl" for mol_id in mol_ids if (parts_dir / f"{mol_id}.pkl").exists()]

    def _load_part(part_path: Path) -> dict[str, object]:
        with part_path.open("rb") as handle:
            return pickle.load(handle)

    load_workers = min(8, max(1, len(part_paths)))
    if load_workers == 1:
        payloads = [_load_part(path) for path in part_paths]
    else:
        with ThreadPoolExecutor(max_workers=load_workers) as pool:
            payloads = list(pool.map(_load_part, part_paths))

    frames = [pd.DataFrame(payload["metrics"]) for payload in payloads]
    energy_frames = [pd.DataFrame(payload.get("energy_values", [])) for payload in payloads]
    metrics_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    energy_df = pd.concat(energy_frames, ignore_index=True) if energy_frames else pd.DataFrame()
    return metrics_df, energy_df


def _manifest_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _mean_per_ligand_rate(numerators: pd.Series, denominators: pd.Series) -> float:
    rates = [
        float(numerator / denominator)
        for numerator, denominator in zip(numerators.tolist(), denominators.tolist())
        if denominator > 0
    ]
    return safe_mean(rates)


def format_ligands_scope(ligands: int, mean_confs: float) -> str:
    if not math.isfinite(float(mean_confs)):
        return f"{int(ligands)}/-"
    return f"{int(ligands)}/{int(round(float(mean_confs)))}"


def add_ligands_scope_column(df: pd.DataFrame, confs_col: str = "total_confs") -> pd.DataFrame:
    if df.empty or "ligands" not in df.columns:
        return df
    out = df.copy()
    if confs_col in out.columns:
        mean_confs = pd.to_numeric(out[confs_col], errors="coerce") / pd.to_numeric(out["ligands"], errors="coerce")
    elif "mean_confs_per_ligand" in out.columns:
        mean_confs = pd.to_numeric(out["mean_confs_per_ligand"], errors="coerce")
    elif "selected_pool_total" in out.columns:
        mean_confs = pd.to_numeric(out["selected_pool_total"], errors="coerce") / pd.to_numeric(out["ligands"], errors="coerce")
    elif "kept_confs_total" in out.columns:
        mean_confs = pd.to_numeric(out["kept_confs_total"], errors="coerce") / pd.to_numeric(out["ligands"], errors="coerce")
    else:
        mean_confs = pd.Series([math.nan] * len(out), index=out.index)
    out["ligands_scope"] = [
        format_ligands_scope(int(ligands), float(mean_conf) if pd.notna(mean_conf) else math.nan)
        for ligands, mean_conf in zip(out["ligands"], mean_confs)
    ]
    return out


def build_generation_filter_summary(manifest_df: pd.DataFrame) -> pd.DataFrame:
    if manifest_df.empty:
        return pd.DataFrame()
    manifest_df = manifest_df.copy()
    if "post_min_clash_rejected" not in manifest_df.columns and "minimization_invalid" in manifest_df.columns:
        manifest_df["post_min_clash_rejected"] = manifest_df["minimization_invalid"]
        manifest_df["minimization_invalid"] = 0
    if "minimization_error" not in manifest_df.columns:
        manifest_df["minimization_error"] = 0
    if "num_target_confs" not in manifest_df.columns:
        manifest_df["num_target_confs"] = 0

    rows = []
    for method, frame in manifest_df.groupby("generation_method", sort=False):
        family, stage, tier = parse_method_metadata(method)
        ligands = int(frame["mol_id"].nunique())
        target = _manifest_numeric(frame, "num_target_confs")
        generated = _manifest_numeric(frame, "generated_candidates")
        pb_input = _manifest_numeric(frame, "pb_input_confs")
        pb_fail = _manifest_numeric(frame, "pb_fail_confs")
        kept = _manifest_numeric(frame, "kept_confs")
        finite_rejected = _manifest_numeric(frame, "finite_rejected")
        clash_rejected = _manifest_numeric(frame, "clash_rejected")
        bond_rejected = _manifest_numeric(frame, "bond_rejected")
        stereo_rejected = _manifest_numeric(frame, "stereo_rejected")
        rmsd_rejected = _manifest_numeric(frame, "rmsd_rejected")
        minimization_input = _manifest_numeric(frame, "minimization_input_confs")
        post_min_clash_rejected = _manifest_numeric(frame, "post_min_clash_rejected")
        geometry_rejected = bond_rejected + stereo_rejected + rmsd_rejected

        mean_kept = safe_mean(kept.tolist())
        row: dict[str, object] = {
            "source": method,
            "family": family,
            "stage": stage,
            "tier": tier,
            "ligands": ligands,
            "ligands_scope": format_ligands_scope(ligands, mean_kept),
            "target_confs_mean": safe_mean(target.tolist()),
            "selected_pool_total": float(pb_input.sum(skipna=True)),
            "kept_confs_total": float(kept.sum(skipna=True)),
            "pb_fail_total": float(pb_fail.sum(skipna=True)),
            "pb_fail_rate_mean": _mean_per_ligand_rate(pb_fail, pb_input),
            "kept_vs_target_rate_mean": _mean_per_ligand_rate(kept, target),
        }

        if tier == "fixed":
            generation_pool = generated.where(generated > 0, minimization_input.where(minimization_input > 0, pb_input))
            row["generation_pool_total"] = float(generation_pool.sum(skipna=True))
            row["finite_fail_total"] = float(finite_rejected.sum(skipna=True))
            row["clash_fail_total"] = float(clash_rejected.sum(skipna=True))
            row["geometry_fail_total"] = float(geometry_rejected.sum(skipna=True))
            row["finite_fail_rate_mean"] = _mean_per_ligand_rate(finite_rejected, generation_pool)
            row["clash_fail_rate_mean"] = _mean_per_ligand_rate(clash_rejected, generation_pool)
            row["geometry_fail_rate_mean"] = _mean_per_ligand_rate(geometry_rejected, generation_pool)
        else:
            row["generation_pool_total"] = math.nan
            row["finite_fail_total"] = math.nan
            row["clash_fail_total"] = math.nan
            row["geometry_fail_total"] = math.nan
            row["finite_fail_rate_mean"] = math.nan
            row["clash_fail_rate_mean"] = math.nan
            row["geometry_fail_rate_mean"] = math.nan

        if stage == "minimized":
            row["post_min_clash_fail_total"] = float(post_min_clash_rejected.sum(skipna=True))
            row["post_min_clash_fail_rate_mean"] = _mean_per_ligand_rate(post_min_clash_rejected, minimization_input)
        else:
            row["post_min_clash_fail_total"] = math.nan
            row["post_min_clash_fail_rate_mean"] = math.nan

        rows.append(row)
    return sort_and_finalize_table(pd.DataFrame(rows))


def aggregate_numeric(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
    include_total_confs: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for keys, frame in df.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["ligands"] = int(frame["mol_id"].nunique()) if "mol_id" in frame else len(frame)
        if include_total_confs and "conformer_count" in frame:
            row["total_confs"] = safe_sum(frame["conformer_count"].tolist())
        for col in metric_cols:
            if col in frame.columns:
                row[col] = safe_mean(pd.to_numeric(frame[col], errors="coerce").tolist())
        rows.append(row)
    out = sort_table_by_source(pd.DataFrame(rows))
    return add_ligands_scope_column(finalize_table_types(out), confs_col="total_confs")


def aggregate_json_count_column(values: Sequence[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        for key, count in parsed.items():
            try:
                counts[str(key)] = counts.get(str(key), 0) + int(count)
            except (TypeError, ValueError):
                continue
    return dict(sorted(counts.items()))


def json_dumps_compact(value: dict[str, int | float]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def build_generation_pb_failure_summary(
    manifest_df: pd.DataFrame,
    generation_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if manifest_df.empty or "pb_check_fail_counts_json" not in manifest_df.columns:
        return pd.DataFrame()
    scope_lookup = {}
    if generation_df is not None and not generation_df.empty:
        scope_lookup = {
            str(row.source): str(row.ligands_scope)
            for row in generation_df.itertuples(index=False)
            if hasattr(row, "ligands_scope")
        }
    rows = []
    for method, frame in manifest_df.groupby("generation_method", sort=False):
        pb_input = _manifest_numeric(frame, "pb_input_confs")
        total_pb_input = safe_sum(pb_input.tolist())
        fail_counts = aggregate_json_count_column(frame["pb_check_fail_counts_json"].tolist())
        for test_name, fail_count in fail_counts.items():
            if fail_count <= 0:
                continue
            rows.append(
                {
                    "source": method,
                    "ligands_scope": scope_lookup.get(method, ""),
                    "selected_pool_total": int(round(total_pb_input)),
                    "pb_test": test_name,
                    "pb_fail_count": int(fail_count),
                    "pb_fail_rate": fail_count / total_pb_input if total_pb_input else math.nan,
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["__source_order"] = out["source"].map(source_order_index)
    out = out.sort_values(["__source_order", "pb_fail_count", "pb_test"], ascending=[True, False, True])
    return finalize_table_types(out.drop(columns=["__source_order"]).reset_index(drop=True))


def build_reference_pb_failure_summary(
    metrics_df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    refs = metrics_df[metrics_df["source"].isin(FILTER_REFERENCE_SOURCES)].copy()
    if refs.empty or "pb_check_fail_counts_json" not in refs.columns:
        return pd.DataFrame()
    scope_lookup = {}
    if reference_df is not None and not reference_df.empty:
        scope_lookup = {
            str(row.source): str(row.ligands_scope)
            for row in reference_df.itertuples(index=False)
            if hasattr(row, "ligands_scope")
        }
    rows = []
    for source, frame in refs.groupby("source", sort=False):
        pb_input = _manifest_numeric(frame, "pb_input_confs")
        total_pb_input = safe_sum(pb_input.tolist())
        fail_counts = aggregate_json_count_column(frame["pb_check_fail_counts_json"].tolist())
        for test_name, fail_count in fail_counts.items():
            if fail_count <= 0:
                continue
            rows.append(
                {
                    "source": source,
                    "ligands_scope": scope_lookup.get(source, ""),
                    "selected_pool_total": int(round(total_pb_input)),
                    "pb_test": test_name,
                    "pb_fail_count": int(fail_count),
                    "pb_fail_rate": fail_count / total_pb_input if total_pb_input else math.nan,
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["__source_order"] = out["source"].map(source_order_index)
    out = out.sort_values(["__source_order", "pb_fail_count", "pb_test"], ascending=[True, False, True])
    return finalize_table_types(out.drop(columns=["__source_order"]).reset_index(drop=True))


def build_reference_filter_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    refs = metrics_df[metrics_df["source"].isin(FILTER_REFERENCE_SOURCES)].copy()
    if refs.empty:
        return pd.DataFrame()
    rows = []
    for source, frame in refs.groupby("source", sort=False):
        ligands = int(frame["mol_id"].nunique())
        clash_input = safe_sum(frame["clash_input_confs"].tolist())
        pb_input = safe_sum(frame["pb_input_confs"].tolist())
        clash_fail = safe_sum(frame["clash_fail_confs"].tolist()) if "clash_fail_confs" in frame else clash_input - safe_sum(frame["clash_pass_confs"].tolist())
        pb_fail = safe_sum(frame["pb_fail_confs"].tolist()) if "pb_fail_confs" in frame else pb_input - safe_sum(frame["pb_pass_confs"].tolist())
        kept = safe_sum(frame["conformer_count"].tolist()) if "conformer_count" in frame else pb_input - pb_fail
        mean_confs = kept / ligands if ligands else math.nan
        fail_counts = (
            aggregate_json_count_column(frame["pb_check_fail_counts_json"].tolist())
            if "pb_check_fail_counts_json" in frame
            else {}
        )
        fail_rates = {
            name: (count / pb_input if pb_input else math.nan)
            for name, count in fail_counts.items()
        }
        rows.append(
            {
                "source": source,
                "ligands": ligands,
                "ligands_scope": format_ligands_scope(ligands, mean_confs),
                "selected_pool_total": int(round(clash_input)),
                "clash_fail_total": int(round(clash_fail)),
                "pb_fail_total": int(round(pb_fail)),
                "clash_fail_rate": clash_fail / clash_input if clash_input else math.nan,
                "pb_fail_rate": pb_fail / pb_input if pb_input else math.nan,
                "pb_check_fail_counts_json": json_dumps_compact(fail_counts),
                "pb_check_fail_rates_json": json_dumps_compact(fail_rates),
            }
        )
    return sort_and_finalize_table(pd.DataFrame(rows))


def build_cluster_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    cluster_sources = metrics_df[
        metrics_df["source"].isin(CLUSTER_COMPARE_SOURCES)
    ].copy()
    if cluster_sources.empty:
        return pd.DataFrame()
    rows = []
    for keys, frame in cluster_sources.groupby(["source", "family", "stage", "tier"], sort=False):
        source, family, stage, tier = keys
        row: dict[str, object] = {
            "source": source,
            "family": family,
            "stage": stage,
            "tier": tier,
            "ligands": int(frame["mol_id"].nunique()),
            "total_confs": safe_sum(frame["conformer_count"].tolist()),
            "mean_confs_per_ligand": safe_mean(frame["conformer_count"].tolist()),
            "pairwise_mean": safe_mean(frame["pairwise_mean"].tolist()) if "pairwise_mean" in frame else math.nan,
            "pairwise_p90": safe_mean(frame["pairwise_p90"].tolist()) if "pairwise_p90" in frame else math.nan,
            "mean_torsion_std_deg": safe_mean(frame["mean_torsion_std_deg"].tolist())
            if "mean_torsion_std_deg" in frame
            else math.nan,
        }
        for threshold in CASF_GEOMETRIC_CLUSTER_THRESHOLDS:
            tag = threshold_tag(threshold)
            cluster_col = f"greedy_clusters_{tag}"
            row[f"total_clusters_{tag}"] = (
                safe_sum(frame[cluster_col].tolist()) if cluster_col in frame else math.nan
            )
            row[f"mean_clusters_{tag}"] = (
                safe_mean(frame[cluster_col].tolist()) if cluster_col in frame else math.nan
            )
            for col in (
                f"clusters_per_100_{tag}",
                f"cluster_entropy_{tag}",
                f"largest_cluster_fraction_{tag}",
                f"effective_clusters_{tag}",
                f"simpson_concentration_{tag}",
                f"singleton_fraction_{tag}",
            ):
                row[col] = safe_mean(frame[col].tolist()) if col in frame else math.nan
        rows.append(row)
    return add_ligands_scope_column(sort_and_finalize_table(pd.DataFrame(rows)), confs_col="total_confs")


def build_energy_summary(energy_df: pd.DataFrame) -> pd.DataFrame:
    if energy_df.empty:
        return pd.DataFrame()
    rows = []
    for keys, frame in energy_df.groupby(["source", "family", "stage", "tier"], sort=False):
        source, family, stage, tier = keys
        values = pd.to_numeric(frame["energy"], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        rows.append(
            {
                "source": source,
                "family": family,
                "stage": stage,
                "tier": tier,
                "ligands": int(frame["mol_id"].nunique()),
                "total_confs": int(values.size),
                "energy_min": safe_mean(
                    frame.groupby("mol_id")["energy"].min().to_numpy(dtype=float).tolist()
                ),
                "energy_max": safe_mean(
                    frame.groupby("mol_id")["energy"].max().to_numpy(dtype=float).tolist()
                ),
                "energy_median": safe_mean(
                    frame.groupby("mol_id")["energy"].median().to_numpy(dtype=float).tolist()
                ),
                "energy_std": safe_mean(
                    frame.groupby("mol_id")["energy"].std(ddof=0).to_numpy(dtype=float).tolist()
                ),
            }
        )
    return add_ligands_scope_column(sort_and_finalize_table(pd.DataFrame(rows)), confs_col="total_confs")


def build_casf_hit_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    generated = metrics_df[metrics_df["source"].isin(CASF_HIT_SOURCES)].copy()
    cols = [
        "casf_best_rmsd",
        "casf_median_rmsd",
        *[f"casf_hit_{threshold_tag(threshold)}" for threshold in CASF_HIT_THRESHOLDS],
    ]
    return add_ligands_scope_column(
        sort_and_finalize_table(aggregate_numeric(generated, ["source", "family", "stage", "tier"], cols)),
        confs_col="total_confs",
    )


def build_casf_opt_hit_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    generated = metrics_df[metrics_df["source"].isin(CASF_HIT_SOURCES)].copy()
    cols = [
        "casf_opt_best_rmsd",
        "casf_opt_median_rmsd",
        *[f"casf_opt_hit_{threshold_tag(threshold)}" for threshold in CASF_HIT_THRESHOLDS],
    ]
    return add_ligands_scope_column(
        sort_and_finalize_table(aggregate_numeric(generated, ["source", "family", "stage", "tier"], cols)),
        confs_col="total_confs",
    )


def _is_count_column(column: str) -> bool:
    return (
        column
        in {
            "ligands",
            "num_ligands",
            "num_confs",
            "total_confs",
            "energy_count",
            "final_kept_num",
            "rotatable_bonds",
            "conformer_index",
        }
        or column.startswith("num_")
        or column.endswith("_total")
        or column.endswith("_confs")
        or column.endswith("_count")
        or column.endswith("_num")
        or column.endswith("_mean_per_ligand")
        or column.endswith("_median_per_ligand")
    )


def fmt(value: object, column: str = "", digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, (int, float, np.floating)):
        if not math.isfinite(float(value)):
            return "-"
        if _is_count_column(column):
            return str(int(round(float(value))))
        return f"{float(value):.{digits}f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, max_cols: int | None = None) -> str:
    if df.empty:
        return "_No data._"
    view = df.copy()
    if max_cols is not None:
        view = view.iloc[:, :max_cols]
    headers = [str(col) for col in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt(row[col], str(col)) for col in view.columns) + " |")
    return "\n".join(lines)


def table_to_markdown(df: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = [column for column in columns if column in df.columns]
    if len(selected) > 10:
        raise ValueError(f"Report table has {len(selected)} columns; max is 10")
    return dataframe_to_markdown(df[selected] if selected else pd.DataFrame())


COLUMN_DOCS: dict[str, str] = {
    "source": "Generation method or reference source label.",
    "ligands_scope": "Eligible ligands / mean conformers aggregated in the row (for example `94/1000`).",
    "target_confs_mean": "Mean per-ligand target set size: 1000 for fixed, dynamic formula (-20 + 22 * rotatable bonds), or ChEMBL3D conformer count.",
    "selected_pool_total": "Total conformers in the selected set entering the reported filter stage (PoseBusters input pool).",
    "kept_confs_total": "Final conformers kept after all filters for that method row.",
    "pb_fail_total": "Total conformers failing PoseBusters in the selected pool.",
    "pb_fail_rate_mean": "Per-ligand PoseBusters failure rate (`pb_fail / pb_input`), averaged across ligands.",
    "kept_vs_target_rate_mean": "Per-ligand final yield (`kept / target_confs`), averaged across ligands.",
    "generation_pool_total": "Fixed-tier only: total conformers generated or entering minimization before subset selection.",
    "finite_fail_total": "Fixed-tier only: conformers rejected for non-finite coordinates during generation.",
    "clash_fail_total": "Fixed-tier only: conformers rejected for steric clash during generation.",
    "geometry_fail_total": "Fixed-tier only: bond/stereo/RMSD rejects during generation.",
    "finite_fail_rate_mean": "Fixed-tier only: finite rejects divided by the generation pool, averaged per ligand.",
    "clash_fail_rate_mean": "Fixed-tier only: clash rejects divided by the generation pool, averaged per ligand.",
    "geometry_fail_rate_mean": "Fixed-tier only: geometry rejects divided by the generation pool, averaged per ligand.",
    "post_min_clash_fail_total": "Minimized methods only: conformers rejected for clash after minimization.",
    "post_min_clash_fail_rate_mean": "Minimized methods only: post-minimization clash rejects divided by minimization input, averaged per ligand.",
    "pb_test": "PoseBusters boolean test name.",
    "pb_fail_count": "Number of conformers failing this PoseBusters test in the selected pool.",
    "pb_fail_rate": "Failures divided by the row selected-pool total (PoseBusters input conformers).",
    "clash_fail_rate": "Clash failures divided by conformers checked for clashes.",
    "mean_confs_per_ligand": "Mean final conformer count per ligand.",
    "total_confs": "Total conformers aggregated across ligands.",
    "casf_best_rmsd": "Mean per-ligand best heavy-atom aligned RMSD to CASF crystal (`ligands`).",
    "casf_median_rmsd": "Mean per-ligand median heavy-atom aligned RMSD to CASF crystal.",
    "casf_opt_best_rmsd": "Mean per-ligand best heavy-atom aligned RMSD to CASF optimized ligand (`ligands_opt`).",
    "casf_opt_median_rmsd": "Mean per-ligand median heavy-atom aligned RMSD to CASF optimized ligand.",
    "casf_hit_0p25": "Fraction of ligands whose best CASF crystal RMSD is <= 0.25 Å.",
    "casf_hit_0p5": "Fraction of ligands whose best CASF crystal RMSD is <= 0.5 Å.",
    "casf_hit_0p75": "Fraction of ligands whose best CASF crystal RMSD is <= 0.75 Å.",
    "casf_hit_2p0": "Fraction of ligands whose best CASF crystal RMSD is <= 2.0 Å.",
    "casf_opt_hit_0p25": "Fraction of ligands whose best CASF optimized RMSD is <= 0.25 Å.",
    "casf_opt_hit_0p5": "Fraction of ligands whose best CASF optimized RMSD is <= 0.5 Å.",
    "casf_opt_hit_0p75": "Fraction of ligands whose best CASF optimized RMSD is <= 0.75 Å.",
    "casf_opt_hit_2p0": "Fraction of ligands whose best CASF optimized RMSD is <= 2.0 Å.",
    "mean_clusters_0p5": "Mean greedy cluster count per ligand at 0.5 Å.",
    "mean_clusters_1p0": "Mean greedy cluster count per ligand at 1.0 Å.",
    "mean_clusters_2p0": "Mean greedy cluster count per ligand at 2.0 Å.",
    "mean_clusters_3p0": "Mean greedy cluster count per ligand at 3.0 Å.",
    "clusters_per_100_0p5": "Mean per-ligand clusters per 100 conformers at 0.5 Å.",
    "clusters_per_100_1p0": "Mean per-ligand clusters per 100 conformers at 1.0 Å.",
    "clusters_per_100_2p0": "Mean per-ligand clusters per 100 conformers at 2.0 Å.",
    "clusters_per_100_3p0": "Mean per-ligand clusters per 100 conformers at 3.0 Å.",
    "cluster_entropy_1p0": "Mean normalized Shannon entropy of 1.0 Å cluster occupancies.",
    "largest_cluster_fraction_1p0": "Mean fraction of conformers in the largest 1.0 Å cluster.",
    "energy_min": "Mean per-ligand minimum PB-passing MMFF94s energy, averaged across ligands.",
    "energy_max": "Mean per-ligand maximum PB-passing MMFF94s energy, averaged across ligands.",
    "energy_median": "Mean per-ligand median PB-passing MMFF94s energy, averaged across ligands.",
    "energy_std": "Mean per-ligand energy standard deviation, averaged across ligands.",
}


def render_table_section(title: str, df: pd.DataFrame, columns: Sequence[str], intro: str = "") -> list[str]:
    lines = [f"### {title}", ""]
    if intro:
        lines.extend([intro, ""])
    docs = [(column, COLUMN_DOCS[column]) for column in columns if column in COLUMN_DOCS]
    if docs:
        lines.append("Column definitions:")
        lines.append("")
        lines.append(definition_list(docs))
        lines.append("")
    lines.append(table_to_markdown(df, columns))
    lines.append("")
    return lines


def definition_list(items: Sequence[tuple[str, str]]) -> str:
    return "\n".join(f"- `{name}`: {description}" for name, description in items)


def write_report(
    paths: GeometricAnalysisPaths,
    mol_ids: Sequence[str],
    n_with_chembl: int,
    casf_opt_ligand_dir: Path,
    generation_df: pd.DataFrame,
    generation_pb_fail_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    reference_pb_fail_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    energy_df: pd.DataFrame,
    casf_hit_df: pd.DataFrame,
    casf_opt_hit_df: pd.DataFrame,
) -> None:
    fixed_generation_df = generation_df[generation_df["tier"] == "fixed"].copy() if not generation_df.empty else generation_df
    lines = [
        f"# CASF Geometric Analysis Revision ({len(mol_ids)} ligands)",
        "",
        f"- Root: `{paths.root_dir}`",
        f"- Generation: `{paths.generation_dir}`",
        f"- CASF crystal ligands: `{paths.casf_ligand_dir}`",
        f"- CASF optimized ligands: `{casf_opt_ligand_dir}`",
        f"- Ligands with ChEMBL3D map: {n_with_chembl} / {len(mol_ids)}",
        f"- Clash cutoff: {CLASH_CUTOFF}",
        "",
        "## Generation Pipeline Overview",
        "",
        "Per-ligand rates are averaged across ligands. PoseBusters and kept-yield rates use the selected conformer pool for that method row (`pb_input_confs`), not the full fixed-pool generation count.",
        "",
        "ChEMBL3D ground truth appears twice in comparison tables: `chembl3d_gt` uses all loaded conformers; `chembl3d_gt_pb` uses only conformers passing PoseBusters against the CASF crystal reference (fair comparison to filtered generation sets).",
        "",
    ]
    lines.extend(
        render_table_section(
            "Target Set Sizes And Selected-Pool Outcomes",
            generation_df,
            [
                "source",
                "ligands_scope",
                "target_confs_mean",
                "selected_pool_total",
                "kept_confs_total",
                "pb_fail_total",
                "pb_fail_rate_mean",
                "kept_vs_target_rate_mean",
            ],
        )
    )
    lines.extend(
        render_table_section(
            "Fixed-Pool Generation Rejects",
            fixed_generation_df,
            [
                "source",
                "ligands_scope",
                "generation_pool_total",
                "finite_fail_total",
                "clash_fail_total",
                "geometry_fail_total",
                "finite_fail_rate_mean",
                "clash_fail_rate_mean",
                "geometry_fail_rate_mean",
            ],
            intro="Only fixed-tier rows are shown here. Rates are computed against the generation/minimization input pool, not the dynamic or ChEMBL-count subsets.",
        )
    )
    lines.extend(
        render_table_section(
            "Selected-Set Filter Counts",
            generation_df,
            [
                "source",
                "ligands_scope",
                "selected_pool_total",
                "pb_fail_total",
                "post_min_clash_fail_total",
                "kept_confs_total",
            ],
            intro="Dynamic and ChEMBL-count rows report only the selected subset entering PoseBusters. They do not repeat fixed-pool generation rejects.",
        )
    )
    lines.extend(
        render_table_section(
            "Selected-Set Filter Rates",
            generation_df,
            [
                "source",
                "ligands_scope",
                "pb_fail_rate_mean",
                "post_min_clash_fail_rate_mean",
                "kept_vs_target_rate_mean",
            ],
        )
    )
    lines.extend(
        render_table_section(
            "Generation PoseBusters Failing Tests",
            generation_pb_fail_df,
            [
                "source",
                "ligands_scope",
                "selected_pool_total",
                "pb_test",
                "pb_fail_count",
                "pb_fail_rate",
            ],
            intro="Only tests with at least one failure are shown. Denominators are the selected PoseBusters input pool for each method row.",
        )
    )
    lines.extend(["## Reference Filter Checks", ""])
    lines.extend(
        render_table_section(
            "Reference Clash And PoseBusters Failures",
            reference_df,
            [
                "source",
                "ligands_scope",
                "selected_pool_total",
                "clash_fail_total",
                "pb_fail_total",
                "clash_fail_rate",
                "pb_fail_rate",
            ],
            intro="`casf_crystal` uses crystal poses from the CASF ligand directory. `casf_opt` is included only when an optimized pose exists under `ligands_opt`.",
        )
    )
    lines.extend(
        render_table_section(
            "Reference PoseBusters Failing Tests",
            reference_pb_fail_df,
            [
                "source",
                "ligands_scope",
                "selected_pool_total",
                "pb_test",
                "pb_fail_count",
                "pb_fail_rate",
            ],
        )
    )
    lines.extend(["## Clustering Diversity", ""])
    lines.extend(
        render_table_section(
            "Mean Cluster Counts Per Ligand",
            cluster_df,
            [
                "source",
                "ligands_scope",
                "mean_clusters_0p5",
                "mean_clusters_1p0",
                "mean_clusters_2p0",
                "mean_clusters_3p0",
            ],
            intro="Includes `chembl3d_gt` (all ChEMBL3D conformers) and `chembl3d_gt_pb` (PoseBusters-passing subset only).",
        )
    )
    lines.extend(
        render_table_section(
            "Normalized Cluster Density",
            cluster_df,
            [
                "source",
                "ligands_scope",
                "clusters_per_100_0p5",
                "clusters_per_100_1p0",
                "clusters_per_100_2p0",
                "clusters_per_100_3p0",
                "cluster_entropy_1p0",
                "largest_cluster_fraction_1p0",
            ],
        )
    )
    lines.extend(["## Energy", ""])
    lines.extend(
        render_table_section(
            "PB-Passing Conformer Energies",
            energy_df,
            [
                "source",
                "ligands_scope",
                "energy_min",
                "energy_max",
                "energy_median",
                "energy_std",
            ],
            intro="Energies for generation methods and `chembl3d_gt_pb` use PoseBusters-passing conformers. `chembl3d_gt` uses all loaded ChEMBL3D conformers.",
        )
    )
    lines.extend(["## CASF16 Crystal RMSD And Hits", ""])
    lines.extend(
        render_table_section(
            "Crystal Ground Truth",
            casf_hit_df,
            [
                "source",
                "ligands_scope",
                "casf_best_rmsd",
                "casf_median_rmsd",
                "casf_hit_0p25",
                "casf_hit_0p5",
                "casf_hit_0p75",
                "casf_hit_2p0",
            ],
            intro="RMSD and hits are computed against CASF crystal poses from the ligand directory. `chembl3d_gt_pb` rows compare only PoseBusters-passing ChEMBL3D conformers.",
        )
    )
    lines.extend(["## CASF16 Optimized Ligand RMSD And Hits", ""])
    lines.extend(
        render_table_section(
            "Optimized Ground Truth",
            casf_opt_hit_df,
            [
                "source",
                "ligands_scope",
                "casf_opt_best_rmsd",
                "casf_opt_median_rmsd",
                "casf_opt_hit_0p25",
                "casf_opt_hit_0p5",
                "casf_opt_hit_0p75",
                "casf_opt_hit_2p0",
            ],
            intro="RMSD and hits are computed against optimized poses from `ligands_opt` when available for a ligand.",
        )
    )
    paths.report_md.parent.mkdir(parents=True, exist_ok=True)
    paths.report_md.write_text("\n".join(lines), encoding="utf-8")


def _table_path(paths: GeometricAnalysisPaths, name: str) -> Path:
    return paths.tables_dir / name


def has_mol2_files(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.mol2"))


def _preview_items(items: Sequence[str], limit: int = 5) -> str:
    preview = ", ".join(items[:limit])
    suffix = f" (+{len(items) - limit} more)" if len(items) > limit else ""
    return f"{preview}{suffix}"


def validate_required_generation_outputs(generation_dir: Path, mol_ids: Sequence[str]) -> None:
    missing_dirs = [method for method in METHODS if not (generation_dir / method).is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            "Missing generation method directories: " + _preview_items(missing_dirs)
        )

    missing_files: list[str] = []
    for mol_id in mol_ids:
        for method in METHODS:
            path = generation_dir / method / f"{mol_id}.sdf"
            if not path.exists():
                missing_files.append(f"{method}/{mol_id}.sdf")
    if missing_files:
        raise FileNotFoundError(
            "Missing generation SDF output(s): " + _preview_items(missing_files)
        )


def validate_metrics_completeness(
    metrics_df: pd.DataFrame,
    mol_ids: Sequence[str],
    chembl_map: pd.DataFrame,
) -> None:
    required_cols = {
        "mol_id",
        "source",
        "family",
        "stage",
        "tier",
        "pb_check_fail_counts_json",
        "pb_check_fail_rates_json",
        "casf_best_rmsd",
        "casf_median_rmsd",
        *[f"casf_hit_{threshold_tag(threshold)}" for threshold in CASF_HIT_THRESHOLDS],
        "casf_opt_best_rmsd",
        "casf_opt_median_rmsd",
        *[f"casf_opt_hit_{threshold_tag(threshold)}" for threshold in CASF_HIT_THRESHOLDS],
    }
    if metrics_df.empty:
        raise RuntimeError("No per-ligand metrics were loaded or computed.")
    missing_cols = required_cols - set(metrics_df.columns)
    if missing_cols:
        raise RuntimeError("Per-ligand metrics missing required column(s): " + ", ".join(sorted(missing_cols)))

    expected_sources = tuple(METHODS) + ("casf_crystal", "chembl3d_sdf", "chembl3d_gt", "chembl3d_gt_pb")
    observed = {
        (str(row.mol_id), str(row.source))
        for row in metrics_df[["mol_id", "source"]].itertuples(index=False)
    }
    missing_rows = [
        f"{mol_id}:{source}"
        for mol_id in mol_ids
        for source in expected_sources
        if (str(mol_id), source) not in observed
    ]
    if missing_rows:
        raise RuntimeError("Missing per-ligand metric row(s): " + _preview_items(missing_rows))

    if chembl_map.empty or "ligand_id" not in chembl_map.columns:
        return
    mapped_ids = {str(value) for value in chembl_map["ligand_id"].dropna().tolist()} & {
        str(mol_id) for mol_id in mol_ids
    }
    if not mapped_ids:
        return

    chembl_rows = metrics_df[metrics_df["source"] == "chembl3d_gt"].copy()
    bad_chembl: list[str] = []
    for mol_id in sorted(mapped_ids):
        frame = chembl_rows[chembl_rows["mol_id"].astype(str) == mol_id]
        if frame.empty:
            bad_chembl.append(f"{mol_id}:missing")
            continue
        conformer_count = pd.to_numeric(frame["conformer_count"], errors="coerce").max()
        status = str(frame["chembl_load_status"].iloc[0]) if "chembl_load_status" in frame else "unknown"
        if not math.isfinite(float(conformer_count)) or float(conformer_count) <= 0 or status != "ok":
            bad_chembl.append(f"{mol_id}:status={status},confs={fmt(conformer_count, 'conformer_count')}")
    if bad_chembl:
        raise RuntimeError("Mapped ChEMBL3D ligand(s) did not load conformers: " + _preview_items(bad_chembl))


def run(
    paths: GeometricAnalysisPaths,
    casf_opt_ligand_dir: Path,
    chembl_map_path: Path,
    chembl_dataset_root: Path,
    limit_molecules: int | None,
    molecule_offset: int,
    skip_geometry: bool,
    geometry_only: bool,
    report_only: bool,
    use_posebusters: bool,
    workers: int | None,
    posebusters_workers: int | None,
    posebusters_energy_threads: int | None,
) -> None:
    del skip_geometry, use_posebusters  # The revised analysis always computes the requested metrics.
    manifest_df = load_manifest(paths)
    chembl_map = load_chembl_map(chembl_map_path)
    mol_ids, excluded = select_analysis_mol_ids(chembl_map, paths.generation_dir, manifest_df)
    _log_excluded_ligands(excluded)
    if molecule_offset:
        mol_ids = mol_ids[molecule_offset:]
    if limit_molecules is not None:
        mol_ids = mol_ids[:limit_molecules]
    if not mol_ids:
        raise SystemExit("No eligible molecules found for analysis after filtering")
    print(f"Selected {len(mol_ids)} ligand(s) for analysis", flush=True)
    validate_required_generation_outputs(paths.generation_dir, mol_ids)

    selected_manifest = manifest_df[manifest_df["mol_id"].isin(mol_ids)].copy()
    topology_root = chembl_dataset_root / "topologies"
    zarr_root = chembl_dataset_root / "zarr_database"

    metrics_df, energy_values_df = load_or_compute_ligand_metrics(
        paths,
        casf_opt_ligand_dir,
        mol_ids,
        selected_manifest,
        chembl_map,
        topology_root,
        zarr_root,
        report_only=report_only,
        workers=workers,
        posebusters_workers=posebusters_workers,
        posebusters_energy_threads=posebusters_energy_threads,
    )

    validate_metrics_completeness(metrics_df, mol_ids, chembl_map)
    metrics_df = sort_and_finalize_table(metrics_df, extra_cols=("mol_id",))
    energy_values_df = sort_and_finalize_table(energy_values_df, extra_cols=("mol_id", "conformer_index"))

    if geometry_only:
        print(f"Per-ligand analysis parts written for {len(mol_ids)} ligand(s)")
        return

    generation_df = build_generation_filter_summary(selected_manifest)
    generation_pb_fail_df = build_generation_pb_failure_summary(selected_manifest, generation_df)
    reference_df = build_reference_filter_summary(metrics_df)
    reference_pb_fail_df = build_reference_pb_failure_summary(metrics_df, reference_df)
    cluster_df = build_cluster_summary(metrics_df)
    energy_df = build_energy_summary(energy_values_df)
    casf_hit_df = build_casf_hit_summary(metrics_df)
    casf_opt_hit_df = build_casf_opt_hit_summary(metrics_df)

    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    generation_df.to_csv(_table_path(paths, "geometric_generation_filter_summary.csv"), index=False)
    generation_pb_fail_df.to_csv(_table_path(paths, "geometric_generation_pb_failure_summary.csv"), index=False)
    reference_df.to_csv(_table_path(paths, "geometric_reference_filter_summary.csv"), index=False)
    reference_pb_fail_df.to_csv(_table_path(paths, "geometric_reference_pb_failure_summary.csv"), index=False)
    cluster_df.to_csv(_table_path(paths, "geometric_cluster_summary.csv"), index=False)
    energy_df.to_csv(_table_path(paths, "geometric_energy_summary.csv"), index=False)
    casf_hit_df.to_csv(_table_path(paths, "geometric_casf_hit_summary.csv"), index=False)
    casf_opt_hit_df.to_csv(_table_path(paths, "geometric_casf_opt_hit_summary.csv"), index=False)
    metrics_df.to_csv(_table_path(paths, "geometric_per_ligand_metrics.csv"), index=False)

    n_with_chembl = len(
        [mol_id for mol_id in mol_ids if not chembl_map.empty and mol_id in set(chembl_map["ligand_id"])]
    )
    write_report(
        paths,
        mol_ids=mol_ids,
        n_with_chembl=n_with_chembl,
        casf_opt_ligand_dir=casf_opt_ligand_dir,
        generation_df=generation_df,
        generation_pb_fail_df=generation_pb_fail_df,
        reference_df=reference_df,
        reference_pb_fail_df=reference_pb_fail_df,
        cluster_df=cluster_df,
        energy_df=energy_df,
        casf_hit_df=casf_hit_df,
        casf_opt_hit_df=casf_opt_hit_df,
    )
    print(f"Analyzed {len(mol_ids)} ligands ({n_with_chembl} with ChEMBL3D map)")
    print(f"Report: {paths.report_md}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_CORE_INTERSECTION_ROOT)
    parser.add_argument("--casf-ligand-dir", type=Path, default=None)
    parser.add_argument("--casf-opt-ligand-dir", type=Path, default=DEFAULT_CASF_OPT_LIGAND_DIR)
    parser.add_argument("--chembl-map-csv", type=Path, default=DEFAULT_CHEMBL_MAP_CSV)
    parser.add_argument("--chembl-dataset-root", type=Path, default=DEFAULT_CHEMBL_DATASET_ROOT)
    parser.add_argument("--limit-molecules", type=int, default=None)
    parser.add_argument("--molecule-offset", type=int, default=0)
    parser.add_argument("--skip-geometry", action="store_true", help="Deprecated; ignored by revised analysis.")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--use-posebusters", action="store_true", help="Deprecated; reference PB always runs.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel ligand workers (default: all CPU cores; use 1 for serial)",
    )
    parser.add_argument(
        "--posebusters-workers",
        type=int,
        default=0,
        help=(
            "PoseBusters workers per ligand job. Default auto splits available CPUs "
            "across ligand workers (all CPUs when --workers 1)."
        ),
    )
    parser.add_argument(
        "--posebusters-energy-threads",
        type=int,
        default=0,
        help=(
            "Energy-ratio threads per PoseBusters job. Default auto uses 1 when "
            "PoseBusters workers > 1, otherwise splits CPUs across ligand workers."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    casf_ligand_dir = args.casf_ligand_dir
    if casf_ligand_dir is None:
        casf_ligand_dir = DEFAULT_CASF16_DATA / "core_chembl3d_exact_intersection_ligands"
        if not has_mol2_files(casf_ligand_dir):
            casf_ligand_dir = args.output_dir.parent / "core_chembl3d_exact_intersection_ligands"
        if not has_mol2_files(casf_ligand_dir):
            casf_ligand_dir = DEFAULT_CASF_LIGAND_DIR
    paths = resolve_geometric_paths(args.output_dir.resolve(), casf_ligand_dir.resolve())
    run(
        paths=paths,
        casf_opt_ligand_dir=args.casf_opt_ligand_dir.resolve(),
        chembl_map_path=args.chembl_map_csv.resolve(),
        chembl_dataset_root=args.chembl_dataset_root.resolve(),
        limit_molecules=args.limit_molecules,
        molecule_offset=args.molecule_offset,
        skip_geometry=args.skip_geometry,
        geometry_only=args.geometry_only,
        report_only=args.report_only,
        use_posebusters=args.use_posebusters,
        workers=None if args.workers == 0 else args.workers,
        posebusters_workers=None if args.posebusters_workers == 0 else args.posebusters_workers,
        posebusters_energy_threads=None
        if args.posebusters_energy_threads == 0
        else args.posebusters_energy_threads,
    )


if __name__ == "__main__":
    main()
