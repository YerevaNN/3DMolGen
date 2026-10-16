#!/usr/bin/env python3
"""Rescore saved PoseBench poses without superposing the crystal ligand.

This is a corrective audit for the PoseBench v0.6.0 primary-ligand-selection
side effect.  It reads the already generated original and CoordToken-refined
SDFs, selects predicted fragments using defensive molecule copies, and uses
RDKit CalcRMS for the reported pose RMSD.  It does not rerun CoordToken.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator, rdMolAlign


WORKSPACE = Path("/mnt/weka/fgeikyan/3dmolgen_bigdata")
COORDTOKEN = Path("/mnt/weka/fgeikyan/CoordToken")
DIFFDOCK_ROOT = COORDTOKEN / "posebench_diffdock_no_postprocessing_reproduction"
VINA_ROOT = COORDTOKEN / "posebench_vina_no_postprocessing_reproduction"
REMAINING_ROOT = COORDTOKEN / "posebench_remaining_methods_no_postprocessing_reproduction"

CHEMICAL_CHECKS = (
    "sanitization",
    "all_atoms_connected",
    "molecular_formula",
    "molecular_bonds",
    "double_bond_stereochemistry",
    "tetrahedral_chirality",
)
INTRAMOLECULAR_CHECKS = (
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "double_bond_flatness",
    "internal_energy",
)
LIGAND_ONLY_CHECKS = CHEMICAL_CHECKS + INTRAMOLECULAR_CHECKS


@dataclass(frozen=True)
class MethodSpec:
    slug: str
    label: str
    root: Path
    original_dir: Path
    refined_dir: Path
    repeats: tuple[int, ...] = (1, 2, 3)


METHODS = (
    MethodSpec(
        "neuralplexer",
        "NeuralPLexer",
        REMAINING_ROOT,
        REMAINING_ROOT / "outputs/neuralplexer/coordinate_rerun",
        REMAINING_ROOT / "outputs/neuralplexer/coordtoken_evaluation",
    ),
    MethodSpec(
        "rfaa",
        "RFAA",
        REMAINING_ROOT,
        REMAINING_ROOT / "outputs/rfaa/coordinate_rerun",
        REMAINING_ROOT / "outputs/rfaa/coordtoken_evaluation",
        (1,),
    ),
    MethodSpec(
        "dynamicbind",
        "DynamicBind",
        REMAINING_ROOT,
        REMAINING_ROOT / "outputs/dynamicbind/coordinate_rerun",
        REMAINING_ROOT / "outputs/dynamicbind/coordtoken_evaluation",
    ),
    MethodSpec(
        "diffdock",
        "DiffDock",
        DIFFDOCK_ROOT,
        DIFFDOCK_ROOT / "outputs/coordinate_rerun",
        DIFFDOCK_ROOT / "outputs/coordtoken_coordinate_evaluation",
    ),
    MethodSpec(
        "af3_ss",
        "AF3-Single-Seq",
        REMAINING_ROOT,
        REMAINING_ROOT / "outputs/af3_ss/coordinate_rerun",
        REMAINING_ROOT / "outputs/af3_ss/coordtoken_evaluation",
    ),
    MethodSpec(
        "vina",
        "AutoDock Vina",
        VINA_ROOT,
        VINA_ROOT / "outputs/coordinate_rerun",
        VINA_ROOT / "outputs/coordtoken_coordinate_evaluation",
    ),
    MethodSpec(
        "chai",
        "Chai-1",
        REMAINING_ROOT,
        REMAINING_ROOT / "outputs/chai/coordinate_rerun",
        REMAINING_ROOT / "outputs/chai/coordtoken_evaluation",
    ),
    MethodSpec(
        "af3",
        "AF3",
        REMAINING_ROOT,
        REMAINING_ROOT / "outputs/af3/coordinate_rerun",
        REMAINING_ROOT / "outputs/af3/coordtoken_evaluation",
    ),
    MethodSpec(
        "chai_ss",
        "Chai-1-Single-Seq",
        REMAINING_ROOT,
        REMAINING_ROOT / "outputs/chai_ss/coordinate_rerun",
        REMAINING_ROOT / "outputs/chai_ss/coordtoken_evaluation",
    ),
)


def parse_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "1.0"}


def ligand_validity(frame: pd.DataFrame) -> dict[str, bool]:
    missing = sorted(set(("mol_id", *LIGAND_ONLY_CHECKS)) - set(frame.columns))
    if missing:
        raise ValueError(f"ligand-validity input lacks columns: {missing}")
    if frame["mol_id"].duplicated().any():
        raise ValueError("duplicate mol_id in ligand-validity input")
    return {
        str(row.mol_id): all(parse_bool(getattr(row, check)) for check in LIGAND_ONLY_CHECKS)
        for row in frame.itertuples(index=False)
    }


def numeric_by_id(frame: pd.DataFrame, column: str) -> dict[str, float]:
    if column not in frame.columns:
        return {}
    values = pd.to_numeric(frame[column], errors="coerce")
    return dict(zip(frame["mol_id"].astype(str), values))


def resolve_path(value: Any, base: Path) -> Path | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    raw = Path(str(value))
    candidates = (
        [raw]
        if raw.is_absolute()
        else [
            base / raw,
            DIFFDOCK_ROOT / raw,
            VINA_ROOT / raw,
            REMAINING_ROOT / raw,
            COORDTOKEN / raw,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def coordinates(mol: Chem.Mol) -> np.ndarray:
    if mol.GetNumConformers() != 1:
        raise ValueError(f"expected one conformer, found {mol.GetNumConformers()}")
    return np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64).copy()


def heavy_copy(mol: Chem.Mol) -> Chem.Mol:
    # The released method outputs include some chemically invalid structures.
    # Removing H without sanitizing lets those rows reach RMSD scoring; their
    # ligand-validity flag remains false independently.
    return Chem.RemoveHs(Chem.Mol(mol), sanitize=False)


def centroid(mol: Chem.Mol) -> np.ndarray:
    return coordinates(heavy_copy(mol)).mean(axis=0)


def select_primary_fragment(
    mol_true_fragment: Chem.Mol,
    mol_pred_fragments: tuple[Chem.Mol, ...],
) -> Chem.Mol | None:
    """Apply the v0.6.0 selection rule without mutating either input."""
    true_before = coordinates(mol_true_fragment)
    pred_before = [coordinates(fragment) for fragment in mol_pred_fragments]
    mol_true_fragment.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol_true_fragment)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    true_fingerprint = generator.GetFingerprint(mol_true_fragment)
    best_index: int | None = None
    max_similarity = -1.0
    min_rmsd = float("inf")

    for index, fragment in enumerate(mol_pred_fragments):
        if fragment.GetNumAtoms() != mol_true_fragment.GetNumAtoms():
            continue
        candidate = Chem.Mol(fragment)
        candidate.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(candidate)
        similarity = DataStructs.TanimotoSimilarity(
            true_fingerprint, generator.GetFingerprint(candidate)
        )
        aligned_rmsd = float("inf")
        if similarity > 0.5:
            try:
                aligned_rmsd = rdMolAlign.GetBestRMS(
                    Chem.Mol(mol_true_fragment),
                    candidate,
                    symmetrizeConjugatedTerminalGroups=True,
                )
            except (RuntimeError, ValueError):
                pass
        if similarity > max_similarity or (
            similarity == max_similarity and aligned_rmsd < min_rmsd
        ):
            best_index = index
            max_similarity = similarity
            min_rmsd = aligned_rmsd

    if not np.array_equal(coordinates(mol_true_fragment), true_before):
        raise AssertionError("fragment selection mutated reference coordinates")
    for fragment, before in zip(mol_pred_fragments, pred_before):
        if not np.array_equal(coordinates(fragment), before):
            raise AssertionError("fragment selection mutated prediction coordinates")
    return None if best_index is None else mol_pred_fragments[best_index]


def restore_centroid(mol: Chem.Mol, target: np.ndarray) -> tuple[Chem.Mol, float]:
    """Rigidly translate a decoded ligand back to its input heavy-atom centroid."""
    restored = Chem.Mol(mol)
    before = centroid(restored)
    shift = np.asarray(target, dtype=np.float64) - before
    conformer = restored.GetConformer()
    for atom_index in range(restored.GetNumAtoms()):
        point = np.asarray(conformer.GetAtomPosition(atom_index), dtype=np.float64)
        conformer.SetAtomPosition(atom_index, point + shift)
    return restored, float(np.linalg.norm(shift))


def load_primary_pair(mol_true_path: Path, mol_pred_path: Path) -> tuple[Chem.Mol, Chem.Mol]:
    mol_true = Chem.MolFromMolFile(str(mol_true_path), removeHs=False)
    mol_pred = Chem.MolFromMolFile(str(mol_pred_path), removeHs=False)
    if mol_true is None or mol_pred is None:
        raise ValueError("RDKit failed to load reference or prediction")
    true_before = coordinates(mol_true)
    true_fragments = Chem.GetMolFrags(mol_true, asMols=True, sanitizeFrags=False)
    pred_fragments = Chem.GetMolFrags(mol_pred, asMols=True, sanitizeFrags=False)
    if len(true_fragments) != 1:
        raise ValueError(f"expected one crystal-ligand fragment, found {len(true_fragments)}")
    selected = select_primary_fragment(true_fragments[0], pred_fragments)
    if selected is None:
        raise ValueError("no predicted fragment with the crystal atom count")
    if not np.array_equal(coordinates(mol_true), true_before):
        raise AssertionError("primary-ligand loading mutated reference coordinates")
    return true_fragments[0], selected


def score_pair(mol_true: Chem.Mol, mol_pred: Chem.Mol) -> dict[str, Any]:
    true_before = coordinates(mol_true)
    true_heavy = heavy_copy(mol_true)
    pred_heavy = heavy_copy(mol_pred)
    true_centroid = centroid(true_heavy)
    pred_centroid = centroid(pred_heavy)
    try:
        rmsd = float(
            rdMolAlign.CalcRMS(
                Chem.Mol(true_heavy),
                Chem.Mol(pred_heavy),
                symmetrizeConjugatedTerminalGroups=True,
            )
        )
    except (RuntimeError, ValueError):
        rmsd = float("nan")
    try:
        kabsch_rmsd = float(
            rdMolAlign.GetBestRMS(
                Chem.Mol(true_heavy),
                Chem.Mol(pred_heavy),
                symmetrizeConjugatedTerminalGroups=True,
            )
        )
    except (RuntimeError, ValueError):
        kabsch_rmsd = float("nan")
    if not np.array_equal(coordinates(mol_true), true_before):
        raise AssertionError("RMSD scoring mutated reference coordinates")
    return {
        "rmsd": rmsd,
        "kabsch_rmsd": kabsch_rmsd,
        "centroid_distance": float(np.linalg.norm(true_centroid - pred_centroid)),
        "pred_centroid_x": float(pred_centroid[0]),
        "pred_centroid_y": float(pred_centroid[1]),
        "pred_centroid_z": float(pred_centroid[2]),
        "reference_unchanged": True,
    }


def score_task(task: dict[str, Any]) -> dict[str, Any]:
    result = {key: task[key] for key in ("method", "label", "repeat", "mol_id")}
    original_path = Path(task["original_path"]) if task.get("original_path") else None
    refined_path = Path(task["refined_path"]) if task.get("refined_path") else None
    true_path = Path(task["true_path"]) if task.get("true_path") else None
    result.update(
        original_path=str(original_path or ""),
        refined_path=str(refined_path or ""),
        true_path=str(true_path or ""),
        refined_fallback_to_original=False,
    )
    if true_path is None or not true_path.is_file() or original_path is None or not original_path.is_file():
        result["status"] = "missing_original_or_reference"
        return result
    try:
        true, original = load_primary_pair(true_path, original_path)
        for key, value in score_pair(true, original).items():
            result[f"original_{key}"] = value
        effective_refined_path = refined_path
        if effective_refined_path is None or not effective_refined_path.is_file():
            effective_refined_path = original_path
            result["refined_fallback_to_original"] = True
        true_refined, refined_raw = load_primary_pair(true_path, effective_refined_path)
        original_centroid = centroid(original)
        refined, pre_restore_delta = restore_centroid(refined_raw, original_centroid)
        result["refined_centroid_correction"] = pre_restore_delta
        for key, value in score_pair(true_refined, refined).items():
            result[f"refined_{key}"] = value
        scored_original_centroid = np.array(
            [result[f"original_pred_centroid_{axis}"] for axis in "xyz"]
        )
        scored_refined_centroid = np.array(
            [result[f"refined_pred_centroid_{axis}"] for axis in "xyz"]
        )
        result["original_refined_centroid_delta"] = float(
            np.linalg.norm(scored_original_centroid - scored_refined_centroid)
        )
        result["status"] = "success"
    except Exception as exc:  # preserve every failure for the audit table
        result["status"] = "failed"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"[:500]
    return result


def load_tasks(spec: MethodSpec) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for repeat in spec.repeats:
        original_paths_file = spec.original_dir / f"repeat_{repeat}_paths.csv"
        refined_paths_file = spec.refined_dir / f"repeat_{repeat}_paths.csv"
        original_eval_file = spec.original_dir / f"repeat_{repeat}.csv"
        refined_eval_file = spec.refined_dir / f"repeat_{repeat}.csv"
        original_paths = pd.read_csv(original_paths_file)
        refined_paths = pd.read_csv(refined_paths_file)
        original_eval = pd.read_csv(original_eval_file)
        refined_eval = pd.read_csv(refined_eval_file)
        original_valid = ligand_validity(original_eval)
        refined_valid = ligand_validity(refined_eval)
        original_recorded_kabsch = numeric_by_id(original_eval, "kabsch_rmsd")
        refined_recorded_kabsch = numeric_by_id(refined_eval, "kabsch_rmsd")
        original_evaluated = set(original_eval["mol_id"].astype(str))
        refined_evaluated = set(refined_eval["mol_id"].astype(str))
        if original_paths["mol_id"].duplicated().any() or refined_paths["mol_id"].duplicated().any():
            raise ValueError(f"{spec.slug} repeat {repeat}: duplicate path rows")
        original_by_id = original_paths.set_index("mol_id", drop=False)
        refined_by_id = refined_paths.set_index("mol_id", drop=False)
        ids = sorted(set(original_by_id.index) | set(refined_by_id.index))
        for mol_id in ids:
            original_row = original_by_id.loc[mol_id] if mol_id in original_by_id.index else None
            refined_row = refined_by_id.loc[mol_id] if mol_id in refined_by_id.index else None
            original_path = (
                resolve_path(original_row.mol_pred, spec.root)
                if original_row is not None and str(mol_id) in original_evaluated
                else None
            )
            refined_path = (
                resolve_path(refined_row.mol_pred, spec.root)
                if refined_row is not None and str(mol_id) in refined_evaluated
                else None
            )
            true_value = (
                original_row.mol_true
                if original_row is not None
                else refined_row.mol_true
            )
            tasks.append(
                {
                    "method": spec.slug,
                    "label": spec.label,
                    "repeat": repeat,
                    "mol_id": str(mol_id),
                    "original_path": str(original_path) if original_path else "",
                    "refined_path": str(refined_path) if refined_path else "",
                    "true_path": str(resolve_path(true_value, spec.root) or ""),
                    "original_ligand_valid": original_valid.get(str(mol_id), False),
                    "refined_ligand_valid": refined_valid.get(
                        str(mol_id), original_valid.get(str(mol_id), False)
                    ),
                    "original_recorded_kabsch_rmsd": original_recorded_kabsch.get(
                        str(mol_id), float("nan")
                    ),
                    "refined_recorded_kabsch_rmsd": refined_recorded_kabsch.get(
                        str(mol_id), original_recorded_kabsch.get(str(mol_id), float("nan"))
                    ),
                }
            )
    return tasks


def add_endpoints(frame: pd.DataFrame, task_frame: pd.DataFrame) -> pd.DataFrame:
    frame = task_frame.merge(
        frame,
        on=["method", "label", "repeat", "mol_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_scored"),
    )
    for condition in ("original", "refined"):
        frame[f"{condition}_rmsd_le_2"] = (
            pd.to_numeric(frame[f"{condition}_rmsd"], errors="coerce") <= 2.0
        )
        frame[f"{condition}_joint_success"] = (
            frame[f"{condition}_rmsd_le_2"]
            & frame[f"{condition}_ligand_valid"].map(parse_bool)
        )
    original_valid = frame["original_ligand_valid"].map(parse_bool)
    frame["gated_ligand_valid"] = np.where(
        original_valid,
        True,
        frame["refined_ligand_valid"].map(parse_bool),
    )
    frame["gated_rmsd"] = np.where(
        original_valid, frame["original_rmsd"], frame["refined_rmsd"]
    )
    frame["gated_kabsch_rmsd"] = np.where(
        original_valid,
        frame["original_kabsch_rmsd"],
        frame["refined_kabsch_rmsd"],
    )
    frame["gated_rmsd_le_2"] = pd.to_numeric(
        frame["gated_rmsd"], errors="coerce"
    ) <= 2.0
    frame["gated_joint_success"] = (
        frame["gated_rmsd_le_2"] & frame["gated_ligand_valid"]
    )
    return frame


def make_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in METHODS:
        group = frame[frame["method"] == spec.slug]
        row: dict[str, Any] = {
            "method": spec.label,
            "requested_denominator": 390,
            "available_original_predictions": int(
                group["original_path"]
                .map(lambda path: Path(path).is_file() if path else False)
                .sum()
            ),
            "corrected_rmsd_available": int(pd.to_numeric(group["original_rmsd"], errors="coerce").notna().sum()),
            "refinement_fallbacks": int(group["refined_fallback_to_original"].map(parse_bool).sum()),
        }
        for condition in ("original", "refined", "gated"):
            joint = int(group[f"{condition}_joint_success"].fillna(False).sum())
            valid = int(group[f"{condition}_ligand_valid"].map(parse_bool).sum())
            rmsd = int(group[f"{condition}_rmsd_le_2"].fillna(False).sum())
            row[f"{condition}_rmsd_le_2_n"] = rmsd
            row[f"{condition}_rmsd_le_2_pct"] = 100.0 * rmsd / 390
            row[f"{condition}_joint_n"] = joint
            row[f"{condition}_joint_pct"] = 100.0 * joint / 390
            row[f"{condition}_ligand_valid_n"] = valid
            row[f"{condition}_ligand_valid_pct"] = 100.0 * valid / 390
        rows.append(row)
    return pd.DataFrame(rows)


def run_regressions(frame: pd.DataFrame) -> dict[str, Any]:
    successful = frame[frame["status"] == "success"].copy()
    if successful.empty:
        raise AssertionError("no poses were rescored")
    if not successful["original_reference_unchanged"].map(parse_bool).all():
        raise AssertionError("an original reference was mutated")
    if not successful["refined_reference_unchanged"].map(parse_bool).all():
        raise AssertionError("a refined-pose reference was mutated")

    centroid_delta = pd.to_numeric(
        successful["original_refined_centroid_delta"], errors="coerce"
    )
    centroid_tolerance = 1e-9
    if centroid_delta.max() > centroid_tolerance:
        offenders = successful.loc[
            centroid_delta > centroid_tolerance,
            ["method", "repeat", "mol_id", "original_refined_centroid_delta"],
        ]
        raise AssertionError(
            "saved refined centroids do not match input-pose centroids:\n"
            + offenders.head(20).to_string(index=False)
        )

    finite = successful[
        pd.to_numeric(successful["original_rmsd"], errors="coerce").notna()
        & pd.to_numeric(successful["original_kabsch_rmsd"], errors="coerce").notna()
    ]
    equal = np.isclose(
        finite["original_rmsd"].astype(float),
        finite["original_kabsch_rmsd"].astype(float),
        atol=1e-6,
        rtol=1e-6,
    )
    equality_fraction = float(equal.mean())
    if equality_fraction >= 0.95:
        raise AssertionError(
            f"CalcRMS and GetBestRMS remain nearly universal matches: {equality_fraction:.3%}"
        )
    if (finite["original_rmsd"] + 1e-6 < finite["original_kabsch_rmsd"]).any():
        raise AssertionError("CalcRMS is unexpectedly smaller than GetBestRMS")

    kabsch_differences = []
    for condition in ("original", "refined"):
        observed = pd.to_numeric(
            successful[f"{condition}_kabsch_rmsd"], errors="coerce"
        )
        recorded = pd.to_numeric(
            successful[f"{condition}_recorded_kabsch_rmsd"], errors="coerce"
        )
        comparable = observed.notna() & recorded.notna()
        kabsch_differences.extend((observed[comparable] - recorded[comparable]).abs())
    maximum_kabsch_difference = float(max(kabsch_differences))
    if maximum_kabsch_difference > 5e-5:
        raise AssertionError(
            "recomputed Kabsch RMSD does not reproduce the saved scorer values: "
            f"max difference {maximum_kabsch_difference:.8f} A"
        )

    known = successful[
        (successful["method"] == "diffdock")
        & (successful["repeat"] == 1)
        & (successful["mol_id"] == "7WUY_76N")
    ]
    if len(known) != 1:
        raise AssertionError("known DiffDock 7WUY_76N repeat-1 example is missing")
    known_row = known.iloc[0]
    expected_centroid_distance = 32.17748946247586
    expected_calc_rms = 32.61122115049256
    if not math.isclose(
        float(known_row.original_centroid_distance),
        expected_centroid_distance,
        rel_tol=0,
        abs_tol=1e-6,
    ):
        raise AssertionError("known DiffDock centroid-distance regression failed")
    if not math.isclose(
        float(known_row.original_rmsd), expected_calc_rms, rel_tol=0, abs_tol=1e-6
    ):
        raise AssertionError("known DiffDock CalcRMS regression failed")

    return {
        "reference_coordinates_unchanged": True,
        "reference_checks": int(len(successful) * 2),
        "centroid_match_tolerance_angstrom": centroid_tolerance,
        "maximum_original_refined_centroid_delta_angstrom": float(centroid_delta.max()),
        "rmsd_kabsch_finite_pairs": int(len(finite)),
        "rmsd_kabsch_equal_within_1e-6": int(equal.sum()),
        "rmsd_kabsch_equality_fraction": equality_fraction,
        "calc_rms_not_less_than_kabsch": True,
        "maximum_difference_from_recorded_kabsch_angstrom": maximum_kabsch_difference,
        "rdkit_version": rdBase.rdkitVersion,
        "known_diffdock_example": {
            "repeat": 1,
            "mol_id": "7WUY_76N",
            "centroid_distance_angstrom": float(known_row.original_centroid_distance),
            "calc_rms_angstrom": float(known_row.original_rmsd),
            "kabsch_rmsd_angstrom": float(known_row.original_kabsch_rmsd),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/posebench_rmsd_rescoring",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--methods",
        nargs="*",
        choices=[spec.slug for spec in METHODS],
        help="Optional subset for development; a full run is required for the paper table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [spec for spec in METHODS if not args.methods or spec.slug in args.methods]
    tasks = [task for spec in selected for task in load_tasks(spec)]
    print(f"Rescoring {len(tasks)} requested method/repeat/complex rows; no refinement is run")
    if args.workers == 1:
        scored = [score_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            scored = list(executor.map(score_task, tasks, chunksize=4))
    task_frame = pd.DataFrame(tasks)
    frame = add_endpoints(pd.DataFrame(scored), task_frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "corrected_per_pose.csv", index=False)
    failures = frame[~frame["status"].isin(["success", "missing_original_or_reference"])]
    failures.to_csv(args.output_dir / "scoring_failures.csv", index=False)
    unavailable = frame[
        frame["original_path"].map(bool)
        & pd.to_numeric(frame["original_rmsd"], errors="coerce").isna()
    ]
    unavailable.to_csv(args.output_dir / "rmsd_unavailable.csv", index=False)
    if args.methods:
        print(frame.groupby(["method", "status"]).size().to_string())
        return
    summary = make_summary(frame)
    regressions = run_regressions(frame)
    summary.to_csv(args.output_dir / "corrected_summary.csv", index=False)
    (args.output_dir / "regression_checks.json").write_text(
        json.dumps(regressions, indent=2) + "\n"
    )
    print(summary.to_string(index=False))
    print(json.dumps(regressions, indent=2))


if __name__ == "__main__":
    main()
