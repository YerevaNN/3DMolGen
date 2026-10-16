#!/usr/bin/env python3
"""Read-only forensic audit of the PoseBench 375 denominator.

This script never modifies source results.  It reads the official metadata,
reference tree, raw method outputs, rerun manifests/results, and aggregation
scripts, then writes diagnostic artifacts beside this file (or --output-dir).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


DIFF = Path("/mnt/weka/fgeikyan/CoordToken/posebench_diffdock_no_postprocessing_reproduction")
VINA = Path("/mnt/weka/fgeikyan/CoordToken/posebench_vina_no_postprocessing_reproduction")
RELEASE = DIFF / "releases/14629652/data/extracted"
INPUT = RELEASE / "forks/DiffDock/inference/diffdock_posebusters_benchmark_inputs.csv"
VINA_INPUT = VINA / "releases/14629652/data/extracted/forks/Vina/inference/vina_posebusters_benchmark_inputs.csv"
DATES = RELEASE / "notebooks/posebusters_benchmark_complex_pdb_deposition_dates.csv"
PAPER_IDS = DIFF / "posebench-source/data/posebusters_pdb_ccd_ids.txt"
DATASET = RELEASE / "data/posebusters_benchmark_set"
CUTOFF = pd.Timestamp("2021-09-30", tz="UTC")
PLOT_DENOMINATOR = 125

METHODS = {
    "DiffDock": {
        "root": DIFF,
        "inference": RELEASE / "forks/DiffDock/inference",
        "dir_template": "diffdock_posebusters_benchmark_output_{repeat}",
        "filename_template": "rank1.sdf",
    },
    "Vina": {
        "root": VINA,
        "inference": VINA / "releases/14629652/data/extracted/forks/Vina/inference",
        "dir_template": "vina_p2rank_posebusters_benchmark_outputs_{repeat}",
        "filename_template": "{complex_id}.sdf",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def first_sdf_molecule(path: Path):
    if not path.is_file():
        return None
    try:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        return supplier[0] if len(supplier) else None
    except Exception:
        return None


def expected_prediction_path(method: str, repeat: int, complex_id: str) -> Path:
    cfg = METHODS[method]
    directory = cfg["inference"] / cfg["dir_template"].format(repeat=repeat)
    filename = cfg["filename_template"].format(complex_id=complex_id)
    return directory / complex_id / filename


def dataframe_stats(df: pd.DataFrame, file_column: str | None = None) -> dict:
    result = {
        "rows": int(len(df)),
        "unique_complex_ids": int(df["complex_id"].nunique()) if "complex_id" in df else None,
        "unique_pdb_ids": int(df["pdb_id"].nunique()) if "pdb_id" in df else None,
        "unique_ligand_ids": int(df["ligand_id"].nunique()) if "ligand_id" in df else None,
        "unique_complex_ligand_pairs": (
            int(df[["complex_id", "ligand_id"]].drop_duplicates().shape[0])
            if {"complex_id", "ligand_id"}.issubset(df.columns)
            else None
        ),
        "duplicate_complex_ids": (
            int(df.duplicated("complex_id").sum()) if "complex_id" in df else None
        ),
    }
    if file_column and file_column in df:
        result["unique_prediction_files"] = int(df[file_column].nunique())
        result["duplicate_prediction_paths"] = int(df.duplicated(file_column).sum())
    return result


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    official = pd.read_csv(INPUT).rename(columns={"complex_name": "complex_id"})
    vina_input = pd.read_csv(VINA_INPUT).rename(columns={"complex_name": "complex_id"})
    official["pdb_id"] = official["complex_id"].str.split("_").str[0].str.upper()
    official["ligand_id"] = official["complex_id"].str.split("_", n=1).str[1].str.upper()
    vina_input["pdb_id"] = vina_input["complex_id"].str.split("_").str[0].str.upper()
    vina_input["ligand_id"] = vina_input["complex_id"].str.split("_", n=1).str[1].str.upper()

    paper_ids = [line.strip().upper() for line in PAPER_IDS.read_text().splitlines() if line.strip()]
    dates = pd.read_csv(DATES)
    dates["pdb_id"] = dates["PDB ID"].str.upper()
    dates["deposition_date"] = pd.to_datetime(dates["Deposition Date"], utc=True)
    postcutoff_pdbs = set(dates.loc[dates["deposition_date"] > CUTOFF, "pdb_id"])
    postcutoff = official[official["pdb_id"].isin(postcutoff_pdbs)].copy()

    # Assertions deliberately detect duplicate keys and mismatched metadata joins.
    assert len(official) == 308
    assert official["complex_id"].is_unique
    assert official["pdb_id"].is_unique
    assert official[["complex_id", "ligand_id"]].duplicated().sum() == 0
    assert len(paper_ids) == len(set(paper_ids)) == 308
    assert set(paper_ids) == set(official["complex_id"].str.upper())
    assert set(vina_input["complex_id"].str.upper()) == set(official["complex_id"].str.upper())
    assert len(postcutoff_pdbs) == len(postcutoff) == 130
    assert postcutoff["complex_id"].is_unique
    assert len(official.merge(dates[["pdb_id"]], on="pdb_id", how="left")) == len(official)
    assert dates["pdb_id"].is_unique, "deposition-date join would be one-to-many"

    reference_rows = []
    for row in official.itertuples(index=False):
        ligand = DATASET / row.complex_id / f"{row.complex_id}_ligand.sdf"
        protein = DATASET / row.complex_id / f"{row.complex_id}_protein.pdb"
        mol = first_sdf_molecule(ligand)
        reference_rows.append(
            {
                "complex_id": row.complex_id,
                "pdb_id": row.pdb_id,
                "ligand_id": row.ligand_id,
                "reference_ligand": str(ligand),
                "reference_ligand_exists": ligand.is_file(),
                "reference_ligand_parsed": mol is not None,
                "reference_fragment_count": len(Chem.GetMolFrags(mol)) if mol is not None else 0,
                "reference_protein": str(protein),
                "reference_protein_exists": protein.is_file(),
            }
        )
    references = pd.DataFrame(reference_rows)
    assert references["reference_ligand_exists"].all()
    assert references["reference_ligand_parsed"].all()

    count_summary: dict = {
        "denominator_finding": {
            "object_counted": "common plotting denominator slots across stochastic repeats",
            "per_repeat_plot_denominator": PLOT_DENOMINATOR,
            "number_of_repeats": 3,
            "reported_denominator": PLOT_DENOMINATOR * 3,
            "is_count_of_distinct_records": False,
            "official_full_entries": len(official),
            "postcutoff_entries": len(postcutoff),
            "arithmetic_difference_375_minus_308": 67,
            "actual_extra_record_count": 0,
        },
        "official_metadata": {
            **dataframe_stats(official),
            "paper_id_lines": len(paper_ids),
            "paper_id_unique": len(set(paper_ids)),
            "paper_ids_equal_input_ids": set(paper_ids) == set(official["complex_id"].str.upper()),
            "deposition_date_rows": len(dates),
            "deposition_date_unique_pdb_ids": dates["pdb_id"].nunique(),
            "official_ids_missing_deposition_date": sorted(set(official["pdb_id"]) - set(dates["pdb_id"])),
            "postcutoff_unique_pdb_ids": len(postcutoff_pdbs),
            "postcutoff_official_entries": len(postcutoff),
        },
        "references": {
            **dataframe_stats(references),
            "reference_directories": sum((DATASET / cid).is_dir() for cid in official["complex_id"]),
            "ligand_files": int(references["reference_ligand_exists"].sum()),
            "parsed_ligands": int(references["reference_ligand_parsed"].sum()),
            "protein_files": int(references["reference_protein_exists"].sum()),
            "multi_fragment_reference_molecules": int((references["reference_fragment_count"] > 1).sum()),
            "maximum_reference_fragment_count": int(references["reference_fragment_count"].max()),
        },
        "methods": {},
    }

    mapping_rows = []
    duplicate_rows = []
    coverage_rows = []

    for method, cfg in METHODS.items():
        method_summary = {"repeats": {}}
        all_eval = []
        full_found_counts = []
        full_parsed_counts = []
        for repeat in (1, 2, 3):
            root = cfg["root"]
            manifest_path = root / f"outputs/coordinate_rerun/repeat_{repeat}_paths.csv"
            evaluation_path = root / f"outputs/coordinate_rerun/repeat_{repeat}.csv"
            manifest = pd.read_csv(manifest_path)
            evaluation = pd.read_csv(evaluation_path)
            manifest["complex_id"] = manifest["mol_id"].str.upper()
            manifest["pdb_id"] = manifest["complex_id"].str.split("_").str[0]
            manifest["ligand_id"] = manifest["complex_id"].str.split("_", n=1).str[1]
            evaluation["complex_id"] = evaluation["mol_id"].str.upper()
            evaluation["pdb_id"] = evaluation["complex_id"].str.split("_").str[0]
            evaluation["ligand_id"] = evaluation["complex_id"].str.split("_", n=1).str[1]
            evaluation["repeat"] = repeat
            evaluation["method"] = method
            all_eval.append(evaluation[["complex_id", "pdb_id", "ligand_id", "repeat", "method"]])

            assert len(manifest) == 130
            assert manifest["complex_id"].is_unique
            assert manifest["mol_pred"].is_unique
            assert set(manifest["complex_id"]) == set(postcutoff["complex_id"].str.upper())
            assert evaluation["complex_id"].is_unique
            assert set(evaluation["complex_id"]).issubset(set(manifest["complex_id"]))
            joined = manifest.merge(
                evaluation[["complex_id"]].assign(evaluated=True),
                on="complex_id",
                how="left",
                validate="one_to_one",
            )
            assert len(joined) == len(manifest), "join changed row count"

            full_paths = [expected_prediction_path(method, repeat, cid) for cid in official["complex_id"]]
            raw_prediction_root = cfg["inference"] / cfg["dir_template"].format(repeat=repeat)
            raw_sdf_files = list(raw_prediction_root.glob("**/*.sdf"))
            raw_prediction_directories = [p for p in raw_prediction_root.iterdir() if p.is_dir()]
            full_found = [p for p in full_paths if p.is_file()]
            full_parsed = sum(first_sdf_molecule(p) is not None for p in full_found)
            full_found_counts.append(len(full_found))
            full_parsed_counts.append(full_parsed)

            post_paths = [expected_prediction_path(method, repeat, cid) for cid in postcutoff["complex_id"]]
            post_found = [p for p in post_paths if p.is_file()]
            post_parsed = sum(first_sdf_molecule(p) is not None for p in post_found)
            result_ids = set(evaluation["complex_id"])
            parsed_ids = {p.parent.name.upper() for p in post_found if first_sdf_molecule(p) is not None}
            unreadable = len(post_found) - post_parsed
            unsupported = len(parsed_ids - result_ids)

            repeat_stats = {
                "raw_prediction_directories": len(raw_prediction_directories),
                "raw_sdf_files_all_ranks_or_outputs": len(raw_sdf_files),
                "raw_expected_prediction_paths_full_308": len(full_paths),
                "raw_top_prediction_files_found_full": len(full_found),
                "parsed_top_prediction_structures_full": full_parsed,
                "postcutoff_manifest_rows": len(manifest),
                "postcutoff_top_prediction_files_found": len(post_found),
                "postcutoff_parsed_top_prediction_structures": post_parsed,
                "evaluation_rows": len(evaluation),
                "unique_complex_ids": evaluation["complex_id"].nunique(),
                "unique_pdb_ids": evaluation["pdb_id"].nunique(),
                "unique_ligand_ids": evaluation["ligand_id"].nunique(),
                "unique_complex_ligand_pairs": evaluation[["complex_id", "ligand_id"]].drop_duplicates().shape[0],
                "unique_expected_prediction_paths": manifest["mol_pred"].nunique(),
                "unique_existing_prediction_files": len(post_found),
                "duplicates_within_repeat": evaluation.duplicated("complex_id").sum(),
                "unreadable_predictions_postcutoff": unreadable,
                "unsupported_or_preprocessing_failed_postcutoff": unsupported,
                "missing_predictions_vs_full_308": len(official) - len(full_found),
                "missing_predictions_vs_postcutoff_130": len(postcutoff) - len(post_found),
                "common_plot_denominator": PLOT_DENOMINATOR,
                "postcutoff_target_minus_plot_denominator": len(postcutoff) - PLOT_DENOMINATOR,
            }
            method_summary["repeats"][str(repeat)] = {k: int(v) for k, v in repeat_stats.items()}

            coverage_rows.append(
                {
                    "prediction_method": method,
                    "repeat": repeat,
                    "total_expected_benchmark_entries": len(official),
                    "postcutoff_expected_entries": len(postcutoff),
                    "raw_sdf_files_all_ranks_or_outputs": len(raw_sdf_files),
                    "prediction_files_found_full": len(full_found),
                    "successfully_parsed_predictions_full": full_parsed,
                    "failed_predictions_postcutoff": unreadable + unsupported,
                    "missing_predictions_full": len(official) - len(full_found),
                    "missing_predictions_postcutoff": len(postcutoff) - len(post_found),
                    "unreadable_predictions_postcutoff": unreadable,
                    "unsupported_molecules_postcutoff": unsupported,
                    "actual_evaluation_rows": len(evaluation),
                    "rows_included_in_metric_numerator_pool": len(evaluation),
                    "scalar_metric_denominator": PLOT_DENOMINATOR,
                    "full_entries_excluded_by_postcutoff_filter": len(official) - len(postcutoff),
                    "postcutoff_minus_scalar_denominator": len(postcutoff) - PLOT_DENOMINATOR,
                    "rows_excluded_from_metric_denominator": "not row-addressable; denominator is a scalar maximum",
                }
            )

            manifest_lookup = joined.set_index("complex_id").to_dict("index")
            for row in official.itertuples(index=False):
                cid = row.complex_id.upper()
                in_subset = cid in set(postcutoff["complex_id"].str.upper())
                source = expected_prediction_path(method, repeat, row.complex_id)
                m = manifest_lookup.get(cid, {})
                evaluated = bool(m.get("evaluated", False))
                source_exists = source.is_file()
                status = (
                    "excluded_pre_or_missing_date"
                    if not in_subset
                    else "evaluated"
                    if evaluated
                    else "missing_prediction"
                    if not source_exists
                    else "unreadable_or_preprocessing_failure"
                )
                mapping_rows.append(
                    {
                        "official_entry_id": cid,
                        "pdb_id": row.pdb_id,
                        "complex_id": cid,
                        "ligand_id": row.ligand_id,
                        "prediction_method": method,
                        "repeat": repeat,
                        "source_file": str(source),
                        "source_file_exists": source_exists,
                        "evaluation_row_id": f"{method}:repeat_{repeat}:{cid}" if evaluated else "",
                        "duplicate_group": f"stochastic_repeats:{method}:{cid}" if evaluated else "",
                        "notes": status,
                    }
                )

        pooled = pd.concat(all_eval, ignore_index=True)
        duplicated = pooled[pooled.duplicated(["method", "complex_id"], keep="first")].copy()
        for row in duplicated.itertuples(index=False):
            duplicate_rows.append(
                {
                    "prediction_method": row.method,
                    "complex_id": row.complex_id,
                    "pdb_id": row.pdb_id,
                    "ligand_id": row.ligand_id,
                    "repeat": row.repeat,
                    "evaluation_row_id": f"{row.method}:repeat_{row.repeat}:{row.complex_id}",
                    "duplicate_key": f"{row.method}:{row.complex_id}",
                    "duplicate_reason": "expected stochastic-repeat observation; not a duplicate within a repeat",
                }
            )
        method_summary["pooled"] = {
            "evaluation_rows": len(pooled),
            "unique_complex_ids": pooled["complex_id"].nunique(),
            "unique_pdb_ids": pooled["pdb_id"].nunique(),
            "unique_ligand_ids": pooled["ligand_id"].nunique(),
            "unique_complex_ligand_pairs": pooled[["complex_id", "ligand_id"]].drop_duplicates().shape[0],
            "duplicate_rows_across_repeats": len(duplicated),
            "unique_prediction_files_full_across_repeats": sum(full_found_counts),
            "parsed_prediction_structures_full_across_repeats": sum(full_parsed_counts),
            "metric_denominator_slots": PLOT_DENOMINATOR * 3,
        }
        method_summary["pooled"] = {k: int(v) for k, v in method_summary["pooled"].items()}
        count_summary["methods"][method] = method_summary

        per_run_metrics = pd.read_csv(
            cfg["root"] / f"outputs/{method.lower()}_unrelaxed_per_run.csv"
        )
        count_summary["methods"][method]["reported_metric_successes"] = {
            "crmsd_le_1a": int(per_run_metrics["cRMSD ≤ 1 Å successes"].sum()),
            "rmsd_le_2a": int(per_run_metrics["RMSD ≤ 2 Å successes"].sum()),
            "rmsd_le_2a_and_pb_valid": int(
                per_run_metrics["RMSD ≤ 2 Å & PB-valid successes"].sum()
            ),
            "denominator": int(per_run_metrics["plot_denominator"].sum()),
        }

    mapping = pd.DataFrame(mapping_rows)
    assert len(mapping) == 308 * 2 * 3
    assert not mapping.duplicated(["official_entry_id", "prediction_method", "repeat"]).any()
    mapping.to_csv(out / "entry_to_rows.csv", index=False)

    duplicates = pd.DataFrame(duplicate_rows)
    duplicates.to_csv(out / "duplicate_rows.csv", index=False)

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(out / "method_coverage.csv", index=False)

    # There are no 67 extra records.  This signed ledger explains the invalid
    # cross-unit subtraction exactly: +250 -178 -5 = +67.
    extra_ledger = pd.DataFrame(
        [
            {
                "component": "repeat_expansion",
                "signed_effect_on_375_minus_308": 250,
                "is_actual_extra_record": False,
                "official_entry_id": "",
                "explanation": "Repeats 2 and 3 add 2 x 125 denominator slots; these are repeat opportunities, not new benchmark entries.",
            },
            {
                "component": "postcutoff_filter",
                "signed_effect_on_375_minus_308": -178,
                "is_actual_extra_record": False,
                "official_entry_id": "",
                "explanation": "The main analysis uses 130 post-cutoff entries rather than all 308 entries.",
            },
            {
                "component": "common_denominator_gap",
                "signed_effect_on_375_minus_308": -5,
                "is_actual_extra_record": False,
                "official_entry_id": "",
                "explanation": "The plotting notebook uses the maximum available method-row count, 125, rather than the 130 metadata targets; the scalar has no five-ID exclusion list.",
            },
        ]
    )
    assert int(extra_ledger["signed_effect_on_375_minus_308"].sum()) == 67
    extra_ledger.to_csv(out / "extra_67_rows.csv", index=False)

    with (out / "count_summary.json").open("w") as handle:
        json.dump(count_summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    inventory = [
        "# Exact source inventory for the denominator audit",
        f"official DiffDock input metadata: {INPUT}",
        f"official Vina input metadata: {VINA_INPUT}",
        f"official PoseBusters 308-ID list: {PAPER_IDS}",
        f"deposition-date metadata: {DATES}",
        f"reference dataset root (308 complex directories): {DATASET}",
        f"reference pattern: {DATASET}/<complex_id>/<complex_id>_ligand.sdf",
        f"reference protein pattern: {DATASET}/<complex_id>/<complex_id>_protein.pdb",
        f"aligned protein root: {DATASET}/posebusters_benchmark_holo_aligned_predicted_structures",
        f"raw benchmark archive: {DIFF}/releases/14629652/data/raw/posebusters_benchmark_set.tar.gz",
        f"raw notebook archive: {DIFF}/releases/14629652/data/raw/posebench_notebooks.tar.gz",
        f"raw DiffDock predictions archive: {DIFF}/releases/14629652/data/raw/diffdock_benchmark_method_predictions.tar.gz",
        f"raw Vina predictions archive: {VINA}/releases/14629652/data/raw/vina_benchmark_method_predictions.tar.gz",
        f"official plotting notebook: {DIFF}/posebench-v0.6.0/notebooks/posebusters_benchmark_inference_results_plotting.ipynb",
        f"copied PoseBench plotting notebook: {DIFF}/posebench-source/notebooks/posebusters_benchmark_inference_results_plotting.ipynb",
        f"script producing 125 and 375 (DiffDock): {DIFF}/reproduce_baseline.py",
        f"script producing 125 and 375 (Vina): {VINA}/reproduce_baseline.py",
        f"coordinate preprocessing/evaluation script: {DIFF}/rerun_coordinates.py",
        f"final table input (DiffDock): {DIFF}/outputs/diffdock_unrelaxed_per_run.csv",
        f"final table output (DiffDock): {DIFF}/outputs/diffdock_unrelaxed_summary.csv",
        f"per-evaluation-row table (DiffDock): {DIFF}/outputs/diffdock_unrelaxed_per_complex.csv",
        f"final table input (Vina): {VINA}/outputs/vina_unrelaxed_per_run.csv",
        f"final table output (Vina): {VINA}/outputs/vina_unrelaxed_summary.csv",
        f"per-evaluation-row table (Vina): {VINA}/outputs/vina_unrelaxed_per_complex.csv",
    ]
    for method, cfg in METHODS.items():
        root = cfg["root"]
        for repeat in (1, 2, 3):
            inventory.extend(
                [
                    f"{method} repeat {repeat} raw prediction root: {cfg['inference'] / cfg['dir_template'].format(repeat=repeat)}",
                    f"{method} repeat {repeat} path manifest: {root}/outputs/coordinate_rerun/repeat_{repeat}_paths.csv",
                    f"{method} repeat {repeat} parsed evaluation: {root}/outputs/coordinate_rerun/repeat_{repeat}.csv",
                ]
            )
    (out / "file_inventory.txt").write_text("\n".join(inventory) + "\n")

    print(json.dumps(count_summary["denominator_finding"], indent=2))
    print(f"Wrote diagnostics to {out}")


if __name__ == "__main__":
    main()
