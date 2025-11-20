import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional
from molgen3D.evaluation.utils import format_float
from pathlib import Path
import pickle

from molgen3D.evaluation.utils import covmat_metrics, DEFAULT_THRESHOLDS
from molgen3D.data_processing.utils import load_pkl
from molgen3D.config.paths import get_data_path

THRESHOLD_075 = 0.75

def format_value_or_na(value, decimals: int = 4) -> str:
    """Format a numeric value or return 'N/A' if None."""
    return format_float(value, decimals) if value is not None else "N/A"

def write_section_header(f, title: str, width: int = 40) -> None:
    """Write a section header with underline."""
    f.write(f"{title}\n")
    f.write("-" * width + "\n")

def write_file_header(f) -> None:
    """Write the main file header."""
    f.write("=" * 80 + "\n")
    f.write("COVMAT EVALUATION RESULTS\n")
    f.write("=" * 80 + "\n\n")

def write_summary_section(f, gen_stats: Dict[str, int], gt_stats: Dict[str, int], missing: List[str], all_nan_keys: List[str]) -> None:
    """Write the evaluation summary section."""
    write_section_header(f, "EVALUATION SUMMARY")
    f.write(f"Processed file: {gen_stats['gen_path']}\n")
    f.write(f"Ground truth file: {gt_stats['gt_path']}\n")
    f.write(f"Total molecules generated: {gen_stats['total_molecules_num']}\n")
    f.write(f"Total conformers generated: {gen_stats['total_conformers_num']}\n")
    f.write(f"Total molecules in ground truth: {gt_stats['total_molecules_num']}\n")
    f.write(f"Total conformers in ground truth: {gt_stats['total_conformers_num']}\n")
    f.write(f"Missing molecules (no conformers): {len(missing)}\n")
    f.write(f"All-NaN RMSD keys: {len(all_nan_keys)}\n\n")

def write_runtime_section(f, durations: Dict[str, float]) -> None:
    """Write the execution runtime section."""
    write_section_header(f, "EXECUTION RUNTIME")
    for k, v in durations.items():
        minutes = float(v) / 60.0
        f.write(f"{k}: {format_float(minutes, 4)} min\n")
    f.write("\n")

def write_metrics_section(f, cov_row_075, matching: Dict[str, float]) -> None:
    """Write the coverage and recall metrics section."""
    write_section_header(f, "COVERAGE AND RECALL METRICS")
    f.write(f"Threshold: {THRESHOLD_075}\n")
    f.write("Coverage-Recall (COV-R):\n")
    f.write(f"  Mean:   {format_float(float(cov_row_075.get('COV-R_mean', np.nan)), 4)}\n")
    f.write(f"  Median: {format_float(float(cov_row_075.get('COV-R_median', np.nan)), 4)}\n")
    f.write("Coverage-Precision (COV-P):\n")
    f.write(f"  Mean:   {format_float(float(cov_row_075.get('COV-P_mean', np.nan)), 4)}\n")
    f.write(f"  Median: {format_float(float(cov_row_075.get('COV-P_median', np.nan)), 4)}\n")
    f.write("Matching-Recall (MAT-R):\n")
    f.write(f"  Mean:   {format_float(matching.get('MAT-R_mean', np.nan), 4)}\n")
    f.write(f"  Median: {format_float(matching.get('MAT-R_median', np.nan), 4)}\n")
    f.write("Matching-Precision (MAT-P):\n")
    f.write(f"  Mean:   {format_float(matching.get('MAT-P_mean', np.nan), 4)}\n")
    f.write(f"  Median: {format_float(matching.get('MAT-P_median', np.nan), 4)}\n\n")

def write_missing_molecules_section(f, missing: List[str]) -> None:
    """Write the missing molecules section."""
    write_section_header(f, "MISSING MOLECULES (NO CONFORMERS)")
    if missing:
        for i, mol in enumerate(missing, 1):
            f.write(f"{i:3d}. {mol}\n")
    else:
        f.write("None - all molecules had conformers\n")
    f.write("\n")

def write_all_nan_section(f, all_nan_keys: List[str]) -> None:
    """Write the all-NaN RMSD keys section."""
    write_section_header(f, "ALL-NaN RMSD KEYS")
    if all_nan_keys:
        for i, key in enumerate(all_nan_keys, 1):
            f.write(f"{i:3d}. {key}\n")
    else:
        f.write("None - all molecules had valid RMSD calculations\n")
    f.write("\n")

def write_generation_info(f, gen_stats: Dict[str, int]) -> None:
    """Write generation results info section."""
    gen_txt_content = read_gen_results_txt(gen_stats['gen_path'])
    if gen_txt_content:
        f.write("GENERATION RESULTS INFO\n")
        f.write("-" * 40 + "\n")
        f.write(gen_txt_content)
        f.write("\n\n")

def write_posebusters_section(
    f,
    posebusters_summary: Optional[pd.DataFrame],
    pass_rate: Optional[float],
    fail_smiles: Optional[List[str]],
    error_smiles: Optional[List[str]]
) -> None:
    """Write PoseBusters evaluation section."""
    # Only write section if we have any PoseBusters data
    has_data = (
        posebusters_summary is not None or
        pass_rate is not None or
        (fail_smiles is not None and len(fail_smiles) > 0) or
        (error_smiles is not None and len(error_smiles) > 0)
    )

    if not has_data:
        return

    write_section_header(f, "POSEBUSTERS EVALUATION")

    # Write summary metrics from DataFrame
    if posebusters_summary is not None:
        try:
            row = posebusters_summary.iloc[0]
            for col in posebusters_summary.columns:
                val = row[col]
                if isinstance(val, (float, np.floating)):
                    f.write(f"{col}: {format_float(float(val), 4)}\n")
                else:
                    f.write(f"{col}: {val}\n")
        except Exception:
            f.write("Summary unavailable.\n")

    # Write pass rate
    if pass_rate is not None:
        pass_pct = pass_rate * 100.0
        f.write(
            f"Pass rate: {format_float(pass_rate, 4)} "
            f"({format_float(pass_pct, 2)}%)\n"
        )

    f.write("\n")

    # Write fail_smiles section
    if fail_smiles is not None:
        if len(fail_smiles) > 0:
            f.write("Failed SMILES (below threshold):\n")
            for i, smiles in enumerate(fail_smiles, 1):
                f.write(f"{i:3d}. {smiles}\n")
            f.write("\n")
        else:
            f.write(
                "Failed SMILES: None - all molecules passed "
                "threshold\n\n"
            )

    # Write error_smiles section
    if error_smiles is not None:
        if len(error_smiles) > 0:
            f.write("Error SMILES (processing errors):\n")
            for i, smiles in enumerate(error_smiles, 1):
                f.write(f"{i:3d}. {smiles}\n")
            f.write("\n")
        else:
            f.write(
                "Error SMILES: None - no processing errors\n\n"
            )


def write_detailed_logging(f, durations: Dict[str, float], missing: List[str], all_nan_keys: List[str], gen_stats: Dict[str, int]) -> None:
    """Write detailed logging info section."""
    f.write("=" * 80 + "\n")
    f.write("DETAILED LOGGING INFO\n")
    f.write("=" * 80 + "\n")
    f.write(f"Results for {gen_stats['gen_path']}\n")
    total_min = float(durations.get('Total processing', 0)) / 60.0
    covmat_min = float(durations.get('CovMat processing', 0)) / 60.0
    f.write(f"Total processing time: {format_float(total_min, 4)} min\n")
    f.write(f"CovMat processing time: {format_float(covmat_min, 4)} min\n")
    if 'PoseBusters processing' in durations:
        pb_min = float(durations['PoseBusters processing']) / 60.0
        f.write(f"PoseBusters processing time: {format_float(pb_min, 4)} min\n")
    f.write(f"Number of missing mols: {len(missing)}\n")
    f.write(f"Missing: {missing}\n")
    f.write(f"Number of all-NaN RMSD keys: {len(all_nan_keys)}\n")
    f.write(f"All-NaN keys: {all_nan_keys}\n")
    f.write("-" * 80 + "\n")

def read_gen_results_txt(gens_path: str) -> Optional[str]:
    """Read txt file content from gen results directory if it exists."""
    txt_files = []
    root = Path(gens_path)
    if not root.exists():
        return None
    for f in root.iterdir():
        if f.is_file() and f.name == "generation_results.txt":
            txt_files.append(f)
    
    if not txt_files:
        return None
    
    # Read the first txt file found
    try:
        with open(txt_files[0], "r") as f:
            return f.read().strip()
    except Exception:
        return None

def save_evaluation_results(cov_df: pd.DataFrame, matching: Dict[str, float], aggregated_metrics: Dict[str, np.ndarray],
                            posebusters_full_results: Optional[pd.DataFrame], posebusters_summary: Optional[pd.DataFrame], pass_rate: Optional[float], fail_smiles: Optional[List[str]], error_smiles: Optional[List[str]],
                            durations: Dict[str, float], rmsd_results: Dict[str, Dict[str, object]], missing: List[str], 
                            all_nan_keys: List[str], results_path: str, gen_stats: Dict[str, int], gt_stats: Dict[str, int],
                            args) -> None:

    os.makedirs(results_path, exist_ok=True)
    save_covmat_results_txt(cov_df, matching, posebusters_summary, posebusters_full_results, pass_rate, fail_smiles, error_smiles, durations, missing, all_nan_keys, results_path, gen_stats, gt_stats)
     
     # Save aggregated metrics (Coverage/Precision/Matching statistics) as pickle file
    rmsd_pickle_path = os.path.join(results_path, "rmsd_matrix.pickle")
    with open(rmsd_pickle_path, "wb") as f:
        pickle.dump(aggregated_metrics, f)
    print(f"Saved aggregated RMSD metrics to: {rmsd_pickle_path}")

    # Save RMSD matrix
    rmsd_matrix_path = os.path.join(results_path, "rmsd_matrix.csv")
    # Load ground truth data once outside the loop for better performance
    gt_dict = load_pkl(get_data_path(f"{args.test_set}_smi"))
    per_molecule_rmsd_stats = []
    for smi, res in rmsd_results.items():
        rmsd_matrix = res["rmsd"]
        # Calculate statistics across all conformer pairs
        all_rmsd_values = rmsd_matrix.flatten()
        valid_rmsd_values = all_rmsd_values[~np.isnan(all_rmsd_values)]
        
        # Calculate RMSD statistics (simplified conditional logic)
        min_rmsd = float(np.min(valid_rmsd_values)) if len(valid_rmsd_values) > 0 else None
        max_rmsd = float(np.max(valid_rmsd_values)) if len(valid_rmsd_values) > 0 else None
        avg_rmsd = float(np.mean(valid_rmsd_values)) if len(valid_rmsd_values) > 0 else None
        
        # Get minimum RMSD for each true conformer (row-wise min)
        min_rmsd_per_true = np.nanmin(rmsd_matrix, axis=1)
        # Calculate per-molecule coverage and matching metrics
        cov_r, mat_r, cov_p, mat_p = covmat_metrics(rmsd_matrix, DEFAULT_THRESHOLDS)
        
        # Get values at 0.75 threshold only
        idx_075 = int(np.argmin(np.abs(DEFAULT_THRESHOLDS - THRESHOLD_075)))
        cov_r_075 = float(cov_r[idx_075]) if len(cov_r) > 0 and idx_075 < len(cov_r) else None
        cov_p_075 = float(cov_p[idx_075]) if len(cov_p) > 0 and idx_075 < len(cov_p) else None
        
        # Get sub smiles from ground truth (loaded once outside loop)
        sub_smiles_list = list(gt_dict.get(smi, {}).get("sub_smiles_counts", {}).keys())
        sub_smiles_str = ";".join(sub_smiles_list) if sub_smiles_list else ""

        per_molecule_rmsd_stats.append({
            "geom_smiles": smi,
            "min_rmsd": format_value_or_na(min_rmsd),
            "max_rmsd": format_value_or_na(max_rmsd),
            "avg_rmsd": format_value_or_na(avg_rmsd),
            "cov_r_075": format_value_or_na(cov_r_075),
            "cov_p_075": format_value_or_na(cov_p_075),
            "mat_r": format_value_or_na(float(mat_r)),
            "mat_p": format_value_or_na(float(mat_p)),
            "num_true_confs": len(min_rmsd_per_true),
            "num_gen_confs": rmsd_matrix.shape[1],
            "num_valid_rmsd_pairs": len(valid_rmsd_values),
            "sub_smiles": sub_smiles_str
        })
    
    rmsd_075_df = pd.DataFrame(per_molecule_rmsd_stats)
    rmsd_075_df.to_csv(rmsd_matrix_path, index=False)
    print(f"Saved RMSD matrix at 0.75 threshold to: {rmsd_matrix_path}")

    # Save full posebusters results as pickle
    if posebusters_full_results is not None:
        posebusters_pickle_path = os.path.join(results_path, "posebusters.pickle")
        with open(posebusters_pickle_path, "wb") as f:
            pickle.dump(posebusters_full_results, f)
        print(f"Saved full PoseBusters results to: {posebusters_pickle_path}")

def save_covmat_results_txt(cov_df: pd.DataFrame, matching: Dict[str, float], posebusters_summary: Optional[pd.DataFrame], 
                            posebusters_full_results: Optional[pd.DataFrame], pass_rate: Optional[float], fail_smiles: Optional[List[str]], error_smiles: Optional[List[str]],
                            durations: Dict[str, float], missing: List[str], all_nan_keys: List[str], results_path: str,
                            gen_stats: Dict[str, int], gt_stats: Dict[str, int]) -> None:
    """Save comprehensive evaluation results to text file."""
    idx075 = int(np.argmin(np.abs(DEFAULT_THRESHOLDS - THRESHOLD_075)))
    cov_row_075 = cov_df.iloc[idx075]

    with open(os.path.join(results_path, "covmat_results.txt"), "w") as f:
        write_file_header(f)
        write_summary_section(f, gen_stats, gt_stats, missing, all_nan_keys)
        write_runtime_section(f, durations)
        write_metrics_section(f, cov_row_075, matching)
        write_missing_molecules_section(f, missing)
        write_all_nan_section(f, all_nan_keys)
        write_generation_info(f, gen_stats)
        write_posebusters_section(f, posebusters_summary, pass_rate, fail_smiles, error_smiles)
        write_detailed_logging(f, durations, missing, all_nan_keys, gen_stats)