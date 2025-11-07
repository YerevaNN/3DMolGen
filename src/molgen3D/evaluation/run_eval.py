import argparse
import math
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    from molgen3D.evaluation.posebusters_check import run_all_posebusters
except Exception:
    run_all_posebusters = None

try:
    import submitit
except Exception:
    submitit = None

BASE_PATH = "/nfs/ap/mnt/sxtn2/chem/GEOM_data"
DEFAULT_TRUE_MOLS_PATH = "/auto/home/menuab/code/3DMolGen/data/inference_set_clean_smiles/drugs_test_dict.pickle"
DEFAULT_THRESHOLDS = np.arange(0.05, 3.05, 0.05)


def format_float(value: float, decimals: int = 4) -> str:
    """Format float to specified decimal places (truncated, not rounded)."""
    if value is None or np.isnan(value):
        return "N/A"
    # Truncate instead of round
    factor = 10 ** decimals
    truncated = math.floor(abs(value) * factor) / factor
    if value < 0:
        truncated = -truncated
    formatted = f"{truncated:.{decimals}f}".rstrip('0').rstrip('.')
    return formatted


@dataclass(frozen=True)
class EvalConfig:
    thresholds: np.ndarray
    num_workers: int
    use_alignmol: bool
    batch_true_confs: int


def load_pickle(path: str) -> Dict:
    """Load molecules from pickle file."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing pickle: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_ground_truth_dict(path: str) -> Dict[str, Dict]:
    """Load ground truth dictionary with conformers."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing pickle: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")
    return data


def preprocess_gen(gen_data: Dict) -> Dict[str, List]:
    """Preprocess generated molecules."""
    processed: Dict[str, List] = {}
    from rdkit.Chem.rdmolops import RemoveHs
    for can, mols in gen_data.items():
        processed[can] = [RemoveHs(m) for m in mols]
    return processed


def _best_rmsd(probe, ref, use_alignmol: bool) -> Optional[float]:
    from rdkit.Chem import rdMolAlign as MA
    try:
        if use_alignmol:
            return float(MA.AlignMol(probe, ref))
        return float(MA.GetBestRMS(probe, ref))
    except Exception:
        return np.nan


def _rmsd_row_for_true(ref_mol, gen_mols: List, use_alignmol: bool) -> np.ndarray:
    return np.array([_best_rmsd(gen_mol, ref_mol, use_alignmol) for gen_mol in gen_mols], dtype=float)


def _compute_key_matrix(key: str, true_confs: List, gen_mols: List, use_alignmol: bool) -> Tuple[str, Dict[str, object], bool]:
    n_true = len(true_confs)
    n_gen = len(gen_mols)
    mat = np.full((n_true, n_gen), np.nan, dtype=float)
    for i_true, ref_mol in enumerate(true_confs):
        row = _rmsd_row_for_true(ref_mol, gen_mols, use_alignmol)
        if row.shape == (n_gen,):
            mat[i_true] = row
    all_nan = bool(np.isnan(mat).all())
    return key, {"n_true": n_true, "n_model": n_gen, "rmsd": mat}, all_nan


def _batch_worker(batch: List[Tuple[str, int, object]], gen_lut: Dict[str, List], use_alignmol: bool) -> List[Tuple[str, int, np.ndarray]]:
    out: List[Tuple[str, int, np.ndarray]] = []
    for smi, i_true, ref_mol in batch:
        gen_mols = gen_lut.get(smi, [])
        out.append((smi, i_true, _rmsd_row_for_true(ref_mol, gen_mols, use_alignmol)))
    return out


def _submit_true_rows_batched(jobs: List[Tuple[str, int, object]], gen_lut: Dict[str, List], use_alignmol: bool,
                              num_workers: int, batch_size: int) -> Iterable[Tuple[str, int, np.ndarray]]:
    batches: List[List[Tuple[str, int, object]]] = [jobs[i:i + batch_size] for i in range(0, len(jobs), batch_size)]

    def task(batch: List[Tuple[str, int, object]]) -> List[Tuple[str, int, np.ndarray]]:
        return _batch_worker(batch, gen_lut, use_alignmol)

    # Sequential processing instead of multiprocessing to avoid RDKit issues
    for batch in batches:
        results = _batch_worker(batch, gen_lut, use_alignmol)
        for item in results:
            yield item


def compute_rmsd_matrix(true_data: Dict, gen_data: Dict[str, List], cfg: EvalConfig) -> Tuple[Dict, List[str], List[str]]:
    missing, all_nan_keys = [], []
    rmsd_results = {}
    work_items: List[Tuple[str, List, List]] = []
    for key in true_data.keys():
        gen_mols = gen_data.get(key, [])
        if not gen_mols:
            missing.append(key)
            continue
        work_items.append((key, true_data[key]["confs"], gen_mols))
    if not work_items:
        return rmsd_results, missing, all_nan_keys
    total_rows = int(sum(len(confs) for _, confs, _ in work_items))
    if cfg.num_workers and cfg.num_workers > 1:
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as ex:
            futures = [ex.submit(_compute_key_matrix, key, confs, gen_mols, cfg.use_alignmol) for key, confs, gen_mols in work_items]
            with tqdm(total=total_rows, desc="RMSD rows", unit="row") as pbar:
                for fut in as_completed(futures):
                    key, res, all_nan = fut.result()
                    rmsd_results[key] = res
                    if all_nan:
                        all_nan_keys.append(key)
                    pbar.update(int(res["n_true"]))
    else:
        with tqdm(total=total_rows, desc="RMSD rows", unit="row") as pbar:
            for key, confs, gen_mols in work_items:
                k, res, all_nan = _compute_key_matrix(key, confs, gen_mols, cfg.use_alignmol)
                rmsd_results[k] = res
                if all_nan:
                    all_nan_keys.append(k)
                pbar.update(int(res["n_true"]))
    return rmsd_results, missing, all_nan_keys


def covmat_metrics(rmsd: np.ndarray, thresholds: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, float]:
    min_true = np.nanmin(rmsd, axis=1)
    min_gen = np.nanmin(rmsd, axis=0)
    cov_recall = np.array([(min_true < t).mean() for t in thresholds], dtype=float)
    cov_precision = np.array([(min_gen < t).mean() for t in thresholds], dtype=float)
    amr_recall = float(np.nanmean(min_true))
    amr_precision = float(np.nanmean(min_gen))
    return cov_recall, amr_recall, cov_precision, amr_precision


def aggregate_metrics(rmsd_results: Dict[str, Dict[str, object]], thresholds: np.ndarray) -> Dict[str, np.ndarray]:
    cov_r_list: List[np.ndarray] = []
    cov_p_list: List[np.ndarray] = []
    mat_r_list: List[float] = []
    mat_p_list: List[float] = []
    for res in rmsd_results.values():
        cov_r, mat_r, cov_p, mat_p = covmat_metrics(res["rmsd"], thresholds)
        cov_r_list.append(cov_r)
        cov_p_list.append(cov_p)
        mat_r_list.append(mat_r)
        mat_p_list.append(mat_p)
    if not cov_r_list:
        return {
            "thresholds": thresholds,
            "CoverageR": np.zeros_like(thresholds),
            "CoverageP": np.zeros_like(thresholds),
            "MatchingR": np.array([]),
            "MatchingP": np.array([]),
        }
    return {
        "thresholds": thresholds,
        "CoverageR": np.vstack(cov_r_list),
        "CoverageP": np.vstack(cov_p_list),
        "MatchingR": np.array(mat_r_list),
        "MatchingP": np.array(mat_p_list),
    }


def summarize_metrics(agg: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = pd.DataFrame(
        {
            "Threshold": agg["thresholds"],
            "COV-R_mean": np.mean(agg["CoverageR"], axis=0) if agg["CoverageR"].size else np.zeros_like(agg["thresholds"]),
            "COV-R_median": np.median(agg["CoverageR"], axis=0) if agg["CoverageR"].size else np.zeros_like(agg["thresholds"]),
            "COV-P_mean": np.mean(agg["CoverageP"], axis=0) if agg["CoverageP"].size else np.zeros_like(agg["thresholds"]),
            "COV-P_median": np.median(agg["CoverageP"], axis=0) if agg["CoverageP"].size else np.zeros_like(agg["thresholds"]),
        }
    )
    stats = {
        "COV-R_mean": float(np.mean(agg["CoverageR"])) if agg["CoverageR"].size else float("nan"),
        "COV-R_median": float(np.median(agg["CoverageR"])) if agg["CoverageR"].size else float("nan"),
        "COV-P_mean": float(np.mean(agg["CoverageP"])) if agg["CoverageP"].size else float("nan"),
        "COV-P_median": float(np.median(agg["CoverageP"])) if agg["CoverageP"].size else float("nan"),
        "MAT-R_mean": float(np.mean(agg["MatchingR"])) if agg["MatchingR"].size else float("nan"),
        "MAT-R_median": float(np.median(agg["MatchingR"])) if agg["MatchingR"].size else float("nan"),
        "MAT-P_mean": float(np.mean(agg["MatchingP"])) if agg["MatchingP"].size else float("nan"),
        "MAT-P_median": float(np.median(agg["MatchingP"])) if agg["MatchingP"].size else float("nan"),
    }
    return df, stats


def find_threshold_index(thresholds: np.ndarray, target: float) -> int:
    idx = int(np.argmin(np.abs(thresholds - target)))
    return idx


def run_posebusters_wrapper(gen_data: Dict[str, List], config: str, max_workers: int) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if run_all_posebusters is None:
        print("PoseBusters not available (import failed)")
        return None, None
    
    if not gen_data:
        print("PoseBusters: No data to process")
        return None, None
    
    num_molecules = len(gen_data)
    num_conformers = sum(len(mols) for mols in gen_data.values())
    print(f"Starting PoseBusters evaluation on {num_molecules} molecules with {num_conformers} total conformers")
    
    # Validate data
    empty_keys = [k for k, v in gen_data.items() if not v]
    if empty_keys:
        print(f"Warning: {len(empty_keys)} molecules have no conformers: {empty_keys[:5]}...")
    
    invalid_mols = []
    for k, mols in gen_data.items():
        for i, mol in enumerate(mols):
            if mol is None:
                invalid_mols.append(f"{k}_conf{i}")
    if invalid_mols:
        print(f"Warning: {len(invalid_mols)} invalid molecules found")
    
    if num_conformers == 0:
        print("No conformers to evaluate, skipping PoseBusters")
        return None, None
    
    try:
        df, summary, fail_smiles, error_smiles = run_all_posebusters(data=gen_data, config=config, full_report=False, max_workers=max_workers)
        print(f"PoseBusters completed successfully")
        if error_smiles:
            print(f"Warning: {len(error_smiles)} molecules had errors")
        return df, summary
    except Exception as e:
        print(f"PoseBusters failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def save_covmat_results(out_dir: str, cov_df: pd.DataFrame, matching: Dict[str, float], 
                       cov_row_075: pd.Series, posebusters_summary: Optional[pd.DataFrame], 
                       durations: Dict[str, float], missing_mols: List[str], all_nan_keys: List[str],
                       gen_txt_content: Optional[str], pickle_file: str, num_gt_molecules: int) -> None:
    """Save covmat_results.txt with organized, readable format."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "covmat_results.txt"), "w") as f:
        f.write("=" * 80 + "\n")
        f.write("COVMAT EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # File information
        f.write("EVALUATION SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Processed file: {pickle_file}\n")
        f.write(f"Total molecules in ground truth: {num_gt_molecules}\n")
        f.write(f"Missing molecules (no conformers): {len(missing_mols)}\n")
        f.write(f"All-NaN RMSD keys: {len(all_nan_keys)}\n\n")
        
        # Execution runtime
        f.write("EXECUTION RUNTIME\n")
        f.write("-" * 40 + "\n")
        for k, v in durations.items():
            minutes = float(v) / 60.0
            f.write(f"{k}: {format_float(minutes, 4)} min\n")
        f.write("\n")
        
        # Coverage and Recall metrics at 0.75 threshold
        f.write("COVERAGE AND RECALL METRICS\n")
        f.write("-" * 40 + "\n")
        f.write("Threshold: 0.75\n")
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
        
        # Remove separate 0.75 threshold block to avoid repetition
        
        # Missing molecules
        f.write("MISSING MOLECULES (NO CONFORMERS)\n")
        f.write("-" * 40 + "\n")
        if missing_mols:
            for i, mol in enumerate(missing_mols, 1):
                f.write(f"{i:3d}. {mol}\n")
        else:
            f.write("None - all molecules had conformers\n")
        f.write("\n")
        
        # All-NaN keys
        f.write("ALL-NaN RMSD KEYS\n")
        f.write("-" * 40 + "\n")
        if all_nan_keys:
            for i, key in enumerate(all_nan_keys, 1):
                f.write(f"{i:3d}. {key}\n")
        else:
            f.write("None - all molecules had valid RMSD calculations\n")
        f.write("\n")
        
        # Generation results txt content
        if gen_txt_content:
            f.write("GENERATION RESULTS INFO\n")
            f.write("-" * 40 + "\n")
            f.write(gen_txt_content)
            f.write("\n\n")
        
        # PoseBusters summary
        if posebusters_summary is not None:
            f.write("POSEBUSTERS EVALUATION\n")
            f.write("-" * 40 + "\n")
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
            f.write("\n")

        f.write("=" * 80 + "\n")

        f.write("\n\nDETAILED LOGGING INFO\n")
        f.write("=" * 80 + "\n")
        f.write(f"Results for {pickle_file}\n")
        total_min = float(durations.get('Total processing', 0)) / 60.0
        covmat_min = float(durations.get('CovMat processing', 0)) / 60.0
        f.write(f"Total processing time: {format_float(total_min, 4)} min\n")
        f.write(f"CovMat processing time: {format_float(covmat_min, 4)} min\n")
        if 'PoseBusters processing' in durations:
            pb_min = float(durations['PoseBusters processing']) / 60.0
            f.write(f"PoseBusters processing time: {format_float(pb_min, 4)} min\n")
        f.write(f"Number of missing mols: {len(missing_mols)}\n")
        f.write(f"Missing: {missing_mols}\n")
        f.write(f"Number of all-NaN RMSD keys: {len(all_nan_keys)}\n")
        f.write(f"All-NaN keys: {all_nan_keys}\n")
        # Omit PoseBusters repeated block to avoid duplication
        f.write("-" * 80 + "\n")



def get_pickle_files(directory_path: str) -> List[str]:
    files: List[str] = []
    root = Path(directory_path)
    if not root.exists():
        return files
    for r, _, fs in os.walk(directory_path):
        for f in fs:
            if f.endswith(".pickle"):
                rel = os.path.relpath(os.path.join(r, f), directory_path)
                files.append(rel)
    return files


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


def get_missing_evaluation_dirs(gen_base: str, eval_base: str, max_recent: Optional[int]) -> List[str]:
    gen_path = Path(gen_base)
    eval_path = Path(eval_base)
    if not gen_path.exists():
        return []
    gen_dirs: List[Tuple[str, float]] = []
    for item in gen_path.iterdir():
        if item.is_dir():
            gen_dirs.append((item.name, item.stat().st_mtime))
    gen_dirs.sort(key=lambda x: x[1], reverse=True)
    missing: List[str] = []
    for name, _ in gen_dirs:
        if not (eval_path / f"{name}_parallel").exists():
            missing.append(name)
    if max_recent and len(missing) > max_recent:
        missing = missing[:max_recent]
    return missing


def derive_eval_base_from_gen(gen_base: str) -> str:
    p = Path(gen_base).resolve()
    parts = list(p.parts)
    if "gen_results" in parts:
        idx = parts.index("gen_results")
        parts[idx] = "eval_results"
        return str(Path(*parts))
    return str(p.parent / "eval_results")

def process_single_pickle(pickle_file: str, gens_path: str, gt_dict: Dict, 
                         results_file, cfg: EvalConfig, run_posebusters: bool, posebusters_config: str) -> bool:
    try:
        full_path = os.path.join(gens_path, pickle_file)
        model_preds = load_pickle(full_path)
        if not model_preds:
            return False
        
        # Read txt file content from gen results directory
        gen_txt_content = read_gen_results_txt(gens_path)
        
        t0 = time.time()
        gen_proc = preprocess_gen(model_preds)
        
        t_prep = time.time()
        rmsd_results, missing, all_nan_keys = compute_rmsd_matrix(gt_dict, gen_proc, cfg)
        t_rmsd = time.time()
        
        # Save full RMSD matrix as pickle file
        rmsd_pickle_path = os.path.join(os.path.dirname(results_file.name), "rmsd_matrix.pickle")
        agg = aggregate_metrics(rmsd_results, cfg.thresholds)
        with open(rmsd_pickle_path, "wb") as f:
            pickle.dump(agg, f)
        print(f"Saved full RMSD aggregate matrix to: {rmsd_pickle_path}")
        cov_df, matching = summarize_metrics(agg)
        idx075 = find_threshold_index(cfg.thresholds, 0.75)
        cov_row_075 = cov_df.iloc[idx075]
        
        # Save RMSD matrix as CSV (0.75 threshold only)
        rmsd_075_csv_path = os.path.join(os.path.dirname(results_file.name), "rmsd_matrix.csv")
        rmsd_data_075 = []
        for smi, res in rmsd_results.items():
            rmsd_matrix = res["rmsd"]
            # Calculate statistics across all conformer pairs
            all_rmsd_values = rmsd_matrix.flatten()
            valid_rmsd_values = all_rmsd_values[~np.isnan(all_rmsd_values)]
            
            if len(valid_rmsd_values) > 0:
                min_rmsd = float(np.min(valid_rmsd_values))
                max_rmsd = float(np.max(valid_rmsd_values))
                avg_rmsd = float(np.mean(valid_rmsd_values))
            else:
                min_rmsd = None
                max_rmsd = None
                avg_rmsd = None
            
            # Get minimum RMSD for each true conformer (row-wise min)
            min_rmsd_per_true = np.nanmin(rmsd_matrix, axis=1)
            # Check if any true conformer is below 0.75 threshold
            below_threshold = min_rmsd_per_true < 0.75
            # Calculate per-molecule coverage and matching metrics
            cov_r, mat_r, cov_p, mat_p = covmat_metrics(rmsd_matrix, cfg.thresholds)
            
            # Get values at 0.75 threshold only
            idx_075 = find_threshold_index(cfg.thresholds, 0.75)
            cov_r_075 = float(cov_r[idx_075]) if len(cov_r) > 0 and idx_075 < len(cov_r) else None
            cov_p_075 = float(cov_p[idx_075]) if len(cov_p) > 0 and idx_075 < len(cov_p) else None
            
            # Get sub smiles from ground truth
            sub_smiles_list = list(gt_dict.get(smi, {}).get("sub_smiles_counts", {}).keys())
            sub_smiles_str = ";".join(sub_smiles_list) if sub_smiles_list else ""
            
            rmsd_data_075.append({
                "geom_smiles": smi,
                "min_rmsd": format_float(min_rmsd, 4) if min_rmsd is not None else "N/A",
                "max_rmsd": format_float(max_rmsd, 4) if max_rmsd is not None else "N/A",
                "avg_rmsd": format_float(avg_rmsd, 4) if avg_rmsd is not None else "N/A",
                "cov_r_075": format_float(cov_r_075, 4) if cov_r_075 is not None else "N/A",
                "cov_p_075": format_float(cov_p_075, 4) if cov_p_075 is not None else "N/A",
                "mat_r": format_float(float(mat_r), 4) if mat_r is not None else "N/A",
                "mat_p": format_float(float(mat_p), 4) if mat_p is not None else "N/A",
                "num_true_confs": len(min_rmsd_per_true),
                "num_gen_confs": rmsd_matrix.shape[1],
                "num_valid_rmsd_pairs": len(valid_rmsd_values),
                "sub_smiles": sub_smiles_str
            })
        
        
        rmsd_075_df = pd.DataFrame(rmsd_data_075)
        rmsd_075_df.to_csv(rmsd_075_csv_path, index=False)
        print(f"Saved RMSD matrix at 0.75 threshold to: {rmsd_075_csv_path}")
        posebusters_duration = 0.0
        posebusters_summary = None
        posebusters_full_results = None
        if run_posebusters:
            print("\n" + "="*80)
            print("STARTING POSEBUSTERS EVALUATION")
            print("="*80)
            pb_start = time.time()
            posebusters_full_results, posebusters_summary = run_posebusters_wrapper(gen_proc, posebusters_config, cfg.num_workers)
            print("\n" + "="*80)
            print(f"Pass percentage: {posebusters_summary['pass_percentage'].iloc[0]:.2f}%\n")
            print("POSEBUSTERS EVALUATION COMPLETED")
            print("="*80)
            posebusters_duration = time.time() - pb_start
            
            # Save full posebusters results as pickle
            if posebusters_full_results is not None:
                posebusters_pickle_path = os.path.join(os.path.dirname(results_file.name), "posebusters.pickle")
                with open(posebusters_pickle_path, "wb") as f:
                    pickle.dump(posebusters_full_results, f)
                print(f"Saved full PoseBusters results to: {posebusters_pickle_path}")
        total = time.time() - t0
        covmat_duration = t_rmsd - t_prep
        
        # Prepare durations dictionary
        durations = {
            "Total processing": total,
            "CovMat processing": covmat_duration,
        }
        if run_posebusters:
            durations["PoseBusters processing"] = posebusters_duration
        
        # Save organized covmat_results.txt
        save_covmat_results(
            out_dir=os.path.dirname(results_file.name),
            cov_df=cov_df,
            matching=matching,
            cov_row_075=cov_row_075,
            posebusters_summary=posebusters_summary,
            durations=durations,
            missing_mols=missing,
            all_nan_keys=all_nan_keys,
            gen_txt_content=gen_txt_content,
            pickle_file=pickle_file,
            num_gt_molecules=len(gt_dict)
        )

        results_file.close()
        return True
    except Exception as e:
        results_file.write(f"ERROR processing {pickle_file}: {e}\n")
        results_file.write("-" * 80 + "\n")
        results_file.flush()
        return False


def run_evaluation(directory_name: str, gen_base: str, eval_base: str, cfg: EvalConfig,
                  run_posebusters: bool, posebusters_config: str, true_mols_path: str = DEFAULT_TRUE_MOLS_PATH) -> bool:
    print(f"Starting evaluation for: {directory_name}")
    
    gens_path = os.path.join(gen_base, directory_name)
    if not os.path.exists(gens_path):
        print(f"Directory does not exist: {gens_path}")
        return False
    
    pickle_files = get_pickle_files(gens_path)
    if not pickle_files:
        print(f"No pickle files found in {directory_name}")
        return False
    if len(pickle_files) != 1:
        print(f"Expected exactly one pickle in {directory_name}, found {len(pickle_files)}")
        return False

    gt_dict = load_ground_truth_dict(true_mols_path)
    print(f"Loaded {len(gt_dict)} ground truth geom_smiles")
    
    results_path = os.path.join(eval_base, f"{directory_name}")
    os.makedirs(results_path, exist_ok=True)
    results_file = open(os.path.join(results_path, "covmat_results.txt"), "w")
    
    total_start = time.time()
    try:
        pickle_file = pickle_files[0]
        ok = process_single_pickle(pickle_file, gens_path, gt_dict, results_file, cfg, run_posebusters, posebusters_config)
        total_duration = time.time() - total_start
        if ok:
            print(f"Evaluation completed for {directory_name} in {total_duration:.4f}s")
            print(f"Results saved to {results_path}")
            return True
        print(f"Failed to process {pickle_file} for {directory_name}")
        try:
            results_file.close()
        except Exception:
            pass
        return False
    except Exception as e:
        print(f"Critical error evaluating {directory_name}: {e}")
        results_file.close()
        return False


def create_slurm_executor(device: str, num_workers: int):
    if submitit is None:
        raise RuntimeError("submitit is not available")
    if device == "local":
        executor = submitit.LocalExecutor(folder=str(Path.home() / "slurm_jobs/eval/job_%j"))
    else:
        executor = submitit.AutoExecutor(folder=str(Path.home() / "slurm_jobs/eval/job_%j"))
    executor.update_parameters(
        name="gen_evals",
        timeout_min=40 * 60,
        cpus_per_task=num_workers,
        mem_gb=80,
        nodes=1,
        slurm_additional_parameters={"partition": device} if device != "local" else {},
    )
    return executor


def submit_evaluation_jobs(directories: List[str], gen_base: str, device: str,
                         cfg: EvalConfig, run_posebusters: bool, posebusters_config: str,
                         true_mols_path: str = DEFAULT_TRUE_MOLS_PATH) -> List[Tuple[str, object]]:
    if submitit is None:
        raise RuntimeError("submitit is not available")
    eval_base = derive_eval_base_from_gen(gen_base)
    executor = create_slurm_executor(device, cfg.num_workers)
    submitted: List[Tuple[str, object]] = []
    for directory in directories:
        job = executor.submit(
            run_evaluation,
            directory_name=directory,
            gen_base=gen_base,
            eval_base=eval_base,
            cfg=cfg,
            run_posebusters=run_posebusters,
            posebusters_config=posebusters_config,
            true_mols_path=true_mols_path,
        )
        submitted.append((directory, job))
    return submitted


def run_directory_mode(args) -> None:
    cfg = EvalConfig(
        thresholds=DEFAULT_THRESHOLDS,
        num_workers=args.num_workers,
        use_alignmol=args.align,
        batch_true_confs=max(1, args.batch_size),
    )
    run_posebusters = not args.no_posebusters
    gen_base = args.gen_results_base
    eval_base = derive_eval_base_from_gen(gen_base)
    if args.specific_dir:
        if args.no_slurm:
            success = run_evaluation(
                directory_name=args.specific_dir,
                gen_base=gen_base,
                eval_base=eval_base,
                cfg=cfg,
                run_posebusters=run_posebusters,
                posebusters_config=args.posebusters_config,
                true_mols_path=args.true_mols_path,
            )
            print(f"{'Successfully' if success else 'Failed to'} evaluate {args.specific_dir}")
        else:
            jobs = submit_evaluation_jobs(
                directories=[args.specific_dir],
                gen_base=gen_base,
                device=args.device,
                cfg=cfg,
                run_posebusters=run_posebusters,
                posebusters_config=args.posebusters_config,
                true_mols_path=args.true_mols_path,
            )
            for d, j in jobs:
                print(f"Submitted {d}: Job ID {j.job_id}")
    else:
        missing = get_missing_evaluation_dirs(gen_base, eval_base, args.max_recent)
        if not missing:
            print("All recent generation directories have been evaluated")
            return
        if args.no_slurm:
            for d in missing:
                run_evaluation(
                    directory_name=d,
                    gen_base=gen_base,
                    eval_base=eval_base,
                    cfg=cfg,
                    run_posebusters=run_posebusters,
                    posebusters_config=args.posebusters_config,
                    true_mols_path=args.true_mols_path,
                )
        else:
            jobs = submit_evaluation_jobs(
                directories=missing,
                gen_base=gen_base,
                device=args.device,
                cfg=cfg,
                run_posebusters=run_posebusters,
                posebusters_config=args.posebusters_config,
                true_mols_path=args.true_mols_path,
            )
            print(f"Submitted {len(jobs)} jobs to {args.device}")
            for d, j in jobs:
                print(f"  - {d}: Job ID {j.job_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="High-performance evaluation for 3DMolGen with dictionary ground truth")
    
    parser.add_argument("--align", action="store_true", help="Use AlignMol instead of GetBestRMS")
    parser.add_argument("--posebusters-config", type=str, default="mol", choices=["mol", "redock"], help="PoseBusters config")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size of true-conformer rows per worker task")
    parser.add_argument("--device", type=str, choices=["local", "a100", "h100", "all"], default="local", help="Slurm partition")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for evaluation")
    parser.add_argument("--max-recent", type=int, default=3, help="Max recent missing directories to evaluate")
    parser.add_argument("--specific-dir", type=str, default=None, help="Specific directory to evaluate")
    parser.add_argument("--gen-results-base", type=str, default="./gen_results/", help="Base directory for generation results")
    parser.add_argument("--true-mols-path", type=str, default=DEFAULT_TRUE_MOLS_PATH, help="Path to ground truth dictionary pickle file")
    parser.add_argument("--no-slurm", action="store_true", help="Run locally instead of submitting to slurm")
    parser.add_argument("--no-posebusters", action="store_true", help="Skip PoseBusters evaluation")
    
    args = parser.parse_args()
    run_directory_mode(args)


if __name__ == "__main__":
    main()
