import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from molgen3D.config.paths import get_base_path, get_data_path
from molgen3D.data_processing.utils import load_pkl
from molgen3D.evaluation import rdkit_utils
from molgen3D.evaluation.posebusters_check import bust_full_gens
from molgen3D.evaluation.utils import (
    DEFAULT_THRESHOLDS,
    covmat_metrics,
    create_slurm_executor,
    find_generation_pickles_path,
)
from molgen3D.evaluation.write_eval_results import save_evaluation_results

def _compute_key_matrix(key: str, true_confs: List, gen_mols: List, use_alignmol: bool) -> Tuple[str, Dict[str, object], bool]:
    n_true = len(true_confs)
    n_gen = len(gen_mols)
    mat = np.full((n_true, n_gen), np.nan, dtype=float)
    for i_true, ref_mol in enumerate(true_confs):
        row = np.array([rdkit_utils._best_rmsd(gen_mol, ref_mol, use_alignmol) for gen_mol in gen_mols], dtype=float)
        if row.shape == (n_gen,):
            mat[i_true] = row
    all_nan = bool(np.isnan(mat).all())
    return key, {"n_true": n_true, "n_model": n_gen, "rmsd": mat}, all_nan

def compute_rmsd_matrix(true_data: Dict, gen_data: Dict[str, List], args: argparse.Namespace) -> Tuple[Dict, List[str], List[str]]:
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
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(_compute_key_matrix, key, confs, gen_mols, args.use_alignmol) for key, confs, gen_mols in work_items]
        with tqdm(total=total_rows, desc="RMSD rows", unit="row") as pbar:
            for fut in as_completed(futures):
                key, res, all_nan = fut.result()
                rmsd_results[key] = res
                if all_nan:
                    all_nan_keys.append(key)
                pbar.update(int(res["n_true"]))
    return rmsd_results, missing, all_nan_keys

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

def run_posebusters_wrapper(gen_data: Dict[str, List], config: str, max_workers: int) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[float]]:
    """Wrapper for PoseBusters evaluation.
    
    Returns:
        Tuple of (df_by_smiles, df_summary, pass_rate).
        Note: fail_smiles and error_smiles removed - users can filter by_smiles_df by pass_percentage.
    """
    
    if not gen_data:
        print("PoseBusters: No data to process")
        return None, None, None
    
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
        return None, None, None
    
    try:
        df_by_smiles, df_summary, pass_rate = bust_full_gens(
            smiles_to_confs=gen_data, config=config, full_report=False, num_workers=max_workers
        )
        print("PoseBusters completed successfully")
        return df_by_smiles, df_summary, pass_rate
    except Exception as e:
        print(f"PoseBusters failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

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

def process_generation_pickle(gens_dict: Dict, gt_dict: Dict, gens_path: str,
                              results_path: str, args: argparse.Namespace) -> bool:

    t0 = time.time()
    # Process generated molecules and calculate total count
    processed_gen_data = rdkit_utils.process_molecules_remove_hs(gens_dict)
    gen_stats = {
        "total_molecules_num": len(processed_gen_data),
        "total_conformers_num": sum(len(confs) for confs in processed_gen_data.values()),
        "gen_path": gens_path,
    }
    gt_stats = {
        "total_molecules_num": len(gt_dict),
        "total_conformers_num": sum(len(confs) for confs in gt_dict.values()),
        "gt_path": get_data_path(f"{args.test_set}_smi"),
    }
    
    t_prep = time.time()
    rmsd_results, missing, all_nan_keys = compute_rmsd_matrix(gt_dict, processed_gen_data, args)
    t_rmsd = time.time()
    agg = aggregate_metrics(rmsd_results, DEFAULT_THRESHOLDS)
    cov_df, matching = summarize_metrics(agg)
    
    posebusters_duration = 0.0
    posebusters_summary = None 
    posebusters_by_smiles = None
    pass_rate = None
    if args.posebusters != "None":
        pb_start = time.time()
        posebusters_by_smiles, posebusters_summary, pass_rate = run_posebusters_wrapper(
            processed_gen_data, args.posebusters, args.num_workers
        )
        if pass_rate is not None:
            print(f"Overall Pass percentage: {pass_rate:2f}%\n")
        posebusters_duration = time.time() - pb_start
        
    durations = {
        "Total processing": time.time() - t0,
        "CovMat processing": t_rmsd - t_prep,
        "PoseBusters processing": posebusters_duration
    }

    save_evaluation_results(
        cov_df=cov_df,
        matching=matching,
        aggregated_metrics=agg,
        posebusters_full_results=posebusters_by_smiles,
        posebusters_summary=posebusters_summary,
        pass_rate=pass_rate,
        durations=durations,
        rmsd_results=rmsd_results,
        missing=missing,
        all_nan_keys=all_nan_keys,
        results_path=results_path,
        gen_stats=gen_stats,
        gt_stats=gt_stats,
        args=args
    )
    return True


def run_evaluation(directory_name: str, gen_base: str, eval_base: str, args: argparse.Namespace) -> bool:
    print(f"Starting evaluation for: {directory_name}")
    
    gens_path = os.path.join(gen_base, directory_name)
    if not os.path.exists(gens_path):
        print(f"Directory does not exist: {gens_path}")
        return False
    gen_pickle_path = find_generation_pickles_path(gens_path)
    if not gen_pickle_path:
        print(f"No pickle files found in {directory_name}")
        return False
    gens_dict = load_pkl(gen_pickle_path)

    gt_dict = load_pkl(get_data_path(f"{args.test_set}_smi"))
    print(f"Loaded {len(gt_dict)} ground truth geom_smiles")
    
    results_path = os.path.join(eval_base, f"{directory_name}")
    process_generation_pickle(gens_dict, gt_dict, gens_path, results_path, args)

def run_directory_mode(args) -> None:
    gen_base = get_base_path("gen_results_root")
    eval_base = derive_eval_base_from_gen(gen_base)

    if args.specific_dir:
        # Check if the specific directory exists before proceeding
        gens_path = os.path.join(gen_base, args.specific_dir)
        if not os.path.exists(gens_path):
            print(f"Error: Specified directory does not exist: {gens_path}")
            return
        directories = [args.specific_dir]
    else:
        directories = get_missing_evaluation_dirs(gen_base, eval_base, args.max_recent)
    if not directories:
        print("All recent generation directories have been evaluated")
        return

    if args.device == "local":
        # Run locally without submitit to avoid RDKit pickling issues
        print(f"Running {len(directories)} evaluations locally")
        for directory in directories:
            print(f"Processing: {directory}")
            success = run_evaluation(directory, gen_base, eval_base, args)
            if not success:
                print(f"Failed to evaluate: {directory}")
    else:
        # Use submitit for remote execution
        executor = create_slurm_executor(device=args.device, job_type="eval", num_gpus=0, num_cpus=args.num_workers)
        jobs = []
        for directory in directories:
            job = executor.submit(
                run_evaluation,
                directory_name=directory,
                gen_base=gen_base,
                eval_base=eval_base,
                args=args,
            )
            jobs.append((directory, job))
        print(f"Submitted {len(jobs)} jobs to {args.device}")
        for directory, job in jobs:
            print(f"  - {directory}: Job ID {job.job_id}")

def main() -> None:
    parser = argparse.ArgumentParser(description="High-performance evaluation for 3DMolGen with dictionary ground truth")
    parser.add_argument("--use-alignmol", action="store_true", help="Use AlignMol instead of GetBestRMS")
    parser.add_argument("--posebusters", type=str, default="None", choices=["mol", "redock", "None"], help="PoseBusters config")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size of true-conformer rows per worker task")
    parser.add_argument("--device", type=str, choices=["local", "a100", "h100", "all"], default="local", help="Slurm partition")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for evaluation")
    parser.add_argument("--max-recent", type=int, default=3, help="Max recent missing directories to evaluate")
    parser.add_argument("--specific-dir", type=str, default=None, help="Specific directory to evaluate")
    parser.add_argument("--test_set", type=str, default="distinct", choices=["clean", "distinct"], help="Test set to evaluate")
    args = parser.parse_args()
    run_directory_mode(args)

if __name__ == "__main__":
    main()
