import pickle
import os
import multiprocessing as mp
from molgen3D.evaluation.improved_covmat import MemoryEfficientCovMatEvaluator, print_covmat_results
from molgen3D.evaluation.posebusters_check import run_all_posebusters
import time
from pathlib import Path
import argparse
import submitit
import sys
from datetime import datetime

# Default Configuration
DEFAULT_GEN_RESULTS_BASE = "/auto/home/menuab/code/3DMolGen/gen_results/"
DEFAULT_EVAL_RESULTS_BASE = "/auto/home/vover/code/3DMolGen/eval_results/"
DEFAULT_TRUE_MOLS_PATH = "/auto/home/menuab/code/3DMolGen/data/geom_drugs_test_set/drugs_test_inference.pickle"

def get_pickle_files(directory_path: str) -> list[str]:
    """Get all .pickle files in directory."""
    pickle_files = []
    if not Path(directory_path).exists():
        return pickle_files
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".pickle"):
                relative_path = os.path.relpath(os.path.join(root, file), directory_path)
                pickle_files.append(relative_path)
    
    return pickle_files

def get_missing_evaluation_dirs(gen_base: str, eval_base: str, max_recent: int = None) -> list[str]:
    """Find generation directories without corresponding evaluation results."""
    gen_path = Path(gen_base)
    eval_path = Path(eval_base)
    
    if not gen_path.exists():
        print(f"Generation directory not found: {gen_base}")
        return []
    
    # Get generation directories sorted by modification time
    gen_dirs = []
    for item in gen_path.iterdir():
        if item.is_dir():
            mod_time = item.stat().st_mtime
            gen_dirs.append((item.name, mod_time))
    
    gen_dirs.sort(key=lambda x: x[1], reverse=True)
    print(f"Found {len(gen_dirs)} generation directories")
    
    # Find missing evaluations
    missing_dirs = []
    for gen_dir, mod_time in gen_dirs:
        eval_dir = eval_path / f"{gen_dir}_parallel"
        if not eval_dir.exists():
            missing_dirs.append(gen_dir)
            readable_time = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Missing evaluation: {gen_dir} (modified: {readable_time})")
    
    # Limit to most recent if specified
    if max_recent and len(missing_dirs) > max_recent:
        print(f"Limiting to {max_recent} most recent missing evaluations")
        missing_dirs = missing_dirs[:max_recent]
    
    return missing_dirs

def load_molecules(pickle_path: str):
    """Load molecules from pickle file."""
    try:
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Failed to load molecules from {pickle_path}: {e}")
        return None

def process_single_pickle(evaluator, pickle_file: str, gens_path: str, true_mols, results_file, 
                         run_posebusters: bool = True, posebusters_config: str = "mol"):
    """Process a single pickle file and write results."""
    print(f"Processing: {pickle_file}")
    start_time = time.time()
    
    try:
        # Load generated molecules
        full_path = os.path.join(gens_path, pickle_file)
        model_preds = load_molecules(full_path)
        
        if not model_preds:
            print(f"Empty data in {pickle_file}, skipping...")
            return False
        
        # Run CovMat evaluation
        results, rmsd_results, missing = evaluator(ref_data=true_mols, gen_data=model_preds)
        covmat_duration = time.time() - start_time
        
        # Run PoseBusters evaluation if enabled
        posebusters_results = None
        posebusters_summary = None
        posebusters_duration = 0
        
        if run_posebusters:
            print(f"Running PoseBusters evaluation for {pickle_file}...")
            pb_start_time = time.time()
            try:
                _, pb_summary, _, _ = run_all_posebusters(
                    data=model_preds, 
                    config=posebusters_config,
                    full_report=False,
                    max_workers=min(8, mp.cpu_count())  # Limit workers to avoid memory issues
                )
                posebusters_summary = pb_summary
                posebusters_duration = time.time() - pb_start_time
                print(f"PoseBusters completed for {pickle_file} in {posebusters_duration:.2f}s")
            except Exception as e:
                print(f"PoseBusters failed for {pickle_file}: {e}")
                posebusters_duration = time.time() - pb_start_time
        
        total_duration = time.time() - start_time
        print(f"Completed {pickle_file} in {total_duration:.2f}s")

        # Save results
        cov_df, matching_metrics = print_covmat_results(results)
        
        results_file.write(f"Results for {pickle_file}\n")
        results_file.write(f"Total processing time: {total_duration:.2f}s\n")
        results_file.write(f"CovMat processing time: {covmat_duration:.2f}s\n")
        if run_posebusters:
            results_file.write(f"PoseBusters processing time: {posebusters_duration:.2f}s\n")
        results_file.write(f"Number of missing mols: {len(missing)}\n")
        results_file.write(f"Missing: {missing}\n")
        results_file.write(f"Matching metrics: {matching_metrics}\n")
        results_file.write(f"Coverage at threshold 0.75: {cov_df.iloc[14]}\n")
        
        # Add PoseBusters results
        if run_posebusters and posebusters_summary is not None:
            results_file.write(f"\nPoseBusters Results:\n")
            results_file.write(f"Pass percentage: {posebusters_summary['pass_percentage'].iloc[0]:.2f}%\n")
            results_file.write(f"Number of molecules: {posebusters_summary['num_smiles'].iloc[0]}\n")
            results_file.write(f"Number of conformers: {posebusters_summary['num_conformers'].iloc[0]}\n")
            
        
            
        results_file.write("-" * 80 + "\n")
        results_file.flush()
        
        return True
        
    except Exception as e:
        print(f"Error processing {pickle_file}: {e}")
        results_file.write(f"ERROR processing {pickle_file}: {e}\n")
        results_file.write("-" * 80 + "\n")
        results_file.flush()
        return False

def run_evaluation(directory_name: str, gen_base: str, eval_base: str, 
                  true_mols_path: str, num_workers: int = 24, 
                  run_posebusters: bool = True, posebusters_config: str = "mol") -> bool:
    """Run evaluation for a single directory."""
    print(f"Starting evaluation for: {directory_name}")
    
    gens_path = os.path.join(gen_base, directory_name)
    if not os.path.exists(gens_path):
        print(f"Directory does not exist: {gens_path}")
        return False
    
    # Get pickle files
    pickle_files = get_pickle_files(gens_path)
    if not pickle_files:
        print(f"No pickle files found in {directory_name}")
        return False
    
    print(f"Found {len(pickle_files)} pickle files to process")
    
    # Load true molecules
    true_mols = load_molecules(true_mols_path)
    if true_mols is None:
        return False
    
    # Setup results directory
    results_path = os.path.join(eval_base, f"{directory_name}_parallel")
    try:
        os.makedirs(results_path, exist_ok=True)
        results_file = open(os.path.join(results_path, "covmat_results.txt"), "w")
    except Exception as e:
        print(f"Failed to create results directory {results_path}: {e}")
        return False
    
    # Create evaluator
    evaluator = MemoryEfficientCovMatEvaluator(
        num_workers=num_workers,
        use_force_field=False,
        use_alignmol=True,
        max_molecules_in_memory=1000
    )
    
    # Process all pickle files
    total_start = time.time()
    successful_files = 0
    
    try:
        for pickle_file in sorted(pickle_files):
            if process_single_pickle(evaluator, pickle_file, gens_path, true_mols, results_file,
                                   run_posebusters, posebusters_config):
                successful_files += 1
        
        # Write summary
        total_duration = time.time() - total_start
        results_file.write(f"\nEvaluation Summary:\n")
        results_file.write(f"Total processing time: {total_duration:.2f}s\n")
        results_file.write(f"Successful files: {successful_files}/{len(pickle_files)}\n")
        if successful_files > 0:
            results_file.write(f"Average time per file: {total_duration/successful_files:.2f}s\n")
        
        results_file.close()
        
        if successful_files > 0:
            print(f"Evaluation completed for {directory_name} in {total_duration:.2f}s")
            print(f"Successfully processed {successful_files}/{len(pickle_files)} files")
            print(f"Results saved to {results_path}")
            return True
        else:
            print(f"No files were successfully processed for {directory_name}")
            return False
            
    except Exception as e:
        print(f"Critical error evaluating {directory_name}: {e}")
        results_file.close()
        return False

def create_slurm_executor(device: str, num_workers: int):
    """Create and configure slurm executor."""
    if device == "local":
        executor = submitit.LocalExecutor(folder=str(Path.home() / "slurm_jobs/eval/job_%j"))
    else:
        executor = submitit.AutoExecutor(folder=str(Path.home() / "slurm_jobs/eval/job_%j"))
    
    executor.update_parameters(
        name="gen_evals",
        timeout_min=40 * 60,  # 40 hours timeout
        cpus_per_task=num_workers,
        mem_gb=80,
        nodes=1,
        slurm_additional_parameters={"partition": device} if device != "local" else {}
    )
    
    return executor

def submit_evaluation_jobs(directories: list[str], gen_base: str, eval_base: str, 
                          true_mols_path: str, device: str, num_workers: int,
                          run_posebusters: bool = True, posebusters_config: str = "mol") -> list:
    """Submit evaluation jobs to slurm."""
    print(f"Submitting {len(directories)} jobs to slurm on {device}")
    
    executor = create_slurm_executor(device, num_workers)
    submitted_jobs = []
    
    for directory in directories:
        job = executor.submit(
            run_evaluation,
            directory_name=directory,
            gen_base=gen_base,
            eval_base=eval_base,
            true_mols_path=true_mols_path,
            num_workers=num_workers,
            run_posebusters=run_posebusters,
            posebusters_config=posebusters_config
        )
        submitted_jobs.append((directory, job))
        print(f"Submitted job {job.job_id} for directory: {directory}")
    
    return submitted_jobs

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Molecular generation evaluation with slurm support")
    
    parser.add_argument("--device", type=str, choices=["local", "a100", "h100"], default="local",
                       help="Device type for slurm submission")
    parser.add_argument("--num-workers", type=int, default=24,
                       help="Number of workers for evaluation")
    parser.add_argument("--max-recent", type=int, default=3,
                       help="Maximum number of recent directories to evaluate")
    parser.add_argument("--specific-dir", type=str, default=None,
                       help="Specific directory name to evaluate (overrides auto-discovery)")
    parser.add_argument("--gen-results-base", type=str, default=DEFAULT_GEN_RESULTS_BASE,
                       help="Base directory for generation results")
    parser.add_argument("--eval-results-base", type=str, default=DEFAULT_EVAL_RESULTS_BASE,
                       help="Base directory for evaluation results")
    parser.add_argument("--true-mols-path", type=str, default=DEFAULT_TRUE_MOLS_PATH,
                       help="Path to true molecules pickle file")
    parser.add_argument("--no-slurm", action="store_true",
                       help="Run locally instead of submitting to slurm")
    parser.add_argument("--no-posebusters", default=False, action="store_true",
                       help="Skip PoseBusters evaluation")
    parser.add_argument("--posebusters-config", type=str, default="mol", choices=["mol", "redock"],
                       help="PoseBusters configuration to use")
    
    args = parser.parse_args()
    
    run_posebusters = not args.no_posebusters
    
    if args.specific_dir:
        # Evaluate specific directory
        print(f"Evaluating specific directory: {args.specific_dir}")
        
        if args.no_slurm:
            # Run locally
            success = run_evaluation(
                directory_name=args.specific_dir,
                gen_base=args.gen_results_base,
                eval_base=args.eval_results_base,
                true_mols_path=args.true_mols_path,
                num_workers=args.num_workers,
                run_posebusters=run_posebusters,
                posebusters_config=args.posebusters_config
            )
            print(f"{'Successfully' if success else 'Failed to'} evaluate {args.specific_dir}")
        else:
            # Submit to slurm
            jobs = submit_evaluation_jobs(
                directories=[args.specific_dir],
                gen_base=args.gen_results_base,
                eval_base=args.eval_results_base,
                true_mols_path=args.true_mols_path,
                device=args.device,
                num_workers=args.num_workers,
                run_posebusters=run_posebusters,
                posebusters_config=args.posebusters_config
            )
            print(f"Submitted slurm job for {args.specific_dir}")
            print(f"Job ID: {jobs[0][1].job_id}")
    
    else:
        # Auto-discover and evaluate missing directories
        print("Finding directories needing evaluation...")
        missing_dirs = get_missing_evaluation_dirs(
            gen_base=args.gen_results_base,
            eval_base=args.eval_results_base,
            max_recent=args.max_recent
        )
        
        if not missing_dirs:
            print("All recent generation directories have been evaluated!")
            return
        
        print(f"Found {len(missing_dirs)} directories needing evaluation:")
        for dir_name in missing_dirs:
            print(f"  - {dir_name}")
        
        if args.max_recent and len(missing_dirs) == args.max_recent:
            print(f"ℹLimited to {args.max_recent} most recent. Increase --max-recent to evaluate more.")
        
        # Submit to slurm
        submitted_jobs = submit_evaluation_jobs(
            directories=missing_dirs,
            gen_base=args.gen_results_base,
            eval_base=args.eval_results_base,
            true_mols_path=args.true_mols_path,
            device=args.device,
            num_workers=args.num_workers,
            run_posebusters=run_posebusters,
            posebusters_config=args.posebusters_config
        )
        
        print(f"\nSlurm submission summary:")
        print(f"Submitted {len(submitted_jobs)} jobs to {args.device}")
        print(f"Workers per job: {args.num_workers}")
        print(f"PoseBusters enabled: {run_posebusters}")
        if run_posebusters:
            print(f"PoseBusters config: {args.posebusters_config}")
        print(f"Results will be saved in: {args.eval_results_base}")
        print(f"\nJob details:")
        for directory, job in submitted_jobs:
            print(f"  - {directory}: Job ID {job.job_id}")

if __name__ == "__main__":
    main()