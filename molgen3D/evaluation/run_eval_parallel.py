import pickle
import os
import multiprocessing as mp
from improved_covmat import MemoryEfficientCovMatEvaluator, print_covmat_results
from loguru import logger as log
import time
from pathlib import Path
import argparse
import submitit
import sys
from datetime import datetime

# Default Configuration
DEFAULT_GEN_RESULTS_BASE = "/auto/home/menuab/code/3DMolGen/gen_results/"
DEFAULT_EVAL_RESULTS_BASE = "/auto/home/menuab/code/3DMolGen/eval_results/"
DEFAULT_TRUE_MOLS_PATH = "/auto/home/menuab/code/3DMolGen/data/geom_drugs_test_set/drugs_test_inference.pickle"

def setup_logging():
    log.remove()
    log.add(
        sys.stderr,
        format="<green>{time:HH:mm}</green> | <level>{level: <4}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

def find_pickle_files(base_dir):
    """Finds all .pickle files recursively and returns their relative paths."""
    pickle_files = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        log.warning(f"Directory does not exist: {base_dir}")
        return pickle_files
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".pickle"):
                relative_path = os.path.relpath(os.path.join(root, file), base_dir)
                pickle_files.append(relative_path)
    
    return pickle_files

def find_missing_evaluations(gen_results_base: str, eval_results_base: str, max_recent: int = None):
    """Find generation directories that don't have corresponding evaluation results."""
    gen_base = Path(gen_results_base)
    eval_base = Path(eval_results_base)
    
    if not gen_base.exists():
        log.error(f"Generation results directory not found: {gen_base}")
        return []
    
    # Get all generation directories
    gen_dirs = []
    for item in gen_base.iterdir():
        if item.is_dir():
            mod_time = item.stat().st_mtime
            gen_dirs.append((item.name, mod_time))
    
    log.info(f"Found {len(gen_dirs)} generation directories")
    
    # Sort by modification time (most recent first)
    gen_dirs.sort(key=lambda x: x[1], reverse=True)
    
    # Find missing evaluations (simple directory name check)
    missing_dirs = []
    for gen_dir, mod_time in gen_dirs:
        eval_dir = eval_base / f"{gen_dir}_parallel"
        
        if not eval_dir.exists():
            missing_dirs.append(gen_dir)
            # Convert timestamp to readable format
            import datetime
            readable_time = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            log.info(f"Missing evaluation: {gen_dir} (modified: {readable_time})")
    
    # Limit to most recent ones if specified
    if max_recent is not None and len(missing_dirs) > max_recent:
        log.info(f"Limiting to {max_recent} most recent missing evaluations")
        missing_dirs = missing_dirs[:max_recent]
    
    log.info(f"Will check {len(missing_dirs)} recent directories")
    return missing_dirs

def evaluate_directory(directory_name: str, gen_results_base: str, eval_results_base: str, 
                      true_mols_path: str, num_workers: int = 24):
    """Evaluate a specific directory."""
    log.info(f"Starting evaluation for: {directory_name}")
    
    gens_path = os.path.join(gen_results_base, directory_name)
    
    # Quick check if directory exists
    if not os.path.exists(gens_path):
        log.error(f"Directory does not exist: {gens_path}")
        return False
    
    # Quick check for pickle files
    pickle_files = find_pickle_files(gens_path)
    if not pickle_files:
        print(f"No pickle files found in {directory_name}")
        return False
    
    log.info(f"Found {len(pickle_files)} pickle files to process")
    
    # Load true molecules
    try:
        with open(true_mols_path, 'rb') as f:
            true_mols = pickle.load(f)
    except Exception as e:
        log.error(f"Failed to load true molecules from {true_mols_path}: {e}")
        return False
    
    results_path = os.path.join(eval_results_base, f"{directory_name}_parallel")
    
    try:
        os.makedirs(results_path, exist_ok=True)
        results_file = open(os.path.join(results_path, "covmat_results.txt"), "w")
    except Exception as e:
        log.error(f"Failed to create results directory {results_path}: {e}")
        return False
    
    # Create the best performing evaluator
    evaluator = MemoryEfficientCovMatEvaluator(
        num_workers=num_workers,
        use_force_field=False,
        use_alignmol=False,
        max_molecules_in_memory=500
    )
    
    total_start_time = time.time()
    successful_files = 0
    failed_files = 0
    
    try:
        for i, pickle_path in enumerate(sorted(pickle_files)):
            log.info(f"Processing file {i+1}/{len(pickle_files)}: {pickle_path}")
            
            file_start_time = time.time()
            
            try:
                # Load generated molecules
                full_path = os.path.join(gens_path, pickle_path)
                with open(full_path, 'rb') as file:
                    model_preds = pickle.load(file)
                
                # Validate the loaded data
                if not model_preds:
                    log.warning(f"Empty data in {pickle_path}, skipping...")
                    failed_files += 1
                    continue
                
                # Run evaluation with the optimized evaluator
                results, rmsd_results, missing = evaluator(ref_data=true_mols, gen_data=model_preds)
                
                file_duration = time.time() - file_start_time
                log.info(f"Completed {pickle_path} in {file_duration:.2f}s")

                # Process and save results
                cov_df, matching_metrics = print_covmat_results(results)

                results_file.write(f"Results for {pickle_path}\n")
                results_file.write(f"Processing time: {file_duration:.2f}s\n")
                results_file.write(f"Number of missing mols: {len(missing)}\n")
                results_file.write(f"Missing: {missing}\n")
                results_file.write(f"Matching metrics: {matching_metrics}\n")
                results_file.write(f"Coverage at threshold 0.75: {cov_df.iloc[14]}\n")
                results_file.write("-" * 80 + "\n")
                results_file.flush()
                
                successful_files += 1
                
            except Exception as e:
                log.error(f"Error processing {pickle_path}: {e}")
                results_file.write(f"ERROR processing {pickle_path}: {e}\n")
                results_file.write("-" * 80 + "\n")
                results_file.flush()
                failed_files += 1

        total_duration = time.time() - total_start_time
        results_file.write(f"\nEvaluation Summary:\n")
        results_file.write(f"Total processing time: {total_duration:.2f}s\n")
        results_file.write(f"Successful files: {successful_files}\n")
        results_file.write(f"Failed files: {failed_files}\n")
        if successful_files > 0:
            results_file.write(f"Average time per successful file: {total_duration/successful_files:.2f}s\n")
        results_file.close()

        if successful_files > 0:
            log.info(f"Evaluation completed for {directory_name} in {total_duration:.2f}s")
            log.info(f"Successfully processed {successful_files}/{len(pickle_files)} files")
            log.info(f"Results saved to {results_path}")
            return True
        else:
            log.error(f"No files were successfully processed for {directory_name}")
            return False
        
    except Exception as e:
        log.error(f"Critical error evaluating {directory_name}: {e}")
        results_file.close()
        return False

def evaluation_job(directory_name: str, gen_results_base: str, eval_results_base: str, 
                  true_mols_path: str, num_workers: int):
    """Job function that can be submitted to slurm."""
    setup_logging()
    log.info(f"Starting slurm job for directory: {directory_name}")
    
    success = evaluate_directory(
        directory_name=directory_name,
        gen_results_base=gen_results_base,
        eval_results_base=eval_results_base,
        true_mols_path=true_mols_path,
        num_workers=num_workers
    )
    
    if success:
        log.info(f"Slurm job completed successfully for {directory_name}")
    else:
        log.error(f"Slurm job failed for {directory_name}")
    
    return success

def setup_argparse():
    """Setup command line argument parsing."""
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
    
    return parser

def submit_slurm_jobs(directories: list, args):
    """Submit evaluation jobs to slurm."""
    log.info(f"Submitting {len(directories)} jobs to slurm on {args.device}")
    
    if args.device == "local":
        executor = submitit.LocalExecutor(folder=str(Path.home() / "slurm_jobs/eval/job_%j"))
    else:
        executor = submitit.AutoExecutor(folder=str(Path.home() / "slurm_jobs/eval/job_%j"))
    
    
    executor.update_parameters(
        name=f"eval_{args.device}",
        timeout_min=4 * 60,  # 4 hours timeout
        cpus_per_task=args.cpus_per_task,
        mem_gb=20,
        nodes=1,
        slurm_additional_parameters={"partition": args.device} if args.device != "local" else {}
    )
    
    job = executor.submit(
        evaluation_job,
        directory_name=directory,
        gen_results_base=args.gen_results_base,
        eval_results_base=args.eval_results_base,
        true_mols_path=args.true_mols_path,
        num_workers=args.num_workers
    )
    submitted_jobs.append((directory, job))
    log.info(f"Submitted job {job.job_id} for directory: {directory}")
    
    return submitted_jobs

if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.device == "local":
            executor = submitit.LocalExecutor(folder=str(Path.home() / "slurm_jobs/gen_evals/job_%j"))
    else:
        executor = submitit.AutoExecutor(folder=str(Path.home() / "slurm_jobs/gen_evals/job_%j"))

    executor.update_parameters(
        name=args.run_name,
        timeout_min=24 * 24 * 60,
        nodes=1,
        mem_gb=80,
        cpus_per_task=args.num_cpus,
        slurm_additional_parameters={"partition": args.device_name,},
    )
    job = executor.submit()
    
    if args.specific_dir:
        # Evaluate specific directory
        log.info(f"Evaluating specific directory: {args.specific_dir}")
        
        if args.no_slurm:
            # Run locally
            success = evaluate_directory(
                directory_name=args.specific_dir,
                gen_results_base=args.gen_results_base,
                eval_results_base=args.eval_results_base,
                true_mols_path=args.true_mols_path,
                num_workers=args.num_workers
            )
            if success:
                print(f"Successfully evaluated {args.specific_dir}")
            else:
                print(f"Failed to evaluate {args.specific_dir}")
        else:
            # Submit to slurm
            submitted_jobs = submit_slurm_jobs([args.specific_dir], args)
            print(f"Submitted slurm job for {args.specific_dir}")
            print(f"Job ID: {submitted_jobs[0][1].job_id}")
    
    else:
        # Auto-discover and evaluate missing directories
        log.info("Finding directories needing evaluation...")
        missing_dirs = find_missing_evaluations(
            gen_results_base=args.gen_results_base,
            eval_results_base=args.eval_results_base,
            max_recent=args.max_recent
        )
        
        if not missing_dirs:
            print("All recent generation directories have been evaluated!")
            sys.exit(0)
        
        print(f"Found {len(missing_dirs)} directories needing evaluation:")
        for dir_name in missing_dirs:
            print(f"  - {dir_name}")
        
        if args.max_recent and len(missing_dirs) == args.max_recent:
            print(f"ℹLimited to {args.max_recent} most recent. Increase --max-recent to evaluate more.")
        
        if args.no_slurm:
            # Run locally (original behavior)
            successful = 0
            failed = 0
            total_start = time.time()
            
            for i, dir_name in enumerate(missing_dirs):
                print(f"\nChecking {i+1}/{len(missing_dirs)}: {dir_name}")
                success = evaluate_directory(
                    directory_name=dir_name,
                    gen_results_base=args.gen_results_base,
                    eval_results_base=args.eval_results_base,
                    true_mols_path=args.true_mols_path,
                    num_workers=args.num_workers
                )
                
                if success:
                    successful += 1
                    print(f"Completed {dir_name}")
                else:
                    failed += 1
                    print(f"Skipped {dir_name}")
            
            total_time = time.time() - total_start
            
            print(f"\nSummary:")
            print(f"Evaluated: {successful}")
            print(f"Skipped: {failed}")
            print(f"Total time: {total_time:.1f}s")
            if successful > 0:
                print(f"Results saved in: {args.eval_results_base}")
        else:
            # Submit to slurm
            submitted_jobs = submit_slurm_jobs(missing_dirs, args)
            
            print(f"\nSlurm submission summary:")
            print(f"Submitted {len(submitted_jobs)} jobs to {args.device}")
            print(f"Workers per job: {args.num_workers}")
            print(f"Results will be saved in: {args.eval_results_base}")
            print(f"\nJob details:")
            for directory, job in submitted_jobs:
                print(f"  - {directory}: Job ID {job.job_id}")
            
            print(f"\nMonitor jobs with: squeue -u $USER")
            print(f"Check job logs in: ~/slurm_jobs/eval/") 