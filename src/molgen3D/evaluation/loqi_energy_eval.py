"""
LOQI-style energy evaluation for generated molecular conformers.

This script evaluates generated conformers using AIMNet2 to compute:
- Initial energies for generated and reference conformers
- Relative energies (generated vs reference)
- Optimized geometries and energies
- Geometry metrics (bond lengths, angles, torsions)
- Topology preservation after optimization

Adapted from LoQI repository evaluation methodology.
"""

import argparse
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm
import submitit

from molgen3D.config.paths import get_data_path, get_base_path
from molgen3D.data_processing.utils import load_pkl
from molgen3D.evaluation import rdkit_utils
from molgen3D.evaluation.aimnet2_metrics import MoleculeAIMNet2Metrics, is_valid
from molgen3D.evaluation.utils import find_generation_pickles_path, create_slurm_executor


def prepare_loqi_molecules(
    gen_data: Dict[str, List[Chem.Mol]],
    gt_data: Dict[str, Dict]
) -> Tuple[List[str], List[Chem.Mol], List[Chem.Mol]]:
    """
    Prepare molecules for LOQI evaluation by pairing generated with reference conformers.

    Args:
        gen_data: Dictionary mapping SMILES to list of generated molecules
        gt_data: Dictionary mapping SMILES to dict containing reference conformers

    Returns:
        Tuple of (smiles_list, gen_mols, ref_mols)
    """
    smiles_list = []
    gen_mols = []
    ref_mols = []

    # Sort molecules by size (biggest first) to catch memory issues early
    sorted_smiles = sorted(
        gt_data.keys(),
        key=lambda s: gt_data[s].get('confs', [None])[0].GetNumAtoms() if gt_data[s].get('confs') else 0,
        reverse=True
    )

    for smi in tqdm(sorted_smiles, desc="Preparing molecules"):
        gt_entry = gt_data[smi]
        ref_confs = gt_entry.get('confs', [])

        if not ref_confs:
            continue

        gen_confs = gen_data.get(smi, [])
        if not gen_confs:
            continue

        # Filter valid molecules
        valid_gen = [mol for mol in gen_confs if is_valid(mol)]
        if not valid_gen:
            continue

        # Use first reference conformer (typically lowest energy)
        ref_mol = ref_confs[0]
        if not is_valid(ref_mol):
            continue

        # Add each generated conformer with the same reference
        for gen_mol in valid_gen:
            smiles_list.append(smi)
            gen_mols.append(gen_mol)
            ref_mols.append(ref_mol)

    print(f"Prepared {len(gen_mols)} conformers from {len(set(smiles_list))} molecules")
    return smiles_list, gen_mols, ref_mols


def compute_loqi_per_molecule_stats(
    smiles_list: List[str],
    gen_energies: np.ndarray,
    ref_energies: np.ndarray,
    opt_energies: Optional[np.ndarray] = None,
    topology_mask: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compute per-molecule LOQI statistics using vectorized pandas operations.

    Args:
        smiles_list: List of SMILES for each conformer
        gen_energies: Initial energies (eV)
        ref_energies: Reference energies (eV)
        opt_energies: Optimized energies (eV), optional
        topology_mask: Topology preservation mask, optional

    Returns:
        DataFrame with per-molecule statistics
    """
    ev2kcalpermol = 23.060547830619026

    # Create DataFrame for vectorized operations
    df = pd.DataFrame({
        'smiles': smiles_list,
        'gen_energy_eV': gen_energies,
        'ref_energy_eV': ref_energies,
    })
    df['relative_energy_kcal'] = (df['gen_energy_eV'] - df['ref_energy_eV']) * ev2kcalpermol

    if opt_energies is not None:
        df['opt_energy_eV'] = opt_energies
        df['opt_relative_energy_kcal'] = (df['opt_energy_eV'] - df['ref_energy_eV']) * ev2kcalpermol

    if topology_mask is not None:
        df['topology_preserved'] = topology_mask

    # Aggregate using groupby (much faster than manual loops)
    agg_dict = {
        'gen_energy_eV': ['mean', 'min'],
        'ref_energy_eV': 'first',
        'relative_energy_kcal': ['mean', 'min', 'median'],
        'smiles': 'count'  # for n_conformers
    }

    if opt_energies is not None:
        agg_dict['opt_energy_eV'] = ['mean', 'min']
        agg_dict['opt_relative_energy_kcal'] = ['mean', 'min']

    if topology_mask is not None:
        agg_dict['topology_preserved'] = 'mean'

    result = df.groupby('smiles', sort=False).agg(agg_dict)

    # Flatten column names
    result.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                      for col in result.columns]
    result = result.rename(columns={
        'gen_energy_eV_mean': 'mean_gen_energy_eV',
        'gen_energy_eV_min': 'min_gen_energy_eV',
        'ref_energy_eV_first': 'ref_energy_eV',
        'relative_energy_kcal_mean': 'mean_relative_energy_kcal',
        'relative_energy_kcal_min': 'min_relative_energy_kcal',
        'relative_energy_kcal_median': 'median_relative_energy_kcal',
        'smiles_count': 'n_conformers',
    })

    if opt_energies is not None:
        result = result.rename(columns={
            'opt_energy_eV_mean': 'mean_opt_energy_eV',
            'opt_energy_eV_min': 'min_opt_energy_eV',
            'opt_relative_energy_kcal_mean': 'mean_opt_relative_energy_kcal',
            'opt_relative_energy_kcal_min': 'min_opt_relative_energy_kcal',
        })
        result['topology_preservation_rate'] = result.get('topology_preserved_mean', np.nan)

        # Compute valid optimized metrics (only where topology is preserved)
        if topology_mask is not None:
            valid_df = df[df['topology_preserved']].copy()
            if len(valid_df) > 0:
                valid_agg = valid_df.groupby('smiles')['opt_relative_energy_kcal'].agg([
                    ('mean_opt_rel_valid_kcal', 'mean'),
                    ('min_opt_rel_valid_kcal', 'min'),
                    ('within_01kcal', lambda x: (x < 0.1).mean()),
                    ('better_than_ref', lambda x: (x < -0.1).mean())
                ])
                result = result.join(valid_agg, how='left')

    result = result.reset_index()
    return result


def process_loqi_evaluation(
    gens_dict: Dict,
    gt_dict: Dict,
    gens_path: str,
    results_path: str,
    args: argparse.Namespace
) -> Dict:
    """
    Process LOQI evaluation for a single generation directory.

    Args:
        gens_dict: Generated conformers dictionary
        gt_dict: Ground truth dictionary
        gens_path: Path to generation directory
        results_path: Path to save results
        args: Command line arguments

    Returns:
        Dictionary with LOQI metrics
    """
    t0 = time.time()

    # Process generated molecules
    processed_gen_data = rdkit_utils.process_molecules_remove_hs(gens_dict)
    print(f"Processed {len(processed_gen_data)} molecules, removing hydrogens")

    # Prepare molecules for LOQI evaluation
    print("Preparing molecules for LOQI evaluation...")
    smiles_list, gen_mols, ref_mols = prepare_loqi_molecules(processed_gen_data, gt_dict)

    if not gen_mols:
        print("No valid molecules to evaluate!")
        return {}

    print(f"Evaluating {len(gen_mols)} conformers from {len(set(smiles_list))} molecules")

    # Initialize AIMNet2 calculator
    aimnet2_model_path = get_data_path("aimnet2_model")
    print(f"Using AIMNet2 model: {aimnet2_model_path}")

    opt_params = {
        'fmax': args.fmax,
        'max_nstep': args.max_steps,
    }

    calculator = MoleculeAIMNet2Metrics(
        model_path=str(aimnet2_model_path),
        device=args.device,
        batchsize=args.batch_size,
        opt_metrics=not args.no_opt,
        opt_params=opt_params,
        chunked=True  # Groups molecules by size for efficient batching
    )

    # Compute metrics
    t_calc_start = time.time()
    print(f"Calculating energies and {'optimizing' if not args.no_opt else 'forces'}...")

    if not args.no_opt:
        gen_metrics, valid_mols, opt_mols, opt_energies = calculator(
            gen_mols,
            reference_molecules=ref_mols,
            return_molecules=True
        )
    else:
        gen_metrics = calculator(gen_mols, reference_molecules=ref_mols)
        opt_energies = None

    calc_time = time.time() - t_calc_start
    print(f"AIMNet2 calculation completed in {calc_time:.2f} seconds")

    # Extract results
    gen_energies = np.array(gen_metrics['energies'])

    # Compute reference energies efficiently (cache unique references)
    # Many conformers may share the same reference molecule
    unique_refs = {}
    ref_indices = []
    for i, ref_mol in enumerate(ref_mols):
        smi = smiles_list[i]
        if smi not in unique_refs:
            unique_refs[smi] = ref_mol
        ref_indices.append(smi)

    print(f"Computing energies for {len(unique_refs)} unique reference molecules...")
    unique_ref_mols = list(unique_refs.values())
    unique_ref_smiles = list(unique_refs.keys())

    ref_metrics = calculator(unique_ref_mols, return_molecules=False)
    unique_ref_energies = np.array(ref_metrics['energies'])

    # Map back to full reference energy array
    ref_energy_map = dict(zip(unique_ref_smiles, unique_ref_energies))
    ref_energies = np.array([ref_energy_map[smi] for smi in ref_indices])

    # Compute statistics
    ev2kcalpermol = 23.060547830619026
    relative_energies = (gen_energies - ref_energies) * ev2kcalpermol
    valid_rel_energies = relative_energies[~np.isnan(relative_energies)]

    loqi_metrics = {
        'n_molecules': len(set(smiles_list)),
        'n_conformers': len(gen_mols),
        'mean_relative_energy': float(np.mean(valid_rel_energies)) if len(valid_rel_energies) > 0 else 0.0,
        'median_relative_energy': float(np.median(valid_rel_energies)) if len(valid_rel_energies) > 0 else 0.0,
        'within_01kcal_gt': float(np.mean(valid_rel_energies < 0.1)) if len(valid_rel_energies) > 0 else 0.0,
        'better_than_gt': float(np.mean(valid_rel_energies < 0.0)) if len(valid_rel_energies) > 0 else 0.0,
        'std_relative_energy': float(np.std(valid_rel_energies)) if len(valid_rel_energies) > 0 else 0.0,
        'min_relative_energy': float(np.min(valid_rel_energies)) if len(valid_rel_energies) > 0 else 0.0,
        'avg_max_forces': gen_metrics.get('avg_max_forces', np.nan),
        'median_max_forces': gen_metrics.get('median_max_forces', np.nan),
    }

    # Add optimization metrics if computed
    if not args.no_opt:
        loqi_metrics.update({
            'opt_converged': gen_metrics.get('opt_converged', 0.0),
            'opt_steps': gen_metrics.get('opt_steps', 0.0),
            'preserved_topology': gen_metrics.get('preserved_topology', 0.0),
            'opt_avg_energy_drop': gen_metrics.get('opt_avg_energy_drop', 0.0),
            'opt_median_energy_drop': gen_metrics.get('opt_median_energy_drop', 0.0),
            'opt_median_relative_energy': gen_metrics.get('opt_median_relative_energy', 0.0),
            'opt_within_01kcal_gt': gen_metrics.get('opt_within_01kcal_gt', 0.0),
            'opt_better_than_gt': gen_metrics.get('opt_better_than_gt', 0.0),
            'opt_bond_lengths_diff': gen_metrics.get('opt_bond_lengths_diff', 0.0),
            'opt_bond_angles_diff': gen_metrics.get('opt_bond_angles_diff', 0.0),
            'opt_dihedrals_diff': gen_metrics.get('opt_dihedrals_diff', 0.0),
        })

        # Compute per-molecule statistics
        topology_mask = gen_metrics.get('topology_mask')
        if topology_mask is not None:
            topology_mask = topology_mask.cpu().numpy()
        per_mol_df = compute_loqi_per_molecule_stats(
            smiles_list,
            gen_energies,
            ref_energies,
            opt_energies.cpu().numpy() if opt_energies is not None else None,
            topology_mask
        )
    else:
        per_mol_df = compute_loqi_per_molecule_stats(
            smiles_list,
            gen_energies,
            ref_energies
        )

    # Save results
    os.makedirs(results_path, exist_ok=True)

    # Save summary
    summary_path = os.path.join(results_path, "loqi_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LOQI (AIMNet2) ENERGY EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Number of molecules: {loqi_metrics['n_molecules']}\n")
        f.write(f"Number of conformers: {loqi_metrics['n_conformers']}\n\n")
        
        f.write("INITIAL CONFORMER METRICS:\n")
        f.write(f"- Median relative energy (gen - gt): {loqi_metrics['median_relative_energy']:.4f} kcal/mol\n")
        f.write(f"- % of conformers within 0.1 kcal/mol of GT: {loqi_metrics['within_01kcal_gt']*100:.2f}%\n")
        f.write(f"- % of conformers with lower energy than GT: {loqi_metrics['better_than_gt']*100:.2f}%\n")
        f.write(f"- Mean relative energy: {loqi_metrics['mean_relative_energy']:.4f} kcal/mol\n")
        f.write(f"- Min relative energy: {loqi_metrics['min_relative_energy']:.4f} kcal/mol\n\n")
        
        if not args.no_opt:
            f.write("OPTIMIZED CONFORMER METRICS:\n")
            f.write(f"- % minimization converged: {loqi_metrics['opt_converged']*100:.2f}%\n")
            f.write(f"- % topology preserved: {loqi_metrics['preserved_topology']*100:.2f}%\n")
            f.write(f"- Median energy drop (gen -> opt): {loqi_metrics['opt_median_energy_drop']:.4f} kcal/mol\n")
            f.write(f"- Median relative energy (opt - gt): {loqi_metrics['opt_median_relative_energy']:.4f} kcal/mol\n")
            f.write(f"- % of optimized conformers within 0.1 kcal/mol of GT: {loqi_metrics['opt_within_01kcal_gt']*100:.2f}%\n")
            f.write(f"- % of optimized conformers with lower energy than GT: {loqi_metrics['opt_better_than_gt']*100:.2f}%\n")
            f.write(f"- Mean energy drop: {loqi_metrics['opt_avg_energy_drop']:.4f} kcal/mol\n")
            f.write(f"- Avg optimization steps: {loqi_metrics['opt_steps']:.1f}\n\n")
            
            f.write("GEOMETRY DIFFERENCES (gen vs opt):\n")
            f.write(f"- Bond lengths RMSD: {loqi_metrics['opt_bond_lengths_diff']:.4f} A\n")
            f.write(f"- Bond angles RMSD: {loqi_metrics['opt_bond_angles_diff']:.4f} deg\n")
            f.write(f"- Dihedral angles RMSD: {loqi_metrics['opt_dihedrals_diff']:.4f} deg\n\n")

        f.write("FORCE METRICS:\n")
        f.write(f"- Mean max force: {loqi_metrics['avg_max_forces']:.6f} eV/A\n")
        f.write(f"- Median max force: {loqi_metrics['median_max_forces']:.6f} eV/A\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"Evaluation completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")

    # Save per-molecule results
    per_mol_path = os.path.join(results_path, "loqi_per_molecule.csv")
    per_mol_df.to_csv(per_mol_path, index=False)
    print(f"Saved per-molecule results to {per_mol_path}")

    # Save full metrics as pickle
    full_results = {
        'loqi_metrics': loqi_metrics,
        'per_molecule': per_mol_df,
        'gen_metrics': gen_metrics,
        'ref_metrics': ref_metrics,
    }
    results_pickle = os.path.join(results_path, "loqi_results.pickle")
    with open(results_pickle, 'wb') as f:
        pickle.dump(full_results, f, protocol=4)

    total_time = time.time() - t0
    print(f"\nTotal LOQI evaluation time: {total_time:.2f} seconds")
    print(f"Mean relative energy: {loqi_metrics['mean_relative_energy']:.3f} kcal/mol")
    print(f"Median relative energy: {loqi_metrics['median_relative_energy']:.3f} kcal/mol")

    if not args.no_opt:
        print(f"Optimization converged: {loqi_metrics['opt_converged']*100:.1f}%")
        print(f"Topology preserved: {loqi_metrics['preserved_topology']*100:.1f}%")

    return loqi_metrics


def get_missing_loqi_evaluation_dirs(gen_base: str, max_recent: Optional[int] = None) -> List[str]:
    """
    Find generation directories that don't have LOQI evaluation results yet.

    Args:
        gen_base: Base directory for generation results
        max_recent: Maximum number of recent directories to return

    Returns:
        List of directory names missing LOQI evaluation
    """
    gen_path = Path(gen_base)
    if not gen_path.exists():
        return []

    gen_dirs: List[Tuple[str, float]] = []
    for item in gen_path.iterdir():
        if item.is_dir():
            gen_dirs.append((item.name, item.stat().st_mtime))

    # Sort by modification time (most recent first)
    gen_dirs.sort(key=lambda x: x[1], reverse=True)

    # Find directories without loqi_eval subdirectory
    missing: List[str] = []
    for name, _ in gen_dirs:
        loqi_eval_path = gen_path / name / "loqi_eval"
        if not loqi_eval_path.exists():
            missing.append(name)

    if max_recent and len(missing) > max_recent:
        missing = missing[:max_recent]

    return missing


def run_loqi_evaluation_for_dir(
    directory_name: str,
    gen_base: Optional[str] = None,
    args: Optional[argparse.Namespace] = None
) -> bool:
    """
    Run LOQI evaluation for a single generation directory.

    Args:
        directory_name: Name of generation directory
        gen_base: Base path for generation results (uses config if None)
        args: Command line arguments

    Returns:
        True if successful, False otherwise
    """
    print(f"Starting LOQI evaluation for: {directory_name}")

    if gen_base is None:
        gen_base = str(get_base_path("gen_results_root"))

    gens_path = os.path.join(gen_base, directory_name)

    if not os.path.exists(gens_path):
        print(f"Directory does not exist: {gens_path}")
        return False

    # Find generation pickle
    gen_pickle_path = find_generation_pickles_path(gens_path)
    if not gen_pickle_path:
        print(f"No pickle files found in {directory_name}")
        return False

    print(f"Loading generations from: {gen_pickle_path}")
    gens_dict = load_pkl(gen_pickle_path)
    print(f"Loaded {len(gens_dict)} generated molecules")

    # Load ground truth
    gt_dict = load_pkl(get_data_path("loqi_smi"))
    print(f"Loaded {len(gt_dict)} ground truth molecules")

    # Create results directory
    results_path = os.path.join(gens_path, "loqi_eval")

    # Run evaluation
    loqi_metrics = process_loqi_evaluation(
        gens_dict=gens_dict,
        gt_dict=gt_dict,
        gens_path=gens_path,
        results_path=results_path,
        args=args
    )

    if loqi_metrics:
        print(f"\nResults saved to: {results_path}")
        return True
    else:
        print("Evaluation failed")
        return False


def monitor_slurm_jobs(jobs: List[Tuple[str, submitit.Job]], args: argparse.Namespace, executor=None, gen_base=None, slurm_args=None) -> None:
    """
    Monitor SLURM jobs and report completion status.

    Args:
        jobs: List of (directory_name, job) tuples
        args: Command line arguments
    """
    import time

    print(f"\nMonitoring {len(jobs)} SLURM jobs...")
    print("Jobs will run asynchronously. Check SLURM queue with: squeue -u $USER")

    # Print job summary
    print("\nJob Summary:")
    for directory, job in jobs:
        print(f"  {directory}: {job.job_id}")

    print(f"\nTo check job status: squeue -j {','.join([str(job.job_id) for _, job in jobs])}")
    print("To cancel jobs: scancel " + " ".join([str(job.job_id) for _, job in jobs]))

    # Optionally wait for completion if requested
    if args.wait_for_completion:
        print("\nWaiting for all jobs to complete...")
        completed = []
        failed = []
        retry_counts = {directory: 0 for directory, _ in jobs}

        while len(completed) + len(failed) < len(jobs):
            time.sleep(60)  # Check every minute

            jobs_to_retry = []
            for directory, job in jobs:
                if directory in completed or directory in failed:
                    continue

                try:
                    result = job.result()  # This will raise exception if job failed
                    if result:  # Assuming function returns True on success
                        completed.append(directory)
                        print(f"✓ {directory} completed successfully")
                    else:
                        if retry_counts[directory] < args.max_retries:
                            retry_counts[directory] += 1
                            print(f"⚠ {directory} failed, will retry ({retry_counts[directory]}/{args.max_retries})")
                            jobs_to_retry.append(directory)
                        else:
                            failed.append(directory)
                            print(f"✗ {directory} failed permanently after {args.max_retries} retries")
                except Exception as e:
                    if "Job not completed" in str(e):
                        continue  # Still running
                    else:
                        if retry_counts[directory] < args.max_retries:
                            retry_counts[directory] += 1
                            print(f"⚠ {directory} failed with error, will retry ({retry_counts[directory]}/{args.max_retries}): {e}")
                            jobs_to_retry.append(directory)
                        else:
                            failed.append(directory)
                            print(f"✗ {directory} failed permanently after {args.max_retries} retries: {e}")

            # Resubmit failed jobs
            for directory in jobs_to_retry:
                new_job = executor.submit(
                    run_loqi_evaluation_for_dir,
                    directory_name=directory,
                    gen_base=gen_base,
                    args=slurm_args,
                )
                # Update job in jobs list
                for i, (dir_name, _) in enumerate(jobs):
                    if dir_name == directory:
                        jobs[i] = (directory, new_job)
                        break

        print("\nFinal Results:")
        print(f"  Completed: {len(completed)}")
        print(f"  Failed: {len(failed)}")

        if failed:
            print(f"  Failed directories: {', '.join(failed)}")


def run_directory_mode(args: argparse.Namespace) -> None:
    """
    Run LOQI evaluation on multiple directories, either locally or via slurm.

    Args:
        args: Command line arguments with device, specific_dir, max_recent, etc.
    """
    gen_base = str(get_base_path("gen_results_root"))

    # Determine which directories to evaluate
    if args.specific_dir:
        gens_path = os.path.join(gen_base, args.specific_dir)
        if not os.path.exists(gens_path):
            print(f"Error: Specified directory does not exist: {gens_path}")
            return
        directories = [args.specific_dir]
        print(f"Evaluating specific directory: {args.specific_dir}")
    else:
        directories = get_missing_loqi_evaluation_dirs(gen_base, args.max_recent)
        if not directories:
            print("All recent generation directories have been evaluated")
            return
        print(f"Found {len(directories)} directories without LOQI evaluation")

    if args.device == "local":
        # Run locally
        print(f"Running {len(directories)} LOQI evaluations locally")
        # Set device to cuda if available, otherwise cpu
        local_args = argparse.Namespace(**vars(args))
        local_args.device = "cuda" if torch.cuda.is_available() else "cpu"
        for directory in directories:
            print(f"\nProcessing: {directory}")
            success = run_loqi_evaluation_for_dir(directory, gen_base, local_args)
            if not success:
                print(f"Failed to evaluate: {directory}")
    else:
        # Submit to slurm - jobs will use cuda when they run
        print(f"Submitting {len(directories)} LOQI evaluation jobs to {args.device}")
        executor = create_slurm_executor(
            device=args.device,
            job_type="loqi_eval",
            num_gpus=args.slurm_gpus,
            num_cpus=args.slurm_cpus,
            job_name=args.slurm_job_name,
            memory_gb=args.slurm_memory_gb,
            timeout_min=args.slurm_time_hours * 60,
        )

        # Create args for slurm jobs with cuda device
        slurm_args = argparse.Namespace(**vars(args))
        slurm_args.device = "cuda"  # Jobs run on GPU nodes

        jobs = []
        for directory in directories:
            job = executor.submit(
                run_loqi_evaluation_for_dir,
                directory_name=directory,
                gen_base=gen_base,
                args=slurm_args,
            )
            jobs.append((directory, job))

        print(f"Submitted {len(jobs)} jobs to {args.device}:")
        for directory, job in jobs:
            print(f"  - {directory}: Job ID {job.job_id}")

        # Monitor job completion if requested
        if args.wait_for_completion or args.max_retries > 0:
            monitor_slurm_jobs(jobs, args, executor, gen_base, slurm_args)


def main():
    parser = argparse.ArgumentParser(
        description="LOQI-style energy evaluation for generated conformers"
    )
    parser.add_argument(
        "--gen-dir",
        "--specific-dir",
        type=str,
        dest="specific_dir",
        default=None,
        help="Specific generation directory to evaluate (relative to gen_results_root)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="local",
        choices=["local", "cuda", "cpu", "a100", "h100", "all"],
        help="Execution mode: local (run locally), cuda/cpu (device for local), or a100/h100/all (slurm partition)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for AIMNet2 calculations (default: 64, increase for faster GPU utilization)"
    )
    parser.add_argument(
        "--no-opt",
        action="store_true",
        help="Skip optimization metrics (faster but less comprehensive)"
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=2e-3,
        help="Force convergence criterion for optimization (default: 2e-3 eV/Å)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Maximum optimization steps (default: 5000)"
    )
    parser.add_argument(
        "--max-recent",
        type=int,
        default=3,
        help="Max recent missing directories to evaluate (default: 3)"
    )

    # SLURM-specific arguments
    parser.add_argument(
        "--slurm-cpus",
        type=int,
        default=4,
        help="CPUs per SLURM task (default: 4)"
    )
    parser.add_argument(
        "--slurm-gpus",
        type=int,
        default=1,
        help="GPUs per SLURM task (default: 1)"
    )
    parser.add_argument(
        "--slurm-memory-gb",
        type=int,
        default=80,
        help="Memory per SLURM task in GB (default: 80)"
    )
    parser.add_argument(
        "--slurm-time-hours",
        type=int,
        default=72,
        help="Time limit for SLURM jobs in hours (default: 72)"
    )
    parser.add_argument(
        "--slurm-job-name",
        type=str,
        default="loqi_eval",
        help="Base name for SLURM jobs (default: loqi_eval)"
    )
    parser.add_argument(
        "--wait-for-completion",
        action="store_true",
        help="Wait for SLURM jobs to complete and report results (default: False)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum number of retries for failed SLURM jobs (default: 1)"
    )

    args = parser.parse_args()

    # Determine execution mode
    slurm_devices = ["a100", "h100", "all"]
    is_slurm_mode = args.device in slurm_devices

    if is_slurm_mode:
        # SLURM mode - will submit jobs to SLURM
        print(f"SLURM mode: submitting to {args.device} partition")
    else:
        # Local mode - determine compute device
        if args.device == "local":
            compute_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            # Explicit cuda or cpu
            compute_device = args.device
        print(f"Local mode: using device {compute_device}")

    # Run in appropriate mode
    if args.specific_dir:
        # Single directory mode
        print(f"Single directory mode: {args.specific_dir}")
        if is_slurm_mode:
            # Submit single job to SLURM
            run_directory_mode(args)
        else:
            # Run locally
            args.device = compute_device
            success = run_loqi_evaluation_for_dir(args.specific_dir, None, args)
            if not success:
                exit(1)
    else:
        # Batch directory mode
        print("Batch directory mode")
        if not is_slurm_mode:
            # Local batch mode
            args.device = compute_device
        run_directory_mode(args)


if __name__ == "__main__":
    main()
