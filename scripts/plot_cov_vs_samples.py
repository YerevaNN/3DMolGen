#!/usr/bin/env python3
"""
Generate COV-R/COV-P vs Number of Samples charts.

Shows how coverage metrics change as the number of generated conformer samples
increases (1, 2, 4, 8, ..., S). Uses first n conformers (deterministic selection,
simulates "stopping early").

Key insight: Compute full RMSD matrices once, then subselect columns to simulate
different sample counts. This is O(1) for each sample count after the initial
O(n_true * n_gen) RMSD computation.

Example usage:
    # Local (np/ap desktop)
    python scripts/plot_cov_vs_samples.py \\
        -g outputs/gen_results/20260121_run/generation_results.pickle \\
        -S 256 -t 0.75 --save-csv

    # Slurm (more CPUs)
    python scripts/plot_cov_vs_samples.py \\
        -g outputs/gen_results/20260121_run/generation_results.pickle \\
        -S 256 --device a100 --num-workers 24
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from molgen3D.config.paths import get_data_path
from molgen3D.data_processing.utils import load_pkl
from molgen3D.evaluation import rdkit_utils
from molgen3D.evaluation.utils import covmat_metrics, create_slurm_executor


# =============================================================================
# Core Computation Functions
# =============================================================================


def get_sample_counts(max_samples: int) -> List[int]:
    """Generate powers of 2 from 1 to max_samples (inclusive).

    Args:
        max_samples: Maximum sample count (will be included if it's a power of 2,
                     otherwise the largest power of 2 <= max_samples is included)

    Returns:
        List of sample counts [1, 2, 4, 8, ..., 2^k] where 2^k <= max_samples
    """
    counts = []
    n = 1
    while n <= max_samples:
        counts.append(n)
        n *= 2
    return counts


def compute_full_rmsd_matrices(
    true_data: Dict,
    gen_data: Dict[str, List],
    num_workers: int,
    use_alignmol: bool,
) -> Tuple[Dict[str, np.ndarray], List[str], List[str]]:
    """Compute full RMSD matrices for all molecules.

    Args:
        true_data: Ground truth data {smiles: {"confs": [mol, ...]}}
        gen_data: Generated conformers {smiles: [mol, ...]}
        num_workers: Number of parallel workers
        use_alignmol: If True, use AlignMol; otherwise use GetBestRMS

    Returns:
        Tuple of:
            - rmsd_matrices: {smiles: ndarray of shape (n_true, n_gen)}
            - missing_keys: SMILES with no generated conformers
            - all_nan_keys: SMILES with all-NaN RMSD matrices
    """
    missing, all_nan_keys = [], []
    rmsd_matrices: Dict[str, np.ndarray] = {}
    work_items: List[Tuple[str, List, List]] = []

    for key in true_data.keys():
        gen_mols = gen_data.get(key, [])
        if not gen_mols:
            missing.append(key)
            continue
        work_items.append((key, true_data[key]["confs"], gen_mols))

    if not work_items:
        return rmsd_matrices, missing, all_nan_keys

    total_rows = sum(len(confs) for _, confs, _ in work_items)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(rdkit_utils.compute_key_matrix, key, confs, gen_mols, use_alignmol)
            for key, confs, gen_mols in work_items
        ]
        with tqdm(total=total_rows, desc="RMSD rows", unit="row") as pbar:
            for fut in as_completed(futures):
                key, res, all_nan = fut.result()
                rmsd_matrices[key] = res["rmsd"]
                if all_nan:
                    all_nan_keys.append(key)
                pbar.update(res["n_true"])

    return rmsd_matrices, missing, all_nan_keys


def compute_metrics_at_sample_count(
    rmsd_matrices: Dict[str, np.ndarray],
    n_samples: int,
    thresholds: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute COV-R and COV-P for a specific sample count.

    Subselects first n_samples columns from each RMSD matrix.

    Args:
        rmsd_matrices: {smiles: ndarray of shape (n_true, n_gen)}
        n_samples: Number of samples to use (first n columns)
        thresholds: RMSD thresholds for coverage computation

    Returns:
        Dict with keys "cov_r" and "cov_p", each an array of shape (n_molecules, n_thresholds)
    """
    cov_r_list: List[np.ndarray] = []
    cov_p_list: List[np.ndarray] = []

    for key, mat in rmsd_matrices.items():
        # Subselect first n_samples columns (or all if fewer available)
        n_available = mat.shape[1]
        n_use = min(n_samples, n_available)
        mat_subset = mat[:, :n_use]

        # Compute coverage metrics
        cov_r, _, cov_p, _ = covmat_metrics(mat_subset, thresholds)
        cov_r_list.append(cov_r)
        cov_p_list.append(cov_p)

    return {
        "cov_r": np.vstack(cov_r_list) if cov_r_list else np.array([]),
        "cov_p": np.vstack(cov_p_list) if cov_p_list else np.array([]),
    }


def compute_cov_vs_samples(
    rmsd_matrices: Dict[str, np.ndarray],
    sample_counts: List[int],
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Compute COV-R and COV-P for each sample count.

    Args:
        rmsd_matrices: Full RMSD matrices {smiles: ndarray}
        sample_counts: List of sample counts [1, 2, 4, ...]
        thresholds: RMSD thresholds

    Returns:
        DataFrame with columns: n_samples, threshold, cov_r_mean, cov_r_std,
                                cov_p_mean, cov_p_std
    """
    rows = []

    for n_samples in tqdm(sample_counts, desc="Sample counts"):
        metrics = compute_metrics_at_sample_count(rmsd_matrices, n_samples, thresholds)

        for i, threshold in enumerate(thresholds):
            cov_r_vals = metrics["cov_r"][:, i] if metrics["cov_r"].size else np.array([])
            cov_p_vals = metrics["cov_p"][:, i] if metrics["cov_p"].size else np.array([])

            rows.append({
                "n_samples": n_samples,
                "threshold": threshold,
                "cov_r_mean": np.nanmean(cov_r_vals) if cov_r_vals.size else np.nan,
                "cov_r_std": np.nanstd(cov_r_vals) if cov_r_vals.size else np.nan,
                "cov_p_mean": np.nanmean(cov_p_vals) if cov_p_vals.size else np.nan,
                "cov_p_std": np.nanstd(cov_p_vals) if cov_p_vals.size else np.nan,
            })

    return pd.DataFrame(rows)


# =============================================================================
# Plotting Function
# =============================================================================


def plot_cov_vs_samples(
    df: pd.DataFrame,
    threshold: float,
    output_dir: Path,
    output_base: str,
    formats: List[str],
    dpi: int = 300,
) -> None:
    """Generate publication-quality COV-R/COV-P vs samples chart.

    Args:
        df: DataFrame from compute_cov_vs_samples()
        threshold: RMSD threshold to plot
        output_dir: Output directory
        output_base: Base filename (without extension)
        formats: Output formats (e.g., ["png", "pdf"])
        dpi: Resolution for raster formats
    """
    # Filter to selected threshold
    df_thresh = df[np.isclose(df["threshold"], threshold)].copy()
    df_thresh = df_thresh.sort_values("n_samples")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    x = df_thresh["n_samples"].values

    # COV-R (blue)
    cov_r_mean = df_thresh["cov_r_mean"].values
    cov_r_std = df_thresh["cov_r_std"].values
    ax.plot(x, cov_r_mean, "o-", color="#1f77b4", linewidth=2, markersize=6, label="COV-R")
    ax.fill_between(x, cov_r_mean - cov_r_std, cov_r_mean + cov_r_std,
                    color="#1f77b4", alpha=0.2)

    # COV-P (red)
    cov_p_mean = df_thresh["cov_p_mean"].values
    cov_p_std = df_thresh["cov_p_std"].values
    ax.plot(x, cov_p_mean, "s-", color="#d62728", linewidth=2, markersize=6, label="COV-P")
    ax.fill_between(x, cov_p_mean - cov_p_std, cov_p_mean + cov_p_std,
                    color="#d62728", alpha=0.2)

    # Formatting
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_ylabel("Coverage", fontsize=12)
    ax.set_title(f"Coverage vs Number of Samples (threshold = {threshold} Å)", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Set x-axis ticks to powers of 2
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in x])

    plt.tight_layout()

    # Save in all formats
    for fmt in formats:
        save_path = output_dir / f"{output_base}.{fmt}"
        fig.savefig(save_path, dpi=dpi if fmt == "png" else None, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)


# =============================================================================
# Verification (Brute Force vs Efficient)
# =============================================================================


def compute_metrics_brute_force(
    true_data: Dict,
    gen_data: Dict[str, List],
    n_samples: int,
    thresholds: np.ndarray,
    use_alignmol: bool,
) -> Dict[str, np.ndarray]:
    """Brute force: compute RMSD matrix using only first n_samples conformers.

    This recomputes RMSD from scratch for each sample count, serving as ground
    truth for verifying the efficient column-subselection approach.
    """
    cov_r_list: List[np.ndarray] = []
    cov_p_list: List[np.ndarray] = []

    for key in true_data.keys():
        gen_mols = gen_data.get(key, [])
        if not gen_mols:
            continue

        true_confs = true_data[key]["confs"]
        # Take only first n_samples generated conformers
        gen_mols_subset = gen_mols[:n_samples]

        n_true = len(true_confs)
        n_gen = len(gen_mols_subset)
        mat = np.full((n_true, n_gen), np.nan, dtype=float)

        for i_true, ref_mol in enumerate(true_confs):
            for j_gen, gen_mol in enumerate(gen_mols_subset):
                mat[i_true, j_gen] = rdkit_utils._best_rmsd(gen_mol, ref_mol, use_alignmol)

        cov_r, _, cov_p, _ = covmat_metrics(mat, thresholds)
        cov_r_list.append(cov_r)
        cov_p_list.append(cov_p)

    return {
        "cov_r": np.vstack(cov_r_list) if cov_r_list else np.array([]),
        "cov_p": np.vstack(cov_p_list) if cov_p_list else np.array([]),
    }


def run_verification(args: argparse.Namespace) -> None:
    """Verify efficient approach matches brute force on a small subset."""
    print("=" * 60)
    print("VERIFICATION MODE: Comparing efficient vs brute force")
    print("=" * 60)

    thresholds = np.array(args.thresholds, dtype=float)

    # Load data
    print(f"\nLoading generations from: {args.generations_pickle}")
    gens_dict = load_pkl(args.generations_pickle)

    print(f"Loading ground truth for test set: {args.test_set}")
    gt_dict = load_pkl(get_data_path(f"{args.test_set}_smi"))

    # Take small subset for verification (first N molecules)
    n_mols = min(args.verify_n_mols, len(gt_dict))
    gt_keys = list(gt_dict.keys())[:n_mols]
    gt_subset = {k: gt_dict[k] for k in gt_keys}
    gen_subset = {k: gens_dict.get(k, []) for k in gt_keys}

    print(f"\nUsing {n_mols} molecules for verification")

    # Process molecules
    processed_gen = rdkit_utils.process_molecules_remove_hs(gen_subset)

    # Compute full RMSD matrices (efficient approach)
    print("\n[Efficient] Computing full RMSD matrices...")
    rmsd_matrices, _, _ = compute_full_rmsd_matrices(
        gt_subset, processed_gen, args.num_workers, args.use_alignmol
    )

    sample_counts = get_sample_counts(args.max_samples)
    print(f"Sample counts to verify: {sample_counts}")

    all_passed = True
    print("\n" + "-" * 60)
    print(f"{'n_samples':>10} | {'Method':>10} | {'COV-R':>10} | {'COV-P':>10} | {'Match':>6}")
    print("-" * 60)

    for n_samples in sample_counts:
        # Efficient approach (column subselection)
        metrics_efficient = compute_metrics_at_sample_count(rmsd_matrices, n_samples, thresholds)
        cov_r_eff = np.nanmean(metrics_efficient["cov_r"][:, 0]) if metrics_efficient["cov_r"].size else np.nan
        cov_p_eff = np.nanmean(metrics_efficient["cov_p"][:, 0]) if metrics_efficient["cov_p"].size else np.nan

        # Brute force (recompute from scratch)
        metrics_brute = compute_metrics_brute_force(
            gt_subset, processed_gen, n_samples, thresholds, args.use_alignmol
        )
        cov_r_brute = np.nanmean(metrics_brute["cov_r"][:, 0]) if metrics_brute["cov_r"].size else np.nan
        cov_p_brute = np.nanmean(metrics_brute["cov_p"][:, 0]) if metrics_brute["cov_p"].size else np.nan

        # Compare
        r_match = np.isclose(cov_r_eff, cov_r_brute, rtol=1e-9)
        p_match = np.isclose(cov_p_eff, cov_p_brute, rtol=1e-9)
        match = r_match and p_match

        if not match:
            all_passed = False

        status = "✓" if match else "✗"
        print(f"{n_samples:>10} | {'Efficient':>10} | {cov_r_eff:>10.6f} | {cov_p_eff:>10.6f} |")
        print(f"{'':>10} | {'Brute':>10} | {cov_r_brute:>10.6f} | {cov_p_brute:>10.6f} | {status:>6}")

    print("-" * 60)
    if all_passed:
        print("\n✓ VERIFICATION PASSED: Efficient approach matches brute force exactly!")
    else:
        print("\n✗ VERIFICATION FAILED: Results differ between approaches!")

    return all_passed


# =============================================================================
# CLI and Execution
# =============================================================================


def run_analysis(args: argparse.Namespace) -> None:
    """Main analysis logic."""
    t_start = time.time()

    # Parse thresholds
    thresholds = np.array(args.thresholds, dtype=float)

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.generations_pickle).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading generations from: {args.generations_pickle}")
    gens_dict = load_pkl(args.generations_pickle)
    print(f"  Loaded {len(gens_dict)} molecules")

    print(f"Loading ground truth for test set: {args.test_set}")
    gt_dict = load_pkl(get_data_path(f"{args.test_set}_smi"))
    print(f"  Loaded {len(gt_dict)} ground truth molecules")

    # Process generated molecules (remove Hs for RMSD computation)
    print("Processing molecules (removing Hs)...")
    processed_gen_data = rdkit_utils.process_molecules_remove_hs(gens_dict)

    # Compute full RMSD matrices
    print(f"Computing RMSD matrices with {args.num_workers} workers...")
    t_rmsd_start = time.time()
    rmsd_matrices, missing, all_nan_keys = compute_full_rmsd_matrices(
        gt_dict, processed_gen_data, args.num_workers, args.use_alignmol
    )
    t_rmsd = time.time() - t_rmsd_start

    print(f"  Computed {len(rmsd_matrices)} matrices in {t_rmsd:.1f}s")
    if missing:
        print(f"  Missing: {len(missing)} molecules")
    if all_nan_keys:
        print(f"  All-NaN: {len(all_nan_keys)} molecules")

    # Compute metrics for each sample count
    sample_counts = get_sample_counts(args.max_samples)
    print(f"Computing metrics for sample counts: {sample_counts}")
    df = compute_cov_vs_samples(rmsd_matrices, sample_counts, thresholds)

    # Save CSV if requested
    if args.save_csv:
        csv_path = output_dir / "cov_vs_samples_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    # Generate plots for each threshold
    for threshold in thresholds:
        # Use underscore instead of dot to avoid pathlib treating it as extension
        threshold_str = f"{threshold:.2f}".replace(".", "_")
        output_base = f"cov_vs_samples_t{threshold_str}"
        plot_cov_vs_samples(df, threshold, output_dir, output_base, args.output_formats, args.dpi)

    # Write summary
    t_total = time.time() - t_start
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("COV vs Samples Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Generations pickle: {args.generations_pickle}\n")
        f.write(f"Test set: {args.test_set}\n")
        f.write(f"Max samples: {args.max_samples}\n")
        f.write(f"Sample counts: {sample_counts}\n")
        f.write(f"Thresholds: {list(thresholds)}\n")
        f.write(f"Use AlignMol: {args.use_alignmol}\n")
        f.write(f"Num workers: {args.num_workers}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Molecules processed: {len(rmsd_matrices)}\n")
        f.write(f"  Missing molecules: {len(missing)}\n")
        f.write(f"  All-NaN molecules: {len(all_nan_keys)}\n\n")
        f.write(f"Timing:\n")
        f.write(f"  RMSD computation: {t_rmsd:.1f}s\n")
        f.write(f"  Total: {t_total:.1f}s\n")
    print(f"Saved: {summary_path}")

    print(f"\nDone! Total time: {t_total:.1f}s")


def run_with_slurm(args: argparse.Namespace) -> None:
    """Submit job to Slurm cluster."""
    executor = create_slurm_executor(
        device=args.device,
        job_type="cov_vs_samples",
        num_gpus=0,
        num_cpus=args.num_workers,
        job_name="cov_vs_samples",
        memory_gb=args.memory_gb,
    )

    # Submit the job
    job = executor.submit(run_analysis, args)
    print(f"Submitted job {job.job_id} to {args.device}")
    print(f"Logs: ~/slurm_jobs/cov_vs_samples/job_{job.job_id}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate COV-R/COV-P vs Number of Samples charts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "-g", "--generations-pickle",
        type=str,
        required=True,
        help="Path to generations pickle file",
    )
    parser.add_argument(
        "-S", "--max-samples",
        type=int,
        required=True,
        help="Maximum samples (powers of 2 up to this value)",
    )

    # Optional arguments
    parser.add_argument(
        "--test-set",
        type=str,
        default="distinct",
        choices=["clean", "distinct", "xl", "qm9"],
        help="Test set to use (default: distinct)",
    )
    parser.add_argument(
        "-t", "--thresholds",
        type=float,
        nargs="+",
        default=[0.75],
        help="RMSD thresholds for coverage (default: 0.75)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: alongside pickle)",
    )
    parser.add_argument(
        "--output-formats",
        type=str,
        nargs="+",
        default=["png", "pdf"],
        help="Output formats (default: png pdf)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PNG output (default: 300)",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save intermediate data as CSV",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--use-alignmol",
        action="store_true",
        help="Use AlignMol instead of GetBestRMS",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="local",
        choices=["local", "a100", "h100"],
        help="Execution device (default: local)",
    )
    parser.add_argument(
        "--memory-gb",
        type=int,
        default=80,
        help="Memory allocation for Slurm (default: 80GB)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification mode: compare efficient vs brute force",
    )
    parser.add_argument(
        "--verify-n-mols",
        type=int,
        default=10,
        help="Number of molecules to use for verification (default: 10)",
    )

    args = parser.parse_args()

    if args.verify:
        if args.device == "local":
            run_verification(args)
        else:
            executor = create_slurm_executor(
                device=args.device,
                job_type="cov_vs_samples",
                num_gpus=0,
                num_cpus=args.num_workers,
                job_name="cov_vs_samples_verify",
                memory_gb=args.memory_gb,
            )
            job = executor.submit(run_verification, args)
            print(f"Submitted verification job {job.job_id} to {args.device}")
            print(f"Logs: ~/slurm_jobs/cov_vs_samples/job_{job.job_id}/")
    elif args.device == "local":
        run_analysis(args)
    else:
        run_with_slurm(args)


if __name__ == "__main__":
    main()
