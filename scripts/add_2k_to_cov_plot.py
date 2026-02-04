#!/usr/bin/env python3
"""
Add "2*k" point to existing COV vs Samples plot.

Regenerates the plot from existing cov_vs_samples_data.csv without re-running
the full RMSD computation. Automatically finds the corresponding eval results
by swapping gen_results → eval_results in the path.

Usage:
    python scripts/add_2k_to_cov_plot.py \
        outputs/gen_results/20260124_124554_qw600_pre_binned_filtered_4e_top_k_r3_70_t13_distinct
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_covmat_results(filepath: Path) -> dict | None:
    """Parse covmat_results.txt to extract COV-R and COV-P mean values."""
    if not filepath.exists():
        return None

    try:
        text = filepath.read_text()
        lines = text.splitlines()

        cov_r_mean = None
        cov_p_mean = None
        total_conformers = None

        for i, line in enumerate(lines):
            if "Coverage-Recall (COV-R):" in line and i + 1 < len(lines):
                mean_line = lines[i + 1]
                if "Mean:" in mean_line:
                    cov_r_mean = float(mean_line.split(":")[1].strip())
            elif "Coverage-Precision (COV-P):" in line and i + 1 < len(lines):
                mean_line = lines[i + 1]
                if "Mean:" in mean_line:
                    cov_p_mean = float(mean_line.split(":")[1].strip())
            elif "Total conformers generated:" in line:
                total_conformers = int(line.split(":")[1].strip())

        if cov_r_mean is None or cov_p_mean is None:
            return None

        return {
            "cov_r_mean": cov_r_mean,
            "cov_p_mean": cov_p_mean,
            "total_conformers": total_conformers,
        }
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def plot_with_2k(
    df: pd.DataFrame,
    full_eval_data: dict,
    output_dir: Path,
    threshold: float = 0.75,
) -> None:
    """Generate plot with "2*k" point."""
    df_thresh = df[np.isclose(df["threshold"], threshold)].copy()
    df_thresh = df_thresh.sort_values("n_samples")

    fig, ax = plt.subplots(figsize=(8, 6))

    x = df_thresh["n_samples"].values.tolist()
    cov_r_mean = df_thresh["cov_r_mean"].values.tolist()
    cov_p_mean = df_thresh["cov_p_mean"].values.tolist()
    x_labels = [str(n) for n in x]

    # Add "2*k" point - position at 2x last sample for visual spacing
    # (actual value is total_conformers, but that creates huge gap on log scale)
    x_2k = x[-1] * 2
    x.append(x_2k)
    cov_r_mean.append(full_eval_data["cov_r_mean"])
    cov_p_mean.append(full_eval_data["cov_p_mean"])
    x_labels.append("2*k")

    x = np.array(x)
    cov_r_mean = np.array(cov_r_mean)
    cov_p_mean = np.array(cov_p_mean)

    # Plot
    ax.plot(x, cov_r_mean, "o-", color="#1f77b4", linewidth=2, markersize=6, label="COV-R")
    ax.plot(x, cov_p_mean, "s-", color="#d62728", linewidth=2, markersize=6, label="COV-P")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_ylabel("Coverage", fontsize=12)
    ax.set_title(f"Coverage vs Number of Samples (threshold = {threshold} Å)", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    plt.tight_layout()

    # Save
    for fmt in ["png", "pdf"]:
        save_path = output_dir / f"cov_vs_samples_final_2k.{fmt}"
        fig.savefig(save_path, dpi=300 if fmt == "png" else None, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Add '2*k' point to existing COV vs Samples plot"
    )
    parser.add_argument(
        "gen_dir",
        type=str,
        help="Directory containing cov_vs_samples_data.csv (inside gen_results)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="CSV filename (default: auto-detect cov_vs_samples_data*.csv)",
    )
    args = parser.parse_args()

    gen_dir = Path(args.gen_dir)

    # Find CSV file
    if args.csv:
        csv_path = gen_dir / args.csv
    else:
        # Auto-detect: look for cov_vs_samples_data*.csv
        candidates = list(gen_dir.glob("cov_vs_samples_data*.csv"))
        if not candidates:
            print(f"Error: No cov_vs_samples_data*.csv found in {gen_dir}")
            return 1
        csv_path = candidates[0]
        if len(candidates) > 1:
            print(f"  Note: Multiple CSVs found, using {csv_path.name}")

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return 1

    # Derive eval_results path by swapping gen_results → eval_results
    eval_dir = Path(str(gen_dir).replace("gen_results", "eval_results"))
    eval_txt = eval_dir / "covmat_results.txt"

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Loading eval results: {eval_txt}")
    full_eval_data = parse_covmat_results(eval_txt)

    if full_eval_data is None:
        print(f"Error: Could not parse {eval_txt}")
        return 1

    print(f"  COV-R: {full_eval_data['cov_r_mean']:.4f}, "
          f"COV-P: {full_eval_data['cov_p_mean']:.4f}, "
          f"total conformers: {full_eval_data['total_conformers']}")

    plot_with_2k(df, full_eval_data, gen_dir)
    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
