#!/usr/bin/env python
"""Extract HP sweep evaluation results into a CSV summary.

Iterates through eval_results directories, parses covmat_results.txt files,
and builds a summary CSV with sampling config metadata and metrics.
"""

import csv
import re
from datetime import datetime
from pathlib import Path

# Sampling config parameter mappings
# Round 1: Vary temperature, fix other params
ROUND1_CONFIGS = {
    "top_p_sweep1": {"temperature": 0.8, "top_p": 0.9, "min_p": None, "top_k": None},
    "top_p_sweep2": {"temperature": 1.0, "top_p": 0.9, "min_p": None, "top_k": None},
    "top_p_sweep3": {"temperature": 1.2, "top_p": 0.9, "min_p": None, "top_k": None},
    "min_p_sweep1": {"temperature": 0.8, "top_p": None, "min_p": 0.1, "top_k": None},
    "min_p_sweep2": {"temperature": 1.0, "top_p": None, "min_p": 0.1, "top_k": None},
    "min_p_sweep3": {"temperature": 1.2, "top_p": None, "min_p": 0.1, "top_k": None},
    "top_k_sweep1": {"temperature": 0.8, "top_p": None, "min_p": None, "top_k": 50},
    "top_k_sweep2": {"temperature": 1.0, "top_p": None, "min_p": None, "top_k": 50},
    "top_k_sweep3": {"temperature": 1.2, "top_p": None, "min_p": None, "top_k": 50},
}

# Round 2: Fix temperature at 1.0, vary other params
ROUND2_CONFIGS = {
    "top_p_r2_1": {"temperature": 1.0, "top_p": 0.8, "min_p": None, "top_k": None},
    "top_p_r2_2": {"temperature": 1.0, "top_p": 0.9, "min_p": None, "top_k": None},
    "top_p_r2_3": {"temperature": 1.0, "top_p": 0.95, "min_p": None, "top_k": None},
    "min_p_r2_1": {"temperature": 1.0, "top_p": None, "min_p": 0.05, "top_k": None},
    "min_p_r2_2": {"temperature": 1.0, "top_p": None, "min_p": 0.1, "top_k": None},
    "min_p_r2_3": {"temperature": 1.0, "top_p": None, "min_p": 0.15, "top_k": None},
    "top_k_r2_1": {"temperature": 1.0, "top_p": None, "min_p": None, "top_k": 20},
    "top_k_r2_2": {"temperature": 1.0, "top_p": None, "min_p": None, "top_k": 50},
    "top_k_r2_3": {"temperature": 1.0, "top_p": None, "min_p": None, "top_k": 100},
}

ALL_CONFIGS = {**ROUND1_CONFIGS, **ROUND2_CONFIGS}


def extract_config_from_dirname(dirname: str) -> tuple[str | None, int | None]:
    """Extract config name and round from directory name.

    Returns:
        (config_name, round_number) or (None, None) if not a sweep dir
    """
    # Round 1 pattern: *_<method>_sweep<N>_*
    r1_match = re.search(r"(top_p_sweep\d|min_p_sweep\d|top_k_sweep\d)", dirname)
    if r1_match:
        return r1_match.group(1), 1

    # Round 2 pattern: *_<method>_r2_<N>_*
    r2_match = re.search(r"(top_p_r2_\d|min_p_r2_\d|top_k_r2_\d)", dirname)
    if r2_match:
        return r2_match.group(1), 2

    return None, None


def extract_runtime_from_gen_logs(gen_results_dir: Path) -> float | None:
    """Extract runtime in minutes from generation logs.txt.

    Parses first and last timestamps from the log file.
    """
    logs_file = gen_results_dir / "logs.txt"
    if not logs_file.exists():
        return None

    try:
        content = logs_file.read_text()
        # Find all timestamps
        timestamps = re.findall(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", content, re.MULTILINE)
        if len(timestamps) < 2:
            return None

        start = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S")
        runtime_minutes = (end - start).total_seconds() / 60
        return round(runtime_minutes, 2)
    except Exception:
        return None


def parse_covmat_results(filepath: Path) -> dict:
    """Parse covmat_results.txt and extract metrics."""
    results = {
        "cov_r_mean": None,
        "cov_p_mean": None,
        "mat_r_mean": None,
        "mat_p_mean": None,
        "smiles_mismatch": None,
        "mol_parse_fail": None,
        "no_eos": None,
        "total_mols": None,
        "total_conformers": None,
    }

    if not filepath.exists():
        return results

    content = filepath.read_text()

    # Extract COV-R Mean
    match = re.search(r"Coverage-Recall \(COV-R\):\s*\n\s*Mean:\s*([\d.]+)", content)
    if match:
        results["cov_r_mean"] = float(match.group(1))

    # Extract COV-P Mean
    match = re.search(r"Coverage-Precision \(COV-P\):\s*\n\s*Mean:\s*([\d.]+)", content)
    if match:
        results["cov_p_mean"] = float(match.group(1))

    # Extract MAT-R Mean
    match = re.search(r"Matching-Recall \(MAT-R\):\s*\n\s*Mean:\s*([\d.]+)", content)
    if match:
        results["mat_r_mean"] = float(match.group(1))

    # Extract MAT-P Mean
    match = re.search(r"Matching-Precision \(MAT-P\):\s*\n\s*Mean:\s*([\d.]+)", content)
    if match:
        results["mat_p_mean"] = float(match.group(1))

    # Extract stats from Counter
    match = re.search(r"stats=Counter\(\{([^}]+)\}\)", content)
    if match:
        stats_str = match.group(1)
        for key in ["smiles_mismatch", "mol_parse_fail", "no_eos"]:
            stat_match = re.search(rf"'{key}':\s*(\d+)", stats_str)
            if stat_match:
                results[key] = int(stat_match.group(1))
            else:
                results[key] = 0

    # Extract total molecules and conformers
    match = re.search(r"Total molecules generated:\s*(\d+)", content)
    if match:
        results["total_mols"] = int(match.group(1))

    match = re.search(r"Total conformers generated:\s*(\d+)", content)
    if match:
        results["total_conformers"] = int(match.group(1))

    return results


def main():
    # Paths
    eval_results_dir = Path("outputs/eval_results")
    gen_results_dir = Path("outputs/gen_results")
    output_dir = Path("outputs/eval_sweep_results_extraction")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "hp_sweep_results.csv"

    # Collect results
    rows = []

    for eval_dir in sorted(eval_results_dir.iterdir()):
        if not eval_dir.is_dir():
            continue

        config_name, round_num = extract_config_from_dirname(eval_dir.name)
        if config_name is None:
            continue

        # Get config parameters
        config_params = ALL_CONFIGS.get(config_name, {})
        if not config_params:
            print(f"Warning: Unknown config {config_name}, skipping")
            continue

        # Parse metrics
        covmat_file = eval_dir / "covmat_results.txt"
        metrics = parse_covmat_results(covmat_file)

        # Find corresponding gen_results directory and extract runtime
        # The eval_dir name matches the gen_results dir name
        gen_dir = gen_results_dir / eval_dir.name
        runtime_minutes = extract_runtime_from_gen_logs(gen_dir)

        # Determine sampling method
        if "top_p" in config_name:
            method = "top_p"
        elif "min_p" in config_name:
            method = "min_p"
        elif "top_k" in config_name:
            method = "top_k"
        else:
            method = "unknown"

        row = {
            "directory": eval_dir.name,
            "config_name": config_name,
            "round": round_num,
            "method": method,
            "temperature": config_params.get("temperature"),
            "top_p": config_params.get("top_p"),
            "min_p": config_params.get("min_p"),
            "top_k": config_params.get("top_k"),
            "runtime_minutes": runtime_minutes,
            "cov_r_mean": metrics["cov_r_mean"],
            "cov_p_mean": metrics["cov_p_mean"],
            "mat_r_mean": metrics["mat_r_mean"],
            "mat_p_mean": metrics["mat_p_mean"],
            "smiles_mismatch": metrics["smiles_mismatch"],
            "mol_parse_fail": metrics["mol_parse_fail"],
            "no_eos": metrics["no_eos"],
            "total_mols": metrics["total_mols"],
            "total_conformers": metrics["total_conformers"],
        }
        rows.append(row)
        runtime_str = f"{runtime_minutes:.1f} min" if runtime_minutes else "N/A"
        print(f"Processed: {config_name} (Round {round_num}) - Runtime: {runtime_str}")

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nWrote {len(rows)} results to {output_csv}")
    else:
        print("No sweep results found")


if __name__ == "__main__":
    main()
