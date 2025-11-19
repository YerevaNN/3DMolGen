import os.path as osp
from statistics import mode, StatisticsError
import numpy as np
import pandas as pd
from collections import Counter
import submitit
from pathlib import Path
from typing import List, Tuple, Optional, Dict  
import math
import os
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolHash

DEFAULT_THRESHOLDS = np.arange(0.05, 3.05, 0.05)


def extract_between(text, start_marker, end_marker):
    start = text.find(start_marker)
    if start != -1:
        start += len(start_marker)  # Move to the end of the start marker
        end = text.find(end_marker, start)
        if end != -1:
            return text[start:end]
    return ""  # Return empty if markers are not found


def create_slurm_executor(
    device: str = "local",
    job_type: str = "eval",
    num_gpus: int = 1,
    num_cpus: int = 8,
    job_name: str = "test_run",
    memory_gb: int = 80,
    timeout_min: int = 3 * 24 * 60, # 3 days
) -> submitit.AutoExecutor:
    if submitit is None:
        raise RuntimeError("submitit is not available")

    folder = str(Path.home() / "slurm_jobs" / job_type / "job_%j")

    if device == "local":
        executor = submitit.LocalExecutor(folder=folder)
    else:
        executor = submitit.AutoExecutor(folder=folder)

    params = dict(
        name=job_name,
        timeout_min=timeout_min,
        cpus_per_task=num_cpus,
        gpus_per_node=num_gpus,
        mem_gb=memory_gb,
        nodes=1,
        slurm_additional_parameters={"partition": device},
    )

    executor.update_parameters(**params)
    return executor

def find_generation_pickles_path(directory_path: str) -> str:
    for r, _, fs in os.walk(directory_path):
        for f in fs:
            if f.endswith(".pickle") or f.endswith(".pkl"):
                return os.path.join(r, f)
    return None  # No pickle files found

def same_molecular_graph(gt: str, gen: str) -> bool:
    m1 = Chem.MolFromSmiles(gt)
    m2 = Chem.MolFromSmiles(gen)
    if m2 is None:
        return False
    else:
        # isomericSmiles=False => ignore stereochemistry in the canonical string
        c1 = Chem.MolToSmiles(m1, canonical=True, isomericSmiles=False)
        c2 = Chem.MolToSmiles(m2, canonical=True, isomericSmiles=False)
        return c1 == c2

def format_float(value: Optional[float], decimals: int = 4) -> str:
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


def covmat_metrics(rmsd: np.ndarray, thresholds: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, float]:
    min_true = np.nanmin(rmsd, axis=1)
    min_gen = np.nanmin(rmsd, axis=0)
    cov_recall = np.array([(min_true < t).mean() for t in thresholds], dtype=float)
    cov_precision = np.array([(min_gen < t).mean() for t in thresholds], dtype=float)
    amr_recall = float(np.nanmean(min_true))
    amr_precision = float(np.nanmean(min_gen))
    return cov_recall, amr_recall, cov_precision, amr_precision
