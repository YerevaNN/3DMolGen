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

from loguru import logger
import torch
from rdkit import Chem
from rdkit.Chem import rdMolHash

DEFAULT_THRESHOLDS = np.arange(0.05, 3.05, 0.05)
PEAK_TFLOPS = {
    "a100": 312.0,
    "nvidia a100": 312.0,
    "nvidia h100": 1979.0,
    "h100": 1979.0,
    "rtx a6000": 78.0,
    "nvidia rtx a6000": 78.0,
}


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
    if m1 is None or m2 is None:
        return False
    c1 = Chem.MolToSmiles(Chem.RemoveHs(m1), canonical=True, isomericSmiles=False)
    c2 = Chem.MolToSmiles(Chem.RemoveHs(m2), canonical=True, isomericSmiles=False)
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


def log_cuda_memory(prefix: str) -> None:
    if not torch.cuda.is_available():
        logger.info(f"{prefix} CUDA memory: CUDA not available.")
        return
    device = torch.cuda.current_device()
    alloc_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    logger.info(
        f"{prefix} CUDA mem (GB): "
        f"allocated={alloc_gb:.2f}, reserved={reserved_gb:.2f}, "
        f"peak_alloc={peak_alloc_gb:.2f}, peak_reserved={peak_reserved_gb:.2f}"
    )


def log_cuda_summary(prefix: str, max_lines: int = 12) -> None:
    if not torch.cuda.is_available():
        return
    summary = torch.cuda.memory_summary(device=torch.cuda.current_device(), abbreviated=True)
    lines = summary.splitlines()
    snippet = "\n".join(lines[:max_lines])
    logger.info(f"{prefix} CUDA memory summary (first {max_lines} lines):\n{snippet}")


def estimate_decoder_flops_per_token(config) -> float:
    """Approximate FLOPs per generated token for decoder-only transformers."""
    hidden = getattr(config, "hidden_size", None)
    layers = getattr(config, "num_hidden_layers", None)
    vocab = getattr(config, "vocab_size", None)
    if not all((hidden, layers, vocab)):
        return 0.0
    core = 24 * layers * (hidden ** 2)
    proj = 4 * hidden * vocab
    return float(core + proj)


def detect_peak_flops(device: torch.device) -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        name = torch.cuda.get_device_name(device)
    except Exception:
        name = ""
    name_lower = name.lower()
    for key, value in PEAK_TFLOPS.items():
        if key in name_lower:
            return value * 1e12
    logger.warning(f"Unknown GPU '{name}', skipping MFU calculation.")
    return None


def log_mfu(model, generated_tokens: int, elapsed_sec: float) -> None:
    if generated_tokens <= 0 or elapsed_sec <= 0:
        return
    flops_per_token = getattr(model, "_flops_per_token", None)
    peak_flops = getattr(model, "_peak_device_flops", None)
    if not flops_per_token or not peak_flops:
        return
    tokens_per_sec = generated_tokens / elapsed_sec
    achieved_flops = flops_per_token * tokens_per_sec
    mfu = achieved_flops / peak_flops
    logger.info(
        f"Throughput: {tokens_per_sec:.2f} tok/s, FLOPs/token≈{flops_per_token/1e12:.2f} TF, "
        f"achieved≈{achieved_flops/1e12:.2f} TF/s, MFU≈{mfu*100:.2f}%"
    )
