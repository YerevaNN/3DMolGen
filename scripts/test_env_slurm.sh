#!/bin/bash
#SBATCH --job-name=test-env
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/auto/home/%u/slurm_jobs/env_tests/test-env-%j.out
#SBATCH --error=/auto/home/%u/slurm_jobs/env_tests/test-env-%j.err

# =============================================================================
# Environment Test Job
# Tests that all dependencies can be loaded and GPU is accessible
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "  Environment Test Job"
echo "  $(date)"
echo "=============================================="
echo ""
echo "Node: $(hostname)"
echo "User: $USER"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Activate the uv environment
export UV_CACHE_DIR="/scratch/${USER}/.cache/uv"
source /scratch/${USER}/3dmolgen/.venv/bin/activate

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Run comprehensive import test
python << 'EOF'
import sys
import time

print("=" * 70)
print("  Comprehensive Dependency Import Test")
print("=" * 70)
print()

results = []
start_total = time.time()

def test_import(name, import_stmt, test_func=None):
    """Test importing a module and optionally run a test function."""
    start = time.time()
    try:
        exec(import_stmt)
        if test_func:
            test_func()
        elapsed = time.time() - start
        results.append((name, "PASS", f"{elapsed:.2f}s"))
        print(f"  [PASS] {name:<25} ({elapsed:.2f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - start
        results.append((name, "FAIL", str(e)[:50]))
        print(f"  [FAIL] {name:<25} - {str(e)[:50]}")
        return False

# =============================================================================
# Core ML Libraries
# =============================================================================
print("--- Core ML Libraries ---")

test_import("torch", """
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f"         PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"         GPU: {torch.cuda.get_device_name(0)}")
print(f"         Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
""")

test_import("flash_attn", """
import flash_attn
from flash_attn import flash_attn_func
print(f"         Flash Attention {flash_attn.__version__}")
""")

test_import("transformers", """
import transformers
print(f"         transformers {transformers.__version__}")
""")

test_import("accelerate", """
import accelerate
print(f"         accelerate {accelerate.__version__}")
""")

test_import("datasets", """
import datasets
print(f"         datasets {datasets.__version__}")
""")

test_import("safetensors", """
import safetensors
print(f"         safetensors {safetensors.__version__}")
""")

# =============================================================================
# Training Libraries
# =============================================================================
print("\n--- Training Libraries ---")

test_import("torchtitan", """
import torchtitan
print(f"         torchtitan installed")
""")

test_import("torchdata", """
import torchdata
print(f"         torchdata {torchdata.__version__}")
""")

test_import("trl", """
import trl
print(f"         trl {trl.__version__}")
""")

# =============================================================================
# Evaluation
# =============================================================================
print("\n--- Evaluation ---")

test_import("posebusters", """
import posebusters
print(f"         posebusters {posebusters.__version__}")
""")

test_import("rdkit", """
from rdkit import Chem
from rdkit import rdBase
print(f"         rdkit {rdBase.rdkitVersion}")
# Quick sanity check
mol = Chem.MolFromSmiles('CCO')
assert mol is not None, 'Failed to parse SMILES'
""")

# =============================================================================
# Utilities
# =============================================================================
print("\n--- Utilities ---")

test_import("scipy", """
import scipy
from scipy.optimize import linear_sum_assignment
print(f"         scipy {scipy.__version__}")
""")

test_import("sklearn", """
import sklearn
print(f"         scikit-learn {sklearn.__version__}")
""")

test_import("numpy", """
import numpy as np
print(f"         numpy {np.__version__}")
""")

test_import("pandas", """
import pandas as pd
print(f"         pandas {pd.__version__}")
""")

test_import("matplotlib", """
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
print(f"         matplotlib {matplotlib.__version__}")
""")

test_import("pillow", """
from PIL import Image
import PIL
print(f"         pillow {PIL.__version__}")
""")

test_import("pydantic", """
import pydantic
print(f"         pydantic {pydantic.__version__}")
""")

test_import("pyarrow", """
import pyarrow
print(f"         pyarrow {pyarrow.__version__}")
""")

# =============================================================================
# CLI / Config
# =============================================================================
print("\n--- CLI / Config ---")

test_import("tyro", """
import tyro
print(f"         tyro installed")
""")

test_import("yaml", """
import yaml
print(f"         pyyaml installed")
""")

test_import("loguru", """
from loguru import logger
print(f"         loguru installed")
""")

test_import("rich", """
import rich
from rich.console import Console
print(f"         rich installed")
""")

# =============================================================================
# Experiment Tracking
# =============================================================================
print("\n--- Experiment Tracking ---")

test_import("wandb", """
import wandb
print(f"         wandb {wandb.__version__}")
""")

test_import("tensorboard", """
import tensorboard
print(f"         tensorboard {tensorboard.__version__}")
""")

# =============================================================================
# Job Submission
# =============================================================================
print("\n--- Job Submission ---")

test_import("submitit", """
import submitit
print(f"         submitit {submitit.__version__}")
""")

test_import("cloudpickle", """
import cloudpickle
print(f"         cloudpickle {cloudpickle.__version__}")
""")

# =============================================================================
# Dev Tools (if installed)
# =============================================================================
print("\n--- Dev Tools (optional) ---")

test_import("einops", """
import einops
print(f"         einops {einops.__version__}")
""")

test_import("bitsandbytes", """
import bitsandbytes
print(f"         bitsandbytes {bitsandbytes.__version__}")
""")

test_import("liger_kernel", """
import liger_kernel
print(f"         liger-kernel installed")
""")

# =============================================================================
# Local Package
# =============================================================================
print("\n--- Local Package ---")

test_import("molgen3D", """
import molgen3D
print(f"         molgen3D installed")
""")

test_import("molgen3D.data_processing", """
from molgen3D.data_processing import smiles_encoder_decoder
print(f"         molgen3D.data_processing loaded")
""")

# =============================================================================
# GPU Sanity Check
# =============================================================================
print("\n--- GPU Sanity Check ---")

test_import("GPU tensor ops", """
import torch
x = torch.randn(1000, 1000, device='cuda')
y = torch.randn(1000, 1000, device='cuda')
z = torch.matmul(x, y)
torch.cuda.synchronize()
print(f"         Matrix multiply on GPU: OK")
""")

test_import("Flash Attention forward", """
import torch
from flash_attn import flash_attn_func
batch, seqlen, nheads, headdim = 2, 128, 8, 64
q = torch.randn(batch, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
k = torch.randn(batch, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
v = torch.randn(batch, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
out = flash_attn_func(q, k, v)
torch.cuda.synchronize()
print(f"         Flash Attention forward pass: OK")
""")

# =============================================================================
# Summary
# =============================================================================
elapsed_total = time.time() - start_total
print()
print("=" * 70)
passed = sum(1 for _, status, _ in results if status == "PASS")
failed = sum(1 for _, status, _ in results if status == "FAIL")
print(f"  Summary: {passed} passed, {failed} failed ({elapsed_total:.1f}s total)")
print("=" * 70)

if failed > 0:
    print("\nFailed imports:")
    for name, status, msg in results:
        if status == "FAIL":
            print(f"  - {name}: {msg}")
    sys.exit(1)
else:
    print("\nAll imports successful!")
    sys.exit(0)
EOF

echo ""
echo "=============================================="
echo "  Test Complete"
echo "  $(date)"
echo "=============================================="
