#!/bin/bash
#SBATCH --job-name=flash_attn_build
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/flash_attn_build_%j.out
#SBATCH --error=/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/flash_attn_build_%j.err

# Build flash-attn from source on h100 (has nvcc)
# After this completes, submit benchmark to a100

set -e

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dmolgen

# Set CUDA environment - prefer CUDA 12.8 for PyTorch 2.9 compatibility
# CUDA_HOME for cleaner detection (less flaky builds)
if [ -f "/usr/local/cuda-12.8/bin/nvcc" ]; then
    export CUDA_HOME=/usr/local/cuda-12.8
elif [ -f "/usr/local/cuda/bin/nvcc" ]; then
    export CUDA_HOME=/usr/local/cuda
else
    echo "ERROR: nvcc not found at /usr/local/cuda-12.8 or /usr/local/cuda"
    exit 1
fi
export PATH=$CUDA_HOME/bin:$PATH

# Ensure pip subprocess can find conda packages
export PYTHONPATH=$CONDA_PREFIX/lib/python3.10/site-packages:$PYTHONPATH

echo ""
echo "=== Environment ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
echo "CUDA_HOME: $CUDA_HOME"
nvcc --version | head -4

# Uninstall any existing broken installation
echo ""
echo "=== Uninstalling old flash-attn ==="
pip uninstall flash-attn -y 2>/dev/null || true

# Build from source
echo ""
echo "=== Building flash-attn from source ==="
# https://www.reddit.com/r/LocalLLaMA/comments/1no4ho1/some_things_i_learned_about_installing_flashattn/
export FLASH_ATTENTION_FORCE_BUILD=TRUE # compile even when a wheel already exists
export FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE # if your base image/toolchain needs C++11 ABI to match PyTorch
# Reduce parallelism to avoid OOM - flash-attn compilation is memory-hungry
# MAX_JOBS=8 + --threads 4 was getting killed
export MAX_JOBS=4
export NVCC_THREADS=2

pip install flash-attn --no-build-isolation

# Verify installation
echo ""
echo "=== Verifying flash-attn ==="
python -c "
import flash_attn
print(f'flash-attn version: {flash_attn.__version__}')
from flash_attn import flash_attn_func
import torch
q = torch.randn(2, 32, 8, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 32, 8, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 32, 8, 64, device='cuda', dtype=torch.float16)
out = flash_attn_func(q, k, v, causal=True)
print(f'Flash attention test: OK (output shape: {out.shape})')
"

echo ""
echo "=== Build complete ==="
echo "Now submit benchmark: sbatch scripts/diagnostics/build_and_benchmark_flash_attn.sh"
echo "End time: $(date)"
