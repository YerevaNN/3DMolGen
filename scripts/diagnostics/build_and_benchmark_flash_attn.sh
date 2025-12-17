#!/bin/bash
#SBATCH --job-name=flash_attn_build_bench
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/flash_attn_build_%j.out
#SBATCH --error=/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/flash_attn_build_%j.err

# Build flash-attn from source and run attention benchmark on A100
# Date: 2025-12-16

set -e

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dmolgen

# Note: flash-attn should be pre-built on np before submitting this job
# Build command: FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE MAX_JOBS=4 pip install flash-attn --no-build-isolation

echo "=== Environment ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
echo ""

# Check if flash-attn works (should be pre-built on np)
echo "=== Checking flash-attn ==="
if python -c "import flash_attn; print(f'flash-attn {flash_attn.__version__} available')" 2>/dev/null; then
    echo "flash-attn is working"
    HAVE_FLASH_ATTN=true
else
    echo "WARNING: flash-attn not available, will skip flash_attention_2 benchmark"
    echo "Build it on np first with: FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE MAX_JOBS=4 pip install flash-attn --no-build-isolation"
    HAVE_FLASH_ATTN=false
fi

echo ""
echo "=== Running Attention Benchmark ==="
cd /auto/home/aram.dovlatyan/3DMolGen-new/3DMolGen

# Run benchmark - include flash_attention_2 only if available
if [ "$HAVE_FLASH_ATTN" = true ]; then
    echo "Running with flash_attention_2"
    python scripts/diagnostics/benchmark_inference_attention.py \
        --model m600_qwen \
        --num-samples 64 \
        --batch-size 16 \
        --attention sdpa sdpa_cudnn sdpa_efficient eager flash_attention_2 \
        --device local
else
    echo "Running without flash_attention_2"
    python scripts/diagnostics/benchmark_inference_attention.py \
        --model m600_qwen \
        --num-samples 64 \
        --batch-size 16 \
        --attention sdpa sdpa_cudnn sdpa_efficient eager \
        --device local
fi

echo ""
echo "=== End time: $(date) ==="
