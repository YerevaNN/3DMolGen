#!/bin/bash
#SBATCH --job-name=attn_bench
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/%j_attn_bench.out
#SBATCH --error=/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/%j_attn_bench.err

# Inference benchmark comparing attention implementations
# Usage: sbatch scripts/diagnostics/submit_inference_benchmark.sh

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dmolgen

cd /auto/home/aram.dovlatyan/3DMolGen-new/3DMolGen

# Print GPU info
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
echo ""

# ============================================================
# Benchmark Qwen model (m600_qwen)
# ============================================================
echo "=== Benchmarking Qwen Model (m600_qwen) ==="
python scripts/diagnostics/benchmark_inference_attention.py \
    --model m600_qwen \
    --model-step 2e \
    --tokenizer qwen3_0.6b_custom \
    --num-samples 64 \
    --batch-size 16 \
    --attention sdpa eager

echo ""
echo "=== End time: $(date) ==="
