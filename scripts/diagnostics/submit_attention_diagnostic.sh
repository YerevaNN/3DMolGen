#!/bin/bash
#SBATCH --job-name=attn_diag
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/%j_attn_diag.out
#SBATCH --error=/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/%j_attn_diag.err

# Attention implementation diagnostic for H100
# Run from 3DMolGen root: sbatch scripts/diagnostics/submit_attention_diagnostic.sh

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dmolgen

# Ensure we're in the right directory
cd /auto/home/aram.dovlatyan/3DMolGen-new/3DMolGen

# Print GPU info
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
echo ""

# Print environment info
echo "=== Environment ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
echo ""

# Run the diagnostic with Qwen model (correct model/tokenizer pairing)
echo "=== Running Attention Diagnostic (Qwen model) ==="
python scripts/diagnostics/attention_diagnostic.py --model m600_qwen --model-step 2e --tokenizer qwen3_0.6b_custom

# Uncomment below to also test Llama model:
# echo "=== Running Attention Diagnostic (Llama model) ==="
# python scripts/diagnostics/attention_diagnostic.py --model m380_conf_v2 --model-step 2e --tokenizer llama3_chem_v1

echo ""
echo "=== End time: $(date) ==="
