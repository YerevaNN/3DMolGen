#!/bin/bash
#SBATCH --job-name=ct_dropout
#SBATCH --partition=research8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/weka/fgeikyan/CoordToken/logs/final_and_nabla_20261012/ct_dropout-%j.log
#SBATCH --gpus=1
#SBATCH --array=0-7%8
set -euo pipefail
cd /mnt/weka/fgeikyan/CoordToken
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/coordtoken_final_${SLURM_JOB_ID}
exec /mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python -u /mnt/weka/fgeikyan/CoordToken/research/final_and_nabla_20261012/run_case.py dropout "$SLURM_ARRAY_TASK_ID"
