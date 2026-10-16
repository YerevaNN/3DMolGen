#!/bin/bash
#SBATCH --job-name=ct_mixed_full
#SBATCH --partition=research8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=1-12:00:00
#SBATCH --output=/mnt/weka/fgeikyan/CoordToken/logs/final_and_nabla_20261012/ct_mixed_full-%j.log
#SBATCH --gpus=8
set -euo pipefail
cd /mnt/weka/fgeikyan/CoordToken
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/coordtoken_final_${SLURM_JOB_ID}
exec /mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python -u /mnt/weka/fgeikyan/CoordToken/research/final_and_nabla_20261012/run_case.py mixed
