#!/bin/bash
#SBATCH --job-name=ct_epoch2_shuffle
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/weka/fgeikyan/CoordToken/logs/mixed_epoch2_20261014/prepare-%j.log
set -euo pipefail
cd /mnt/weka/fgeikyan/CoordToken
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python -u research/mixed_epoch2_20261014/prepare_epoch.py
