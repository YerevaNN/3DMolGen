#!/bin/bash
#SBATCH --job-name=ct_mixed_epoch2
#SBATCH --partition=research8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/weka/fgeikyan/CoordToken/logs/mixed_epoch2_20261014/train-%j.log
set -euo pipefail
cd /mnt/weka/fgeikyan/CoordToken
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python -u research/mixed_epoch2_20261014_v3/launch.py
