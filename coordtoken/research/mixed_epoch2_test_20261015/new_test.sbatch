#!/bin/bash
#SBATCH --job-name=ct_epoch2_test
#SBATCH --partition=research8
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/weka/fgeikyan/CoordToken/logs/mixed_epoch2_test_20261015/eval-%j.log
set -euo pipefail
cd /mnt/weka/fgeikyan/CoordToken
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python -u research/mixed_epoch2_test_20261015/evaluate_new_test.py
/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python -u research/mixed_epoch2_test_20261015/compare_full_tests.py
