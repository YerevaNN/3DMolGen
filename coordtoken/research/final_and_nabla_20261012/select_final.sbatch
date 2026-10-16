#!/bin/bash
#SBATCH --job-name=ct_select_final
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/weka/fgeikyan/CoordToken/logs/final_and_nabla_20261012/ct_select_final-%j.log

set -euo pipefail
cd /mnt/weka/fgeikyan/CoordToken
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/coordtoken_final_${SLURM_JOB_ID}
/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python -u /mnt/weka/fgeikyan/CoordToken/research/final_and_nabla_20261012/selection.py final
exec /mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python -u /mnt/weka/fgeikyan/CoordToken/research/final_and_nabla_20261012/report.py
