#!/bin/bash
#SBATCH --account=research
#SBATCH --partition=research_cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=coordtoken_group_c_final
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${PYTHON:-python}" -u "$1/run_nabla.py" --contract "$2" --output "$3" --workers "${SLURM_CPUS_PER_TASK:-32}"
