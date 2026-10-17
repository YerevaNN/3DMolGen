#!/bin/bash
#SBATCH --account=research
#SBATCH --partition=research_cpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=03:00:00
#SBATCH --job-name=coordtoken_exact_duplicates
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${PYTHON:-python}" -u "$1/audit.py" --release "$2" --output "$3" --workers "${SLURM_CPUS_PER_TASK:-64}"
