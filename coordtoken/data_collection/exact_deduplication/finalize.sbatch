#!/bin/bash
#SBATCH --account=research
#SBATCH --partition=research_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=coordtoken_dedup_finalize
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${PYTHON:-python}" -u "$1/finalize.py" --release "$2" --audit "$3" --repo "$4" --paper "$5"
