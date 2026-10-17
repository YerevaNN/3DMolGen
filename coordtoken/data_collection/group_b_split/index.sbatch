#!/bin/bash
#SBATCH --account=research
#SBATCH --partition=research_cpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --job-name=coordtoken_group_b_index
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${PYTHON:-python}" -u "$1/index_inputs.py" --manifest "$2" --output "$3" \
  --workers "${SLURM_CPUS_PER_TASK:-64}"
