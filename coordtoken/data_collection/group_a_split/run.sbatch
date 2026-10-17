#!/bin/bash
#SBATCH --account=research
#SBATCH --partition=research_cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=coordtoken_group_a_keys
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: sbatch [--output LOG] run.sbatch CODE_DIRECTORY MANIFEST NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${PYTHON:-python}" -u "$1/process_group_a.py" \
  --manifest "$2" --output "$3" --workers "${SLURM_CPUS_PER_TASK:-32}"
