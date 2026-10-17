#!/bin/bash
#SBATCH --job-name=verify_nabla_revision
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
set -euo pipefail
COORDTOKEN_ROOT="${COORDTOKEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
COORDTOKEN_PYTHON="${COORDTOKEN_PYTHON:-python}"
cd "$COORDTOKEN_ROOT"
exec "$COORDTOKEN_PYTHON" -u data_collection/verify_filtered_csv.py --output "${COORDTOKEN_OUTPUT:-data_collection/runs/nabla_holdouts_v1}"
