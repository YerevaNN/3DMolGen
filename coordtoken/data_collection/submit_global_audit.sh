#!/bin/bash
#SBATCH --job-name=coordtoken_all_holdouts
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
COORDTOKEN_ROOT="${COORDTOKEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
COORDTOKEN_PYTHON="${COORDTOKEN_PYTHON:-python}"
cd "$COORDTOKEN_ROOT"
exec "$COORDTOKEN_PYTHON" -u data_collection/audit_from_registry.py \
 --registry "${COORDTOKEN_REGISTRY:-data_collection/runs/holdout_registry_v3/registry.json}" \
 --output "${COORDTOKEN_OUTPUT:-data_collection/runs/all_holdouts_audit_v1}" \
 --workers "${SLURM_CPUS_PER_TASK:-32}"
