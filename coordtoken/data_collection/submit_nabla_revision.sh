#!/bin/bash
#SBATCH --job-name=coordtoken_nabla_revision
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
COORDTOKEN_ROOT="${COORDTOKEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
COORDTOKEN_PYTHON="${COORDTOKEN_PYTHON:-python}"
cd "$COORDTOKEN_ROOT"
exec "$COORDTOKEN_PYTHON" -u data_collection/inchikey_filter.py \
 --audit "${COORDTOKEN_AUDIT:-logs/inchikey_overlap/nabla_all_val_test_190m}" \
 --reference-label val/nablaDFT \
 --reference-label test/nablaDFT_scaffolds \
 --reference-label test/nablaDFT_structures \
 --failure-policy quarantine \
 --output "${COORDTOKEN_OUTPUT:-data_collection/runs/nabla_holdouts_v1}" \
 --workers "${SLURM_CPUS_PER_TASK:-16}" --resume
