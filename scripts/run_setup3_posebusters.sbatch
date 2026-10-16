#!/usr/bin/env bash
# PoseBusters validity for every Setup 3 condition (CPU only).
#
# Submit both variants:
#   VARIANT=half sbatch scripts/run_setup3_posebusters.sbatch
#   VARIANT=full sbatch scripts/run_setup3_posebusters.sbatch
#
# Output is per molecule, so re-submitting with the same OUT_DIR resumes.

#SBATCH --job-name=setup3-posebusters
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/posebusters/%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/posebusters/%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
PYTHON_BIN=${PYTHON_BIN:-/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python}
SCRIPT=${SCRIPT:-${REPO_ROOT}/scripts/setup3_posebusters.py}
VARIANT=${VARIANT:-half}
RUN_DIR=${RUN_DIR:-${REPO_ROOT}/outputs/conformer_interpolation/setup3_generated_${VARIANT}_cart}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/outputs/conformer_interpolation/posebusters_setup3_${VARIANT}}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-48}}

mkdir -p "${OUT_DIR}" "${REPO_ROOT}/outputs/slurm_jobs/posebusters"

for f in "${PYTHON_BIN}" "${SCRIPT}" "${RUN_DIR}"; do
  [[ -e "${f}" ]] || { echo "Missing: ${f}" >&2; exit 1; }
done

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Started:  $(date)"
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Node:     $(hostname)"
echo "Variant:  ${VARIANT}"
echo "Run dir:  ${RUN_DIR}"
echo "Out dir:  ${OUT_DIR}"
echo "Workers:  ${WORKERS}"

"${PYTHON_BIN}" "${SCRIPT}" "${RUN_DIR}" "${OUT_DIR}" "${WORKERS}"
echo "Finished: $(date)"
