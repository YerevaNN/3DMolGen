#!/usr/bin/env bash
# PoseBusters validity for the refined Setup 3 conditions (CPU only).
#
# Sharded, because there are seven conditions to bust:
#   VARIANT=full sbatch --array=0-2 scripts/run_setup3_refine_posebusters.sbatch
#
# Every shard writes into the same OUT_DIR but takes a disjoint slice of the
# molecules, so re-submitting resumes and only the last shard to finish writes
# the complete summary.json.

#SBATCH --job-name=setup3-refine-pb
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/posebusters/%A_%a.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/posebusters/%A_%a.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
PYTHON_BIN=${PYTHON_BIN:-/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python}
SCRIPT=${SCRIPT:-${REPO_ROOT}/scripts/setup3_posebusters.py}
VARIANT=${VARIANT:-full}
RUN_DIR=${RUN_DIR:-${REPO_ROOT}/outputs/conformer_interpolation/setup3_refine_${VARIANT}}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/outputs/conformer_interpolation/posebusters_setup3_refine_${VARIANT}}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-32}}
SHARD=${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
# Space-separated condition names; empty busts every condition in the record.
PB_CONDITIONS=${PB_CONDITIONS:-}

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
echo "Shard:    ${SHARD}/${NUM_SHARDS}"
echo "Workers:  ${WORKERS}"
echo "Cond:     ${PB_CONDITIONS:-<all>}"

args=("${RUN_DIR}" "${OUT_DIR}" "${WORKERS}" --shard "${SHARD}" --num-shards "${NUM_SHARDS}")
if [[ -n "${PB_CONDITIONS}" ]]; then
  args+=(--conditions ${PB_CONDITIONS})
fi

"${PYTHON_BIN}" "${SCRIPT}" "${args[@]}"
echo "Finished: $(date)"
