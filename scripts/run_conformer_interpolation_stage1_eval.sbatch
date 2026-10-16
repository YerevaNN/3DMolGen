#!/usr/bin/env bash
# CPU-only evaluation of persisted Stage 1 interpolation conformers.

#SBATCH --job-name=fsq-interp-eval
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/interpolation/%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/interpolation/%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
PYTHON_BIN=${PYTHON_BIN:-/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python}
INPUT=${INPUT:-/mnt/weka/fgeikyan/tokenizer/distinct_smi_filtered_13.pickle}
FSQ_CHECKPOINT=${FSQ_CHECKPOINT:-/mnt/weka/fgeikyan/fsq/new_checkpoints/full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/conformer_interpolation/stage1_ground_truth}
NUM_WORKERS=${NUM_WORKERS:-32}
MAX_MOLECULES=${MAX_MOLECULES:-}
SEED=${SEED:-20260727}
PAIRING_STRATEGY=${PAIRING_STRATEGY:-random}

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${REPO_ROOT}/outputs/slurm_jobs/interpolation"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

args=(
  -m molgen3D.evaluation.conformer_interpolation
  --input "${INPUT}"
  --output-dir "${OUTPUT_DIR}"
  --fsq-checkpoint "${FSQ_CHECKPOINT}"
  --device cuda
  --phase evaluate
  --num-workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --pairing-strategy "${PAIRING_STRATEGY}"
  --resume
)
if [[ -n "${MAX_MOLECULES}" ]]; then
  args+=(--max-molecules "${MAX_MOLECULES}")
fi

echo "Started:        $(date)"
echo "Job ID:         ${SLURM_JOB_ID:-local}"
echo "Input:          ${INPUT}"
echo "Output:         ${OUTPUT_DIR}"
echo "Metric workers: ${NUM_WORKERS}"
echo "Max molecules:  ${MAX_MOLECULES:-all}"
echo "Pairing:        ${PAIRING_STRATEGY}"

"${PYTHON_BIN}" "${args[@]}"
echo "Finished: $(date)"
