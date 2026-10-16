#!/usr/bin/env bash
# Ground-truth FSQ conformer interpolation sanity check.
#
# Submit:
#   sbatch scripts/run_conformer_interpolation_stage1.sbatch
#
# The output is incremental. Re-submitting with the same OUTPUT_DIR resumes
# completed generation and metric records.

#SBATCH --job-name=fsq-interp-stage1
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/interpolation/%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/interpolation/%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
PYTHON_BIN=${PYTHON_BIN:-/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python}
INPUT=${INPUT:-/mnt/weka/fgeikyan/tokenizer/distinct_smi_filtered_13.pickle}
FSQ_CHECKPOINT=${FSQ_CHECKPOINT:-/mnt/weka/fgeikyan/fsq/new_checkpoints/full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/conformer_interpolation/stage1_ground_truth}
MODEL_BATCH_SIZE=${MODEL_BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-1}
MAX_MOLECULES=${MAX_MOLECULES:-}
SEED=${SEED:-20260727}
PHASE=${PHASE:-generate}
PAIRING_STRATEGY=${PAIRING_STRATEGY:-random}
DEVICE=${DEVICE:-cuda}

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${REPO_ROOT}/outputs/slurm_jobs/interpolation"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${INPUT}" ]]; then
  echo "Ground-truth pickle does not exist: ${INPUT}" >&2
  exit 1
fi
if [[ ! -f "${FSQ_CHECKPOINT}" ]]; then
  echo "FSQ checkpoint does not exist: ${FSQ_CHECKPOINT}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

args=(
  -m molgen3D.evaluation.conformer_interpolation
  --input "${INPUT}"
  --output-dir "${OUTPUT_DIR}"
  --fsq-checkpoint "${FSQ_CHECKPOINT}"
  --device "${DEVICE}"
  --phase "${PHASE}"
  --model-batch-size "${MODEL_BATCH_SIZE}"
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
echo "Node:           $(hostname)"
echo "Input:          ${INPUT}"
echo "Checkpoint:     ${FSQ_CHECKPOINT}"
echo "Output:         ${OUTPUT_DIR}"
echo "Model batch:    ${MODEL_BATCH_SIZE}"
echo "Metric workers: ${NUM_WORKERS}"
echo "Max molecules:  ${MAX_MOLECULES:-all}"
echo "Phase:          ${PHASE}"
echo "Pairing:        ${PAIRING_STRATEGY}"
echo "Device:         ${DEVICE}"

if [[ "${DEVICE}" == cuda* ]]; then
  "${PYTHON_BIN}" - <<'PY'
import torch

print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the Slurm allocation.")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
fi

"${PYTHON_BIN}" "${args[@]}"
echo "Finished: $(date)"
