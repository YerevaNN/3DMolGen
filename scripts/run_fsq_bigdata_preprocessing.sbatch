#!/usr/bin/env bash
# Run FSQ preprocessing for one grouped conformer shard.
#
# Direct usage:
#   sbatch --array=0-7 scripts/run_fsq_bigdata_preprocessing.sbatch
#
# Useful overrides:
#   PARTITION cannot be overridden here; use sbatch --partition=...
#   INPUT_DIR, DEST, RUN_NAME, FSQ_CKPT_PATH, FSQ_BATCH_SIZE, BATCH_SIZE,
#   MAX_CONFS, FSQ_MAX_TOKENS, SPLIT_NAME, PYTHON, VENV_PATH

#SBATCH --job-name=fsq-bigdata-preproc
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-7
#SBATCH --output=fsq_preproc_%A_%a.out
#SBATCH --error=fsq_preproc_%A_%a.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
INPUT_DIR=${INPUT_DIR:-/mnt/weka/vtarasov/3DBigData_formatted/grouped_conformers}
DEST=${DEST:-/mnt/weka/fgeikyan/fsq}
RUN_NAME=${RUN_NAME:-3dbigdata_fsq_4096}
SPLIT_NAME=${SPLIT_NAME:-train}
FSQ_CKPT_PATH=${FSQ_CKPT_PATH:-/mnt/weka/fgeikyan/fsq/new_checkpoints/full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt}
FSQ_DEVICE=${FSQ_DEVICE:-cuda}
FSQ_BATCH_SIZE=${FSQ_BATCH_SIZE:-128}
FSQ_MAX_TOKENS=${FSQ_MAX_TOKENS:-218}
MAX_CONFS=${MAX_CONFS:-0}
BATCH_SIZE=${BATCH_SIZE:-100000}
NUM_WORKERS=${NUM_WORKERS:-1}
PRECISION=${PRECISION:-4}
PYTHON=${PYTHON:-/mnt/weka/shared-cache/miniforge3/bin/python3}
SAVE_PICKLES=${SAVE_PICKLES:-0}
ISOMERIC=${ISOMERIC:-1}
CENTER=${CENTER:-0}
FILTER_RANGES=${FILTER_RANGES:-}
SHARD_ID=${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-}}
VENV_PATH=${VENV_PATH:-}

if [[ -z "${SHARD_ID}" ]]; then
  echo "ERROR: SHARD_ID is empty. Submit as an array or set SHARD_ID explicitly." >&2
  exit 2
fi

if [[ "${FSQ_DEVICE}" == cuda* || "${FSQ_DEVICE}" == auto ]]; then
  if [[ "${NUM_WORKERS}" != "1" ]]; then
    echo "ERROR: CUDA FSQ preprocessing requires NUM_WORKERS=1." >&2
    exit 2
  fi
fi

if [[ -n "${VENV_PATH}" ]]; then
  source "${VENV_PATH}"
elif [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  source "${REPO_ROOT}/.venv/bin/activate"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Array task:    ${SHARD_ID}"
echo "Node:          $(hostname)"
echo "Started:       $(date)"
echo "Repo:          ${REPO_ROOT}"
echo "Python:        ${PYTHON}"
echo "Input dir:     ${INPUT_DIR}"
echo "Dest:          ${DEST}/${RUN_NAME}"
echo "Checkpoint:    ${FSQ_CKPT_PATH}"
echo "FSQ device:    ${FSQ_DEVICE}"
echo "FSQ batch:     ${FSQ_BATCH_SIZE}"
echo "Max confs:     ${MAX_CONFS}"
echo "Save pickles:  ${SAVE_PICKLES}"
echo ""

if [[ "${FSQ_DEVICE}" == cuda* || "${FSQ_DEVICE}" == auto ]]; then
  "${PYTHON}" - <<'PY'
import sys
import torch

print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    sys.exit("CUDA is not available inside this Slurm allocation.")
PY
fi

cmd=(
  "${PYTHON}" -u -m molgen3D.data_processing.preprocess_big_data
  --input_dir "${INPUT_DIR}"
  --shard_id "${SHARD_ID}"
  --dest "${DEST}"
  --run_name "${RUN_NAME}"
  --embedding_type fsq
  --fsq_ckpt_path "${FSQ_CKPT_PATH}"
  --fsq_device "${FSQ_DEVICE}"
  --fsq_batch_size "${FSQ_BATCH_SIZE}"
  --fsq_max_tokens "${FSQ_MAX_TOKENS}"
  --num_workers "${NUM_WORKERS}"
  --split_name "${SPLIT_NAME}"
  --max_confs "${MAX_CONFS}"
  --precision "${PRECISION}"
  --batch_size "${BATCH_SIZE}"
)

if [[ -n "${FILTER_RANGES}" ]]; then
  cmd+=(--filter_ranges "${FILTER_RANGES}")
fi
if [[ "${ISOMERIC}" == "1" || "${ISOMERIC}" == "true" || "${ISOMERIC}" == "True" ]]; then
  cmd+=(--isomeric)
fi
if [[ "${CENTER}" == "1" || "${CENTER}" == "true" || "${CENTER}" == "True" ]]; then
  cmd+=(--center)
fi
if [[ "${SAVE_PICKLES}" != "1" && "${SAVE_PICKLES}" != "true" && "${SAVE_PICKLES}" != "True" ]]; then
  cmd+=(--no_pickles)
fi

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n\n'

# This is a single-process job inside an allocated batch shell. Running
# directly avoids node-specific Slurm PMIx plugin failures from srun.
"${cmd[@]}"
