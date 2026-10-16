#!/usr/bin/env bash
#SBATCH --job-name=qwen06b-fsq-bigdata-infer
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/inference/%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/inference/%j.err
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
RUN_ROOT=${RUN_ROOT:-/mnt/weka/fgeikyan/checkpoints/qwen3_06b/260607-0722-33c9-qwen3_06b_fsq_bigdata_from_base_1e}
STEP=${STEP:-70534}
EPOCH=${EPOCH:-1e}
DCP_PATH=${DCP_PATH:-${RUN_ROOT}/step-${STEP}}
MODEL_PATH=${MODEL_PATH:-${DCP_PATH}-hf}

PYTHON_BIN=${PYTHON_BIN:-/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python}
INFERENCE_SCRIPT=${INFERENCE_SCRIPT:-${REPO_ROOT}/src/molgen3D/evaluation/inference.py}
HF_ASSETS_PATH=${HF_ASSETS_PATH:-/mnt/weka/vtarasov/Qwen3-0.6B-Base/patched_for_titan}
TOKENIZER_PATH=${TOKENIZER_PATH:-${REPO_ROOT}/src/molgen3D/training/tokenizers/Qwen3_tokenizer_fsq_4096}
VQ_CKPT_PATH=${VQ_CKPT_PATH:-/mnt/weka/fgeikyan/fsq/new_checkpoints/full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt}
TEST_DATA_PATH=${TEST_DATA_PATH:-/mnt/weka/fgeikyan/tokenizer/distinct_smi_filtered_13.pickle}
RESULTS_ROOT=${RESULTS_ROOT:-${REPO_ROOT}/outputs/gen_results_user}
RUN_NAME=${RUN_NAME:-qwen06b_bigdata_step${STEP}_fsq_distinct}
BATCH_SIZE=${BATCH_SIZE:-256}
PARALLEL_JOBS=${PARALLEL_JOBS:-1}
SAMPLING_CONFIG=${SAMPLING_CONFIG:-top_p_sampling1}
LIMIT=${LIMIT:-}
UNIQUE_ONLY=${UNIQUE_ONLY:-0}
CONVERT_IF_MISSING=${CONVERT_IF_MISSING:-1}

mkdir -p "${REPO_ROOT}/outputs/slurm_jobs/inference"
mkdir -p "${RESULTS_ROOT}"
mkdir -p "/tmp/matplotlib-${USER:-fgeikyan}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${TOKENIZER_PATH}" ]]; then
  echo "Error: tokenizer path not found: ${TOKENIZER_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TEST_DATA_PATH}" ]]; then
  echo "Error: test data not found: ${TEST_DATA_PATH}" >&2
  exit 1
fi
if [[ ! -f "${VQ_CKPT_PATH}" ]]; then
  echo "Error: VQ_CKPT_PATH not found: ${VQ_CKPT_PATH}" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/matplotlib-${USER:-fgeikyan}
export MOLGEN3D_REPO_ROOT="${REPO_ROOT}"
export MOLGEN3D_PROJECT_ROOT="${REPO_ROOT}"
export VQ_CKPT_PATH

cd "${REPO_ROOT}"

echo "Running Qwen3 0.6B FSQ big-data inference"
echo "Repo root: ${REPO_ROOT}"
echo "DCP path: ${DCP_PATH}"
echo "HF model path: ${MODEL_PATH}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Test data: ${TEST_DATA_PATH}"
echo "VQ ckpt: ${VQ_CKPT_PATH}"
echo "Results root: ${RESULTS_ROOT}"
echo "Run name: ${RUN_NAME}"
echo "Batch size: ${BATCH_SIZE}"
echo "Limit: ${LIMIT:-<unset>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

if [[ ! -f "${MODEL_PATH}/model.safetensors" && ! -f "${MODEL_PATH}/model.safetensors.index.json" ]]; then
  if [[ "${CONVERT_IF_MISSING}" != "1" ]]; then
    echo "Error: HF model is missing and CONVERT_IF_MISSING=${CONVERT_IF_MISSING}: ${MODEL_PATH}" >&2
    exit 1
  fi
  if [[ ! -d "${DCP_PATH}" ]]; then
    echo "Error: DCP checkpoint path not found: ${DCP_PATH}" >&2
    exit 1
  fi
  echo "HF checkpoint missing; converting DCP checkpoint first (-> ${DCP_PATH}-hf)."
  PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    -m molgen3D.training.pretraining.helpers.convert_qwen3_dcp_to_hf \
    --dcp-path "${DCP_PATH}" \
    --hf-assets-path "${HF_ASSETS_PATH}" \
    --tokenizer-path "${TOKENIZER_PATH}"
else
  echo "HF checkpoint already exists; skipping conversion."
fi

SRUN_BIN=${SRUN_BIN:-}
if [[ -z "${SRUN_BIN}" ]]; then
  for candidate in /usr/bin/srun /bin/srun "$(command -v srun 2>/dev/null || true)"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      SRUN_BIN="${candidate}"
      break
    fi
  done
fi

DIRECT_CUDA_OK=$("${PYTHON_BIN}" - <<'PY'
import os
import torch
ok = int(torch.cuda.is_available() and torch.cuda.device_count() > 0)
print({"mode": "direct", "cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
print(ok)
PY
)
echo "${DIRECT_CUDA_OK}"

LAUNCH_PREFIX=()
if [[ "$(echo "${DIRECT_CUDA_OK}" | tail -n 1)" == "1" ]]; then
  echo "Using direct Python launch; CUDA is visible."
elif [[ -n "${SRUN_BIN}" ]]; then
  SRUN_CUDA_OK=$("${SRUN_BIN}" --ntasks=1 --gres=gpu:1 "${PYTHON_BIN}" - <<'PY'
import os
import torch
ok = int(torch.cuda.is_available() and torch.cuda.device_count() > 0)
print({"mode": "srun", "cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
print(ok)
PY
)
  echo "${SRUN_CUDA_OK}"
  if [[ "$(echo "${SRUN_CUDA_OK}" | tail -n 1)" == "1" ]]; then
    echo "Using srun launch; CUDA is visible inside a Slurm step."
    LAUNCH_PREFIX=("${SRUN_BIN}" --ntasks=1 --gres=gpu:1)
  else
    echo "Error: CUDA is not visible in direct Python or srun." >&2
    exit 1
  fi
else
  echo "Error: CUDA is not visible and srun was not found." >&2
  exit 1
fi

LIMIT_ARG=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

UNIQUE_ONLY_ARG=()
if [[ "${UNIQUE_ONLY}" == "1" ]]; then
  UNIQUE_ONLY_ARG=(--unique_only)
fi

echo "Starting inference"
PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" \
"${LAUNCH_PREFIX[@]}" "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
  --device local \
  --fsq \
  --epoch "${EPOCH}" \
  --test_set distinct \
  --batch_size "${BATCH_SIZE}" \
  --parallel_jobs "${PARALLEL_JOBS}" \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --test_data_path "${TEST_DATA_PATH}" \
  --results_path "${RESULTS_ROOT}" \
  --run_name "${RUN_NAME}" \
  --sampling_config "${SAMPLING_CONFIG}" \
  "${LIMIT_ARG[@]}" \
  "${UNIQUE_ONLY_ARG[@]}"

echo "Done."
