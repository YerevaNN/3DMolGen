#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
JOB_SCRIPT=${JOB_SCRIPT:-${REPO_ROOT}/scripts/run_qwen3_06b_fsq_bigdata_inference.sbatch}
RUN_ROOT=${RUN_ROOT:-/mnt/weka/fgeikyan/checkpoints/qwen3_06b/260607-0722-33c9-qwen3_06b_fsq_bigdata_from_base_1e}
STEP=${STEP:-70534}
EPOCH=${EPOCH:-1e}
DCP_PATH=${DCP_PATH:-${RUN_ROOT}/step-${STEP}}
MODEL_PATH=${MODEL_PATH:-${DCP_PATH}-hf}
TOKENIZER_PATH=${TOKENIZER_PATH:-${REPO_ROOT}/src/molgen3D/training/tokenizers/Qwen3_tokenizer_fsq_4096}
TEST_DATA_PATH=${TEST_DATA_PATH:-/mnt/weka/fgeikyan/tokenizer/distinct_smi_filtered_13.pickle}
RESULTS_ROOT=${RESULTS_ROOT:-${REPO_ROOT}/outputs/gen_results_user}
RUN_NAME=${RUN_NAME:-qwen06b_bigdata_step${STEP}_fsq_distinct}
BATCH_SIZE=${BATCH_SIZE:-256}
PARALLEL_JOBS=${PARALLEL_JOBS:-1}
SAMPLING_CONFIG=${SAMPLING_CONFIG:-top_p_sampling1}
LIMIT=${LIMIT:-}
UNIQUE_ONLY=${UNIQUE_ONLY:-0}
CONVERT_IF_MISSING=${CONVERT_IF_MISSING:-1}
CPUS=${CPUS:-12}
MEM=${MEM:-128G}
TIME_LIMIT=${TIME_LIMIT:-1-00:00:00}

mkdir -p "${REPO_ROOT}/outputs/slurm_jobs/inference"
mkdir -p "${RESULTS_ROOT}"

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "Error: job script not found: ${JOB_SCRIPT}" >&2
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
if [[ ! -f "${MODEL_PATH}/model.safetensors" && ! -d "${DCP_PATH}" ]]; then
  echo "Error: neither HF model (${MODEL_PATH}) nor DCP checkpoint (${DCP_PATH}) found" >&2
  exit 1
fi

echo "Submitting Qwen3 0.6B FSQ big-data inference"
echo "STEP=${STEP}"
echo "EPOCH=${EPOCH}"
echo "DCP_PATH=${DCP_PATH}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "TEST_DATA_PATH=${TEST_DATA_PATH}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "RUN_NAME=${RUN_NAME}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "LIMIT=${LIMIT:-<unset>}"
echo "UNIQUE_ONLY=${UNIQUE_ONLY}"
echo "CONVERT_IF_MISSING=${CONVERT_IF_MISSING}"
echo "CPUS=${CPUS}"
echo "MEM=${MEM}"
echo "TIME_LIMIT=${TIME_LIMIT}"

job_id=$(
  sbatch --parsable \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    --time="${TIME_LIMIT}" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",RUN_ROOT="${RUN_ROOT}",STEP="${STEP}",EPOCH="${EPOCH}",DCP_PATH="${DCP_PATH}",MODEL_PATH="${MODEL_PATH}",TOKENIZER_PATH="${TOKENIZER_PATH}",TEST_DATA_PATH="${TEST_DATA_PATH}",RESULTS_ROOT="${RESULTS_ROOT}",RUN_NAME="${RUN_NAME}",BATCH_SIZE="${BATCH_SIZE}",PARALLEL_JOBS="${PARALLEL_JOBS}",SAMPLING_CONFIG="${SAMPLING_CONFIG}",LIMIT="${LIMIT}",UNIQUE_ONLY="${UNIQUE_ONLY}",CONVERT_IF_MISSING="${CONVERT_IF_MISSING}" \
    "${JOB_SCRIPT}"
)

echo "Job submitted: ${job_id}"
