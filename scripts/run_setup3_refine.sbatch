#!/usr/bin/env bash
# Setup 3 addendum: one tokenizer encode/decode pass over the interpolated ensembles.
#
#   VARIANT=full sbatch scripts/run_setup3_refine.sbatch
#
# Output is per molecule, so re-submitting with the same OUTPUT_DIR resumes.

#SBATCH --job-name=setup3-refine
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/interpolation/%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/interpolation/%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
PYTHON_BIN=${PYTHON_BIN:-/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python}
VARIANT=${VARIANT:-full}
SOURCE_DIR=${SOURCE_DIR:-${REPO_ROOT}/outputs/conformer_interpolation/setup3_generated_${VARIANT}_cart}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/conformer_interpolation/setup3_refine_${VARIANT}}
GENERATED=${GENERATED:-${REPO_ROOT}/outputs/gen_results_user/20260614_024445_qwen06b_sft_6e_step28320_fsq_distinct/generation_results.pickle}
GROUND_TRUTH=${GROUND_TRUTH:-/mnt/weka/fgeikyan/tokenizer/distinct_smi_filtered_13.pickle}
FSQ_CHECKPOINT=${FSQ_CHECKPOINT:-/mnt/weka/fgeikyan/fsq/new_checkpoints/full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt}
CONDITIONS=${CONDITIONS:-"latent random_token cartesian"}
MODEL_BATCH_SIZE=${MODEL_BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}
MAX_MOLECULES=${MAX_MOLECULES:-}
PHASE=${PHASE:-all}
DEVICE=${DEVICE:-cuda}

mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/outputs/slurm_jobs/interpolation"

for f in "${PYTHON_BIN}" "${SOURCE_DIR}" "${GENERATED}" "${GROUND_TRUTH}" "${FSQ_CHECKPOINT}"; do
  [[ -e "${f}" ]] || { echo "Missing: ${f}" >&2; exit 1; }
done

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

args=(
  -m molgen3D.evaluation.setup3_tokenizer_refinement
  --source-dir "${SOURCE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --fsq-checkpoint "${FSQ_CHECKPOINT}"
  --generated "${GENERATED}"
  --ground-truth "${GROUND_TRUTH}"
  --conditions ${CONDITIONS}
  --device "${DEVICE}"
  --phase "${PHASE}"
  --model-batch-size "${MODEL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --resume
)
if [[ -n "${MAX_MOLECULES}" ]]; then
  args+=(--max-molecules "${MAX_MOLECULES}")
fi

echo "Started:    $(date)"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Variant:    ${VARIANT}"
echo "Source:     ${SOURCE_DIR}"
echo "Output:     ${OUTPUT_DIR}"
echo "Conditions: ${CONDITIONS}"
echo "Phase:      ${PHASE}"

"${PYTHON_BIN}" "${args[@]}"
echo "Finished: $(date)"
