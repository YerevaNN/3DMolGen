#!/usr/bin/env bash
# Setup 3: interpolate between model-generated conformers.
#
# Submit both variants:
#   VARIANT=half sbatch scripts/run_generated_interpolation_setup3.sbatch
#   VARIANT=full sbatch scripts/run_generated_interpolation_setup3.sbatch
#
# Output is incremental; re-submitting with the same OUTPUT_DIR resumes.

#SBATCH --job-name=fsq-interp-setup3
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/interpolation/%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/interpolation/%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
PYTHON_BIN=${PYTHON_BIN:-/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python}
VARIANT=${VARIANT:-half}
GENERATED=${GENERATED:-${REPO_ROOT}/outputs/gen_results_user/20260614_024445_qwen06b_sft_6e_step28320_fsq_distinct/generation_results.pickle}
GROUND_TRUTH=${GROUND_TRUTH:-/mnt/weka/fgeikyan/tokenizer/distinct_smi_filtered_13.pickle}
FSQ_CHECKPOINT=${FSQ_CHECKPOINT:-/mnt/weka/fgeikyan/fsq/new_checkpoints/full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/conformer_interpolation/setup3_generated_${VARIANT}}
MODEL_BATCH_SIZE=${MODEL_BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-16}
MAX_MOLECULES=${MAX_MOLECULES:-}
SEED=${SEED:-20260727}
PHASE=${PHASE:-all}
DEVICE=${DEVICE:-cuda}

mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/outputs/slurm_jobs/interpolation"

for f in "${PYTHON_BIN}" "${GENERATED}" "${GROUND_TRUTH}" "${FSQ_CHECKPOINT}"; do
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
  -m molgen3D.evaluation.generated_conformer_interpolation
  --generated "${GENERATED}"
  --ground-truth "${GROUND_TRUTH}"
  --output-dir "${OUTPUT_DIR}"
  --fsq-checkpoint "${FSQ_CHECKPOINT}"
  --variant "${VARIANT}"
  --device "${DEVICE}"
  --phase "${PHASE}"
  --model-batch-size "${MODEL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --resume
)
if [[ -n "${MAX_MOLECULES}" ]]; then
  args+=(--max-molecules "${MAX_MOLECULES}")
fi

echo "Started:    $(date)"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Variant:    ${VARIANT}"
echo "Generated:  ${GENERATED}"
echo "Output:     ${OUTPUT_DIR}"
echo "Phase:      ${PHASE}"

"${PYTHON_BIN}" "${args[@]}"
echo "Finished: $(date)"