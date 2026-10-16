#!/usr/bin/env bash
# CPU half of the Setup 3 refinement addendum: COV/MAT scoring, gate assembly.
#
# The refine phase needs an H100; scoring and gating do not, so they run here
# instead of holding a GPU idle for the longer of the two.
#
#   PHASE=evaluate sbatch scripts/run_setup3_refine_eval.sbatch
#
#   PHASE=gate GATE_DIR=... GATE_PB_DIR=... \
#     sbatch scripts/run_setup3_refine_eval.sbatch
#
# Metrics are stored per molecule and per condition, so this resumes and only
# computes what is missing.

#SBATCH --job-name=setup3-refine-eval
#SBATCH --partition=research8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
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
NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}
PHASE=${PHASE:-evaluate}
GATE_DIR=${GATE_DIR:-}
GATE_PB_DIR=${GATE_PB_DIR:-}

mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/outputs/slurm_jobs/interpolation"

for f in "${PYTHON_BIN}" "${SOURCE_DIR}" "${GENERATED}" "${GROUND_TRUTH}"; do
  [[ -e "${f}" ]] || { echo "Missing: ${f}" >&2; exit 1; }
done

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

args=(
  -m molgen3D.evaluation.setup3_tokenizer_refinement
  --source-dir "${SOURCE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --fsq-checkpoint "${FSQ_CHECKPOINT}"
  --generated "${GENERATED}"
  --ground-truth "${GROUND_TRUTH}"
  --conditions ${CONDITIONS}
  --phase "${PHASE}"
  --num-workers "${NUM_WORKERS}"
  --resume
)
if [[ -n "${GATE_DIR}" ]]; then
  args+=(--gate-dir "${GATE_DIR}")
fi
if [[ -n "${GATE_PB_DIR}" ]]; then
  args+=(--gate-pb-dir "${GATE_PB_DIR}")
fi

echo "Started:  $(date)"
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Node:     $(hostname)"
echo "Output:   ${OUTPUT_DIR}"
echo "Phase:    ${PHASE}"
echo "Workers:  ${NUM_WORKERS}"

"${PYTHON_BIN}" "${args[@]}"
echo "Finished: $(date)"
