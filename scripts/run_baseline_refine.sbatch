#!/usr/bin/env bash
# One tokenizer encode/decode pass over an external baseline's conformers.
#
#   METHOD=etflow sbatch scripts/run_baseline_refine.sbatch
#
# Output is per molecule, so re-submitting with the same OUTPUT_DIR resumes.

#SBATCH --job-name=baseline-refine
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/baseline_refine/%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/baseline_refine/%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
PYTHON_BIN=${PYTHON_BIN:-/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python}
METHOD=${METHOD:?set METHOD to one of torsional, mcf_s, mcf_b, etflow}
FSQ_CHECKPOINT=${FSQ_CHECKPOINT:-/mnt/weka/fgeikyan/fsq/new_checkpoints/full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt}
MODEL_BATCH_SIZE=${MODEL_BATCH_SIZE:-128}
PHASE=${PHASE:-refine}
MAX_MOLECULES=${MAX_MOLECULES:-}
DEVICE=${DEVICE:-cuda}

case "${METHOD}" in
  torsional) INPUT=/mnt/weka/vtarasov/code/torsional-diffusion/torsional_diff_pred/conformers_20steps_final_geom_keys_2.pkl ;;
  mcf_s)     INPUT=/mnt/weka/vtarasov/code/ml-mcf/artifacts/viz/S/ml-mcf-s/generation_result.pickle ;;
  mcf_b)     INPUT=/mnt/weka/vtarasov/code/ml-mcf/artifacts/viz/B/ml-mcf-b/processed.pickle ;;
  etflow)    INPUT=/mnt/weka/vtarasov/code/3DMolGen/etflow/generated_files_geom_original_no_atom_num.pkl ;;
  qwen17b)   INPUT=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/gen_results_user/20260622_092610_qwen17b_sft_6e_step28320_fsq_distinct/generation_results.pickle ;;
  *) echo "Unknown METHOD: ${METHOD}" >&2; exit 1 ;;
esac
INPUT=${INPUT_OVERRIDE:-${INPUT}}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/baseline_refinement/${METHOD}}

mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/outputs/slurm_jobs/baseline_refine"

for f in "${PYTHON_BIN}" "${INPUT}" "${FSQ_CHECKPOINT}"; do
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
  -m molgen3D.evaluation.baseline_tokenizer_refinement
  --input "${INPUT}"
  --output-dir "${OUTPUT_DIR}"
  --method "${METHOD}"
  --fsq-checkpoint "${FSQ_CHECKPOINT}"
  --phase "${PHASE}"
  --device "${DEVICE}"
  --model-batch-size "${MODEL_BATCH_SIZE}"
  --resume
)
if [[ -n "${MAX_MOLECULES}" ]]; then
  args+=(--max-molecules "${MAX_MOLECULES}")
fi

echo "Started:  $(date)"
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Node:     $(hostname)"
echo "Method:   ${METHOD}"
echo "Input:    ${INPUT}"
echo "Output:   ${OUTPUT_DIR}"
echo "Phase:    ${PHASE}"

"${PYTHON_BIN}" "${args[@]}"
echo "Finished: $(date)"
