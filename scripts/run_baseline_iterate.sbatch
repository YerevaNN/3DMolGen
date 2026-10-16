#!/usr/bin/env bash
# One more tokenizer pass over a baseline's already-refined conformers.
#
#   METHOD=etflow ROUND=2 sbatch scripts/run_baseline_iterate.sbatch
#
# Round 1 is outputs/baseline_refinement/<method>; round r>=2 is
# <method>_r<r> and takes round r-1's `after` conformers as its input, so the
# `before` of every round is the `after` of the previous one and the rounds are
# directly comparable directory to directory.
#
# Export and prepare are CPU work but only a couple of minutes, so they share
# the GPU job rather than costing a second queue wait.

#SBATCH --job-name=baseline-iter
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --exclude=gpu09
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/baseline_refine/%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/baseline_refine/%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
PYTHON_BIN=${PYTHON_BIN:-/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python}
METHOD=${METHOD:?set METHOD to one of torsional, mcf_s, mcf_b, etflow}
ROUND=${ROUND:?set ROUND to 2 or higher}
FSQ_CHECKPOINT=${FSQ_CHECKPOINT:-/mnt/weka/fgeikyan/fsq/new_checkpoints/full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt}
MODEL_BATCH_SIZE=${MODEL_BATCH_SIZE:-128}

BASE="${REPO_ROOT}/outputs/baseline_refinement"
PREV_ROUND=$(( ROUND - 1 ))
if [[ "${PREV_ROUND}" -eq 1 ]]; then
  PREV_DIR="${BASE}/${METHOD}"
else
  PREV_DIR="${BASE}/${METHOD}_r${PREV_ROUND}"
fi
DIR="${BASE}/${METHOD}_r${ROUND}"
EXPORT_DIR="${BASE}/exported_rounds/${METHOD}_r${PREV_ROUND}"

mkdir -p "${DIR}" "${EXPORT_DIR}" "${REPO_ROOT}/outputs/slurm_jobs/baseline_refine"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Started:  $(date)"
echo "Node:     $(hostname)"
echo "Method:   ${METHOD}   round ${ROUND}  (from ${PREV_DIR})"

# Step 1: previous round's `after` conformers back into the input format.
shopt -s nullglob
existing=("${EXPORT_DIR}"/*_after.pkl)
if [[ ${#existing[@]} -eq 0 ]]; then
  "${PYTHON_BIN}" scripts/export_baseline_conformers.py "${PREV_DIR}" "${EXPORT_DIR}" --conditions after
  existing=("${EXPORT_DIR}"/*_after.pkl)
fi
[[ ${#existing[@]} -gt 0 ]] || { echo "export produced no pickle in ${EXPORT_DIR}" >&2; exit 1; }
INPUT="${existing[0]}"
echo "Input:    ${INPUT}"

# Step 2 and 3: lay out this round's records, then the pass itself.
for phase in prepare refine; do
  echo "--- phase ${phase}  $(date)"
  "${PYTHON_BIN}" -m molgen3D.evaluation.baseline_tokenizer_refinement \
    --input "${INPUT}" \
    --output-dir "${DIR}" \
    --method "${METHOD}" \
    --fsq-checkpoint "${FSQ_CHECKPOINT}" \
    --phase "${phase}" \
    --device cuda \
    --model-batch-size "${MODEL_BATCH_SIZE}" \
    --resume
done
echo "Finished: $(date)"
