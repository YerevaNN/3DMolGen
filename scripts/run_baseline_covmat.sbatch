#!/usr/bin/env bash
# COV/MAT for a baseline refinement run: before, after (refine-all), gated.
#
#   METHOD=etflow sbatch scripts/run_baseline_covmat.sbatch
#
# CPU only. Metrics are stored per molecule and per condition, so this resumes
# and recomputes only what is missing.

#SBATCH --job-name=baseline-covmat
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/baseline_refine/%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/baseline_refine/%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
PYTHON_BIN=${PYTHON_BIN:-/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python}
METHOD=${METHOD:?set METHOD}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/baseline_refinement/${METHOD}}
GROUND_TRUTH=${GROUND_TRUTH:-/mnt/weka/fgeikyan/tokenizer/distinct_smi_filtered_13.pickle}
FSQ_CHECKPOINT=${FSQ_CHECKPOINT:-/mnt/weka/fgeikyan/fsq/new_checkpoints/full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt}
NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Started:  $(date)"
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Node:     $(hostname)"
echo "Method:   ${METHOD}"
echo "Workers:  ${NUM_WORKERS}"

"${PYTHON_BIN}" -m molgen3D.evaluation.baseline_tokenizer_refinement \
  --output-dir "${OUTPUT_DIR}" --method "${METHOD}" \
  --fsq-checkpoint "${FSQ_CHECKPOINT}" \
  --phase evaluate --ground-truth "${GROUND_TRUTH}" --num-workers "${NUM_WORKERS}"
echo "Finished: $(date)"
