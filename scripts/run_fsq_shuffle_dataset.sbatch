#!/bin/bash
#SBATCH --job-name=fsq-shuffle
#SBATCH --partition=research
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/preprocessing/fsq_shuffle_%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/preprocessing/fsq_shuffle_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
INPUT_DIR=${INPUT_DIR:-/mnt/weka/fgeikyan/fsq/3dbigdata_fsq_4096}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/weka/fgeikyan/fsq/3dbigdata_fsq_4096_shuffled/processed_strings/train}
BUCKET_DIR=${BUCKET_DIR:-/mnt/weka/fgeikyan/fsq/3dbigdata_fsq_4096_shuffle_buckets_train}
PYTHON=${PYTHON:-/mnt/weka/shared-cache/miniforge3/bin/python3}
NUM_BUCKETS=${NUM_BUCKETS:-400}
WORKERS=${WORKERS:-64}

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

"${PYTHON}" scripts/shuffle_jsonl_dataset.py \
  "${INPUT_DIR}" \
  "${OUTPUT_DIR}" \
  --input-glob "shard_*/processed_strings/train/*.jsonl" \
  --num-buckets "${NUM_BUCKETS}" \
  --workers "${WORKERS}" \
  --bucket-dir "${BUCKET_DIR}" \
  --cleanup
