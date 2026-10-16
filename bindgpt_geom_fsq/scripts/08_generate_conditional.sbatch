#!/usr/bin/env bash
#SBATCH --job-name=bgfsq-cond
#SBATCH --partition=premium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu09
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_geom_fsq/logs/%x-%A_%a.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_geom_fsq/logs/%x-%A_%a.err

# Job array over test-molecule shards; each task writes its own attempts file.
# Slurm runs a spooled copy of this script, so BASH_SOURCE points at the spool
# directory, not at this file. Resolve common.sh from the repo root instead.
REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
source "${REPO_ROOT}/bindgpt_geom_fsq/scripts/common.sh"
ARM=${ARM:?set ARM=coord or ARM=fsq}
TRAIN_SEED=${TRAIN_SEED:-17}
CHECKPOINT=${CHECKPOINT:-}
if [[ -z "${CHECKPOINT}" ]]; then
  # Default to the end-of-training checkpoint for this arm and seed.
  CHECKPOINT=$(ls -d "${BINDGPT_FSQ_ROOT}/checkpoints/${ARM}/seed_${TRAIN_SEED}"/step-*-final 2>/dev/null | tail -1)
fi
if [[ -z "${CHECKPOINT}" || ! -d "${CHECKPOINT}" ]]; then
  echo "ERROR: no checkpoint for arm=${ARM} seed=${TRAIN_SEED}; set CHECKPOINT explicitly" >&2
  exit 2
fi
echo "checkpoint: ${CHECKPOINT}"
NUM_SHARDS=${NUM_SHARDS:-40}
capture_environment "cond-${ARM}"

${CLI} --data-root "${DATA_ROOT}" generate-conditional \
  --config "${BINDGPT_FSQ_ROOT}/configs/${ARM}_seed${TRAIN_SEED}.yaml" \
  --checkpoint "${CHECKPOINT}" --device cuda \
  --shard "${SLURM_ARRAY_TASK_ID}" --num-shards "${NUM_SHARDS}" \
  --batch-size "${GEN_BATCH_SIZE:-64}"
