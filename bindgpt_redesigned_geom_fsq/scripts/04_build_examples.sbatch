#!/usr/bin/env bash
#SBATCH --job-name=bgfsq-examples
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu09
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_redesigned_geom_fsq/logs/%x-%A_%a.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_redesigned_geom_fsq/logs/%x-%A_%a.err

# Slurm runs a spooled copy of this script, so BASH_SOURCE points at the spool
# directory, not at this file. Resolve common.sh from the repo root instead.
REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
source "${REPO_ROOT}/bindgpt_redesigned_geom_fsq/scripts/common.sh"
capture_environment "examples-${SPLIT:-train}"

# One array task per extracted shard; tasks never share an output file.
${CLI} --data-root "${DATA_ROOT}" build-examples \
  --split "${SPLIT:-train}" \
  --shard "${SLURM_ARRAY_TASK_ID}" \
  --device cuda --batch-size "${FSQ_BATCH_SIZE:-256}" \
  --layout "${LAYOUT:-suffix}"
