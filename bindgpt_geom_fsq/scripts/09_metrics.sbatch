#!/usr/bin/env bash
#SBATCH --job-name=bgfsq-metrics
#SBATCH --partition=premium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu09
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_geom_fsq/logs/%x-%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_geom_fsq/logs/%x-%j.err

# Slurm runs a spooled copy of this script, so BASH_SOURCE points at the spool
# directory, not at this file. Resolve common.sh from the repo root instead.
REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
source "${REPO_ROOT}/bindgpt_geom_fsq/scripts/common.sh"
ARM=${ARM:?set ARM=coord or ARM=fsq}
TRAIN_SEED=${TRAIN_SEED:-17}
SAMPLING_SEED=${SAMPLING_SEED:-2023}
capture_environment "metrics-${ARM}"

TAG="${ARM}_train${TRAIN_SEED}_sample${SAMPLING_SEED}"
UNCOND="${BINDGPT_FSQ_ROOT}/generations/unconditional/${ARM}/train_seed_${TRAIN_SEED}/sample_seed_${SAMPLING_SEED}"
COND="${BINDGPT_FSQ_ROOT}/generations/conformer/${ARM}/train_seed_${TRAIN_SEED}/sample_seed_${SAMPLING_SEED}"

${CLI} metrics-unconditional --attempts "${UNCOND}/attempts.jsonl" \
  --tag "${TAG}" --both-references --parallel
${CLI} metrics-conformer --generations "${COND}" --tag "${TAG}" --threshold 0.75 --num-workers "${SLURM_CPUS_PER_TASK:-64}"
${CLI} report
