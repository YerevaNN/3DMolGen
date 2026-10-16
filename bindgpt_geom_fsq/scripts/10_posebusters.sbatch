#!/usr/bin/env bash
#SBATCH --job-name=bgfsq-pb
#SBATCH --partition=premium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --exclude=gpu09
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_geom_fsq/logs/%x-%A_%a.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_geom_fsq/logs/%x-%A_%a.err

# PoseBusters validity for one arm. CPU-only, and it depends only on the saved
# conditional generations -- not on COV/MAT -- so it runs concurrently with
# 09_metrics rather than after it.
REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
source "${REPO_ROOT}/bindgpt_geom_fsq/scripts/common.sh"
ARM=${ARM:?set ARM=coord or ARM=fsq}
TRAIN_SEED=${TRAIN_SEED:-17}
SAMPLING_SEED=${SAMPLING_SEED:-2023}
capture_environment "pb-${ARM}"

# Pin every numeric library to one thread. Without this each of the 64 worker
# processes spawns its own RDKit/BLAS thread pool on a 224-core node, and the
# job burns 10-25x the CPU it needs while making almost no forward progress.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RDKIT_NUM_THREADS=1

TAG="${ARM}_train${TRAIN_SEED}_sample${SAMPLING_SEED}"
COND="${BINDGPT_FSQ_ROOT}/generations/conformer/${ARM}/train_seed_${TRAIN_SEED}/sample_seed_${SAMPLING_SEED}"

${CLI} metrics-posebusters \
  --generations "${COND}" --tag "${TAG}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-64}" \
  --shard-size "${PB_SHARD_SIZE:-25}" \
  --shard-stride "${PB_SHARD_STRIDE:-${SLURM_ARRAY_TASK_COUNT:-1}}" \
  --shard-offset "${SLURM_ARRAY_TASK_ID:-0}"
