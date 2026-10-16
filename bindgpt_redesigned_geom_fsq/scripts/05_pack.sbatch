#!/usr/bin/env bash
#SBATCH --job-name=bgfsq-pack
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --exclude=gpu09
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_redesigned_geom_fsq/logs/%x-%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_redesigned_geom_fsq/logs/%x-%j.err

# Slurm runs a spooled copy of this script, so BASH_SOURCE points at the spool
# directory, not at this file. Resolve common.sh from the repo root instead.
REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
source "${REPO_ROOT}/bindgpt_redesigned_geom_fsq/scripts/common.sh"
capture_environment pack

for split in valid train; do
  ${CLI} --data-root "${DATA_ROOT}" merge-examples --split "${split}"
  ${CLI} --data-root "${DATA_ROOT}" pack --split "${split}"
done
${CLI} write-configs --layout "${LAYOUT:-suffix}" --seed "${TRAIN_SEED:-17}" \
  --sampling-seed "${SAMPLING_SEED:-2023}" --cluster "${CLUSTER_NAME:-unknown}"
