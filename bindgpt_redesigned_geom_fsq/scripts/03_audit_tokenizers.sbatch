#!/usr/bin/env bash
#SBATCH --job-name=bgfsq-tokaudit
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --exclude=gpu09
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_redesigned_geom_fsq/logs/%x-%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_redesigned_geom_fsq/logs/%x-%j.err

# Slurm runs a spooled copy of this script, so BASH_SOURCE points at the spool
# directory, not at this file. Resolve common.sh from the repo root instead.
REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
source "${REPO_ROOT}/bindgpt_redesigned_geom_fsq/scripts/common.sh"
capture_environment tokaudit

# Round-trip both coordinate tokenizers on the same 10,000 validation
# conformers: public-tokenizer agreement plus reconstruction error for coord,
# fingerprint plus reconstruction error for the frozen FSQ checkpoint.
${CLI} --data-root "${DATA_ROOT}" audit-tokenizers \
  --conformers "${AUDIT_CONFORMERS:-10000}" \
  --public-checks "${AUDIT_PUBLIC_CHECKS:-10000}" \
  --device cuda --batch-size 128
