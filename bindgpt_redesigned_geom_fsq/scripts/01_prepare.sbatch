#!/usr/bin/env bash
#SBATCH --job-name=bgfsq-prepare
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu09
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_redesigned_geom_fsq/logs/%x-%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_redesigned_geom_fsq/logs/%x-%j.err

# Slurm runs a spooled copy of this script, so BASH_SOURCE points at the spool
# directory, not at this file. Resolve common.sh from the repo root instead.
REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
source "${REPO_ROOT}/bindgpt_redesigned_geom_fsq/scripts/common.sh"
capture_environment prepare

# Pin the public BindGPT tokenizer, derive the SMILES token set, build the union
# vocabulary, then pull the decimal GEOM conformers out of the source corpora.
${CLI} prepare-tokenizers --offline
${CLI} --data-root "${DATA_ROOT}" extract --split valid
${CLI} --data-root "${DATA_ROOT}" extract --split train
