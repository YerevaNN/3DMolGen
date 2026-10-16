#!/usr/bin/env bash
#SBATCH --job-name=bgfsq-manifests
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
capture_environment manifests

# Identity manifest, the three pairwise overlap audits, and the FSQ provenance
# audit. Intersections are recorded but not filtered, matching existing SFT.
${CLI} --data-root "${DATA_ROOT}" manifests --workers 64
