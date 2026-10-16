#!/usr/bin/env bash
# CPU half of the baseline refinement: write the `before` conditions.
#
#   METHOD=etflow sbatch scripts/run_baseline_prepare.sbatch
#
# No GPU: this only strips hydrogens and lays out per-molecule records, so the
# baseline validity can be scored while the GPU refine phase is still queued.

#SBATCH --job-name=baseline-prepare
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/baseline_refine/%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/baseline_refine/%j.err

set -euo pipefail
export PHASE=prepare
export DEVICE=cpu
exec bash "${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}/scripts/run_baseline_refine.sbatch"
