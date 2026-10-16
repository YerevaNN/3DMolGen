#!/usr/bin/env bash
# Assemble round R's gated ensemble. Deliberately separate from scoring: the next
# round needs these conformers, and nothing else, so COV/MAT must not sit on the
# critical path of the chain.
#SBATCH --job-name=gated-asm
#SBATCH --partition=premium
#SBATCH --qos=researcher
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/baseline_refine/ga_%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/baseline_refine/ga_%j.err
set -euo pipefail
cd /mnt/weka/fgeikyan/3dmolgen_bigdata
export PYTHONPATH=src PYTHONUNBUFFERED=1
/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python scripts/assemble_gated.py "${METHOD:?}" "${ROUND:?}"
