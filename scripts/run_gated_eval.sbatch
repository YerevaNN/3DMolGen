#!/usr/bin/env bash
# COV/MAT for one iterative-gating round. Off the critical path.
#SBATCH --job-name=gated-eval
#SBATCH --partition=premium
#SBATCH --qos=researcher
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/baseline_refine/ge_%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/baseline_refine/ge_%j.err
set -euo pipefail
cd /mnt/weka/fgeikyan/3dmolgen_bigdata
export PYTHONPATH=src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python -m molgen3D.evaluation.baseline_tokenizer_refinement \
  --output-dir outputs/baseline_refinement/${METHOD:?}_g${ROUND:?} --method "${METHOD}" \
  --fsq-checkpoint /mnt/weka/fgeikyan/fsq/new_checkpoints/full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt \
  --phase evaluate --num-workers 16
