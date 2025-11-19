#!/bin/bash
#SBATCH --job-name=torchtitan-qwen3
#SBATCH --cpus-per-task=16
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=~/slurm_jobs/titan/%j.out
#SBATCH --error=~/slurm_jobs/titan/%j.err

export WANDB_ENTITY=${WANDB_ENTITY:-menuab_team}
export WANDB_PROJECT=${WANDB_PROJECT:-3dmolgen}
export WANDB_GROUP=${WANDB_GROUP:-pretrain}
export WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-pretrain}
export WANDB_CONFIG=${WANDB_CONFIG:-'{"run_type": "pretrain"}'}

exec torchrun \
    --nproc_per_node=4 \
    -m torchtitan.train \
    --job.config_file src/molgen3D/training/pretraining/config/qwen3_06b_custom.toml