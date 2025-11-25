#!/bin/bash
#SBATCH --job-name=torchtitan-basic
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=~/slurm_jobs/titan/test_run_%j.out
#SBATCH --error=~/slurm_jobs/titan/test_run_%j.err

export WANDB_ENTITY=${WANDB_ENTITY:-menuab_team}
export WANDB_PROJECT=${WANDB_PROJECT:-3dmolgen}
export WANDB_GROUP=${WANDB_GROUP:-basic_test}
export WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-test}

exec torchrun \
    --master_port=29800 \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    -m torchtitan.train \
    --job.config_file src/molgen3D/training/pretraining/config/test_run_config.toml

