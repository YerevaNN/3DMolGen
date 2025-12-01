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

export WANDB_ENTITY="${WANDB_ENTITY:-menuab_team}"
export WANDB_PROJECT="${WANDB_PROJECT:-3dmolgen}"
export WANDB_GROUP="${WANDB_GROUP:-basic_test}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-test}"

RUN_DESC="${RUN_DESC:-torchtitan-test}"
TRAIN_TOML=${TRAIN_TOML:-src/molgen3D/config/pretrain/qwen3_06b_wsds.toml}
MASTER_PORT=${MASTER_PORT:-$(( (RANDOM % 20000) + 20000 ))}
MASTER_ADDR=${MASTER_ADDR:-$(hostname)}

export MASTER_ADDR MASTER_PORT

exec torchrun \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    -m molgen3D.training.pretraining.torchtitan_runner \
    --run-desc "${RUN_DESC}" \
    --train-toml "${TRAIN_TOML}"
