#!/bin/bash
#SBATCH --job-name=torchtitan-qwen3
#SBATCH --cpus-per-task=64
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=~/slurm_jobs/titan/%j.out
#SBATCH --error=~/slurm_jobs/titan/%j.err

export WANDB_ENTITY=${WANDB_ENTITY:-menuab_team}
export WANDB_PROJECT=${WANDB_PROJECT:-3dmolgen}
export WANDB_GROUP=${WANDB_GROUP:-pretrain}
export WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-pretrain}
export WANDB_CONFIG=${WANDB_CONFIG:-'{"run_type": "pretrain"}'}
export TORCH_COMPILE=0

TRAIN_TOML=${TRAIN_TOML:-src/molgen3D/config/pretrain/qwen3_06b.toml}
DEFAULT_RUN_DESC=$(basename "${TRAIN_TOML}" .toml)
RUN_DESC=${RUN_DESC:-${DEFAULT_RUN_DESC}}

MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
MASTER_PORT=${MASTER_PORT:-$(( (RANDOM % 20000) + 20000 ))}
NGPU_PER_NODE=${NGPU_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}
NNODES=${SLURM_NNODES:-1}
NODE_RANK=${SLURM_NODEID:-0}

export MASTER_ADDR MASTER_PORT

exec torchrun \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node="${NGPU_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    -m molgen3D.training.pretraining.torchtitan_runner \
    --run-desc "${RUN_DESC}" \
    --train-toml "${TRAIN_TOML}"
