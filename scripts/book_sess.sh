#!/usr/bin/env bash
#SBATCH --job-name=debugging
#SBATCH --cpus-per-task=16
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=02:00:00


srun \
  --job-name=$SLURM_JOB_NAME \
  --mail-type=ALL \
  --partition=$SLURM_JOB_PARTITION \
  --nodes=$SLURM_JOB_NUM_NODES \
  --ntasks=$SLURM_NTASKS \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --mem=$SLURM_MEM_PER_NODE \
  --time=$SLURM_TIMELIMIT \
  --gres=$SLURM_JOB_GPUS \
  --pty bash