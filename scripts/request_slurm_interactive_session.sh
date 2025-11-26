#!/usr/bin/env bash
#SBATCH --job-name=debugging
#SBATCH --cpus-per-task=16
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=02:00:00


srun \
  --job-name=debugging \
  --partition=a100 \
  --nodes=1 \
  --cpus-per-task=16 \
  --gres=gpu:4 \
  --mem=32G \
  --time=24:00:00 \
  --pty bash -l