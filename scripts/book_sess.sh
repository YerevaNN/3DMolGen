#!/usr/bin/env bash

srun \
  --job-name=debugging \
  --mail-type=ALL \
  --partition=a100 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=20 \
  --mem=64G \
  --time=48:00:00 \
  --gres=gpu:2 \
  --pty bash