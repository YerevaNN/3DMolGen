#!/bin/bash
#SBATCH --job-name=split_group_c
#SBATCH --output=/auto/home/filya/fsq/logs/split_group_c_%j.out
#SBATCH --error=/auto/home/filya/fsq/logs/split_group_c_%j.err
#SBATCH --partition=h100
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:0

set -euo pipefail

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titan

python -u /auto/home/filya/fsq/split_group_c.py \
  --reference-root /nfs/h100/raid/chem/3D_big_data_new \
  --group-c-root /nfs/h100/raid/chem/3D_big_data/grp_c \
  --output-root /nfs/h100/raid/chem/3D_big_data_new \
  --target-rows 63381
