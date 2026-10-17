#!/bin/bash
#SBATCH --job-name=split_grp_b_merged
#SBATCH --output=/auto/home/filya/fsq/logs/split_grp_b_merged_%j.out
#SBATCH --error=/auto/home/filya/fsq/logs/split_grp_b_merged_%j.err
#SBATCH --partition=h100
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0

set -euo pipefail

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titan

python -u /auto/home/filya/fsq/split_group_b_merged.py \
  --base-root /nfs/h100/raid/chem/3D_big_data_new \
  --group-b-root /nfs/h100/raid/chem/3D_big_data/grp_b \
  --output-root /nfs/h100/raid/chem/3D_big_data_new
