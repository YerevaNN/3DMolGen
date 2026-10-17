#!/bin/bash
#SBATCH --job-name=merge_train_new
#SBATCH --output=/auto/home/filya/fsq/logs/merge_train_new_%j.out
#SBATCH --error=/auto/home/filya/fsq/logs/merge_train_new_%j.err
#SBATCH --partition=a100
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0

set -euo pipefail

source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate titan
fi

python -u /auto/home/filya/fsq/merge_train_only_streaming.py \
  --root /nfs/h100/raid/chem/3D_big_data_new_dedup_no_geom_revisited \
  --output-csv /nfs/h100/raid/chem/3D_big_data_new_dedup_no_geom_revisited/merged_train.csv
