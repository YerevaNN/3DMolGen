#!/bin/bash
#SBATCH --job-name=build_grp_d_geom_revisited
#SBATCH --output=/auto/home/filya/fsq/logs/build_grp_d_geom_revisited_%j.out
#SBATCH --error=/auto/home/filya/fsq/logs/build_grp_d_geom_revisited_%j.err
#SBATCH --partition=a100
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:0

set -euo pipefail

source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate titan
fi

python -u /auto/home/filya/fsq/build_group_d_geom_revisited_test.py \
  --revisited-pickle /auto/home/vover/revisited.pickle \
  --output-root /nfs/h100/raid/chem/3D_big_data/grp_d \
  --dataset-name geom_revisited_test \
  --summary-csv /nfs/h100/raid/chem/3D_big_data/grp_d/grp_d_summary.csv
