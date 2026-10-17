#!/bin/bash
#SBATCH --job-name=sample_dup_stats
#SBATCH --output=/auto/home/filya/fsq/logs/sample_dup_stats_%j.out
#SBATCH --error=/auto/home/filya/fsq/logs/sample_dup_stats_%j.err
#SBATCH --partition=h100
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0

set -euo pipefail

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titan

python -u /auto/home/filya/fsq/dedupe_samples_and_report_intersections.py \
  --phase stats \
  --source-root /nfs/h100/raid/chem/3D_big_data_new \
  --output-root /nfs/h100/raid/chem/3D_big_data_new_dedup \
  --summary-csv /auto/home/filya/fsq/3d_big_data_new_exact_duplicate_stats.csv \
  --intersections-csv /auto/home/filya/fsq/3d_big_data_new_exact_sample_intersections.csv \
  --sqlite-db /nfs/h100/raid/chem/3D_big_data_new_dedup_state/sample_duplicate_stats.sqlite \
  --drop-lists-root /nfs/h100/raid/chem/3D_big_data_new_dedup_state/drop_lists
