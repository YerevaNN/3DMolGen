#!/bin/bash
#SBATCH --job-name=rm_geom_revisited
#SBATCH --output=/auto/home/filya/fsq/logs/remove_geom_revisited_%j.out
#SBATCH --error=/auto/home/filya/fsq/logs/remove_geom_revisited_%j.err
#SBATCH --partition=a100

#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:0

set -euo pipefail

source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate titan
fi

python -u /auto/home/filya/fsq/remove_geom_revisited_test_smiles.py \
  --input-root /nfs/h100/raid/chem/3D_big_data_new_dedup \
  --reference-csv /nfs/h100/raid/chem/3D_big_data/grp_d/test/geom_revisited_test.csv \
  --output-root /nfs/h100/raid/chem/3D_big_data_new_dedup_no_geom_revisited \
  --summary-csv /nfs/h100/raid/chem/3D_big_data_new_dedup_no_geom_revisited/geom_revisited_smiles_removal_summary.csv \
  --workers "${SLURM_CPUS_PER_TASK}"
