#!/bin/bash
#SBATCH --job-name=add_chembl_group_a
#SBATCH --output=add_chembl_group_a_%j.out
#SBATCH --error=add_chembl_group_a_%j.err
#SBATCH --partition=h100
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:0

set -euo pipefail

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titan

python /auto/home/filya/fsq/add_chembl_to_group_a.py \
  --base-root /nfs/h100/raid/chem/3D_big_data \
  --chembl-root /nfs/dgx/raid/chem/TokenizerData \
  --output /nfs/h100/raid/chem/3D_big_data_new \
  --workers "${SLURM_CPUS_PER_TASK}"
