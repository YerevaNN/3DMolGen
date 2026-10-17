#!/bin/bash
#SBATCH --job-name=remove_smiles
#SBATCH --output=remove_test_val_smiles_%j.out
#SBATCH --error=remove_test_val_smiles_%j.err
#SBATCH --partition=all
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titan

python /auto/home/filya/fsq/remove_test_val_smiles.py \
  --input /nfs/dgx/raid/chem/TokenizerData \
  --output /nfs/h100/raid/chem/3D_big_data \
  --workers 30
