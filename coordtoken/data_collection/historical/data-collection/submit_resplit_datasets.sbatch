#!/bin/bash
#SBATCH --job-name=resplit_chembl_kiba
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --partition=h100

python3 resplit_datasets.py --data-dir /nfs/dgx/raid/chem/TokenizerData
