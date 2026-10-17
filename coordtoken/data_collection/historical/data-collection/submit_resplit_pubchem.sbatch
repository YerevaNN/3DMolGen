#!/bin/bash
#SBATCH --job-name=resplit_pubchem
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --output=logs1/resplit_pubchem_%j.out
#SBATCH --error=logs1/resplit_pubchem_%j.err

cd /auto/home/filya/fsq/data-collection
mkdir -p logs1

python resplit_pubchem.py --output-dir /nfs/dgx/raid/chem/TokenizerData_v2/pubchem_only
