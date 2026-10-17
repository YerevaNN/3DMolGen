#!/bin/bash
#SBATCH --job-name=shuffle_pubchem
#SBATCH --output=/auto/home/filya/fsq/data-collection/logs/shuffle_pubchem_%j.out
#SBATCH --error=/auto/home/filya/fsq/data-collection/logs/shuffle_pubchem_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=600G
#SBATCH --time=12:00:00
#SBATCH --partition=h100

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Start: $(date -Is)"

source ~/miniforge/etc/profile.d/conda.sh
conda activate nab

cd /auto/home/filya/fsq/data-collection

IN_BASE=/nfs/dgx/raid/chem/PubChem3DTokenizerData
OUT_BASE=/nfs/dgx/raid/chem/PubChem3DTokenizerDataShuffled

for split in train val test; do
    echo "Shuffling ${split}..."
    python -u shuffle_shards.py \
        --in-dir "${IN_BASE}/${split}" \
        --out-dir "${OUT_BASE}/${split}" \
        --seed 42 \
        --pattern "pubchem_w*.csv"
done

echo "Done: $(date -Is)"
