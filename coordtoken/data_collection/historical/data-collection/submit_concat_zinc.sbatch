#!/bin/bash
#SBATCH --job-name=concat_zinc
#SBATCH --output=/auto/home/filya/fsq/data-collection/logs/concat_zinc_%j.out
#SBATCH --error=/auto/home/filya/fsq/data-collection/logs/concat_zinc_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --partition=h100

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Start: $(date -Is)"

source ~/miniforge/etc/profile.d/conda.sh
conda activate nab

cd /auto/home/filya/fsq/data-collection

IN_BASE=/nfs/dgx/raid/chem/TokenizerDataShuffled
OUT_BASE=/nfs/dgx/raid/chem/TokenizerData

for split in train val test; do
    echo "Concatenating ${split}..."
    mkdir -p "${OUT_BASE}/${split}"
    python -u concat_shards.py \
        --in-dir "${IN_BASE}/${split}" \
        --out-file "${OUT_BASE}/${split}/zinc.csv" \
        --seed 42 \
        --pattern "zinc_w*.csv"
done

echo "Done: $(date -Is)"
