#!/bin/bash
#SBATCH --job-name=resplit
#SBATCH --output=logs/resplit_%j.out
#SBATCH --error=logs/resplit_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=800G
#SBATCH --cpus-per-task=4
#SBATCH --partition=h100

set -euo pipefail

cd /auto/home/filya/fsq/data-collection
mkdir -p logs

BASE_DIR="/nfs/dgx/raid/chem/TokenizerData"

echo "=== Resplitting ZINC ==="
python resplit.py \
    --base-dir "${BASE_DIR}" \
    --out-dir "${BASE_DIR}" \
    --filename zinc.csv \
    --out-filename zinc1.csv \
    --val-size 500000 \
    --test-size 1000000 \
    --seed 42

echo ""
echo "=== Resplitting PubChem3D ==="
python resplit.py \
    --base-dir "${BASE_DIR}" \
    --out-dir "${BASE_DIR}" \
    --filename pubchem3d.csv \
    --out-filename pubchem3d1.csv \
    --val-size 500000 \
    --test-size 1000000 \
    --seed 42

echo ""
echo "=== Verifying output ==="
for split in train val test; do
    echo "${split}:"
    wc -l "${BASE_DIR}/${split}/zinc1.csv" "${BASE_DIR}/${split}/pubchem3d1.csv"
done

echo "Done!"
