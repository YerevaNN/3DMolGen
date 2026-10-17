#!/bin/bash
#SBATCH --job-name=pubchem_fast
#SBATCH --output=/auto/home/filya/fsq/data-collection/logs/pubchem_fast_%j.out
#SBATCH --error=/auto/home/filya/fsq/data-collection/logs/pubchem_fast_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=51
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --partition=h100

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Start: $(date -Is)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source ~/miniforge/etc/profile.d/conda.sh
conda activate nab

cd /auto/home/filya/fsq/data-collection
mkdir -p /auto/home/filya/fsq/data-collection/logs

python -u csv_pubchem3d_fast.py \
  --shard-dir /nfs/dgx/raid/chem/pubchem_shards \
  --out-dir  /nfs/dgx/raid/chem/PubChem3DTokenizerData \
  --seed 42 \
  --val-size 500000 \
  --test-size 1000000 \
  --n-workers 50 \
  --precision 4 \
  --batch-size 2000

echo "End: $(date -Is)"
