#!/bin/bash
#SBATCH --job-name=merge_pubchem_zinc_all
#SBATCH --output=/auto/home/filya/fsq/data-collection/logs/merge_pubchem_zinc_all_%j.out
#SBATCH --error=/auto/home/filya/fsq/data-collection/logs/merge_pubchem_zinc_all_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --partition=h100

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Start: $(date -Is)"

source ~/miniforge/etc/profile.d/conda.sh
conda activate nab

cd /auto/home/filya/fsq/data-collection
mkdir -p /auto/home/filya/fsq/data-collection/logs

python -u merge_pubchem_zinc_csvs.py \
  --data-root /nfs/dgx/raid/chem/TokenizerData \
  --backup-dir /nfs/dgx/raid/chem/TokenizerData/backup \
  --splits val test \
  --datasets pubchem zinc

echo "End: $(date -Is)"
