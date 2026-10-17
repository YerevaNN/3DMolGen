#!/bin/bash
#SBATCH --job-name=chembl_iso_rm
#SBATCH --output=/auto/home/filya/fsq/data-collection/logs/chembl_iso_rm_%j.out
#SBATCH --error=/auto/home/filya/fsq/data-collection/logs/chembl_iso_rm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --partition=h100

echo "Removing isotopic ChEMBL rows"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Data dir: ${CHEMBL_DATA_DIR:-/nfs/dgx/raid/chem/TokenizerData}"

cd /auto/home/filya/fsq/data-collection
python -u remove_isotopes_chembl.py --data-dir "${CHEMBL_DATA_DIR:-/nfs/dgx/raid/chem/TokenizerData}"

echo "Finished removing isotopic ChEMBL rows"
