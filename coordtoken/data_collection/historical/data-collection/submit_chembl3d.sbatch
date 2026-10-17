#!/bin/bash
#SBATCH --job-name=chembl3d_csv
#SBATCH --output=/auto/home/filya/fsq/data-collection/logs/chembl3d_%j.out
#SBATCH --error=/auto/home/filya/fsq/data-collection/logs/chembl3d_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=h100

echo "Processing ChEMBL3D dataset"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Source dir: ${CHEMBL_SOURCE_DIR:-/nfs/h100/raid/chem/chembl3d/processed}"
echo "Output dir: ${CHEMBL_OUTPUT_DIR:-/nfs/dgx/raid/chem/TokenizerData}"
echo "Splits: ${CHEMBL_SPLITS:-train,val,test}"

cd /auto/home/filya/fsq/data-collection
python -u csv_chembl3d.py

echo "Finished processing ChEMBL3D"
