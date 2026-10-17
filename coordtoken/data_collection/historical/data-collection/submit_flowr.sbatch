#!/bin/bash
#SBATCH --job-name=flowr_csv
#SBATCH --output=/auto/home/filya/fsq/data-collection/logs/flowr_%A_%a.out
#SBATCH --error=/auto/home/filya/fsq/data-collection/logs/flowr_%A_%a.err
#SBATCH --array=0-13
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=h100

DATASETS=(
    "OMol25_small_mols"
    "KIBA-3D"
    "OMol25_bio_mols"
    "SPINDR"
    "Plinder"
    "BindingNet-Mid"
    "BindingNet-Low"
    "DAVIS-3D"
    "BindingNet-High"
    "CrossDocked2020"
    "BindingMoad"
    "HiQBind"
    "SAIR"
    "Kinodata-3D"
)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Processing dataset: ${DATASET}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: ${SLURM_JOB_ID}"

cd /auto/home/filya/fsq/data-collection
python csv_flowr_single_dataset.py ${DATASET}

echo "Finished processing ${DATASET}"
