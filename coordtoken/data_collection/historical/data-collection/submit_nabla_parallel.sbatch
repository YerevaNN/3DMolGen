#!/bin/bash
#SBATCH --job-name=nabla_csv_parallel
#SBATCH --output=/auto/home/filya/fsq/data-collection/logs/nabla_parallel_%j.out
#SBATCH --error=/auto/home/filya/fsq/data-collection/logs/nabla_parallel_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=41
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --partition=h100

echo "Processing NablaDFT dataset with 41 parallel processes"
echo "Job ID: ${SLURM_JOB_ID}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK}"

cd /auto/home/filya/fsq/data-collection
python -u csv_nabla_parallel.py

echo "Finished processing NablaDFT"
