#!/bin/bash
#SBATCH --job-name=preproc
#SBATCH --cpus-per-task=60
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/auto/home/menuab/slurm_jobs/preprocessing/torchdiff_preproc_%j.out
#SBATCH --error=/auto/home/menuab/slurm_jobs/preprocessing/torchdiff_preproc_%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /auto/home/menuab/code/3DMolGen

srun python3 molgen3D/data_processing/data_preprocessing.py \
  --geom_raw_path /nfs/ap/mnt/sxtn2/chem/GEOM_data/rdkit_folder \
  --dest /nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_processed \
  --indices_path /nfs/ap/mnt/sxtn2/chem/GEOM_data/splits/splits/split0.npy \
  --embedding_type cartesian_v2 \
  --num_workers ${SLURM_CPUS_PER_TASK} \
  --precision 4 \
  --dataset_type drugs \
  --run_name geom_mapped \
  --max_confs 30 \
  --splits test \



