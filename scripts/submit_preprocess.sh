#!/bin/bash
#SBATCH --job-name=preprocess_geom
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/auto/home/vover/3DMolGen/logs/preprocess_%j.out
#SBATCH --error=/auto/home/vover/3DMolGen/logs/preprocess_%j.err

mkdir -p /auto/home/vover/3DMolGen/logs
cd /auto/home/vover/3DMolGen

export OMP_NUM_THREADS=1   # avoid extra threading per worker

python src/molgen3D/data_processing/data_preprocessing.py \
    --embedding_type cartesian_binned_v2 \
    --num_workers 96 \
    --run_name binned_paired_stripped
