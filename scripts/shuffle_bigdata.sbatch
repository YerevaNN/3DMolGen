#!/bin/bash
#SBATCH --job-name=shuffle_bigdata
#SBATCH --cpus-per-task=32
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/vtarasov/slurm_jobs/shuffle_%j.out
#SBATCH --error=/home/vtarasov/slurm_jobs/shuffle_%j.err

set -euo pipefail

INPUT_DIR=/mnt/weka/vtarasov/3DBigData_formatted/processed/bigdata_uniform_binned_isomeric/processed_strings/train
OUTPUT_DIR=/mnt/weka/vtarasov/3DBigData_formatted/processed/bigdata_uniform_binned_isomeric/processed_strings_shuffled/train
BUCKET_DIR=/mnt/weka/vtarasov/tmp/shuffle_buckets

mkdir -p "$(dirname "$SLURM_ERROR_FILE")" 2>/dev/null || true
mkdir -p "$OUTPUT_DIR"
mkdir -p "$BUCKET_DIR"

echo "Starting shuffle at $(date)"
echo "Input:   $INPUT_DIR"
echo "Output:  $OUTPUT_DIR"
echo "Buckets: $BUCKET_DIR"
echo "CPUs:    $SLURM_CPUS_PER_TASK"

python3 /home/vtarasov/code/3DMolGen/scripts/shuffle_jsonl_dataset.py \
    "$INPUT_DIR" \
    "$OUTPUT_DIR" \
    --num-buckets 200 \
    --workers "$SLURM_CPUS_PER_TASK" \
    --bucket-dir "$BUCKET_DIR" \
    --seed 42 \
    --cleanup

echo "Shuffle complete at $(date)"
