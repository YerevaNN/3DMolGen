#!/bin/bash
#SBATCH --partition=research8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
# INPUT OUTPUT SPLIT TOKENIZER_DIR NUM_SHARDS; array range is 0..NUM_SHARDS-1.
if [[ $# -ne 5 ]]; then echo "Usage: INPUT OUTPUT SPLIT TOKENIZER_DIR NUM_SHARDS" >&2; exit 2; fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "${PYTHON:-python}" -m molgen3D.data_processing.preprocess_coordtoken_csv \
 --input "$1" --output "$2" --split "$3" --tokenizer "$4" \
 --num-shards "$5" --shard-index "${SLURM_ARRAY_TASK_ID:-0}" --device cuda --precision bf16
