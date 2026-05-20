#!/bin/bash
# Usage:
#   sbatch --partition=research --job-name=convert-big-data \
#          --cpus-per-task=200 --mem=128G --time=48:00:00 \
#          scripts/convert_big_data_format.sh
#
# Environment overrides:
#   REPO_ROOT, VENV_PATH, INPUT, OUTPUT
#   WORKERS       — phase-1 decode workers (default: cpus-per-task - 1)
#   CHUNK_SIZE    — CSV rows per worker task      (default: 500)
#   MAX_LINES     — rows to process, 0=all        (default: 0)
#   START_LINE    — skip N data rows for resume   (default: 0)
#   LOG_EVERY     — progress print interval       (default: 50000)
#   NUM_SHARDS    — output shard pkl count        (default: 16)
#   MERGE_WORKERS — phase-2 shards merged at once (default: 1)
#   STREAMS_DIR   — temp stream files location    (default: {output_dir}/.streams)
#   KEEP_STREAMS  — set to "--keep-streams" to retain stream files (default: "")
#   SORT_OUTPUT   — set to "--sort-output" to sort pkl keys        (default: "")
#
# Memory guide (200 M conformers, ~5 KB/mol):
#   NUM_SHARDS=16 MERGE_WORKERS=1  ->  ~50 GB peak  (--mem=64G)
#   NUM_SHARDS=8  MERGE_WORKERS=2  ->  ~100 GB peak (--mem=128G)
#   NUM_SHARDS=4  MERGE_WORKERS=4  ->  ~200 GB peak (--mem=256G)
#
#SBATCH --output=/home/vtarasov/slurm_logs/convert_big_data/%j.out
#SBATCH --error=/home/vtarasov/slurm_logs/convert_big_data/%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/vtarasov/code/3DMolGen}
VENV_PATH=${VENV_PATH:-$REPO_ROOT/.venv/bin/activate}
INPUT=${INPUT:-/mnt/weka/vtarasov/3DbigData/merged_train.csv}
OUTPUT=${OUTPUT:-/mnt/weka/vtarasov/3DBigData_formatted/grouped_conformers.pkl}
MAX_LINES=${MAX_LINES:-0}
START_LINE=${START_LINE:-0}
LOG_EVERY=${LOG_EVERY:-50000}
CHUNK_SIZE=${CHUNK_SIZE:-500}
NUM_SHARDS=${NUM_SHARDS:-16}
MERGE_WORKERS=${MERGE_WORKERS:-1}
STREAMS_DIR=${STREAMS_DIR:-""}
KEEP_STREAMS=${KEEP_STREAMS:-""}
SORT_OUTPUT=${SORT_OUTPUT:-""}

# Derive worker count from SLURM allocation; leave 1 CPU for main process
if [[ -z "${WORKERS:-}" ]]; then
    AVAIL_CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
    WORKERS=$(( AVAIL_CPUS > 1 ? AVAIL_CPUS - 1 : 1 ))
fi

mkdir -p /home/vtarasov/slurm_logs/convert_big_data

echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "Started:       $(date)"
echo "Input CSV:     $INPUT"
echo "Output:        $OUTPUT"
echo "Workers:       $WORKERS (decode)"
echo "Chunk size:    $CHUNK_SIZE"
echo "Num shards:    $NUM_SHARDS"
echo "Merge workers: $MERGE_WORKERS"
echo "Max lines:     $MAX_LINES"
echo "Start line:    $START_LINE"
echo ""

source "$VENV_PATH"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

STREAMS_ARG=""
[[ -n "$STREAMS_DIR" ]] && STREAMS_ARG="--streams-dir $STREAMS_DIR"

python -u scripts/convert_big_data_format.py \
    --input         "$INPUT"         \
    --output        "$OUTPUT"        \
    --workers       "$WORKERS"       \
    --chunk-size    "$CHUNK_SIZE"    \
    --num-shards    "$NUM_SHARDS"    \
    --merge-workers "$MERGE_WORKERS" \
    --max-lines     "$MAX_LINES"     \
    --start-line    "$START_LINE"    \
    --log-every     "$LOG_EVERY"     \
    ${STREAMS_ARG}                   \
    ${KEEP_STREAMS}                  \
    ${SORT_OUTPUT}
