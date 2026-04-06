#!/bin/bash
# Usage:
#   sbatch --partition=research --job-name=preprocess-revisited \
#          --cpus-per-task=32 --mem=256G --time=48:00:00 \
#          scripts/preprocess_geom_revisited.sh
#
# Memory note: the train split pickle is ~6.5 GB on disk and expands to
# ~50 GB in RAM.  multiprocessing.Pool forks the parent, so Python's
# refcount CoW quickly multiplies that by the worker count.  Cap workers
# at 32 to stay well within 256 GB; more workers give negligible extra
# throughput for this CPU-light, fork-heavy workload.
#SBATCH --output=/home/vtarasov/slurm_logs/preprocess/revisited-%j.out
#SBATCH --error=/home/vtarasov/slurm_logs/preprocess/revisited-%j.err

set -euo pipefail

GEOM_RAW=/data/vtarasov/geom_revisited
DEST=/data/vtarasov
BIN_CONFIGS="/home/vtarasov/code/3DMolGen/src/molgen3D/config/bin_configs"
# Cap at 32: more workers multiply CoW memory without improving throughput.
_RAW_CPUS=${SLURM_CPUS_PER_TASK:-32}
WORKERS=$(( _RAW_CPUS > 32 ? 32 : _RAW_CPUS ))

echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "CPUs:      $WORKERS"
echo "Started:   $(date)"
echo ""

source "/home/vtarasov/code/3DMolGen/.venv/bin/activate"

cd "/home/vtarasov/code/3DMolGen"

echo "=========================================="
echo "  1/3  cartesian_v2"
echo "=========================================="
python -m molgen3D.data_processing.data_preprocessing_revisited \
    --geom_raw_path  "$GEOM_RAW" \
    --dest           "$DEST" \
    --run_name       geom_revisited_cartesian_isomeric \
    --embedding_type cartesian_v2 \
    --num_workers    "$WORKERS" \
    --isomeric

echo "=========================================="
echo "  2/3  quantile_binned"
echo "=========================================="
python -m molgen3D.data_processing.data_preprocessing_revisited \
    --geom_raw_path   "$GEOM_RAW" \
    --dest            "$DEST" \
    --run_name        geom_revisited_quantile_binned_isomeric \
    --embedding_type  quantile_binned \
    --bin_config_path "$BIN_CONFIGS/quantile_bins.json" \
    --num_workers     "$WORKERS" \
    --isomeric

echo "=========================================="
echo "  3/3  uniform_binned"
echo "=========================================="
python -m molgen3D.data_processing.data_preprocessing_revisited \
    --geom_raw_path   "$GEOM_RAW" \
    --dest            "$DEST" \
    --run_name        geom_revisited_uniform_binned_isomeric \
    --embedding_type  uniform_binned \
    --bin_config_path "$BIN_CONFIGS/uniform_bins.json" \
    --num_workers     "$WORKERS" \
    --isomeric

echo "=========================================="
echo "  All 3 runs complete."
echo "  Finished: $(date)"
echo "  Output dirs:"
echo "    $DEST/geom_revisited_cartesian_isomeric"
echo "    $DEST/geom_revisited_quantile_binned_isomeric"
echo "    $DEST/geom_revisited_uniform_binned_isomeric"
echo "=========================================="
