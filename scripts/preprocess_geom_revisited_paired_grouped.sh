#!/bin/bash
# Usage:
#   sbatch --partition=research --job-name=preprocess-revisited-grouped \
#          --cpus-per-task=128 --mem=256G --time=12:00:00 \
#          scripts/preprocess_geom_revisited_paired_grouped.sh
#
# This script runs grouped revisited preprocessing across three encodings:
# cartesian, quantile_binned, and uniform_binned.

set -euo pipefail

GEOM_RAW=${GEOM_RAW:-/data/vtarasov/geom_revisited}
DEST=${DEST:-/data/vtarasov}
BIN_CONFIGS=${BIN_CONFIGS:-/home/vtarasov/code/3DMolGen/src/molgen3D/config/bin_configs}
REPO_ROOT=${REPO_ROOT:-/home/vtarasov/code/3DMolGen}
VENV_PATH=${VENV_PATH:-$REPO_ROOT/.venv/bin/activate}

# Cap at 32: more workers multiply CoW memory without improving throughput.
_RAW_CPUS=${SLURM_CPUS_PER_TASK:-32}
WORKERS=$(( _RAW_CPUS > 32 ? 32 : _RAW_CPUS ))

echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "CPUs:      $WORKERS"
echo "Started:   $(date)"
echo "GEOM_RAW:  $GEOM_RAW"
echo "DEST:      $DEST"
echo ""

source "$VENV_PATH"
cd "$REPO_ROOT"

run_job() {
    local step="$1"
    local total="$2"
    local label="$3"
    local module="$4"
    local run_name="$5"
    local embedding_type="$6"
    local bin_config="${7:-}"

    echo "=========================================="
    echo "  ${step}/${total}  ${label}"
    echo "=========================================="

    if [[ -n "$bin_config" ]]; then
        python -m "$module" \
            --geom_raw_path "$GEOM_RAW" \
            --dest "$DEST" \
            --run_name "$run_name" \
            --embedding_type "$embedding_type" \
            --bin_config_path "$bin_config" \
            --num_workers "$WORKERS" \
            --isomeric
    else
        python -m "$module" \
            --geom_raw_path "$GEOM_RAW" \
            --dest "$DEST" \
            --run_name "$run_name" \
            --embedding_type "$embedding_type" \
            --num_workers "$WORKERS" \
            --isomeric
    fi
}


run_job 1 3 \
    "grouped cartesian" \
    "molgen3D.data_processing.preprocess_geom_grouped_revisited" \
    "geom_revisited_cartesian_isomeric_grouped" \
    "cartesian"

run_job 2 3 \
    "grouped quantile_binned" \
    "molgen3D.data_processing.preprocess_geom_grouped_revisited" \
    "geom_revisited_quantile_binned_isomeric_grouped" \
    "quantile_binned" \
    "$BIN_CONFIGS/quantile_bins.json"

run_job 3 3 \
    "grouped uniform_binned" \
    "molgen3D.data_processing.preprocess_geom_grouped_revisited" \
    "geom_revisited_uniform_binned_isomeric_grouped" \
    "uniform_binned" \
    "$BIN_CONFIGS/uniform_bins.json"

echo "=========================================="
echo "  All 3 grouped runs complete."
echo "  Finished: $(date)"
echo "  Output dirs:"
echo "    $DEST/geom_revisited_cartesian_isomeric_grouped"
echo "    $DEST/geom_revisited_quantile_binned_isomeric_grouped"
echo "    $DEST/geom_revisited_uniform_binned_isomeric_grouped"
echo "=========================================="
