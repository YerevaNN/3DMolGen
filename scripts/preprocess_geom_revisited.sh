#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

GEOM_RAW=/data/molgen/rdkit_folder
DEST=/data/molgen
INDICES=/data/molgen/splits/splits/split0.npy
BIN_CONFIGS="$PROJECT_ROOT/src/molgen3D/config/bin_configs"
WORKERS=80

cd "$PROJECT_ROOT"
source .venv/bin/activate

echo "=========================================="
echo "  1/3  cartesian_v2 (raw)"
echo "=========================================="
python -m molgen3D.data_processing.data_preprocessing \
    --geom_raw_path "$GEOM_RAW" \
    --dest          "$DEST" \
    --run_name      geom_revisited_cartesian_isomeric \
    --embedding_type cartesian_v2 \
    --indices_path  "$INDICES" \
    --num_workers   "$WORKERS" \
    --use_isomeric_smiles

echo "=========================================="
echo "  2/3  quantile_binned"
echo "=========================================="
python -m molgen3D.data_processing.data_preprocessing \
    --geom_raw_path  "$GEOM_RAW" \
    --dest           "$DEST" \
    --run_name       geom_revisited_quantile_binned_isomeric \
    --embedding_type quantile_binned \
    --bin_config_path "$BIN_CONFIGS/quantile_bins.json" \
    --indices_path   "$INDICES" \
    --num_workers    "$WORKERS" \
    --use_isomeric_smiles

echo "=========================================="
echo "  3/3  uniform_binned"
echo "=========================================="
python -m molgen3D.data_processing.data_preprocessing \
    --geom_raw_path  "$GEOM_RAW" \
    --dest           "$DEST" \
    --run_name       geom_revisited_uniform_binned_isomeric \
    --embedding_type uniform_binned \
    --bin_config_path "$BIN_CONFIGS/uniform_bins.json" \
    --indices_path   "$INDICES" \
    --num_workers    "$WORKERS" \
    --use_isomeric_smiles

echo "=========================================="
echo "  All 3 runs complete."
echo "  Output dirs:"
echo "    $DEST/geom_revisited_cartesian"
echo "    $DEST/geom_revisited_quantile_binned"
echo "    $DEST/geom_revisited_uniform_binned"
echo "=========================================="
