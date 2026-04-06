#!/bin/bash
# Usage:
#   sbatch --partition=research --job-name=count-revisited-train-tokens \
#          --cpus-per-task=8 --mem=64G --time=24:00:00 \
#          scripts/count_revisited_train_tokens.sh
#
# Counts tokens on the revisited train splits currently defined in paths.yaml.
# Grouped datasets are counted with serialization_mode=isomer_units and
# non-grouped datasets with serialization_mode=pairs.
# Note: paths.yaml currently contains a duplicate
# `revisited_cartesian_isomeric_grouped_train` entry and does not define
# `revisited_cartesian_isomeric_train`, so this script targets the unique train
# aliases that are available today.
#
# Optional overrides:
#   REPO_ROOT=/path/to/3DMolGen
#   VENV_PATH=/path/to/.venv/bin/activate
#   OUTPUT_ROOT=/path/to/output-dir
#   SEQ_LEN=4096
#   SAMPLE_LINES=1000
#   BATCH_SIZE=4
#   UNIT_BATCH_SIZE=64
#   SAMPLE_UNITS=10000
#   SAMPLE_SAMPLES=1000
#   SAMPLE_LINES_FOR_UNITS=1000
#   SHUFFLE=true
#   GROUPED_ESTIMATE=exact   # exact | fast | sample_only
#
#SBATCH --output=/home/vtarasov/slurm_logs/token_counts/revisited-train-%j.out
#SBATCH --error=/home/vtarasov/slurm_logs/token_counts/revisited-train-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/vtarasov/code/3DMolGen}
VENV_PATH=${VENV_PATH:-$REPO_ROOT/.venv/bin/activate}
SEQ_LEN=${SEQ_LEN:-4096}
SAMPLE_LINES=${SAMPLE_LINES:-1000}
BATCH_SIZE=${BATCH_SIZE:-4}
UNIT_BATCH_SIZE=${UNIT_BATCH_SIZE:-64}
SAMPLE_UNITS=${SAMPLE_UNITS:-10000}
SAMPLE_SAMPLES=${SAMPLE_SAMPLES:-1000}
SAMPLE_LINES_FOR_UNITS=${SAMPLE_LINES_FOR_UNITS:-1000}
SHUFFLE=${SHUFFLE:-false}
GROUPED_ESTIMATE=${GROUPED_ESTIMATE:-exact}

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_TAG=${SLURM_JOB_ID:-local}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/outputs/token_counts/revisited_train/${JOB_TAG}-${TIMESTAMP}}

mkdir -p "$OUTPUT_ROOT"
mkdir -p /home/vtarasov/slurm_logs/token_counts

echo "Job ID:            ${SLURM_JOB_ID:-local}"
echo "Node:              $(hostname)"
echo "Started:           $(date)"
echo "Repo root:         $REPO_ROOT"
echo "Output root:       $OUTPUT_ROOT"
echo "Seq len:           $SEQ_LEN"
echo "Sample lines:      $SAMPLE_LINES"
echo "Batch size:        $BATCH_SIZE"
echo "Unit batch size:   $UNIT_BATCH_SIZE"
echo "Sample units:      $SAMPLE_UNITS"
echo "Sample samples:    $SAMPLE_SAMPLES"
echo "Sample unit lines: $SAMPLE_LINES_FOR_UNITS"
echo "Shuffle:           $SHUFFLE"
echo "Grouped estimate:  $GROUPED_ESTIMATE"
echo ""

source "$VENV_PATH"
cd "$REPO_ROOT"

build_grouped_extra_args() {
    case "$GROUPED_ESTIMATE" in
        exact)
            echo "--exact-estimate"
            ;;
        fast)
            echo "--fast-estimate --sample-units $SAMPLE_UNITS"
            ;;
        sample_only)
            echo "--sample-only --sample-samples $SAMPLE_SAMPLES --sample-lines-for-units $SAMPLE_LINES_FOR_UNITS"
            ;;
        *)
            echo "Unsupported GROUPED_ESTIMATE='$GROUPED_ESTIMATE' (expected: exact, fast, sample_only)" >&2
            exit 1
            ;;
    esac
}

run_count() {
    local dataset_alias="$1"
    local serialization_mode="$2"
    local tokenizer_alias="$3"
    local label="$4"
    local safe_name="${dataset_alias//\//_}"
    local log_path="$OUTPUT_ROOT/${safe_name}.log"
    local extra_args=()

    if [[ "$serialization_mode" == "isomer_units" ]]; then
        read -r -a extra_args <<< "$(build_grouped_extra_args)"
        extra_args+=(
            --unit-batch-size "$UNIT_BATCH_SIZE"
        )
    else
        extra_args+=(
            --sample-lines "$SAMPLE_LINES"
            --batch-size "$BATCH_SIZE"
        )
    fi

    if [[ "$SHUFFLE" == "true" ]]; then
        extra_args+=(--shuffle)
    fi

    echo "============================================================"
    echo "Dataset:            $dataset_alias"
    echo "Label:              $label"
    echo "Serialization mode: $serialization_mode"
    echo "Tokenizer:          $tokenizer_alias"
    echo "Log path:           $log_path"
    echo "Started:            $(date)"
    echo "============================================================"

    python -m molgen3D.training.pretraining.dataprocessing.count_tokens \
        --train-path "$dataset_alias" \
        --skip-validation \
        --seq-len "$SEQ_LEN" \
        --batch-size "$BATCH_SIZE" \
        --serialization-mode "$serialization_mode" \
        --tokenizers "$tokenizer_alias" \
        "${extra_args[@]}" | tee "$log_path"

    echo ""
}

run_count \
    "revisited_cartesian_isomeric_grouped_train" \
    "isomer_units" \
    "qwen3_0.6b_custom" \
    "grouped cartesian"

run_count \
    "revisited_quantile_binned_isomeric_grouped_train" \
    "isomer_units" \
    "qwen3_0.6b_binned_258" \
    "grouped quantile_binned"

run_count \
    "revisited_uniform_binned_isomeric_grouped_train" \
    "isomer_units" \
    "qwen3_0.6b_binned_258" \
    "grouped uniform_binned"

run_count \
    "revisited_quantile_binned_isomeric_train" \
    "pairs" \
    "qwen3_0.6b_binned_258" \
    "non-grouped quantile_binned"

run_count \
    "revisited_uniform_binned_isomeric_train" \
    "pairs" \
    "qwen3_0.6b_binned_258" \
    "non-grouped uniform_binned"

echo "Finished: $(date)"
echo "Logs written to: $OUTPUT_ROOT"
