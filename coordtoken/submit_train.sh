#!/bin/bash
#SBATCH --job-name=fsq_train
#SBATCH --output=logs/training/fsq_train_%j.out
#SBATCH --error=logs/training/fsq_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gpus=8
#SBATCH --partition=research
#SBATCH --mem=512G

set -e

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Starting FSQ training"
echo "============================================"

LAUNCH_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export FSQ_RUNS_DIR="$LAUNCH_DIR/logs"
mkdir -p "$FSQ_RUNS_DIR"

source /home/fgeikyan/tokenizer/conda_envs/train/bin/activate

if [ ! -d "/home/fgeikyan/tokenizer/.aim" ]; then
    echo "Initializing Aim repository at /home/fgeikyan/tokenizer/.aim"
    python - <<'PY'
from aim.sdk.repo import Repo
Repo("/home/fgeikyan/tokenizer", init=True)
PY
fi

# DATA can be set via environment variable: "full_5pct", "full_20pct", "full", or "nablaDFT"
DATA="${DATA:-full}"
LEVELS_SIZE="${LEVELS_SIZE:-12}"
D_MODEL="${D_MODEL:-1024}"
BATCH_BASE="${BATCH_BASE:-128}"
EPOCHS="${EPOCHS:-1}"
CONTINUE_TO_EPOCHS="${CONTINUE_TO_EPOCHS:-}"
LR="${LR:-1e-4}"
SEED="${SEED:-123}"
RUN_NAME="${RUN_NAME:-}"
CKPT_EVERY_PERCENT="${CKPT_EVERY_PERCENT:-}"
STOP_AFTER_EPOCH_FRACTION="${STOP_AFTER_EPOCH_FRACTION:-}"
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"

if [ -n "$CONTINUE_TO_EPOCHS" ]; then
    EPOCHS="$CONTINUE_TO_EPOCHS"
fi

if [ -n "$RESUME_CKPT_PATH" ] && [ ! -f "$RESUME_CKPT_PATH" ]; then
    echo "Error: RESUME_CKPT_PATH not found: $RESUME_CKPT_PATH"
    exit 1
fi
USE_1GPU="${USE_1GPU:-false}"
LIGHTNING_PROFILER="${LIGHTNING_PROFILER:-none}"
ATOMS_TO_DECODER="${ATOMS_TO_DECODER:-true}"
ATOMS_TO_ENCODER="${ATOMS_TO_ENCODER:-true}"
NABLA_SOURCE_DATA="/mnt/weka/fgeikyan/fsq/shuffle_index_nabla"

if [ "$DATA" = "nablaDFT" ]; then
    SOURCE_DATA="$NABLA_SOURCE_DATA"
    TARGET_DATA="$NABLA_SOURCE_DATA"
elif [ "$DATA" = "full_5pct" ]; then
    SOURCE_DATA="/mnt/weka/fgeikyan/fsq/shuffle_index_merged_train_5pct"
    TARGET_DATA="/dev/shm/shuffle_index_merged_train_5pct"
elif [ "$DATA" = "full_20pct" ]; then
    SOURCE_DATA="/mnt/weka/fgeikyan/fsq/shuffle_index_merged_train"
    TARGET_DATA="/dev/shm/shuffle_index_merged_train"
    if [ -z "$STOP_AFTER_EPOCH_FRACTION" ]; then
        STOP_AFTER_EPOCH_FRACTION=0.2
    fi
elif [ "$DATA" = "full" ]; then
    SOURCE_DATA="/mnt/weka/fgeikyan/fsq/shuffle_index_merged_train"
    TARGET_DATA="/dev/shm/shuffle_index_merged_train"
else
    echo "Error: Invalid DATA value: $DATA. Must be 'full_5pct', 'full_20pct', 'full', or 'nablaDFT'"
    exit 1
fi

if [ -z "$CKPT_EVERY_PERCENT" ]; then
    if [ "$DATA" = "full" ]; then
        CKPT_EVERY_PERCENT=10
    else
        CKPT_EVERY_PERCENT=100
    fi
fi

echo "Launch dir: $LAUNCH_DIR"
echo "Using dataset: $DATA"
echo "LEVELS_SIZE: $LEVELS_SIZE"
echo "D_MODEL: $D_MODEL"
echo "BATCH_BASE: $BATCH_BASE"
echo "EPOCHS: $EPOCHS"
echo "LR: $LR"
echo "SEED: $SEED"
echo "RUN_NAME: ${RUN_NAME:-<auto>}"
echo "CKPT_EVERY_PERCENT: $CKPT_EVERY_PERCENT"
echo "STOP_AFTER_EPOCH_FRACTION: ${STOP_AFTER_EPOCH_FRACTION:-<none>}"
echo "RESUME_CKPT_PATH: ${RESUME_CKPT_PATH:-<none>}"
echo "USE_1GPU: $USE_1GPU"
echo "LIGHTNING_PROFILER: $LIGHTNING_PROFILER"
echo "ATOMS_TO_DECODER: $ATOMS_TO_DECODER"
echo "ATOMS_TO_ENCODER: $ATOMS_TO_ENCODER"
if [ ! -d "$SOURCE_DATA" ]; then
    echo "Warning: Source directory $SOURCE_DATA does not exist. Skipping copy."
elif [ "$SOURCE_DATA" = "$TARGET_DATA" ]; then
    echo "Using training data in place at $SOURCE_DATA"
else
    source_bytes="$(du -sb "$SOURCE_DATA" | awk '{print $1}')"
    shm_avail_bytes="$(df -B1 --output=avail /dev/shm | tail -n 1 | tr -d ' ')"
    echo "SOURCE_DATA bytes: $source_bytes"
    echo "/dev/shm available bytes: $shm_avail_bytes"
    if [ "$source_bytes" -gt "$shm_avail_bytes" ]; then
        echo "Error: not enough space in /dev/shm for $SOURCE_DATA"
        exit 1
    fi
    echo "Copying training data from $SOURCE_DATA to $TARGET_DATA..."
    mkdir -p /dev/shm
    rsync -av --progress "$SOURCE_DATA/" "$TARGET_DATA/"
    echo "Data copy completed."
fi

export USE_1GPU
export CKPT_EVERY_PERCENT
export STOP_AFTER_EPOCH_FRACTION
export LIGHTNING_PROFILER
export ATOMS_TO_DECODER
export ATOMS_TO_ENCODER

submit_args=(
    --data "$DATA"
    --levels-size "$LEVELS_SIZE"
    --d-model "$D_MODEL"
    --batch-base "$BATCH_BASE"
    --epochs "$EPOCHS"
    --lr "$LR"
    --seed "$SEED"
)

if [ -n "$RESUME_CKPT_PATH" ]; then
    submit_args+=(--resume-ckpt "$RESUME_CKPT_PATH")
fi

if [ -n "$RUN_NAME" ]; then
    submit_args+=(--run-name "$RUN_NAME")
fi

python submit_train.py \
    "${submit_args[@]}"

echo "Done!"
