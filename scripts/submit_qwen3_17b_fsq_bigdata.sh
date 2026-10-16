#!/usr/bin/env bash
# Submit the Qwen3 1.7B FSQ big-data pretraining.
#
# Reuses the existing torchtitan launch script with its default big-data
# topology (3 nodes x 8 GPUs) -- the same setup the 0.6B big-data run used.
# Only the training TOML changes (1.7B flavor + 1.7B base checkpoint).
#
# The launch script auto-detects the training Python env (the one with
# torchtitan/tyro), so no environment changes are needed here.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_TOML=${TRAIN_TOML:-${REPO_ROOT}/src/molgen3D/config/pretrain/qwen3_17b_fsq_bigdata_from_base.toml}

if [[ ! -f "${TRAIN_TOML}" ]]; then
  echo "Missing TRAIN_TOML: ${TRAIN_TOML}" >&2
  exit 1
fi

# Use the launch script's built-in SBATCH directives (3 nodes x 8 GPUs).
SBATCH_ARGS=(
  --job-name=qwen17b-fsq-bigdata
  --export=ALL,TRAIN_TOML="${TRAIN_TOML}"
)

# Allow extra sbatch args (e.g. --nodelist, --exclude) to be appended.
if [[ $# -gt 0 ]]; then
  SBATCH_ARGS+=("$@")
fi

SBATCH_ARGS+=("${SCRIPT_DIR}/launch_torchtitan_qwen3.sh")

echo "Submitting 1.7B FSQ big-data pretrain job"
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  TRAIN_TOML=${TRAIN_TOML}"
echo "  topology: launch script default (3 nodes x 8 GPUs)"

cd "${REPO_ROOT}"
exec sbatch "${SBATCH_ARGS[@]}"
