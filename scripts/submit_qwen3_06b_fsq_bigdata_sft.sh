#!/usr/bin/env bash
# Submit the Qwen3 0.6B FSQ SFT on top of the big-data pretrained checkpoint.
#
# This reuses the existing torchtitan launch script but mirrors the 1.7B SFT
# resource footprint (1 node x 8 GPUs) instead of the 3-node big-data pretrain
# topology. The 3-node setup was only needed for the big-data pretraining; the
# conformer SFT matches how the 1.7B SFT was run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_TOML=${TRAIN_TOML:-${REPO_ROOT}/src/molgen3D/config/pretrain/qwen3_06b_fsq_bigdata_sft.toml}

if [[ ! -f "${TRAIN_TOML}" ]]; then
  echo "Missing TRAIN_TOML: ${TRAIN_TOML}" >&2
  exit 1
fi

# Resource footprint copied from the 1.7B SFT launch (1 node, 8 GPUs).
SBATCH_ARGS=(
  --job-name=qwen06b-fsq-sft
  --nodes=1
  --ntasks-per-node=1
  --gres=gpu:8
  --cpus-per-task=64
  --mem=200G
  --time=1-00:00:00
  --export=ALL,TRAIN_TOML="${TRAIN_TOML}"
)

# Allow extra sbatch args (e.g. --nodelist, --exclude) to be appended.
if [[ $# -gt 0 ]]; then
  SBATCH_ARGS+=("$@")
fi

SBATCH_ARGS+=("${SCRIPT_DIR}/launch_torchtitan_qwen3.sh")

echo "Submitting 0.6B FSQ SFT job"
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  TRAIN_TOML=${TRAIN_TOML}"
echo "  nodes=1 gres=gpu:8 cpus-per-task=64 mem=200G"

cd "${REPO_ROOT}"
exec sbatch "${SBATCH_ARGS[@]}"
