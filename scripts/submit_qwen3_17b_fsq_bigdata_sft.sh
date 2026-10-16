#!/usr/bin/env bash
# Submit the Qwen3 1.7B FSQ SFT on top of the big-data pretrained checkpoint.
#
# Mirrors the 1.7B conformer-SFT footprint (1 node x 8 GPUs). The 3-node setup
# was only needed for the big-data pretraining; the conformer SFT runs on a
# single node, same as the 3DMolGen 1.7B SFT.
#
# Checkpointing uses async_mode="disabled" (sync save) because the 1.7B sits at
# ~95% GPU memory at local_batch_size=6/seq_len=4096 -- async staging OOMs there
# (verified by smoke test). Sync save writes after the step, so it fits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_TOML=${TRAIN_TOML:-${REPO_ROOT}/src/molgen3D/config/pretrain/qwen3_17b_fsq_bigdata_sft.toml}

if [[ ! -f "${TRAIN_TOML}" ]]; then
  echo "Missing TRAIN_TOML: ${TRAIN_TOML}" >&2
  exit 1
fi

# Resource footprint copied from the 1.7B SFT launch (1 node, 8 GPUs).
SBATCH_ARGS=(
  --job-name=qwen17b-fsq-sft
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

echo "Submitting 1.7B FSQ SFT job"
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  TRAIN_TOML=${TRAIN_TOML}"
echo "  nodes=1 gres=gpu:8 cpus-per-task=64 mem=200G  (async_mode=disabled)"

cd "${REPO_ROOT}"
exec sbatch "${SBATCH_ARGS[@]}"
