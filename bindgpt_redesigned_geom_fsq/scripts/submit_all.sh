#!/usr/bin/env bash
# Submit the whole job graph with Slurm dependencies, so no stage can start
# before its inputs exist. Nothing here encodes a scientific setting.
#
#   ./submit_all.sh                 # prepare data, manifests, tokenizer audit
#   START=examples ./submit_all.sh  # examples through training and evaluation
#   START=pack ./submit_all.sh      # reuse existing examples, then continue
#
# PARTITION defaults to premium (H100, 7-day walltime).
#
# gpu09 is excluded everywhere: it wedges CUDA jobs while still looking healthy.
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
HERE="${REPO_ROOT}/bindgpt_redesigned_geom_fsq/scripts"
source "${HERE}/common.sh"

START=${START:-prepare}
TRAIN_SEED=${TRAIN_SEED:-17}
SAMPLING_SEED=${SAMPLING_SEED:-2023}
NUM_SHARDS=${NUM_SHARDS:-40}
UPSTREAM_JOB_ID=${UPSTREAM_JOB_ID:-}
CLUSTER_NAME=${CLUSTER_NAME:-$(scontrol show config 2>/dev/null | awk -F= '/ClusterName/{gsub(/ /,"",$2); print $2}')}
mkdir -p "${BINDGPT_FSQ_ROOT}/logs"

# Every submission carries the partition and the node exclusion explicitly, so
# PARTITION=... on the command line actually overrides the #SBATCH defaults.
submit() { sbatch --parsable --partition="${PARTITION}" --exclude="${EXCLUDE_NODES}" "$@"; }
dep() { [[ -n "${1:-}" ]] && echo "--dependency=afterok:${1}" || echo ""; }

exports="ALL,REPO_ROOT=${REPO_ROOT},BINDGPT_FSQ_ROOT=${BINDGPT_FSQ_ROOT},DATA_ROOT=${DATA_ROOT}"
exports="${exports},TRAIN_SEED=${TRAIN_SEED},SAMPLING_SEED=${SAMPLING_SEED},CLUSTER_NAME=${CLUSTER_NAME}"
exports="${exports},PARTITION=${PARTITION},CPU_PARTITION=${CPU_PARTITION},EXCLUDE_NODES=${EXCLUDE_NODES}"

PREPARE="" ; MANIFESTS="" ; AUDIT="" ; PACK=""

if [[ "${START}" == "prepare" ]]; then
  PREPARE=$(submit --export="${exports}" "${HERE}/01_prepare.sbatch")
  echo "prepare            ${PREPARE}"
  MANIFESTS=$(submit $(dep "${PREPARE}") --export="${exports}" "${HERE}/02_manifests.sbatch")
  echo "manifests          ${MANIFESTS}"
  AUDIT=$(submit $(dep "${MANIFESTS}") --export="${exports}" "${HERE}/03_audit_tokenizers.sbatch")
  echo "tokenizer audit    ${AUDIT}"

  echo
  echo "Once manifests finish, continue from a login node with:"
  echo "  START=examples ${HERE}/submit_all.sh"
  echo "The shard counts only exist after extraction, and compute nodes do not"
  echo "provide sbatch, so the login-side continuation performs the fan-out."
  exit 0
fi

if [[ "${START}" == "examples" ]]; then
  EXAMPLE_IDS=()
  upstream_args=()
  if [[ -n "${UPSTREAM_JOB_ID}" ]]; then
    upstream_args=("--dependency=afterok:${UPSTREAM_JOB_ID}")
  fi
  for split in valid train; do
    shards=$(find "${DATA_ROOT}/extracted/${split}" -maxdepth 1 -name '*.jsonl' | wc -l)
    if [[ "${shards}" -lt 1 ]]; then
      echo "ERROR: no extracted shards for ${split} under ${DATA_ROOT}/extracted/${split}" >&2
      exit 2
    fi
    id=$(submit "${upstream_args[@]}" --array="0-$((shards - 1))" \
      --export="${exports},SPLIT=${split}" "${HERE}/04_build_examples.sbatch")
    EXAMPLE_IDS+=("${id}")
    echo "examples ${split}      ${id} (${shards} shards)"
  done
  examples_done=$(IFS=:; echo "${EXAMPLE_IDS[*]}")
  PACK=$(submit --dependency=afterok:"${examples_done}" --export="${exports}" \
    "${HERE}/05_pack.sbatch")
  echo "pack + configs     ${PACK}"
elif [[ "${START}" == "pack" ]]; then
  PACK=$(submit --export="${exports}" "${HERE}/05_pack.sbatch")
  echo "pack + configs     ${PACK}"
else
  echo "ERROR: START must be prepare, examples, or pack (got ${START})" >&2
  exit 2
fi

TRAIN_IDS=()
for arm in coord fsq; do
  id=$(submit $(dep "${PACK}") --export="${exports},ARM=${arm}" "${HERE}/06_train.sbatch")
  TRAIN_IDS+=("${id}")
  echo "train ${arm}         ${id}"
done

# Evaluation waits on both arms so the pair is always reported together.
both=$(IFS=:; echo "${TRAIN_IDS[*]}")
for arm in coord fsq; do
  gen_exports="${exports},ARM=${arm}"

  # 07 and 08 resolve the end-of-training checkpoint for this arm themselves.
  u=$(submit --dependency=afterok:"${both}" --export="${gen_exports}" \
      "${HERE}/07_generate_unconditional.sbatch")
  echo "uncond ${arm}        ${u}"

  c=$(submit --dependency=afterok:"${both}" \
      --export="${gen_exports},NUM_SHARDS=${NUM_SHARDS}" \
      --array=0-$((NUM_SHARDS - 1)) \
      "${HERE}/08_generate_conditional.sbatch")
  echo "cond ${arm}          ${c}"

  m=$(submit --dependency=afterok:"${u}":"${c}" --export="${gen_exports}" \
      "${HERE}/09_metrics.sbatch")
  echo "metrics ${arm}       ${m}"
done
