#!/usr/bin/env bash
#SBATCH --job-name=bgfsq-fanout
#SBATCH --partition=premium
#SBATCH --exclude=gpu09
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_geom_fsq/logs/%x-%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_geom_fsq/logs/%x-%j.err

# The number of example-build array tasks equals the number of extracted shards,
# which is only known after extraction has run. This job reads it and submits the
# arrays itself. It is a real script, not an sbatch --wrap string, because --wrap
# runs under /bin/sh and common.sh needs bash.
REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
source "${REPO_ROOT}/bindgpt_geom_fsq/scripts/common.sh"
capture_environment fanout

EXPORTS=${EXPORTS:?submit_all.sh must pass EXPORTS}

for split in valid train; do
  shards=$(find "${DATA_ROOT}/extracted/${split}" -maxdepth 1 -name '*.jsonl' | wc -l)
  if [[ "${shards}" -lt 1 ]]; then
    echo "ERROR: no extracted shards for ${split} under ${DATA_ROOT}/extracted/${split}" >&2
    exit 2
  fi
  echo "${split}: ${shards} shards"
  job=$(sbatch --parsable \
    --partition="${PARTITION}" --exclude="${EXCLUDE_NODES}" \
    --array="0-$((shards - 1))" \
    --export="${EXPORTS},SPLIT=${split}" \
    "${REPO_ROOT}/bindgpt_geom_fsq/scripts/04_build_examples.sbatch")
  echo "${split} example array: ${job}"
done
