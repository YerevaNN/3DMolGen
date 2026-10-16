#!/usr/bin/env bash
# Submit FSQ big-data preprocessing as a Slurm array.
#
# Usage:
#   scripts/submit_fsq_bigdata_preprocessing.sh
#
# Common overrides:
#   PARTITION=research FSQ_BATCH_SIZE=256 scripts/submit_fsq_bigdata_preprocessing.sh

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
INPUT_DIR=${INPUT_DIR:-/mnt/weka/vtarasov/3DBigData_formatted/grouped_conformers}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/outputs/slurm_jobs/preprocessing}
PARTITION=${PARTITION:-research}
MEM=${MEM:-512G}

mkdir -p "${LOG_DIR}"

shard_count=$(find "${INPUT_DIR}" -maxdepth 1 -name 'shard_*.pkl' | wc -l)
shard_count=${shard_count//[[:space:]]/}
if [[ -z "${shard_count}" || "${shard_count}" == "0" ]]; then
  echo "ERROR: no shard_*.pkl files found in ${INPUT_DIR}" >&2
  exit 2
fi

array_range="0-$((shard_count - 1))"

echo "Submitting FSQ preprocessing array"
echo "Input dir:  ${INPUT_DIR}"
echo "Shards:     ${shard_count} (${array_range})"
echo "Partition:  ${PARTITION}"
echo "Memory:     ${MEM}"
echo "Walltime:   3-00:00:00"
echo "Log dir:    ${LOG_DIR}"
echo ""

array_job_id=$(
  sbatch --parsable \
    --partition="${PARTITION}" \
    --mem="${MEM}" \
    --array="${array_range}" \
    --output="${LOG_DIR}/fsq_preproc_%A_%a.out" \
    --error="${LOG_DIR}/fsq_preproc_%A_%a.err" \
    "${REPO_ROOT}/scripts/run_fsq_bigdata_preprocessing.sbatch"
)

echo "Array job:  ${array_job_id}"
