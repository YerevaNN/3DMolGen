#!/usr/bin/env bash
# Submit CASF conformer generation or geometric analysis on Slurm.
#
# Usage:
#   ./scripts/submit_casf.sh generate [core|ref]
#   ./scripts/submit_casf.sh analyze  [core|ref]
#
# Environment overrides: CHEMBL_MAP_CSV, OUTPUT_DIR, CASF_LIGAND_DIR / LIG_DIR,
# MERGE_JOB_ID (analysis), ANALYSIS_WORKERS, PARTITION, PYTHON, etc.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  submit_casf.sh generate [core|ref]
  submit_casf.sh analyze  [core|ref]

Examples:
  ./scripts/submit_casf.sh generate core
  MERGE_JOB_ID=12345 ./scripts/submit_casf.sh analyze ref

Environment:
  MODE=generate|analyze   (alternative to first argument)
  COHORT=core|ref         (alternative to second argument; default: core)
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="${MODE:-${1:-}}"
COHORT="${COHORT:-${2:-core}}"

if [[ -z "${MODE}" ]]; then
  usage >&2
  exit 1
fi

case "${MODE}" in
  generate|gen) MODE=generate ;;
  analyze|analysis|analy) MODE=analyze ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage >&2
    exit 1
    ;;
esac

case "${COHORT}" in
  core|ref) ;;
  *)
    echo "Unknown cohort: ${COHORT} (expected core or ref)" >&2
    exit 1
    ;;
esac

WEKA_DATA="/mnt/weka/mbedrosian/data/casf16"
WEKA_PHARMA="/mnt/weka/mbedrosian/pharma_generation_analysis"

if [[ "${COHORT}" == "core" ]]; then
  DEFAULT_CSV="${WEKA_DATA}/casf16_core_chembl3d_exact_intersection.csv"
  DEFAULT_LIG_DIR="${WEKA_DATA}/CASF16/core_chembl3d_exact_intersection_ligands"
  DEFAULT_OUTPUT_DIR="${WEKA_PHARMA}/core_pb_full_dynamic_chembl_count"
else
  DEFAULT_CSV="${WEKA_DATA}/casf16_ref_chembl3d_exact_intersection.csv"
  DEFAULT_LIG_DIR="${WEKA_DATA}/CASF16_REF/ligands"
  DEFAULT_OUTPUT_DIR="${WEKA_PHARMA}/ref_pb_full_dynamic_chembl_count"
fi

PYTHON="${PYTHON:-/home/mbedrosian/.conda/envs/chembl3d/bin/python}"
PARTITION="${PARTITION:-research}"
CHEMBL_MAP_CSV="${CHEMBL_MAP_CSV:-${CSV:-${DEFAULT_CSV}}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUT_DIR:-${DEFAULT_OUTPUT_DIR}}}"
CASF_LIGAND_DIR="${CASF_LIGAND_DIR:-${LIG_DIR:-${DEFAULT_LIG_DIR}}}"
LIG_DIR="${CASF_LIGAND_DIR}"
CSV="${CHEMBL_MAP_CSV}"

submit_generate() {
  local TOPOLOGY_ROOT="${CHEMBL3D_TOPOLOGY_ROOT:-/mnt/weka/mbedrosian/data/chembl3d/topologies}"
  local LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/slurm_jobs/casf_conformer_sets/${COHORT}}"
  local WORKER_SBATCH="${REPO_ROOT}/scripts/run_casf_ref_conformer_molecule.sbatch"
  local MERGE_SBATCH="${REPO_ROOT}/scripts/run_casf_ref_conformer_merge.sbatch"
  local MAX_ARRAY_JOBS="${MAX_ARRAY_JOBS:-80}"
  local MAX_ARRAY_SIZE="${MAX_ARRAY_SIZE:-1000}"
  local NUM_THREADS="${NUM_THREADS:-8}"
  local MINIMIZE_WORKERS="${MINIMIZE_WORKERS:-${NUM_THREADS}}"
  local SLURM_MEM="${SLURM_MEM:-16G}"
  local SLURM_GRES="${SLURM_GRES:-gpu:0}"

  mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

  if [[ ! -f "${CSV}" ]]; then
    echo "Missing intersection CSV: ${CSV}" >&2
    exit 1
  fi
  if [[ ! -d "${LIG_DIR}" ]]; then
    echo "Missing ligand directory: ${LIG_DIR}" >&2
    exit 1
  fi
  if [[ ! -d "${TOPOLOGY_ROOT}" ]]; then
    echo "Missing ChEMBL3D topology root: ${TOPOLOGY_ROOT}" >&2
    exit 1
  fi
  if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing executable Python: ${PYTHON}" >&2
    exit 1
  fi

  local N
  N="$(
    "${PYTHON}" - <<PY
import csv
from pathlib import Path
with Path("${CSV}").open(newline="") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
  )"
  local PARTS_DIR="${OUTPUT_DIR}/generation/manifest_parts"
  local DONE
  DONE="$(find "${PARTS_DIR}" -maxdepth 1 -name '*.tsv' 2>/dev/null | wc -l || true)"

  echo "mode=generate cohort=${COHORT}"
  echo "total_molecules=${N}"
  echo "existing_manifest_parts=${DONE}"
  echo "chembl_map_csv=${CSV}"
  echo "ligand_dir=${LIG_DIR}"
  echo "topology_root=${TOPOLOGY_ROOT}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "python=${PYTHON}"
  echo "num_threads=${NUM_THREADS}"
  echo "minimize_workers=${MINIMIZE_WORKERS}"

  submit_array_chunk() {
    local base="$1"
    local count="$2"
    local last=$((count - 1))
    echo "Submitting array base=${base} range=0-${last}%${MAX_ARRAY_JOBS}" >&2
    sbatch \
      --partition="${PARTITION}" \
      --cpus-per-task="${NUM_THREADS}" \
      --gres="${SLURM_GRES}" \
      --mem="${SLURM_MEM}" \
      --array="0-${last}%${MAX_ARRAY_JOBS}" \
      --export=ALL,REPO_ROOT="${REPO_ROOT}",PYTHON="${PYTHON}",CHEMBL_MAP_CSV="${CSV}",LIGAND_DIR="${LIG_DIR}",CHEMBL3D_TOPOLOGY_ROOT="${TOPOLOGY_ROOT}",OUTPUT_DIR="${OUTPUT_DIR}",MOLECULE_OFFSET_BASE="${base}",NUM_THREADS="${NUM_THREADS}",MINIMIZE_WORKERS="${MINIMIZE_WORKERS}" \
      "${WORKER_SBATCH}" \
      | awk '{print $NF}'
  }

  local -a ARRAY_JOB_IDS=()
  local offset=0
  while (( offset < N )); do
    local remaining=$((N - offset))
    local chunk=$(( remaining < MAX_ARRAY_SIZE ? remaining : MAX_ARRAY_SIZE ))
    local job_id
    job_id="$(submit_array_chunk "${offset}" "${chunk}")"
    if [[ -z "${job_id}" ]]; then
      echo "Failed to submit array chunk base=${offset}" >&2
      exit 1
    fi
    ARRAY_JOB_IDS+=("${job_id}")
    offset=$((offset + chunk))
  done

  local dependency="afterany:$(IFS=:; echo "${ARRAY_JOB_IDS[*]}")"
  local MERGE_JOB_ID
  MERGE_JOB_ID="$(
    sbatch \
      --partition="${PARTITION}" \
      --dependency="${dependency}" \
      --export=ALL,REPO_ROOT="${REPO_ROOT}",PYTHON="${PYTHON}",CHEMBL_MAP_CSV="${CSV}",OUTPUT_DIR="${OUTPUT_DIR}" \
      "${MERGE_SBATCH}" \
      | awk '{print $NF}'
  )"

  echo "array_job_ids=${ARRAY_JOB_IDS[*]}"
  echo "merge_job_id=${MERGE_JOB_ID}"
  echo "logs=${LOG_DIR}"
  echo "Monitor: squeue -u ${USER} -n casf-ref-conf,casf-ref-merge"
  echo "Next: MERGE_JOB_ID=${MERGE_JOB_ID} ./scripts/submit_casf.sh analyze ${COHORT}"
}

submit_analyze() {
  local ANALYSIS_WORKERS="${ANALYSIS_WORKERS:-32}"
  local MERGE_JOB_ID="${MERGE_JOB_ID:-}"
  local SBATCH_SCRIPT="${REPO_ROOT}/scripts/run_casf_ref_geometric_analysis.sbatch"
  local LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/slurm_jobs/casf_ref_analysis/${COHORT}}"

  mkdir -p "${LOG_DIR}"

  if [[ "${COHORT}" == "ref" && -z "${MERGE_JOB_ID}" ]]; then
    echo "MERGE_JOB_ID must be set for ref analysis (Slurm job id of casf-ref-merge)" >&2
    exit 1
  fi

  local DEPENDENCY=""
  if [[ -n "${MERGE_JOB_ID}" ]]; then
    DEPENDENCY="afterok:${MERGE_JOB_ID}"
  fi

  echo "mode=analyze cohort=${COHORT}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "chembl_map_csv=${CHEMBL_MAP_CSV}"
  echo "casf_ligand_dir=${CASF_LIGAND_DIR}"
  echo "analysis_workers=${ANALYSIS_WORKERS}"
  if [[ -n "${DEPENDENCY}" ]]; then
    echo "dependency=${DEPENDENCY}"
  fi

  local -a SBATCH_ARGS=(--partition="${PARTITION}")
  if [[ -n "${DEPENDENCY}" ]]; then
    SBATCH_ARGS+=(--dependency="${DEPENDENCY}")
  fi

  local ANALYSIS_JOB_ID
  ANALYSIS_JOB_ID="$(
    sbatch \
      "${SBATCH_ARGS[@]}" \
      --export=ALL,REPO_ROOT="${REPO_ROOT}",OUTPUT_DIR="${OUTPUT_DIR}",CHEMBL_MAP_CSV="${CHEMBL_MAP_CSV}",CASF_LIGAND_DIR="${CASF_LIGAND_DIR}",ANALYSIS_WORKERS="${ANALYSIS_WORKERS}",PYTHON="${PYTHON}" \
      "${SBATCH_SCRIPT}" \
      | awk '{print $NF}'
  )"

  echo "analysis_job_id=${ANALYSIS_JOB_ID}"
  echo "logs=${LOG_DIR}"
  echo "Monitor: squeue -u ${USER} -n casf-ref-analy"
}

case "${MODE}" in
  generate) submit_generate ;;
  analyze) submit_analyze ;;
esac
