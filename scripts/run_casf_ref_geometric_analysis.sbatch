#!/bin/bash
#SBATCH --job-name=casf-ref-analy
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --gpus-per-node=0
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/mbedrosian/code/3DMolGen/outputs/slurm_jobs/casf_ref_analysis/%j.out
#SBATCH --error=/home/mbedrosian/code/3DMolGen/outputs/slurm_jobs/casf_ref_analysis/%j.err

set -euo pipefail

: "${REPO_ROOT:=/home/mbedrosian/code/3DMolGen}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
: "${CHEMBL_MAP_CSV:?CHEMBL_MAP_CSV must be set}"
: "${CASF_LIGAND_DIR:?CASF_LIGAND_DIR must be set}"
: "${ANALYSIS_WORKERS:=${SLURM_CPUS_PER_TASK:-}}"
if [[ -z "${ANALYSIS_WORKERS}" ]]; then
  echo "Set ANALYSIS_WORKERS or run under Slurm with --cpus-per-task" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
  :
elif [[ -x /home/mbedrosian/.conda/envs/chembl3d/bin/python ]]; then
  PYTHON=/home/mbedrosian/.conda/envs/chembl3d/bin/python
elif [[ -x /mnt/weka/shared-cache/miniforge3/bin/python ]]; then
  PYTHON=/mnt/weka/shared-cache/miniforge3/bin/python
else
  echo "No usable Python with RDKit and posebusters found" >&2
  exit 127
fi

ANALYZER="${REPO_ROOT}/scripts/analyze_casf_conformer_sets.py"
MANIFEST="${OUTPUT_DIR}/generation/manifest.tsv"

echo "node=$(hostname)"
echo "python=${PYTHON}"
echo "output_dir=${OUTPUT_DIR}"
echo "chembl_map_csv=${CHEMBL_MAP_CSV}"
echo "casf_ligand_dir=${CASF_LIGAND_DIR}"
echo "analysis_workers=${ANALYSIS_WORKERS}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing merged manifest: ${MANIFEST}" >&2
  exit 1
fi

exec "${PYTHON}" -u "${ANALYZER}" \
  --output_dir "${OUTPUT_DIR}" \
  --workers "${ANALYSIS_WORKERS}" \
  --casf-ligand-dir "${CASF_LIGAND_DIR}" \
  --chembl-map-csv "${CHEMBL_MAP_CSV}"
