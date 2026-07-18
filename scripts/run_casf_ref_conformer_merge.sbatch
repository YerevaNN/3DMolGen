#!/bin/bash
#SBATCH --job-name=casf-ref-merge
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=0
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/mbedrosian/code/3DMolGen/outputs/slurm_jobs/casf_ref_conformer_sets/merge_%j.out
#SBATCH --error=/home/mbedrosian/code/3DMolGen/outputs/slurm_jobs/casf_ref_conformer_sets/merge_%j.err

set -euo pipefail

: "${REPO_ROOT:=/home/mbedrosian/code/3DMolGen}"
: "${CHEMBL_MAP_CSV:?CHEMBL_MAP_CSV must be set}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

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

GENERATOR="${REPO_ROOT}/src/molgen3D/pharmacophore/generate_casf_smiles_conformer_sets.py"

exec "${PYTHON}" -u "${GENERATOR}" \
  --chembl_map_csv "${CHEMBL_MAP_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --merge_manifest
