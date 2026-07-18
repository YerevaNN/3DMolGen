#!/bin/bash
#SBATCH --job-name=casf-ref-conf
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --gpus-per-node=0
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/mbedrosian/code/3DMolGen/outputs/slurm_jobs/casf_ref_conformer_sets/%A_%a.out
#SBATCH --error=/home/mbedrosian/code/3DMolGen/outputs/slurm_jobs/casf_ref_conformer_sets/%A_%a.err

set -euo pipefail

: "${REPO_ROOT:=/home/mbedrosian/code/3DMolGen}"
: "${CHEMBL_MAP_CSV:?CHEMBL_MAP_CSV must be set}"
: "${LIGAND_DIR:?LIGAND_DIR must be set}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set}"
: "${MOLECULE_OFFSET_BASE:=0}"
: "${CHEMBL3D_TOPOLOGY_ROOT:=/mnt/weka/mbedrosian/data/chembl3d/topologies}"
: "${NUM_THREADS:=${SLURM_CPUS_PER_TASK:-8}}"
: "${MINIMIZE_WORKERS:=${NUM_THREADS}}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
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
OFFSET=$((MOLECULE_OFFSET_BASE + SLURM_ARRAY_TASK_ID))
MANIFEST_PARTS="${OUTPUT_DIR}/generation/manifest_parts"

TOTAL="$(
  "${PYTHON}" - <<PY
import csv
from pathlib import Path
with Path("${CHEMBL_MAP_CSV}").open(newline="") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)"

if (( OFFSET >= TOTAL )); then
  echo "Offset ${OFFSET} out of range (${TOTAL} molecules); nothing to do."
  exit 0
fi

MOL_ID="$(
  "${PYTHON}" - <<PY
import csv
from pathlib import Path
rows = list(csv.DictReader(Path("${CHEMBL_MAP_CSV}").open(newline="")))
print(rows[${OFFSET}]["ligand_id"])
PY
)"
PART_PATH="${MANIFEST_PARTS}/${MOL_ID}.tsv"

echo "node=$(hostname)"
echo "python=${PYTHON}"
echo "array_task_id=${OFFSET}"
echo "mol_id=${MOL_ID}"
echo "chembl_map_csv=${CHEMBL_MAP_CSV}"
echo "ligand_dir=${LIGAND_DIR}"
echo "topology_root=${CHEMBL3D_TOPOLOGY_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "num_threads=${NUM_THREADS}"
echo "minimize_workers=${MINIMIZE_WORKERS}"

if [[ -f "${PART_PATH}" ]]; then
  echo "Skip ${MOL_ID}; manifest part already exists."
  exit 0
fi

exec "${PYTHON}" -u "${GENERATOR}" \
  --chembl_map_csv "${CHEMBL_MAP_CSV}" \
  --ligand_dir "${LIGAND_DIR}" \
  --chembl3d_topology_root "${CHEMBL3D_TOPOLOGY_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --molecule_offset "${OFFSET}" \
  --limit_molecules 1 \
  --num_threads "${NUM_THREADS}" \
  --minimize_workers "${MINIMIZE_WORKERS}"
