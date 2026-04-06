#!/bin/bash
# Usage:
#   sbatch --partition=research --job-name=count-geom-revisited-train \
#          --cpus-per-task=1 --mem=32G --time=02:00:00 \
#          scripts/count_geom_revisited_train_pickle.sh
#
# Optional overrides:
#   REPO_ROOT=/path/to/3DMolGen
#   VENV_PATH=/path/to/.venv/bin/activate
#   PICKLE_PATH=/data/vtarasov/geom_revisited/train_data.pickle
#
#SBATCH --output=/home/vtarasov/slurm_logs/dataset_counts/geom-revisited-train-%j.out
#SBATCH --error=/home/vtarasov/slurm_logs/dataset_counts/geom-revisited-train-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/vtarasov/code/3DMolGen}
VENV_PATH=${VENV_PATH:-$REPO_ROOT/.venv/bin/activate}
PICKLE_PATH=${PICKLE_PATH:-/data/vtarasov/geom_revisited/train_data.pickle}

mkdir -p /home/vtarasov/slurm_logs/dataset_counts

echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Started:     $(date)"
echo "Repo root:   $REPO_ROOT"
echo "Pickle path: $PICKLE_PATH"
echo ""

source "$VENV_PATH"
cd "$REPO_ROOT"

python -u - <<'PY'
import pickle
from pathlib import Path


pickle_path = Path("/data/vtarasov/geom_revisited/train_data.pickle")
override = Path(__import__("os").environ.get("PICKLE_PATH", str(pickle_path)))
pickle_path = override

if not pickle_path.exists():
    raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

with pickle_path.open("rb") as fh:
    data = pickle.load(fh)


def count_from_entry(entry):
    if isinstance(entry, tuple) and len(entry) >= 2:
        mols = entry[1]
        try:
            return 1, len(mols)
        except TypeError:
            return 1, 0
    if isinstance(entry, dict):
        if "mols" in entry:
            mols = entry["mols"]
            try:
                return 1, len(mols)
            except TypeError:
                return 1, 0
        if "conformers" in entry:
            conformers = entry["conformers"]
            try:
                return 1, len(conformers)
            except TypeError:
                return 1, 0
    return 1, 0


if isinstance(data, dict):
    items = list(data.items())
    molecule_count = len(items)
    conformer_count = 0
    for _, value in items:
        try:
            conformer_count += len(value)
        except TypeError:
            pass
elif isinstance(data, list):
    molecule_count = 0
    conformer_count = 0
    for entry in data:
        mols, confs = count_from_entry(entry)
        molecule_count += mols
        conformer_count += confs
else:
    raise TypeError(f"Unsupported pickle top-level type: {type(data).__name__}")

print("GEOM revisited train counts")
print(f"pickle_path: {pickle_path}")
print(f"molecules:   {molecule_count:,}")
print(f"conformers:  {conformer_count:,}")
PY

echo ""
echo "Finished: $(date)"
