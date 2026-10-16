#!/usr/bin/env bash
# One round of ITERATIVE gating: refine the previous round's gated ensemble.
#   METHOD=torsional ROUND=2 sbatch scripts/run_gated_iterate.sbatch
#SBATCH --job-name=gated-iter
#SBATCH --partition=premium
#SBATCH --qos=researcher
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --exclude=gpu09
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/baseline_refine/gi_%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/baseline_refine/gi_%j.err
set -euo pipefail
cd /mnt/weka/fgeikyan/3dmolgen_bigdata
export PYTHONPATH=src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python
M=${METHOD:?}; R=${ROUND:?}
BASE=outputs/baseline_refinement
PREV=$([ "$R" -eq 2 ] && echo "$BASE/$M" || echo "$BASE/${M}_g$((R-1))")
DIR=$BASE/${M}_g$R
EXP=$BASE/exported_gated/${M}_g$((R-1))
mkdir -p "$DIR" "$EXP"
echo "=== round $R: gated ensemble from $PREV  $(date)"
shopt -s nullglob
f=("$EXP"/*_gated.pkl)
if [ ${#f[@]} -eq 0 ]; then $P scripts/export_baseline_conformers.py "$PREV" "$EXP" --conditions gated; f=("$EXP"/*_gated.pkl); fi
echo "input: ${f[0]}"
for phase in prepare refine; do
  echo "--- $phase $(date)"
  $P -m molgen3D.evaluation.baseline_tokenizer_refinement --input "${f[0]}" \
    --output-dir "$DIR" --method "$M" \
    --fsq-checkpoint /mnt/weka/fgeikyan/fsq/new_checkpoints/full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt \
    --phase "$phase" --device cuda --model-batch-size 128 --resume
done
echo "done $(date)"
