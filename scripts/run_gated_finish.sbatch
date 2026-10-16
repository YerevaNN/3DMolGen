#!/usr/bin/env bash
# Assemble round R's gated ensemble from its before/after flags, then score it.
#   METHOD=torsional ROUND=2 sbatch scripts/run_gated_finish.sbatch
#SBATCH --job-name=gated-fin
#SBATCH --partition=premium
#SBATCH --qos=researcher
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/baseline_refine/gf_%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/baseline_refine/gf_%j.err
set -euo pipefail
cd /mnt/weka/fgeikyan/3dmolgen_bigdata
export PYTHONPATH=src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python
M=${METHOD:?}; R=${ROUND:?}
echo "=== assemble r$R $(date)"
$P scripts/assemble_gated.py "$M" "$R"
echo "=== evaluate r$R $(date)"
$P -m molgen3D.evaluation.baseline_tokenizer_refinement \
  --output-dir outputs/baseline_refinement/${M}_g$R --method "$M" \
  --fsq-checkpoint /mnt/weka/fgeikyan/fsq/new_checkpoints/full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt \
  --phase evaluate --num-workers 16
echo "done $(date)"
