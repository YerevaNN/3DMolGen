#!/usr/bin/env bash
# Another tokenizer pass over an existing refinement round.
#   SRC=<round N-1 dir> DST=<round N dir> sbatch scripts/run_setup3_refine_iterate.sbatch
#SBATCH --job-name=setup3-iterate
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --exclude=gpu09
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/interpolation/%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/slurm_jobs/interpolation/%j.err
set -euo pipefail
REPO_ROOT=/mnt/weka/fgeikyan/3dmolgen_bigdata
PYTHON_BIN=/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python
FSQ_CHECKPOINT=${FSQ_CHECKPOINT:-/mnt/weka/fgeikyan/fsq/new_checkpoints/full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt}
SRC=${SRC:?set SRC}
DST=${DST:?set DST}
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Started: $(date)  node $(hostname)"
echo "SRC=${SRC}"; echo "DST=${DST}"
"${PYTHON_BIN}" scripts/setup3_refine_iterate.py \
  --source-dir "${SRC}" --output-dir "${DST}" \
  --fsq-checkpoint "${FSQ_CHECKPOINT}" --device cuda --resume
echo "Finished: $(date)"
