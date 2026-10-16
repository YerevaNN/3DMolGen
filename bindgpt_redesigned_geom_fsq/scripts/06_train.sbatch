#!/usr/bin/env bash
#SBATCH --job-name=bgfsq-train
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu09
#SBATCH --output=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_redesigned_geom_fsq/logs/%x-%j.out
#SBATCH --error=/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_redesigned_geom_fsq/logs/%x-%j.err

# One job per arm. ARM=coord|fsq, TRAIN_SEED defaults to 17.
# Slurm runs a spooled copy of this script, so BASH_SOURCE points at the spool
# directory, not at this file. Resolve common.sh from the repo root instead.
REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
source "${REPO_ROOT}/bindgpt_redesigned_geom_fsq/scripts/common.sh"
ARM=${ARM:?set ARM=coord or ARM=fsq}
TRAIN_SEED=${TRAIN_SEED:-17}
capture_environment "train-${ARM}"

GPUS=${GPUS:-2}
# torchrun forces OMP_NUM_THREADS=1 unless told otherwise; give the dataloader
# and collate a fair share of the allocated CPUs.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( ${SLURM_CPUS_PER_TASK:-16} / GPUS ))}

# No srun: the Slurm client is not installed on the compute nodes, and this is a
# single-node job, so torchrun's --standalone rendezvous is all that is needed.
"${PYTHON}" -u -m torch.distributed.run \
  --standalone --nproc_per_node="${GPUS}" \
  -m molgen3D.experiments.bindgpt_fsq.cli \
  --data-root "${DATA_ROOT}" train \
  --config "${BINDGPT_FSQ_ROOT}/configs/${ARM}_seed${TRAIN_SEED}.yaml"
