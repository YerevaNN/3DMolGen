#!/usr/bin/env bash
# Shared environment for every stage of the BindGPT-Coord vs BindGPT-FSQ run.
# Scientific settings live in the committed YAML configs, never here.
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/weka/fgeikyan/3dmolgen_bigdata}
export BINDGPT_FSQ_ROOT=${BINDGPT_FSQ_ROOT:-${REPO_ROOT}/bindgpt_redesigned_geom_fsq}
export PYTHONPATH=${REPO_ROOT}/src:${BINDGPT_FSQ_ROOT}/vendor${PYTHONPATH:+:${PYTHONPATH}}
PYTHON=${PYTHON:-/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python}
CLI="${PYTHON} -u -m molgen3D.experiments.bindgpt_fsq.cli"
DATA_ROOT=${DATA_ROOT:-${BINDGPT_FSQ_ROOT}/data}

# Normal partitions: research for GPU work, research_cpu for CPU-only stages.
# gpu09 is excluded everywhere -- it wedges CUDA jobs while still looking healthy.
PARTITION=${PARTITION:-research}
CPU_PARTITION=${CPU_PARTITION:-research_cpu}
EXCLUDE_NODES=${EXCLUDE_NODES:-gpu09}

# Capture the run environment next to the artifacts, as the plan requires.
capture_environment() {
  local tag="$1"
  local dest="${BINDGPT_FSQ_ROOT}/environment/${tag}-${SLURM_JOB_ID:-local}"
  mkdir -p "${dest}"
  {
    echo "date=$(date -Is)"
    echo "host=$(hostname)"
    echo "python=${PYTHON}"
    echo "git_commit=$(cd "${REPO_ROOT}" && git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_status_short=$(cd "${REPO_ROOT}" && git status --short 2>/dev/null | wc -l) modified paths"
  } > "${dest}/run.txt"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi > "${dest}/nvidia-smi.txt" 2>&1 || true
  command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]] \
    && scontrol show job "${SLURM_JOB_ID}" > "${dest}/scontrol.txt" 2>&1 || true
  command -v module >/dev/null 2>&1 && module list > "${dest}/modules.txt" 2>&1 || true
  "${PYTHON}" -m pip freeze > "${dest}/pip-freeze.txt" 2>&1 || true
}
