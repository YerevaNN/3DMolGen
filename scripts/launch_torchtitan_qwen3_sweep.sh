#!/usr/bin/env bash
# Dedicated launcher for hp_sweep runs. Mirrors launch_torchtitan_qwen3.sh
# but keeps outputs/job names isolated for sweeps.
#SBATCH --job-name=torchtitan-qwen3-sweep
#SBATCH --cpus-per-task=32
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=6-00:00:00
#SBATCH --output=outputs/slurm_jobs/titan_sweep/%j.out
#SBATCH --error=outputs/slurm_jobs/titan_sweep/%j.err

export WANDB_ENTITY=${WANDB_ENTITY:-menuab_team}
export WANDB_PROJECT=${WANDB_PROJECT:-3dmolgen}
export WANDB_GROUP=${WANDB_GROUP:-pretrain}
export WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-pretrain}
export WANDB_CONFIG=${WANDB_CONFIG:-'{"run_type": "pretrain"}'}
export TORCH_COMPILE=${TORCH_COMPILE:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
# export MOLGEN3D_REPO_ROOT=${MOLGEN3D_REPO_ROOT:-}

TRAIN_TOML=${TRAIN_TOML:-src/molgen3D/config/pretrain/qwen3_06b.toml}

_DEFAULT_DESCRIPTION=$(python3 - <<'PY' "$TRAIN_TOML"
import pathlib, re, sys
toml_path = pathlib.Path(sys.argv[1])
text = toml_path.read_text()
match = re.search(r'^\s*description\s*=\s*"([^"]+)"', text, re.MULTILINE)
print(match.group(1) if match else toml_path.stem, end="")
PY
)
DESCRIPTION=${JOB_DESCRIPTION:-${RUN_DESC:-${_DEFAULT_DESCRIPTION}}}
if [[ -n "${RUN_DESC:-}" ]]; then
  echo "WARNING: RUN_DESC is deprecated; set JOB_DESCRIPTION or job.description instead." >&2
fi
echo "Using description: ${DESCRIPTION}"

if [[ -z "${RUN_NAME:-}" ]]; then
  STAMP=$(date +%y%m%d-%H%M)
  HASH=$(python3 - <<'PY'
import secrets
print(secrets.token_hex(2), end="")
PY
)
  RUN_NAME="${STAMP}-${HASH}-${DESCRIPTION}"
fi
echo "Run name: ${RUN_NAME}"

# Refresh the Slurm job name to reflect the run description for easier tracking.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  JOB_NAME_BASE="torchtitan-${DESCRIPTION}"
  JOB_NAME_SANITIZED=$(echo "${JOB_NAME_BASE}" | tr -cs '[:alnum:]._-' '-')
  JOB_NAME_TRUNC=${JOB_NAME_SANITIZED:0:128}
  scontrol update JobId="${SLURM_JOB_ID}" JobName="${JOB_NAME_TRUNC}" >/dev/null 2>&1 || true
  echo "Updated Slurm job name to: ${JOB_NAME_TRUNC}"
fi

MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
MASTER_PORT=${MASTER_PORT:-$(( (RANDOM % 20000) + 20000 ))}
if [[ -z "${SLURM_GPUS_ON_NODE:-}" ]]; then
    echo "SLURM_GPUS_ON_NODE is unset; please request GPUs via --gres."
    exit 1
fi
NGPU_PER_NODE=${SLURM_GPUS_ON_NODE}
NNODES=${SLURM_NNODES:-1}
NODE_RANK=${SLURM_NODEID:-0}
export MASTER_ADDR MASTER_PORT

TMP_TOML=$(mktemp /tmp/qwen3_runXXXXXX.toml)
cleanup() {
  rm -f "${TMP_TOML}"
}
trap cleanup EXIT

python3 - <<'PY' "$TRAIN_TOML" "$TMP_TOML" "$RUN_NAME" "$DESCRIPTION"
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
run_name = sys.argv[3]
description = sys.argv[4]

lines = src.read_text().splitlines()
out_lines = []
in_block = False
inserted = False
description_set = False

for line in lines:
    stripped = line.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        if in_block and not inserted:
            out_lines.append(f'run_name = "{run_name}"')
            inserted = True
        in_block = stripped == "[molgen_run]"
    if stripped.startswith("description") and "=" in stripped and "[" not in stripped and "]" not in stripped:
        out_lines.append(f'description = "{description}"')
        description_set = True
        continue
    out_lines.append(line)

if in_block and not inserted:
    out_lines.append(f'run_name = "{run_name}"')
if not description_set:
    patched = []
    added = False
    for line in out_lines:
        patched.append(line)
        if not added and line.strip() == "[job]":
            patched.append(f'description = "{description}"')
            added = True
    out_lines = patched

dst.write_text("\n".join(out_lines) + "\n")
PY

exec torchrun \
  --nproc_per_node="${NGPU_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  -m molgen3D.training.pretraining.torchtitan_runner \
  --train-toml "${TMP_TOML}"

