#!/usr/bin/env bash
#SBATCH --job-name=torchtitan-qwen3
#SBATCH --cpus-per-task=64
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=outputs/slurm_jobs/titan/%j.out
#SBATCH --error=outputs/slurm_jobs/titan/%j.err

set -euo pipefail

export WANDB_ENTITY=${WANDB_ENTITY:-menuab_team}
export WANDB_PROJECT=${WANDB_PROJECT:-3dmolgen}
export WANDB_GROUP=${WANDB_GROUP:-pretrain}
export WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-pretrain}
export WANDB_CONFIG=${WANDB_CONFIG:-'{"run_type": "pretrain"}'}
export TORCH_COMPILE=${TORCH_COMPILE:-0}

TRAIN_TOML=${TRAIN_TOML:-src/molgen3D/config/pretrain/qwen3_06b.toml}

_DEFAULT_RUN_DESC=$(python3 - <<'PY' "$TRAIN_TOML"
import pathlib, re, sys
toml_path = pathlib.Path(sys.argv[1])
text = toml_path.read_text()
match = re.search(r'^\s*run_desc\s*=\s*"([^"]+)"', text, re.MULTILINE)
print(match.group(1) if match else toml_path.stem, end="")
PY
)
RUN_DESC=${RUN_DESC:-${_DEFAULT_RUN_DESC}}
echo "Using run_desc: ${RUN_DESC}"

if [[ -z "${RUN_NAME:-}" ]]; then
  STAMP=$(date +%y%m%d-%H%M)
  HASH=$(python3 - <<'PY'
import secrets
print(secrets.token_hex(2), end="")
PY
)
  RUN_NAME="${STAMP}-${HASH}-${RUN_DESC}"
fi
echo "Run name: ${RUN_NAME}"

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

python3 - <<'PY' "$TRAIN_TOML" "$TMP_TOML" "$RUN_NAME"
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
run_name = sys.argv[3]

lines = src.read_text().splitlines()
out_lines = []
in_block = False
inserted = False

for line in lines:
    stripped = line.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        if in_block and not inserted:
            out_lines.append(f'run_name = "{run_name}"')
            inserted = True
        in_block = stripped == "[molgen_run]"
    out_lines.append(line)

if in_block and not inserted:
    out_lines.append(f'run_name = "{run_name}"')

dst.write_text("\n".join(out_lines) + "\n")
PY

exec torchrun \
  --nproc_per_node="${NGPU_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  -m molgen3D.training.pretraining.torchtitan_runner \
  --run-desc "${RUN_DESC}" \
  --train-toml "${TMP_TOML}"
