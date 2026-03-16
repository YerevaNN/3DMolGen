#!/usr/bin/env bash
#SBATCH --job-name=torchtitan-qwen3
#SBATCH --cpus-per-task=64
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --time=6-00:00:00
#SBATCH --output=outputs/slurm_jobs/titan/%j.out
#SBATCH --error=outputs/slurm_jobs/titan/%j.err

export WANDB_ENTITY=${WANDB_ENTITY:-vover-yerevann}
export WANDB_PROJECT=${WANDB_PROJECT:-3dmolgen}
export WANDB_GROUP=${WANDB_GROUP:-pretrain}
export WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-pretrain}
export WANDB_CONFIG=${WANDB_CONFIG:-'{"run_type": "pretrain"}'}
# Keep per-parameter tensors for FSDP to expose embedding rows (needed for grad probe)
# export TORCH_FSDP_USE_ORIG_PARAMS=${TORCH_FSDP_USE_ORIG_PARAMS:-1}
# Also disable parameter flattening so embedding rows remain addressable
# export TORCH_FSDP_FLATTEN_PARAMETERS=${TORCH_FSDP_FLATTEN_PARAMETERS:-0}
# Disable DTensor FSDP so embedding weights stay real tensors for probes
# export TORCH_FSDP_USE_DTENSOR=${TORCH_FSDP_USE_DTENSOR:-0}
export TORCH_COMPILE=${TORCH_COMPILE:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_TOML=${TRAIN_TOML:-src/molgen3D/config/pretrain/qwen3_06b.toml}

# Prebuild validation .idx files to avoid stalls when validation starts.
if [[ "${PREBUILD_VALIDATION_IDX:-1}" == "1" ]]; then
  python3 - <<'PY' "$TRAIN_TOML"
import pathlib
import sys
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python <3.11

toml_path = pathlib.Path(sys.argv[1]).resolve()
repo_root = toml_path.parent.parent.parent
sys.path.insert(0, str(repo_root))

from molgen3D.config.paths import resolve_tag  # type: ignore
from molgen3D.training.pretraining.dataprocessing.utils import expand_paths, build_line_index  # type: ignore

cfg = toml_path.read_text()
data = tomllib.loads(cfg)
validation = data.get("validation", {}) or {}
dataset_path = validation.get("dataset_path") or ""
if not dataset_path:
    raise SystemExit(0)
if ":" in dataset_path:
    dataset_path = str(resolve_tag(dataset_path))
paths = expand_paths([dataset_path])
for path in paths:
    build_line_index(path)
PY
fi

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
_DEFAULT_GPUS=$(grep -E '^#SBATCH --gres=gpu:' "$0" | sed 's/.*gpu:\([0-9]*\).*/\1/' | head -1)
NGPU_PER_NODE=${SLURM_GPUS_ON_NODE:-${NGPU_PER_NODE:-${_DEFAULT_GPUS:-8}}}
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

# Optional CPU pinning for non-Slurm runs.
CPU_PIN_CMD=()
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if command -v taskset >/dev/null 2>&1; then
    CPU_PIN_CPUS=${CPU_PIN_CPUS:-"0-63"}
    CPU_PIN_CMD=(taskset -c "${CPU_PIN_CPUS}")
  elif command -v numactl >/dev/null 2>&1; then
    CPU_PIN_CMD=(numactl --cpunodebind=0 --membind=0)
  fi
fi

exec "${CPU_PIN_CMD[@]}" torchrun \
  --nproc_per_node="${NGPU_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  -m molgen3D.training.pretraining.torchtitan_runner \
  --train-toml "${TMP_TOML}"
