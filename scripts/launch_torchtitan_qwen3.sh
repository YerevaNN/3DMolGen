#!/usr/bin/env bash
#SBATCH --job-name=torchtitan-qwen3-fsq-bigdata
#SBATCH --cpus-per-task=32
#SBATCH --partition=research
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=256Gb
#SBATCH --time=6-00:00:00
#SBATCH --output=outputs/slurm_jobs/titan/%j.out
#SBATCH --error=outputs/slurm_jobs/titan/%j.err

set -euo pipefail

export WANDB_ENTITY=${WANDB_ENTITY:-filya-team}
export WANDB_PROJECT=${WANDB_PROJECT:-3dmolgen}
export WANDB_GROUP=${WANDB_GROUP:-pretrain}
export WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-pretrain}
export WANDB_CONFIG=${WANDB_CONFIG:-'{"run_type": "pretrain"}'}
export WANDB_MODE=${WANDB_MODE:-online}
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
export ENABLE_NCCL_DEBUG=${ENABLE_NCCL_DEBUG:-0}
if [[ "${ENABLE_NCCL_DEBUG}" == "1" ]]; then
  export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
  export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
fi

# Limit IB Completion Queue allocation to avoid ibv_create_cq ENOMEM when
# multiple ranks set up connections simultaneously on the same HCA. Each NCCL
# channel creates CQs; capping channels and QPs/connection keeps total CQ
# count within the HCA's device-memory limit.
#
# CollNet and NVLS are for specialized switch/NVLink topologies not present
# in this 2-node IB cluster. Disabling them eliminates ~22 unnecessary extra
# channels (6 collnet + 16 nvls) that each create IB CQs.
export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-1}
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-8}
export NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE:-0}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

# Do not auto-force NCCL_IB_HCA across nodes: interface names can differ by
# host. If you need to constrain HCAs, set NCCL_IB_HCA explicitly yourself.

# Slurm batch jobs land on compute nodes with a stripped PATH that may omit
# the Slurm binaries, even when they are installed locally. Prepend any known
# Slurm bin directory we can find so `scontrol`, `srun`, etc. are reachable.
# This is required on heterogeneous clusters (e.g. Bright Cluster Manager)
# where login nodes ship `/opt/slurm/bin` but compute nodes only have
# `/cm/local/apps/slurm/current/bin`.
if ! command -v scontrol >/dev/null 2>&1; then
  _slurm_candidates=(
    /opt/slurm/bin /opt/slurm/sbin
    /usr/local/bin /usr/local/sbin
    /usr/bin /usr/sbin
    /cm/shared/apps/slurm/current/bin /cm/shared/apps/slurm/current/sbin
    /cm/local/apps/slurm/current/bin  /cm/local/apps/slurm/current/sbin
  )
  # Fallback to version-pinned Bright Cluster dirs (e.g. .../slurm/25.05/bin)
  # in case the `current` symlink is missing on this node.
  shopt -s nullglob
  for _vd in /cm/local/apps/slurm/[0-9]* /cm/shared/apps/slurm/[0-9]*; do
    [[ -d "${_vd}" ]] && _slurm_candidates+=("${_vd}/bin" "${_vd}/sbin")
  done
  shopt -u nullglob
  for _slurm_dir in "${_slurm_candidates[@]}"; do
    if [[ -x "${_slurm_dir}/scontrol" ]] && [[ ":${PATH}:" != *":${_slurm_dir}:"* ]]; then
      export PATH="${_slurm_dir}:${PATH}"
      break
    fi
  done
  unset _slurm_candidates _slurm_dir _vd
fi

WORKSPACE_DIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "${WORKSPACE_DIR}"
mkdir -p outputs/slurm_jobs/titan

EXTRA_SITE_PACKAGES=${EXTRA_SITE_PACKAGES:-}
TRAINING_VENV_SITE_PACKAGES=${TRAINING_VENV_SITE_PACKAGES:-/mnt/weka/fgeikyan/fsq_remote/3DMolGen/.venv/lib/python3.10/site-packages}
TRAINING_SHARED_LIB=${TRAINING_SHARED_LIB:-/mnt/weka/shared-cache/miniforge3/lib}
if [[ -d "${TRAINING_SHARED_LIB}" ]]; then
  export LD_LIBRARY_PATH="${TRAINING_SHARED_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

_pythonpath_entries=()
if [[ -n "${EXTRA_SITE_PACKAGES}" ]]; then
  _pythonpath_entries+=("${EXTRA_SITE_PACKAGES}")
fi
if [[ -d "${TRAINING_VENV_SITE_PACKAGES}" ]]; then
  _pythonpath_entries+=("${TRAINING_VENV_SITE_PACKAGES}")
fi
_pythonpath_entries+=("${WORKSPACE_DIR}/src")
if [[ -n "${PYTHONPATH:-}" ]]; then
  _pythonpath_entries+=("${PYTHONPATH}")
fi
PYTHONPATH=$(IFS=:; echo "${_pythonpath_entries[*]}")
export PYTHONPATH
unset _pythonpath_entries

if [[ -n "${EXTRA_SITE_PACKAGES}" ]]; then
  :
else
  EXTRA_SITE_PACKAGES=""
fi

_python_candidates=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
  _python_candidates+=("${PYTHON_BIN}")
fi
_python_candidates+=(
  "${WORKSPACE_DIR}/.venv/bin/python"
  "/mnt/weka/shared-cache/conda/python-3.10.20-h3c07f61_0_cpython/bin/python3.10"
  "/mnt/weka/shared-cache/conda/python-3.10.18-hd6af730_0_cpython/bin/python3.10"
  "/mnt/weka/shared-cache/conda/python-3.10.16-habfa6aa_3_cpython/bin/python3.10"
  "/mnt/weka/fgeikyan/fsq_remote/3DMolGen/.venv/bin/python"
)
_system_python=$(command -v python3 || true)
if [[ -n "${_system_python}" ]]; then
  _python_candidates+=("${_system_python}")
fi
PYTHON_BIN=""
for _python_candidate in "${_python_candidates[@]}"; do
  if [[ ! -x "${_python_candidate}" ]]; then
    continue
  fi
  if "${_python_candidate}" - <<'PY' >/dev/null 2>&1
import importlib
required = [
    "tyro",
    "torch",
    "torchtitan.components",
    "torchtitan.models.qwen3",
    "molgen3D.training.pretraining.torchtitan_runner",
    "molgen3D.training.pretraining.torchtitan_model.qwen3_custom",
]
for module_name in required:
    importlib.import_module(module_name)
PY
  then
    PYTHON_BIN="${_python_candidate}"
    break
  fi
  echo "Skipping Python candidate without required training imports: ${_python_candidate}" >&2
done
unset _python_candidates _python_candidate _system_python
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "No usable PYTHON_BIN found for TorchTitan training." >&2
  exit 1
fi

TRAIN_TOML=${TRAIN_TOML:-src/molgen3D/config/pretrain/qwen3_06b_fsq_bigdata_from_base.toml}

_DEFAULT_DESCRIPTION=$("${PYTHON_BIN}" - <<'PY' "$TRAIN_TOML"
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
  HASH=$("${PYTHON_BIN}" - <<'PY'
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

# === Multi-node topology (no srun: head SSHes into workers) ===
echo "Python: ${PYTHON_BIN}"
echo "PYTHONPATH: ${PYTHONPATH}"
"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

required = [
    "tyro",
    "torch",
    "torchtitan.components",
    "torchtitan.models.qwen3",
    "molgen3D.training.pretraining.torchtitan_runner",
    "molgen3D.training.pretraining.torchtitan_model.qwen3_custom",
]
failed = []
for module_name in required:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        failed.append((module_name, exc))
if failed:
    print("Training environment import check failed:", file=sys.stderr)
    for module_name, exc in failed:
        print(f"  {module_name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
print("Training environment import check passed.", flush=True)
PY

if [[ "${PREBUILD_TRAIN_IDX:-1}" == "1" ]]; then
  echo "Prebuilding train .idx files for ${TRAIN_TOML}"
  "${PYTHON_BIN}" - <<'PY' "$TRAIN_TOML"
import pathlib
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from molgen3D.config.paths import resolve_tag
from molgen3D.training.pretraining.dataprocessing.utils import (
    build_line_index,
    expand_paths,
)

toml_path = pathlib.Path(sys.argv[1])
data = tomllib.loads(toml_path.read_text())
molgen_data = data.get("molgen_data", {}) or {}
training = data.get("training", {}) or {}
dataset_path = molgen_data.get("train_path") or training.get("dataset_path") or ""
if not dataset_path:
    raise SystemExit("No train path found in molgen_data.train_path or training.dataset_path")
if ":" in dataset_path:
    dataset_path = str(resolve_tag(dataset_path))
paths = expand_paths([dataset_path])
print(f"Indexing {len(paths)} JSONL files from {dataset_path}", flush=True)
for idx, path in enumerate(paths, start=1):
    build_line_index(path)
    if idx == 1 or idx % 50 == 0 or idx == len(paths):
        print(f"Indexed {idx}/{len(paths)} files", flush=True)
PY
fi

if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  if ! command -v scontrol >/dev/null 2>&1; then
    echo "ERROR: scontrol not found on PATH; cannot expand SLURM_JOB_NODELIST='${SLURM_JOB_NODELIST}'." >&2
    echo "       Searched: /opt/slurm/{bin,sbin}, /usr/{local/,}{bin,sbin}, /cm/{shared,local}/apps/slurm/current/{bin,sbin}." >&2
    echo "       Update PATH or add the correct slurm bin dir to the candidate list at the top of this script." >&2
    exit 1
  fi
  NODELIST_ARR=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
else
  NODELIST_ARR=( "$(hostname -s)" )
fi
NNODES=${NNODES:-${#NODELIST_ARR[@]}}
if (( NNODES < 1 )); then
  echo "ERROR: NNODES resolved to ${NNODES}; failed to determine cluster topology." >&2
  exit 1
fi
MASTER_ADDR=${MASTER_ADDR:-${NODELIST_ARR[0]}}
MASTER_PORT=${MASTER_PORT:-$(( (RANDOM % 20000) + 20000 ))}
export MASTER_ADDR MASTER_PORT

_autodetect_socket_iface() {
  local _target="$1"
  local _route
  _route=$(ip -o route get "${_target}" 2>/dev/null | awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i == "dev" && (i + 1) <= NF) {
          print $(i + 1)
          exit
        }
      }
    }') || true
  if [[ -n "${_route}" ]]; then
    if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
      export NCCL_SOCKET_IFNAME="${_route}"
      echo "Auto-set NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} (route to ${_target})"
    fi
    if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
      export GLOO_SOCKET_IFNAME="${_route}"
      echo "Auto-set GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} (route to ${_target})"
    fi
  fi
}
_autodetect_socket_iface "${MASTER_ADDR}"

_DEFAULT_GPUS=$(grep -E '^#SBATCH --gres=gpu:' "$0" | sed 's/.*gpu:\([0-9]*\).*/\1/' | head -1)
NGPU_PER_NODE=${SLURM_GPUS_ON_NODE:-${NGPU_PER_NODE:-${_DEFAULT_GPUS:-8}}}

echo "Topology: nnodes=${NNODES} gpus_per_node=${NGPU_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT} nodes=[${NODELIST_ARR[*]}]"

# Place TMP_TOML in the shared workspace so worker nodes can read it (workers
# cannot see /tmp on the head node).
mkdir -p outputs/slurm_jobs/titan/tmp
TMP_TOML=$(mktemp "${WORKSPACE_DIR}/outputs/slurm_jobs/titan/tmp/qwen3_runXXXXXX.toml")

WORKER_BOOTSTRAP=""
if (( NNODES > 1 )) && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  WORKER_BOOTSTRAP="${WORKSPACE_DIR}/outputs/slurm_jobs/titan/tmp/${SLURM_JOB_ID}.worker.sh"
fi

SSH_PIDS=()
cleanup() {
  for pid in "${SSH_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  rm -f "${TMP_TOML}"
  [[ -n "${WORKER_BOOTSTRAP}" ]] && rm -f "${WORKER_BOOTSTRAP}"
}
trap cleanup EXIT INT TERM

"${PYTHON_BIN}" - <<'PY' "$TRAIN_TOML" "$TMP_TOML" "$RUN_NAME" "$DESCRIPTION"
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

# Forward selected comm/fabric env vars to SSH-launched worker nodes.
# SSH typically does not preserve arbitrary env from the head node.
FORWARD_ENV_VARS=(
  CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER
  TORCH_NCCL_ASYNC_ERROR_HANDLING TORCH_DISTRIBUTED_DEBUG GLOO_SOCKET_IFNAME
  LD_LIBRARY_PATH
  NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_SOCKET_IFNAME NCCL_SOCKET_FAMILY
  NCCL_IB_DISABLE NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_IB_TC
  NCCL_IB_QPS_PER_CONNECTION NCCL_MAX_NCHANNELS NCCL_COLLNET_ENABLE NCCL_NVLS_ENABLE
  NCCL_NET NCCL_NET_GDR_LEVEL NCCL_P2P_DISABLE
  UCX_TLS UCX_NET_DEVICES
  FI_PROVIDER FI_EFA_USE_DEVICE_RDMA
)
FORWARDED_ENV_BLOCK=""
for _env_name in "${FORWARD_ENV_VARS[@]}"; do
  _env_val="${!_env_name:-}"
  if [[ -n "${_env_val}" ]]; then
    _escaped_env_val=$(printf "%q" "${_env_val}")
    FORWARDED_ENV_BLOCK+="export ${_env_name}=${_escaped_env_val}"$'\n'
  fi
done
unset _env_name _env_val _escaped_env_val

# === Fan out to worker nodes via SSH (replacement for srun) ===
if [[ -n "${WORKER_BOOTSTRAP}" ]]; then
  # Generate a worker bootstrap script that exports the same env, then runs
  # torch.distributed.run with the appropriate node_rank passed as $1.
  cat > "${WORKER_BOOTSTRAP}" <<EOF
#!/usr/bin/env bash
set -e
cd '${WORKSPACE_DIR}'
export PATH='${PATH}'
export WANDB_ENTITY='${WANDB_ENTITY}'
export WANDB_PROJECT='${WANDB_PROJECT}'
export WANDB_GROUP='${WANDB_GROUP}'
export WANDB_JOB_TYPE='${WANDB_JOB_TYPE}'
export WANDB_CONFIG='${WANDB_CONFIG}'
export WANDB_MODE='${WANDB_MODE}'
export WANDB_RUN_NAME='${RUN_NAME}'
export PYTHONPATH='${PYTHONPATH}'
export TORCH_COMPILE='${TORCH_COMPILE}'
export TOKENIZERS_PARALLELISM='${TOKENIZERS_PARALLELISM}'
export OMP_NUM_THREADS='${OMP_NUM_THREADS}'
export PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF}'
export MASTER_ADDR='${MASTER_ADDR}'
export MASTER_PORT='${MASTER_PORT}'
${FORWARDED_ENV_BLOCK}
# Per-node socket interface selection: pick route-to-master unless user
# explicitly set NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME.
if [[ -z "\${NCCL_SOCKET_IFNAME:-}" ]] || [[ -z "\${GLOO_SOCKET_IFNAME:-}" ]]; then
  _route_if=\$(ip -o route get '${MASTER_ADDR}' 2>/dev/null | awk '
    {
      for (i = 1; i <= NF; i++) {
        if (\$i == "dev" && (i + 1) <= NF) {
          print \$(i + 1)
          exit
        }
      }
    }') || true
  if [[ -n "\${_route_if}" ]]; then
    if [[ -z "\${NCCL_SOCKET_IFNAME:-}" ]]; then
      export NCCL_SOCKET_IFNAME="\${_route_if}"
      echo "[worker] Auto-set NCCL_SOCKET_IFNAME=\${NCCL_SOCKET_IFNAME} (route to ${MASTER_ADDR})"
    fi
    if [[ -z "\${GLOO_SOCKET_IFNAME:-}" ]]; then
      export GLOO_SOCKET_IFNAME="\${_route_if}"
      echo "[worker] Auto-set GLOO_SOCKET_IFNAME=\${GLOO_SOCKET_IFNAME} (route to ${MASTER_ADDR})"
    fi
  fi
  unset _route_if
fi
NODE_RANK=\$1
echo "[worker rank=\${NODE_RANK} host=\$(hostname)] launching torch.distributed.run"
exec '${PYTHON_BIN}' -m torch.distributed.run \\
  --nproc_per_node=${NGPU_PER_NODE} \\
  --master_addr='${MASTER_ADDR}' \\
  --master_port=${MASTER_PORT} \\
  --nnodes=${NNODES} \\
  --node_rank="\${NODE_RANK}" \\
  -m molgen3D.training.pretraining.torchtitan_runner \\
  --train-toml '${TMP_TOML}'
EOF
  chmod +x "${WORKER_BOOTSTRAP}"

  for ((i=1; i<NNODES; i++)); do
    target="${NODELIST_ARR[i]}"
    worker_log="${WORKSPACE_DIR}/outputs/slurm_jobs/titan/${SLURM_JOB_ID}.rank${i}.log"
    echo "Launching node_rank=${i} on ${target} via SSH (log: ${worker_log})"
    ssh -n -o StrictHostKeyChecking=no -o ServerAliveInterval=30 "${target}" \
      "bash '${WORKER_BOOTSTRAP}' ${i}" \
      > "${worker_log}" 2>&1 &
    SSH_PIDS+=("$!")
  done
fi

echo "Launching torch.distributed.run on host=$(hostname) node_rank=0/${NNODES} master=${MASTER_ADDR}:${MASTER_PORT} nproc_per_node=${NGPU_PER_NODE}"

"${CPU_PIN_CMD[@]}" "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NGPU_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nnodes="${NNODES}" \
  --node_rank=0 \
  -m molgen3D.training.pretraining.torchtitan_runner \
  --train-toml "${TMP_TOML}"
TORCH_EXIT=$?

# Wait for SSH-launched workers to finish so their output is collected.
if [[ ${#SSH_PIDS[@]} -gt 0 ]]; then
  for pid in "${SSH_PIDS[@]}"; do
    wait "$pid" || true
  done
fi

exit ${TORCH_EXIT}
