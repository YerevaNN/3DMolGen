#!/usr/bin/env bash
set -euo pipefail

TOML_PATH="src/molgen3D/config/pretrain/qwen3_06b.toml"

# Read run_desc from TOML (fallback to qwen3_06b if missing)
RUN_DESC=$(python3 - <<'PY' "$TOML_PATH"
import pathlib, re, sys
toml_path = pathlib.Path(sys.argv[1])
text = toml_path.read_text()
match = re.search(r'^\s*run_desc\s*=\s*"([^"]+)"', text, re.MULTILINE)
print(match.group(1) if match else "qwen3_06b", end="")
PY
)

echo "Using run_desc: ${RUN_DESC}"

# Build YYMMDD-HHMM prefix + random 4-hex hash
STAMP=$(date +%y%m%d-%H%M)
HASH=$(python3 - <<'PY'
import secrets
print(secrets.token_hex(2), end="")
PY
)
RUN_NAME="${STAMP}-${HASH}-${RUN_DESC}"

echo "Run name: ${RUN_NAME}"

TMP_TOML=$(mktemp /tmp/qwen3_runXXXXXX.toml)
cleanup() {
  rm -f "${TMP_TOML}"
}
trap cleanup EXIT

# Copy base TOML and inject run_name under [molgen_run]
python3 - <<'PY' "$TOML_PATH" "$TMP_TOML" "$RUN_NAME"
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

torchrun --nproc_per_node="${WORLD_SIZE:-4}" --master_port="${MASTER_PORT:-29501}" \
  -m molgen3D.training.pretraining.torchtitan_runner \
  --run-desc "${RUN_DESC}" \
  --train-toml "${TMP_TOML}"
