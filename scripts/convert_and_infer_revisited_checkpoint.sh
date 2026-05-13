#!/usr/bin/env bash
#SBATCH --job-name=qwen3-convert-infer-revisited
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm_jobs/titan/%j.out
#SBATCH --error=outputs/slurm_jobs/titan/%j.err

set -euo pipefail

REPO_ROOT="/home/vtarasov/code/3DMolGen"
cd "${REPO_ROOT}"

source .venv/bin/activate
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

export DCP_STEP_DIR="${DCP_STEP_DIR:-/mnt/weka/vtarasov/checkpoints/qwen3_06b/260330-1541-7425-qwen3_06b_pre_4e_revisited_quantile_binned_isomeric/step-6800}"
export HF_OUT_DIR="${HF_OUT_DIR:-${DCP_STEP_DIR}-hf}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/src/molgen3D/training/tokenizers/Qwen3_tokenizer_binned_258}"
export QUANTILE_BIN_CONFIG="${QUANTILE_BIN_CONFIG:-${REPO_ROOT}/src/molgen3D/config/bin_configs/quantile_bins.json}"

echo "Converting DCP checkpoint: ${DCP_STEP_DIR}"
echo "HF output directory: ${HF_OUT_DIR}"

python -m molgen3D.training.pretraining.helpers.convert_qwen3_dcp_to_hf \
  --dcp-path "${DCP_STEP_DIR}" \
  --tokenizer-path "${TOKENIZER_PATH}"

echo "Running revisited inference from: ${HF_OUT_DIR}"

python - <<'PY'
import os
from pathlib import Path

from molgen3D.config.paths import get_base_path, get_data_path
from molgen3D.config.sampling_config import gen_num_codes, sampling_configs
from molgen3D.evaluation.inference import run_inference

dcp_step_dir = Path(os.environ["DCP_STEP_DIR"])
hf_out_dir = Path(os.environ["HF_OUT_DIR"])
tokenizer_path = os.environ["TOKENIZER_PATH"]
quantile_bin_config = os.environ["QUANTILE_BIN_CONFIG"]

run_inference(
    {
        "model_path": str(hf_out_dir),
        "tokenizer_path": tokenizer_path,
        "torch_dtype": "bfloat16",
        "batch_size": 64,
        "num_gens": gen_num_codes["2k_per_conf"],
        "gen_config": sampling_configs["top_p_sampling1"],
        "device": "cuda",
        "results_path": str(get_base_path("gen_results_root")),
        "run_name": f"{hf_out_dir.name}_revisited",
        "test_data_path": str(get_data_path("revisited_smi")),
        "test_set": "revisited",
        "binned": True,
        "serialization_tag": "quantile",
        "uniform_bin_config_path": quantile_bin_config,
        "quantile_bin_config_path": quantile_bin_config,
        "limit": None,
        "attention_imp": "sdpa",
    }
)
PY
