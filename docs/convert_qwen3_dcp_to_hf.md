# Qwen3 DCP → HF Converter

TorchTitan checkpoints are saved as Distributed CheckPoint (`*.distcp`) shards. The helper at
`src/molgen3D/training/pretraining/helpers/convert_qwen3_dcp_to_hf.py` reconstructs those shards,
verifies them against the recorded training config, and emits a HuggingFace-compatible artifact
(model weights + tokenizer assets).

## Prerequisites

* A Titan run launched via `molgen3D.training.pretraining.torchtitan_runner`. Every launch
  writes `job_config.json` into the run’s log directory and drops `config.json` into the checkpoint
  directory once the first `step-*` folder is created. The converter only consumes the DCP step
  directory plus one of those config files; no manual inputs are required.
* Access to the `hf_assets_path` referenced in the config (e.g. the patched Qwen3 base download).
* Python environment with the project dependencies installed (matching the training environment).

## Usage

From the project root:

```bash
python -m src.molgen3D.training.pretraining.helpers.convert_qwen3_dcp_to_hf \
  --dcp-path /nfs/.../qwen3_06b/<run-root>/step-1800
```

* `--dcp-path` may point to a single `step-*` directory, a run root containing multiple steps, or a
  specific `*.distcp` file (the script will strip back to the parent `step-*` folder).
* Add `--dry-run` to inspect which steps would be processed without writing any output.

For each step the converter:

1. Loads the run config (preferring `<ckpt_root>/config.json`, falling back to the log’s
   `job_config.json`).
2. Reads all tensors from the DCP shards and normalizes their dtype to the training dtype.
3. Verifies embedding/head shapes, vocab padding, tied weights, and NaN/Inf safety.
4. Builds a Qwen3 HF model using the base assets referenced in the config, overriding vocab size,
   max sequence length, and dtype to match the training run.
5. Adapts the Titan state dict to HF layout via `Qwen3StateDictAdapter`, loads it into the HF model,
   and runs a cheap forward-fidelity check.
6. Saves the model (`config.json`, `model.safetensors`, etc.) into `<step-dir>-hf/` and copies the
   tokenizer assets referenced in the config alongside it.

## Output Layout

```
step-1800-hf/
├── config.json
├── generation_config.json
├── model.safetensors
├── special_tokens_map.json   # copied from training tokenizer
├── tokenizer.json            # copied from training tokenizer
└── tokenizer_config.json     # copied from training tokenizer
```

You can now load the checkpoint via standard HuggingFace APIs:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(".../step-1800-hf")
model = AutoModelForCausalLM.from_pretrained(".../step-1800-hf", torch_dtype="auto")
```

## Notes

* The converter refuses to run if the config is missing (e.g. a legacy run) or if the stored
  `hf_assets_path` / tokenizer path no longer exists.
* To accelerate repeated conversions, keep the HF base assets cached locally; the script only reads
  their `config.json` and tokenizer metadata.
* The HF export is deterministic with respect to the DCP and config; rerunning against the same
  step directory will overwrite the existing `step-*-hf` folder.
