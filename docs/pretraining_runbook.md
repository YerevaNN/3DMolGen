# MolGen3D Qwen3 Pretraining Runbook

This document consolidates everything you need to run and reason about the Qwen3‑0.6B pretraining stack that lives under `src/molgen3D`. It covers:

1. Every important TOML option (what it does, recommended values, and interactions).
2. How the launcher, runner, TorchTitan trainer, model, and dataloader call into each other.
3. The actual training objective with concrete label examples so you can debug losses or extend the data pipeline confidently.

The canonical config referenced below is `src/molgen3D/config/pretrain/qwen3_06b.toml`.

---

## 1. Config Reference

The TOML is split into logical sections. Fields marked **(recommended)** are good defaults, while **(required)** means you must set them per run. When a value is resolved via `paths.yaml`, we note the alias.

### `[job]`
- `description` *(recommended)* – becomes part of the run name (`YYMMDD-HHMM-<hash>-description`). Use short, unique text so logs/checkpoints stay readable.
- `dump_folder` *(recommended)* – alias under `paths.yaml.base_paths`. The final checkpoint folder is `<dump_folder>/<run-name>/checkpoint`.
- `print_config` *(optional)* – keep `true` to log the resolved config at startup.
- `custom_config_module = "molgen3D.training.pretraining.config.custom_job_config"` **(required)** – wires in the MolGen-specific path resolver, WSDS scheduler helper, HF checkpoint patch, and dataloader defaults. Removing it reverts to upstream TorchTitan behavior.

### `[molgen_run]`
This section controls initialization mode and tokenizer selection.
- `init_mode = "scratch" | "hf_pretrain" | "resume"` **(required)** – determines how `_configure_initial_load` sets the Titan checkpoint fields.
  - `scratch` → random init (no `initial_load_path`).
  - `hf_pretrain` → loads Hugging Face weights from `base_model_tag`.
  - `resume` → loads a previous Titan DCP; you must also set `resume_run_path_tag` and re-use the same run name.
- `tokenizer_tag` *(required)* – alias under `paths.yaml.tokenizers`. `tokenizers:qwen3_0.6b_custom` lives inside the repo, while `tokenizers:qwen3_0.6b_origin` points at the untouched HF snapshot.
- `base_model_tag` *(required when `init_mode="hf_pretrain"`)* – alias under `paths.yaml.base_paths` for the official checkpoint (`qwen3_0.6b_base_model`).
- `resume_run_path_tag` *(required when `init_mode="resume"`)* – alias pointing to the existing checkpoint directory (e.g. `ckpts:qwen3_06b/<run-name>`).
- `run_name` *(optional)* – force a specific run name (used when resuming a run and you want logs/checkpoints to stay in the same directories).

### `[model]`
- `name = "molgen_qwen3"`, `flavor = "0.6B"` *(required)* – selects the patched Titan train spec registered by `experimental.custom_import`.
- `hf_assets_path` *(required)* – base HF model directory used to hydrate tokenizer/config artifacts. For custom tokenizers this still points to the original weights since only `molgen_run.tokenizer_tag` changes how the vocabulary is widened.

### `[experimental]`
- `custom_import = "molgen3D.training.pretraining.torchtitan_model.qwen3_custom"` **(required)** – registers the custom train spec (handles tokenizer resizing, tied LM head logic, checkpoint hooks, and metric patches).

### `[training]`
- `seq_len = 2048` *(recommended)* – must match the dataloader’s packing length. Larger values require more VRAM and may need smaller `local_batch_size`.
- `local_batch_size = 12` *(recommended starting point)* – per-rank batch size. Combined with `[parallelism]` degrees to form the global batch.
- `global_batch_size = -1` *(recommended)* – `-1` lets TorchTitan derive the correct global batch. Set an explicit value only if you need a specific number for logging.
- `steps` **(required)** – total optimization steps. WSDS scheduler checkpoints must fit within this range.
- `dtype = "bfloat16"` *(recommended)* – ensures compute happens in bf16 while master weights remain fp32.
- `seed` *(optional)* – falls back to 1234. `[molgen_data].seed` overrides dataloader randomness.
- `dataset = "molgen_jsonl"` and `dataset_path = "data:conformers_train"` **(required)** – tell the config manager to instantiate the custom JSONL dataloader and resolve the alias defined in `paths.yaml`.

### `[optimizer]`
- `name = "AdamW"` *(recommended)*.
- `lr` *(optional, defaults to WSDS base_lr when omitted)* – if you leave it unset, `custom_job_config` copies the effective LR into `wsds_scheduler.base_lr` / `lr_max`. Set it explicitly only when you need a non-default learning rate.
- `beta1`, `beta2`, `eps`, `weight_decay` *(optional)* – standard defaults.

### `[lr_scheduler]`
The base TorchTitan scheduler. When WSDS is enabled it still reads the warmup value.
- `warmup_steps` **(required)** – keep it equal to `[wsds_scheduler].warmup_steps`.
- `decay_type` *(optional)* – `"cosine"` works well; you can change it to `"constant"` when experimenting with WSDS-only schedules.

### `[wsds_scheduler]`
Controls the custom warmup/stable/decay schedule.
- `enable = true` *(recommended)* – turn off only if you want to rely solely on Titan’s scheduler.
- `warmup_steps` *(required)* – typically 500.
- `checkpoints` *(optional)* – cosmetic markers for when you expect stage transitions. Adjust when `training.steps` changes.
- `lr_max`, `lr_min` *(required)* – top/bottom LR bounds; `lr_max` normally matches `optimizer.lr`.
- `decay_frac` *(optional)* – fraction of total steps allocated to the decay phase.

### `[checkpoint]`
- `enable = true` *(required)*.
- `interval` *(recommended)* – 2 500 steps keeps a good balance between recovery granularity and disk usage.
- `keep_latest_k` *(recommended)* – set to 3–4 to limit storage.
- `folder = "checkpoint"` *(required)* – directory under `<run-name>` where Titan saves DCPs.
- `initial_load_path`, `initial_load_model_only`, `initial_load_in_hf` *(see `[molgen_run]`)* – these fields are overwritten automatically by `launch_qwen3_pretrain` depending on `init_mode`.
- `last_save_in_hf`, `save_hf_per_checkpoint` *(optional)* – set to `true` to emit Hugging Face safetensors when saving DCPs.
- `async_mode = "async"` *(optional)* – leave `async` for faster checkpointing; switch to `sync` when network contention causes issues.

### `[metrics]`
- `log_freq` *(recommended)* – set to 20 to log every 20 steps; lower values increase I/O.
- `enable_wandb = true` *(optional but recommended)* – uses the environment variables configured in `launch_torchtitan.sh`.
- `save_for_all_ranks = false` – keep false to avoid duplicate logs when running multi-rank inference.
- `save_tb_folder = "tb"` *(optional)* – subfolder under `outputs/pretrain_logs/<run-name>/tb`.

### `[parallelism]`
- `data_parallel_replicate_degree` *(required)* – number of replica groups.
- `data_parallel_shard_degree = -1` *(recommended)* – `-1` asks TorchTitan to infer the shard degree from `torchrun --nproc_per_node`. Set an explicit number only for debugging.
- `tensor_parallel_degree`, `fsdp_reshard_after_forward` *(optional)* – left at 1 / `"default"` for the current runs.

### `[validation]`
- `enable = true` *(recommended)* – turn off only when chasing throughput benchmarks.
- `dataset_path = "data:conformers_valid"` *(required)* – alias to the validation split.
- `local_batch_size`, `seq_len`, `freq`, `steps` *(optional)* – tweak according to available compute. Setting `steps = -1` runs through the entire validation set.

### `[molgen_data]`
Controls how `JsonlTaggedPackedDataset` behaves.
- `train_path_key`, `tokenizer_key` *(required)* – aliases resolved through `paths.yaml`.
- `min_emb_len = 16` *(recommended)* – drop units shorter than 16 characters.
- `shuffle_lines = true`, `infinite = true` *(recommended)* – keep the dataloader streaming.
- `seed` *(optional)* – defaults to `[training].seed` if omitted.
- `num_workers = 8`, `pin_memory = true`, `prefetch_factor = 2`, `persistent_workers = true` *(tunable)* – adjust based on CPU capacity.
- `lookahead_limit = 100` *(recommended)* – controls how many pending units the packer inspects when trying to fill each sequence.

---

## 2. Additional toggles & environment variables

- `WANDB_RUN_NAME` – set this when resuming an existing run so logs, checkpoints, and WandB all use the same run folder.
- `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_GROUP` – can be exported in the launch script or per invocation. Defaults are set in `scripts/launch_torchtitan.sh`.
- `WANDB_DIR` – automatically pointed at `wandb/<run-name>` by `launch_qwen3_pretrain`, but you can override it to stash offline runs elsewhere.
- `RUN_DESC` (launch script) – environment variable forwarded to `--run-desc`; effectively a friendlier name for the run.
- `MASTER_PORT`, `NGPU_PER_NODE` – configurable via the launcher; necessary when running multiple jobs per node.
- `export HF_DATASETS_OFFLINE=1` – optional for air‑gapped environments when you already copied tokenizer/model files locally.

---

## 3. Control Flow & File Map

1. **Launch script (`scripts/launch_torchtitan.sh`)**
   - Sets Slurm/torchrun variables, defaults `TRAIN_TOML`, exports WandB metadata, and finally invokes:
     ```bash
     torchrun ... -m molgen3D.training.pretraining.torchtitan_runner \
       --run-desc "${RUN_DESC}" \
       --train-toml "${TRAIN_TOML}"
     ```

2. **Runner (`src/molgen3D/training/pretraining/torchtitan_runner.py`)**
   - Parses CLI args (`tyro` dataclass `QwenPretrainRunConfig`).
   - Loads the TOML via `ConfigManager`.
   - Plans run directories (`_plan_run_layout`), resolves paths, loads tokenizer info, ensures log/checkpoint folders, and sets environment variables (RUN_NAME, WANDB_Dir, etc.).
   - Calls `Trainer(job_config)` from TorchTitan and handles HF checkpoint export flags.

3. **Custom job config (`src/molgen3D/training/pretraining/config/custom_job_config.py`)**
   - Extends TorchTitan’s `JobConfig` with `MolGenDataConfig`, `MolGenRunConfig`, and `MolGenCheckpointConfig`.
   - Resolves path aliases, injects dataloader options, and ensures WSDS scheduler picks up `optimizer.lr`.

4. **Custom model spec (`src/molgen3D/training/pretraining/torchtitan_model/qwen3_custom.py`)**
   - Registers the `molgen_qwen3` train spec: builds dataloaders (`build_molgen_dataloader` / validator), resizes embeddings to match the tokenizer, ties the LM head, and patches the checkpoint manager / metrics logger.

5. **Dataloader (`src/molgen3D/training/pretraining/dataprocessing/dataloader.py`)**
   - Implements `JsonlTaggedPackedDataset` and the Titan-compatible wrapper `TitanStatefulDataLoader`, which streams `[SMILES]…[/SMILES][CONFORMER]…[/CONFORMER]` units and packs them into fixed-length sequences with <|endoftext|> separators.

6. **Trainer (TorchTitan)**
   - Standard Titan `Trainer` handles forward/backward, FSDP sharding, WSDS scheduler steps, logging, and checkpointing. Our patches ensure `molgen_run` initializations, tokenizer overrides, and HF exports integrate seamlessly.

---

## 4. Training Objective

The model is optimized for next-token prediction over packed sequences of chemical units. Each packed sample is constructed as:

```
[SMILES]...[/SMILES][CONFORMER]...[/CONFORMER] <|endoftext|>
[SMILES]...[/SMILES][CONFORMER]...[/CONFORMER] <|endoftext|>
...
<|endoftext|> (padding until seq_len)
```

- `<|endoftext|>` serves both as a real separator between units and as the padding token.
- Labels are shifted left by one token: `label[t] = input[t+1]` for every position that has a successor. The last real token and the padded tail are set to `ignore_index = -100` so they do not contribute to the loss.
- Example (seq_len=12):

| Position | Input token                                  | Label token                                 |
|----------|----------------------------------------------|---------------------------------------------|
| 0        | `[SMILES]`                                   | Next token (`C`)                            |
| 1        | `C`                                          | Next token (`C`)                            |
| 2        | `C`                                          | `[/SMILES]`                                 |
| 3        | `[/SMILES]`                                  | `[CONFORMER]`                               |
| 4        | `[CONFORMER]`                                | `[H]`                                      |
| ...      | ...                                          | ...                                         |
| 8        | `<|endoftext|>` (separator)                  | First token of the next unit                |
| 11       | `<|endoftext|>` (padding)                    | `ignore_index (-100)` (no successor)        |

During training, TorchTitan flattens logits and labels, drops the `ignore_index` positions, and computes `CrossEntropyLoss(reduction="mean")`. Consequently:
- Frequent tokens like `[SMILES]` and `<|endoftext|>` contribute real gradients until they hit the padded tail.
- Rare tokens (e.g., unusual SMILES fragments) produce higher per-token losses, which explains occasional spikes in `global_max_loss` and `grad_norm`.

---

## 5. Putting It All Together

1. **Create or modify a TOML** by editing `src/molgen3D/config/pretrain/qwen3_06b.toml`. Choose `molgen_run.init_mode` (`scratch`, `hf_pretrain`, or `resume`) and adjust any hyperparameters or dataset aliases.
2. **Launch** via `scripts/launch_torchtitan.sh` (or your own `torchrun` command). Ensure the correct environment variables (WandB, run name) are exported if you need deterministic folder naming.
3. **Monitor** `outputs/pretrain_logs/<run-name>/runtime.log` for the startup banner and metrics, plus `wandb/<run-name>` for offline WandB files. Check `ckpts_root/qwen3_06b/<run-name>/checkpoint` for Titan DCPs and `_hf` folders when HF export is enabled.
4. **Resume** by reusing the run name, pointing `molgen_run.init_mode = "resume"` and `resume_run_path_tag` at the existing checkpoint directory, or leaving `init_mode = "scratch"` but exporting `WANDB_RUN_NAME=<existing>` if you just want Titan to pick up the latest checkpoint automatically.

Refer to `tests/dataloader/` for executable invariants that prove label shifting, metric reductions, and distributed sharding all behave as described. Any change to the dataloader or runner should keep those tests green.
