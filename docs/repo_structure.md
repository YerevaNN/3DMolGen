# 3DMolGen Repository Layout

This note is a concise orientation guide for the 3DMolGen codebase so you can
quickly find the right module, script, or config when debugging or launching a
run.

Key principles:

- Importable Python lives under `src/molgen3D`; everything else is orchestration.
- Qwen3 TorchTitan pretraining and GRPO fine-tuning share the same package but
  have separate configs, entrypoints, and logging stacks.
- Reproducible runs are snapshot-based: every launch writes configs, logs, and
  checkpoints into a dedicated folder under `outputs/`.
- Shell scripts are intentionally thin so all logic stays versioned in Python.

---

## Top-level Layout

| Path | Purpose |
| --- | --- |
| `data/` | Local datasets or symlinks resolved via `paths.yaml`. Not importable. |
| `docs/` | Runbooks and reference material (this file lives here). |
| `notebooks/` | Exploration notebooks—never imported, often rely on ad-hoc deps. |
| `outputs/` | Git-ignored run artifacts (`pretrain_logs`, checkpoints, HF exports, WandB dumps). |
| `scripts/` | Launchers such as `launch_torchtitan_qwen3.sh` or GRPO CLI wrappers. |
| `src/` | Python package root; `pip install -e .` exposes it as `molgen3D`. |
| `tests/` | Unit and contract tests for dataloaders, schedulers, and config helpers. |
| `torchtitan/` | Upstream TorchTitan mirror used for local tweaks. |
| `environment.yml` | Conda environment with CUDA/toolchain pins. |
| `pyproject.toml` | PEP 621 metadata, runtime deps, and entrypoints. |

---

## `src/molgen3D` Package Map

```
src/molgen3D
├─ config/             # TOML + paths.yaml helpers and custom JobConfig extensions
├─ data_processing/    # JSONL packer, tokenizer utilities, validation helpers
├─ evaluation/         # Inference + scoring pipelines for downstream tasks
├─ training/
│  ├─ pretraining/     # Qwen3 TorchTitan runner, specs, schedulers, helpers
│  ├─ grpo/            # RLHF/GRPO trainer, configs, rewards, launch plumbing
│  └─ tokenizers/      # Tokenizer builders and conversions
├─ utils/              # Shared utilities (logging, distributed helpers)
└─ vq_vae/             # Legacy VQ-VAE experiments (kept for comparison)
```

Highlights:

- `training/pretraining/torchtitan_runner.py` is the canonical TorchTitan entry
  point. It resolves `config/pretrain/*.toml`, patches metrics for WandB, wires
  tokenizer overrides, and configures WSDS schedules.
- `config/pretrain/qwen3_06b.toml` is the default Qwen3 config. It feeds the
  custom job config defined in `training/pretraining/config/custom_job_config.py`.
- GRPO code mirrors Hugging Face trainer patterns (`training/grpo/grpo_trainer.py`)
  and exposes CLIs through `scripts/grpo_*.sh`.
- `evaluation/` contains deterministic inference entrypoints for scoring Qwen3
  checkpoints or GRPO outputs.

---

## Scripts & Automation

- `scripts/launch_torchtitan_qwen3.sh` configures Slurm resources, stamps a run
  name, writes a temporary TOML with the resolved `molgen_run.run_name`, and
  executes `torchrun -m molgen3D.training.pretraining.torchtitan_runner`.
- Additional launch scripts follow the same pattern: set environment variables →
  delegate into the Python package. This keeps experimentation reproducible and
  makes it easy to reconstruct commands from `outputs/pretrain_logs/<run>/runtime.log`.

---

## Outputs & Experiment Tracking

- `outputs/pretrain_logs/<run>/runtime.log` mirrors the TorchTitan console plus a
  JSON snapshot of the resolved config (`job_config.json`).
- `ckpts_root/qwen3_06b/<run>/` (configured via `paths.yaml`) stores Titan DCPs;
  HF exports land under the same tree when enabled.
- `wandb_runs/<run>/` is auto-populated when `metrics.enable_wandb = true`. The
  directory is pointed to by `WANDB_DIR` so offline syncs work out of the box.

---

## Docs, Notebooks, and Experiments

- Docs like `docs/pretraining_runbook.md` capture operational guidance (launch
  knobs, failure modes). Keep living documentation here alongside this structure
  overview.
- Jupyter notebooks in `notebooks/` are meant for exploratory analysis or plotting
  existing runs. Anything production-grade should migrate into `src/molgen3D`.
