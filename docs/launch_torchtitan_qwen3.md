# Launching Qwen3 Runs on TorchTitan

`scripts/launch_torchtitan_qwen3.sh` is a SLURM-friendly wrapper around
`molgen3D.training.pretraining.torchtitan_runner`. It sets reasonable defaults for WANDB,
derives a unique run name from `[job].description`, injects it into a temporary TOML, and launches
`torchrun` with the proper distributed settings pulled from SLURM.

## SLURM directives

The script defaults to:

```bash
#SBATCH --job-name=torchtitan-qwen3
#SBATCH --partition=a100    # override as needed
#SBATCH --gres=gpu:2        # matches nproc_per_node
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=~/slurm_jobs/titan/%j.out
#SBATCH --error=~/slurm_jobs/titan/%j.err
```

Adjust those at the top of the script if you need different resources.

## Environment variables

You can override the following before calling `sbatch`:

| Variable         | Default                                    | Purpose                                |
|------------------|--------------------------------------------|----------------------------------------|
| `TRAIN_TOML`     | `src/molgen3D/config/pretrain/qwen3_06b.toml` | Base config copied per run             |
| `RUN_DESC`       | extracted from `[job].description` in the TOML or filename stem | Shown in logs/W&B            |
| `RUN_NAME`       | auto-generated `YYMMDD-HHMM-<rand>-<desc>`  | Used for `molgen_run.run_name`         |
| `WANDB_ENTITY`   | `menuab_team`                              | W&B entity                             |
| `WANDB_PROJECT`  | `3dmolgen`                                 | W&B project                            |
| `WANDB_GROUP`    | `pretrain`                                 | W&B group                              |
| `WANDB_JOB_TYPE` | `pretrain`                                 | W&B job type                           |
| `WANDB_CONFIG`   | `{"run_type": "pretrain"}`                  | Additional run metadata                |
| `TORCH_COMPILE`  | `0`                                        | Torch compile toggle passed through    |

Distributed variables (`MASTER_ADDR`, `MASTER_PORT`, `NGPU_PER_NODE`, `NNODES`, `NODE_RANK`)
are derived from SLURM (`hostname`, `SLURM_GPUS_ON_NODE`, etc.) but can be overridden if needed.

## How the script works

1. **Set defaults**: ensures WANDB env vars and `TORCH_COMPILE` are exported, picks `TRAIN_TOML` and
   derives `RUN_DESC` from `[job].description`.
2. **Generate run name**: unless `RUN_NAME` is set, builds `YYMMDD-HHMM-<4hex>-<RUN_DESC>`.
3. **Populate distributed settings**:
   * `MASTER_ADDR` defaults to `hostname`.
   * `MASTER_PORT` picks a random port between 20000–40000 if unset.
   * `SLURM_GPUS_ON_NODE` drives `NGPU_PER_NODE` (and thus `torchrun --nproc_per_node`).
   * `SLURM_NNODES` and `SLURM_NODEID` populate `--nnodes` / `--node_rank`.
4. **Prepare temporary TOML**: copies the base TOML, injects `run_name = "<RUN_NAME>"` under
   `[molgen_run]`, and writes it to `/tmp/qwen3_runXXXXXX.toml`. The temp file is deleted on exit
   via a trap.
5. **Launch `torchrun`**: invokes

   ```bash
   torchrun \
     --nproc_per_node="${NGPU_PER_NODE}" \
     --master_port="${MASTER_PORT}" \
     --nnodes="${NNODES}" \
     --node_rank="${NODE_RANK}" \
     -m molgen3D.training.pretraining.torchtitan_runner \
     --train-toml "${TMP_TOML}"
   ```

## Example sbatch submission

```bash
sbatch \
  --gres=gpu:4 \
  --partition=h100 \
  --time=72:00:00 \
  --export=TRAIN_TOML=src/molgen3D/config/pretrain/qwen3_06b.toml,RUN_DESC=qwen3_0p6b_hf \
  scripts/launch_torchtitan_qwen3.sh
```

This will:

* create a unique run name (e.g. `251201-1302-abcd-qwen3_0p6b_hf`)
* copy the TOML, inject that `run_name`, and launch TorchTitan on 4 GPUs with the same launcher
  logic as `launch_torchtitan.sh`.
