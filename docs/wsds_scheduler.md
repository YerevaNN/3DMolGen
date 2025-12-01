# WSDS Scheduler Guide

## How It Works

The Warmup–Stable–Decay–Stable (WSDS) scheduler simulates multiple short training runs within a single continuous job. Each checkpoint in the list marks the end of a "mini run" cycle:

1. **Warmup** – Global linear ramp from 0 → `lr_max` over `warmup_steps`.
2. **Stable phase** – Maintain `lr_max` until the decay window for the current checkpoint begins.
3. **Decay phase** – Linearly anneal from `lr_max` → `lr_min` so the learning rate reaches `lr_min` exactly at the checkpoint step.
4. **Reset** – Immediately jump back to `lr_max` for the next checkpoint cycle (after the last checkpoint, LR stays at `lr_min`).

The scheduler automatically integrates with TorchTitan when `[wsds_scheduler].enable = true` is set in a TOML config that uses `molgen3D.training.pretraining.torchtitan_model.qwen3_custom`. The `custom_job_config.JobConfig` automatically copies `[optimizer].lr` into `wsds_scheduler.base_lr` so warmup uses the same starting point.

## How to Call It

### Configuration

Add the `[wsds_scheduler]` section to your TorchTitan TOML config file:

```toml
[wsds_scheduler]
enable = true
warmup_steps = 400
checkpoints = [1000, 2000, 3000, 4000]
lr_max = 2e-4
lr_min = 2e-5
decay_frac = 0.1
```

**Required fields:**
- `enable` – Set to `true` to activate WSDS scheduling
- `warmup_steps` – Number of steps for global linear warmup
- `checkpoints` – Sorted list of step numbers where decay phases end
- `lr_max` – Maximum learning rate (typically matches `[optimizer].lr`)
- `lr_min` – Minimum learning rate at end of each decay phase

**Decay length (choose one):**
- `decay_steps` – Fixed number of steps for every decay window (Priority 1)
- `decay_frac` – Fraction of checkpoint step (T × frac) when `decay_steps` is unset

**Optional:**
- `base_lr` – Override for LambdaLR normalization (defaults to `[optimizer].lr`)

### Required TOML Sections

The WSDS scheduler requires these sections to be configured:

```toml
[job]
custom_config_module = "molgen3D.training.pretraining.config.custom_job_config"

[experimental]
custom_import = "molgen3D.training.pretraining.torchtitan_model.qwen3_custom"

[optimizer]
lr = 2e-4  # This value is automatically copied to wsds_scheduler.base_lr
```

## Example Run Cases

### Example 1: Fractional Decay (10% of each checkpoint)

```toml
[training]
steps = 4000

[optimizer]
lr = 2e-4

[wsds_scheduler]
enable = true
warmup_steps = 400
checkpoints = [1000, 2000, 3000, 4000]
lr_max = 2e-4
lr_min = 2e-5
decay_frac = 0.1
```

**Behavior:**
- Steps 0-400: Warmup from 0 → 2e-4
- Steps 400-900: Stable at 2e-4
- Steps 900-1000: Decay from 2e-4 → 2e-5 (100 steps = 10% of 1000)
- Step 1000: Reset to 2e-4
- Steps 1000-1800: Stable at 2e-4
- Steps 1800-2000: Decay from 2e-4 → 2e-5 (200 steps = 10% of 2000)
- Step 2000: Reset to 2e-4
- Steps 2000-2700: Stable at 2e-4
- Steps 2700-3000: Decay from 2e-4 → 2e-5 (300 steps = 10% of 3000)
- Step 3000: Reset to 2e-4
- Steps 3000-3600: Stable at 2e-4
- Steps 3600-4000: Decay from 2e-4 → 2e-5 (400 steps = 10% of 4000)
- After step 4000: Stay at 2e-5

### Example 2: Fixed Decay Steps (100 steps per checkpoint)

```toml
[training]
steps = 10000

[optimizer]
lr = 2e-4

[wsds_scheduler]
enable = true
warmup_steps = 400
checkpoints = [1250, 2500, 5000, 10000]
lr_max = 2e-4
lr_min = 2e-5
decay_steps = 100
```

**Behavior:**
- Steps 0-400: Warmup from 0 → 2e-4
- Steps 400-1150: Stable at 2e-4
- Steps 1150-1250: Decay from 2e-4 → 2e-5 (100 steps)
- Step 1250: Reset to 2e-4
- Steps 1250-2400: Stable at 2e-4
- Steps 2400-2500: Decay from 2e-4 → 2e-5 (100 steps)
- Step 2500: Reset to 2e-4
- Steps 2500-4900: Stable at 2e-4
- Steps 4900-5000: Decay from 2e-4 → 2e-5 (100 steps)
- Step 5000: Reset to 2e-4
- Steps 5000-9900: Stable at 2e-4
- Steps 9900-10000: Decay from 2e-4 → 2e-5 (100 steps)
- After step 10000: Stay at 2e-5

### Example 3: Real Training Configuration

Based on `src/molgen3D/config/pretrain/qwen3_06b_wsds.toml`:

```toml
[training]
steps = 4000
local_batch_size = 2

[optimizer]
lr = 2e-4

[lr_scheduler]
warmup_steps = 400  # Should match wsds_scheduler.warmup_steps

[wsds_scheduler]
enable = true
warmup_steps = 400
checkpoints = [1000, 2000, 3000, 4000]
lr_max = 2e-4
lr_min = 2e-5
decay_frac = 0.1

[checkpoint]
enable = true
interval = 2000  # TorchTitan checkpoint interval (independent of WSDS checkpoints)
```

**Notes:**
- `[lr_scheduler].warmup_steps` should match `[wsds_scheduler].warmup_steps` for consistency
- WSDS `checkpoints` are learning rate milestones, not TorchTitan checkpoint save points
- TorchTitan checkpoint `interval` is independent and controls when model checkpoints are saved
