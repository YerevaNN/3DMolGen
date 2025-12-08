"""
Utility to submit a grid of hyperparameter runs (LR × global batch size) as
separate Slurm jobs using the TorchTitan Qwen3 launcher.

Usage example:
    python -m molgen3D.training.pretraining.helpers.hp_sweep \
        --lrs 1e-4 3e-4 \
        --micro-batches 2 4 8 \
        --grad-accum 1 2 \
        --train-toml src/molgen3D/config/pretrain/qwen3_06b.toml
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    import tomli as tomllib  # type: ignore[assignment]

try:
    import tomli_w  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - explicit error
    raise SystemExit("tomli_w is required to run hp_sweep") from exc

# ---- Sweep configuration (edit these) ---------------------------------------------------------

# Learning rates to try.
LRS: list[float] = [1e-4, 2e-4, 3e-4, 4e-4]

# Global batch sizes to try (TorchTitan interprets this as total across ranks).
GLOBAL_BATCH_SIZES: list[int] = [96, 144]

# Training steps keyed by global batch size.
TRAIN_STEPS_BY_GB: dict[int, int] = {
    144: 1500,
    96: 1000,
}

# Warmup steps for the scheduler.
WARMUP_STEPS: int = 200

# Directory to store generated sweep configs (relative to repo root or absolute).
SWEEP_CONFIG_DIR: Path = Path("outputs/hp_sweep_configs")

# Base TOML to copy/override per run (relative to repo root or absolute).
TRAIN_TOML: Path = Path("src/molgen3D/config/pretrain/qwen3_06b.toml")

# Slurm launcher script (relative to repo root or absolute).
LAUNCHER: Path = Path("scripts/launch_torchtitan_qwen3_sweep.sh")

# Set to True to print submissions without calling sbatch.
DRY_RUN: bool = False

# Extra env overrides passed to sbatch (edit as needed).
EXTRA_ENV: dict[str, str] = {
    # "WANDB_GROUP": "lr_sweep",
}


def _find_repo_root(start: Path) -> Path:
    """
    Walk upward from `start` until we find a directory containing both `.git`
    and `src/molgen3D`. Raises if not found to avoid surprising relative-path
    behavior in batch jobs.
    """
    env_root = os.environ.get("MOLGEN3D_REPO_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "src" / "molgen3D").exists():
            return candidate
        raise RuntimeError(f"MOLGEN3D_REPO_ROOT does not exist or is missing src/molgen3D: {candidate}")

    for parent in [start] + list(start.parents):
        if (parent / ".git").is_dir() and (parent / "src" / "molgen3D").exists():
            return parent
    cwd = Path.cwd().resolve()
    if (cwd / "src" / "molgen3D").exists():
        return cwd
    raise RuntimeError(
        "Could not locate repo root (expected .git and src/molgen3D). "
        "Set MOLGEN3D_REPO_ROOT=/path/to/3DMolGen if running from an installed wheel."
    )


def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve())
    train_toml = TRAIN_TOML if TRAIN_TOML.is_absolute() else repo_root / TRAIN_TOML
    launcher = LAUNCHER if LAUNCHER.is_absolute() else repo_root / LAUNCHER
    sweep_cfg_dir = (
        SWEEP_CONFIG_DIR if SWEEP_CONFIG_DIR.is_absolute() else repo_root / SWEEP_CONFIG_DIR
    )
    sweep_cfg_dir.mkdir(parents=True, exist_ok=True)

    if not train_toml.exists():
        raise FileNotFoundError(f"train_toml not found: {train_toml}")
    if not launcher.exists():
        raise FileNotFoundError(f"launcher script not found: {launcher}")

    base_cfg = tomllib.loads(train_toml.read_text())

    extra_env = dict(EXTRA_ENV)

    submissions = 0
    for lr in LRS:
        for gb in GLOBAL_BATCH_SIZES:
            cfg = tomllib.loads(train_toml.read_text())
            # Update optimizer.lr
            cfg.setdefault("optimizer", {})["lr"] = float(lr)

            # Update training settings (batch size + steps)
            training_cfg = cfg.setdefault("training", {})
            training_cfg["local_batch_size"] = 4
            training_cfg["global_batch_size"] = int(gb)
            if gb in TRAIN_STEPS_BY_GB:
                training_cfg["steps"] = int(TRAIN_STEPS_BY_GB[gb])
            training_steps = int(training_cfg.get("steps", 0)) or None

            # Ensure validation batch size
            cfg.setdefault("validation", {})["local_batch_size"] = 4

            # Update scheduler warmup/decay checkpoints
            wsds_cfg = cfg.setdefault("wsds_scheduler", {})
            wsds_cfg["warmup_steps"] = int(WARMUP_STEPS)
            if training_steps:
                # Decay at the total planned steps for this run
                wsds_cfg["checkpoints"] = [int(training_steps)]

            # Disable checkpointing for sweep runs
            cfg.setdefault("checkpoint", {})["enable"] = False
            desc = f"lr{lr}_gb{gb}"
            cfg.setdefault("job", {})["description"] = desc

            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".toml",
                delete=False,
                dir=sweep_cfg_dir,
                prefix=f"{desc}_",
            ) as tmp:
                tomli_w.dump(cfg, tmp)
                tmp_path = Path(tmp.name)

            env = os.environ.copy()
            env.update(
                {
                    "TRAIN_TOML": str(tmp_path),
                    "RUN_DESC": desc,
                    "MOLGEN3D_REPO_ROOT": str(repo_root),
                }
            )
            env.update(extra_env)

            cmd = ["sbatch", str(launcher)]
            if DRY_RUN:
                print(f"[dry-run] {' '.join(cmd)} TRAIN_TOML={tmp_path} RUN_DESC={desc}")
            else:
                subprocess.run(cmd, check=True, cwd=repo_root, env=env)
                print(f"submitted: lr={lr} gb={gb} tmp={tmp_path}")
            submissions += 1

    print(f"planned submissions: {submissions}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)

