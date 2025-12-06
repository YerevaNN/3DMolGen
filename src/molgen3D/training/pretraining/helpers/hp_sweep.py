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
GLOBAL_BATCH_SIZES: list[int] = [96, 144, 192]

# Base TOML to copy/override per run (relative to repo root or absolute).
TRAIN_TOML: Path = Path("src/molgen3D/config/pretrain/qwen3_06b.toml")

# Slurm launcher script (relative to repo root or absolute).
LAUNCHER: Path = Path("scripts/launch_torchtitan_qwen3.sh")

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
    for parent in [start] + list(start.parents):
        if (parent / ".git").is_dir() and (parent / "src" / "molgen3D").exists():
            return parent
    raise RuntimeError("Could not locate repo root (expected .git and src/molgen3D).")


def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve())
    train_toml = TRAIN_TOML if TRAIN_TOML.is_absolute() else repo_root / TRAIN_TOML
    launcher = LAUNCHER if LAUNCHER.is_absolute() else repo_root / LAUNCHER

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
            # Update global batch size
            cfg.setdefault("training", {})["global_batch_size"] = int(gb)
            desc = f"lr{lr}_gb{gb}"
            cfg.setdefault("job", {})["description"] = desc

            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".toml", delete=False
            ) as tmp:
                tomli_w.dump(cfg, tmp)
                tmp_path = Path(tmp.name)

            env = os.environ.copy()
            env.update(
                {
                    "TRAIN_TOML": str(tmp_path),
                    "RUN_DESC": desc,
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

