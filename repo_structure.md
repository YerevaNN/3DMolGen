# Repository Structure

This document explains how the 3DMolGen codebase is organized.

The key ideas:

- All importable code lives under `src/molgen3D`.
- LLM pretraining and GRPO training are first-class citizens with their own configs.
- Every serious run uses a **snapshot** into `outputs/` with its own code, config, logs, and checkpoints.
- Slurm/Submitit entrypoints are thin wrappers; logic lives in the package.

---

## Top-level Layout

```text
3DMolGen/
├─ configs/
│  ├─ llm/           # LLM pretraining configs (TorchTitan, architectures, data)
│  └─ grpo/          # GRPO / RL configs
├─ data/             # Local data (or symlinks); not importable code
├─ notebooks/        # Exploration / analysis notebooks (not part of the package)
├─ outputs/          # Run snapshots (gitignored)
├─ scripts/          # CLI and Slurm helpers (minimal, call into molgen3D)
├─ src/
│  └─ molgen3D/      # Main Python package
├─ tests/            # Unit / integration tests
├─ pyproject.toml    # Package + dependencies (Python-level)
└─ environment.yml   # Conda environment (system + CUDA + pinned stack)