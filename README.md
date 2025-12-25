# 3DMolGen

3DMolGen is a Python package for 3D molecular conformer generation using language models. It provides a complete training and inference pipeline for autoregressive models that generate molecular conformations by predicting 3D coordinates alongside SMILES topology.

## Overview

3DMolGen uses an **enriched SMILES representation** that embeds 3D coordinates directly into the molecular string format. This allows language models to:
- Copy SMILES topology tokens verbatim
- Predict only the 3D coordinates (`<x,y,z>` blocks) for each atom
- Generate chemically valid conformers in a single autoregressive pass

The project supports:
- **Pretraining**: Large-scale pretraining of Qwen3-based models using TorchTitan
- **Fine-tuning**: GRPO (Group Relative Policy Optimization) for reward-guided optimization
- **Evaluation**: Comprehensive evaluation pipelines including PoseBusters validation

## Features

- **Enriched SMILES Format**: Lossless encoding of molecular topology and 3D coordinates in a text format
- **TorchTitan Integration**: Distributed pretraining using Meta's TorchTitan framework
- **Custom Dataloader**: Efficient sequence packing with deterministic shuffling and resumability
- **WSDS Scheduler**: Custom warmup-stable-decay learning rate schedule
- **GRPO Training**: Reinforcement learning fine-tuning with multi-component rewards
- **Weights & Biases Integration**: Comprehensive experiment tracking
- **SLURM Support**: Production-ready distributed training on HPC clusters
- **Evaluation Tools**: PoseBusters validation and molecular property scoring

## Installation

### Quick Start (One Command)

```bash
git clone <repository-url>
cd 3DMolGen
./setup.sh
```

This installs everything you need. See details below.

---

### Understanding the Stack

We use a **conda + uv hybrid approach**:

| Tool | Role | Why |
|------|------|-----|
| **Conda** | Environment & Python management | Manages Python 3.10, system deps like rdkit, CUDA libs |
| **uv** | Fast package installer | 10-100x faster than pip, caches aggressively |

**What is uv?** [uv](https://docs.astral.sh/uv/) is a Rust-based Python package manager from Astral (the Ruff team). It's a drop-in pip replacement that's dramatically faster - critical for ephemeral cluster environments where we reinstall packages every job.

### Version Matrix

| Component | Version | Source |
|-----------|---------|--------|
| Python | 3.10.x | Conda |
| PyTorch | 2.9.1+cu128 | [pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128) |
| Flash Attention | 2.8.3+cu128torch2.9 | [Prebuilt wheel](https://github.com/mjun0812/flash-attention-prebuild-wheels) |
| CUDA | 12.8 | System drivers |
| transformers | ≥4.50.0 | PyPI |
| trl | ≥0.15.0 | PyPI |
| torchtitan | ≥0.2.0 | PyPI |

---

### Option 1: Automated Setup (Recommended)

The `setup.sh` script handles everything:

```bash
./setup.sh              # Full installation
./setup.sh --dev        # Include dev tools (pytest, black, etc.)
./setup.sh --verify     # Just verify existing installation
```

**What it does:**
1. Installs uv if not present (via curl)
2. Creates/activates conda environment `3dmolgen` with Python 3.10
3. Uses `uv pip install` for fast package installation
4. Installs PyTorch 2.9.1+cu128 from official CUDA wheel index
5. Installs Flash Attention 2.8.3 from prebuilt wheel
6. Installs molgen3D in editable mode
7. Runs verification checks

### Option 2: Manual Installation

Step-by-step if you want more control:

```bash
# 1. Create conda environment with Python 3.10 and rdkit
conda create -n 3dmolgen python=3.10 rdkit -c conda-forge -y
conda activate 3dmolgen

# 2. Install uv (fast pip replacement)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 3. Install PyTorch with CUDA 12.8 (using uv for speed)
uv pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Install Flash Attention (prebuilt wheel)
# Option A: From local copy (YerevaNN cluster)
uv pip install /nfs/ap/mnt/sxtn2/chem/wheels/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl

# Option B: From GitHub
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl

# 5. Install remaining dependencies and molgen3D
uv pip install -e ".[dev]"
```

### Verifying Installation

```bash
python verify_env.py
```

Expected output:
```
==================================================================
3DMolGen Environment Verification
==================================================================
  [PASS] PyTorch              v2.9.1+cu128             (CUDA 12.8, 8x NVIDIA H100)
  [PASS] Flash Attention      v2.8.3+cu128torch2.9     (flash_attn_func available)
  [PASS] transformers         v4.57.0
  [PASS] trl                  v0.15.0
  [PASS] torchtitan           v0.2.0
  ...
==================================================================
All checks passed! Environment is ready.
==================================================================
```

### Flash Attention Notes

Flash Attention 2 requires prebuilt wheels (compilation takes 2+ hours without ninja). Our wheel is for:
- Python 3.10
- PyTorch 2.9
- CUDA 12.8

**Local copy available at:** `/nfs/ap/mnt/sxtn2/chem/wheels/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl`

**For other configurations:** Download from [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels). See [`docs/python_cuda_packaging_guide.md`](docs/python_cuda_packaging_guide.md) for the wheel compatibility matrix.

### Slurm Job Template

For ephemeral environments on the new DGX cluster:

```bash
#!/bin/bash
#SBATCH --job-name=3dmolgen
#SBATCH --partition=h100
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Fast setup (uv caching makes warm installs <30s)
cd /path/to/3DMolGen
./setup.sh

# Run training
conda activate 3dmolgen
torchrun --nproc_per_node=8 \
  -m molgen3D.training.pretraining.torchtitan_runner \
  --train-toml src/molgen3D/config/pretrain/qwen3_06b.toml
```

## Project Structure

```
3DMolGen/
├── src/molgen3D/          # Main Python package
│   ├── config/            # TOML configs, path resolution (paths.yaml)
│   ├── data_processing/   # SMILES encoding/decoding, preprocessing
│   ├── evaluation/        # Inference and scoring pipelines
│   ├── training/
│   │   ├── pretraining/   # TorchTitan runner, Qwen3 custom spec
│   │   ├── grpo/          # GRPO trainer and rewards
│   │   └── tokenizers/    # Tokenizer builders
│   ├── utils/             # Shared utilities
│   └── vq_vae/            # Legacy VQ-VAE experiments
├── docs/                  # Comprehensive documentation
├── scripts/               # Launch scripts for SLURM
├── notebooks/             # Exploration notebooks
├── tests/                 # Unit and integration tests
└── outputs/               # Run artifacts (checkpoints, logs, WandB)
```

See [`docs/repo_structure.md`](docs/repo_structure.md) for detailed layout information.

## Quick Start

### Pretraining

1. **Configure your run**: Edit `src/molgen3D/config/pretrain/qwen3_06b.toml`:
   - Set `molgen_run.init_mode` to `"scratch"`, `"hf_pretrain"`, or `"resume"`
   - Configure dataset paths via `paths.yaml` aliases
   - Adjust hyperparameters (learning rate, batch size, steps, etc.)

2. **Launch training**:
```bash
sbatch scripts/launch_torchtitan_qwen3.sh
```

Or manually:
```bash
torchrun --nproc_per_node=2 \
  -m molgen3D.training.pretraining.torchtitan_runner \
  --train-toml src/molgen3D/config/pretrain/qwen3_06b.toml
```

3. **Monitor**: Check `outputs/pretrain_logs/<run-name>/runtime.log` and WandB dashboard.

See [`docs/pretraining_runbook.md`](docs/pretraining_runbook.md) for detailed configuration options and [`docs/launch_torchtitan_qwen3.md`](docs/launch_torchtitan_qwen3.md) for launch script details.

### GRPO Fine-tuning

```python
from molgen3D.training.grpo.train_grpo_model import main
from molgen3D.training.grpo.config import Config

# Load configuration
config = Config.from_yaml("path/to/grpo_config.yaml")

# Run GRPO training
main(config, enable_wandb=True)
```

### Inference and Evaluation

```python
from molgen3D.evaluation.inference import generate_conformers
from molgen3D.evaluation.posebusters_check import validate_conformers

# Generate conformers for SMILES
smiles = "CCO"  # Ethanol
conformers = generate_conformers(
    model_path="path/to/checkpoint",
    smiles_list=[smiles],
    num_samples=10
)

# Validate with PoseBusters
results = validate_conformers(conformers)
```

## Data Format

3DMolGen uses an **enriched SMILES format** that embeds 3D coordinates:

```
[C]<1.25,0.03,-0.94>[C]<2.55,0.11,-0.44>(=[O]<3.42,0.85,-1.22>)[N+H2]<1.82,-1.40,0.12>
```

Each atom descriptor `[Element...]` is immediately followed by its 3D coordinates `<x,y,z>`. This format:
- Preserves full chemical fidelity (stereochemistry, charges, isotopes)
- Allows models to copy topology and predict only coordinates
- Enables lossless round-trip encoding/decoding

See [`docs/enriched_smiles.md`](docs/enriched_smiles.md) for the complete specification.

## Training Data

The dataloader expects JSONL files with packed sequences:
- Each line contains `[SMILES]...[/SMILES][CONFORMER]...[/CONFORMER]` units
- Sequences are packed to `seq_len` (default 2048) with `<|endoftext|>` separators
- Supports deterministic shuffling, resumability, and distributed sharding

See [`docs/dataloader.md`](docs/dataloader.md) for dataloader details and diagnostic tools.

## Configuration

3DMolGen uses TOML configuration files with path aliases resolved via `src/molgen3D/config/paths.yaml`. Key sections:

- `[molgen_run]`: Initialization mode, tokenizer selection, run naming
- `[training]`: Sequence length, batch size, optimizer settings
- `[wsds_scheduler]`: Custom warmup-stable-decay schedule
- `[checkpoint]`: Checkpointing frequency and format
- `[metrics]`: WandB and logging configuration

Example config: `src/molgen3D/config/pretrain/qwen3_06b.toml`

## Documentation

- [`docs/repo_structure.md`](docs/repo_structure.md): Codebase organization
- [`docs/pretraining_runbook.md`](docs/pretraining_runbook.md): Complete pretraining guide
- [`docs/dataloader.md`](docs/dataloader.md): Dataloader architecture and usage
- [`docs/enriched_smiles.md`](docs/enriched_smiles.md): Data format specification
- [`docs/launch_torchtitan_qwen3.md`](docs/launch_torchtitan_qwen3.md): Launch script reference

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

The project uses:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

To run all checks:
```bash
black .
flake8
mypy .
```

### Diagnostic Tools

- `smoke_test_dataloader.py`: Validate dataloader with sample batches
- `count_tokens.py`: Estimate dataset token counts and packing efficiency

## Key Components

### Custom Dataloader (`JsonlTaggedPackedDataset`)
- Efficient sequence packing with lookahead optimization
- Deterministic shuffling and distributed sharding
- Stateful resumability for checkpointing

### WSDS Scheduler
- Warmup → Stable → Decay learning rate schedule
- Configurable checkpoints and decay fractions
- Automatic LR synchronization with optimizer

### TorchTitan Integration
- Custom Qwen3 model spec with tokenizer resizing
- Patched checkpoint manager for HF export
- Metric logging integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

Please ensure all tests pass and documentation is updated.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{molgen3d2024,
  author = {Your Name},
  title = {3DMolGen: 3D Molecular Generation using Language Models},
  year = {2024},
  url = {https://github.com/yourusername/3DMolGen}
}
```

## Acknowledgments

- Built on [TorchTitan](https://github.com/pytorch/torchtitan) for distributed training
- Uses [Qwen3](https://github.com/QwenLM/Qwen2.5) as the base language model
- Integrates [PoseBusters](https://github.com/maabuu/posebusters) for conformer validation
