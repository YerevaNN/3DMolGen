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

### Option 1: Development Installation (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd 3DMolGen
```

2. Create and activate a conda environment:

   **Using environment.yml (Recommended):**
   ```bash
   conda env create -f environment.yml
   conda activate 3dmolgen
   ```

   **Manual setup (Alternative):**
   ```bash
   # Create a new conda environment with Python 3.10
   conda create -n 3dmolgen python=3.10 -y
   conda activate 3dmolgen
   
   # Install PyTorch with CUDA support (adjust CUDA version as needed)
   pip install torch -U --index-url https://download.pytorch.org/whl/cu126
   
   # Install TorchTitan
   pip install -U torchtitan
   
   # Install core dependencies
   pip install transformers wandb loguru rdkit
   
   # Install additional dependencies from environment.yml if needed
   ```

3. Install the package in development mode:
```bash
pip install -e .
```

### Option 2: Minimal Installation

```bash
pip install molgen3D
```

**Note**: The minimal installation may not include all dependencies. For full functionality, use the development installation.

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
  --run-desc "my_run" \
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
