# 3DMolGen

3DMolGen is a Python package for 3D molecular generation using language models. It provides tools for training and inference of molecular conformer generation models.

## Features

- Training of language models for 3D molecular generation
- Support for various molecular formats (SMILES, 3D coordinates)
- Integration with Weights & Biases for experiment tracking
- SLURM support for distributed training
- Comprehensive evaluation metrics

## Installation

### Option 1: Using pip (minimal installation)

```bash
pip install molgen3D
```

### Option 2: Development installation (recommended for contributors)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/molgen3D.git
cd molgen3D
```

2. Create and activate a conda environment:
```bash
conda env create -f requirements.yml
conda activate 3dmolgen
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Training

```python
from molgen3D.grpo.grpo_hf.config import Config
from molgen3D.grpo.grpo_hf.run_hf import main

# Load configuration
config = Config.from_yaml("path/to/config.yaml")

# Run training
main(config, enable_wandb=True)
```

### Configuration

The package uses YAML configuration files. Here's an example:

```yaml
model:
  checkpoint_path: "path/to/model"
  tokenizer_path: "path/to/tokenizer"
  mol_tags: ["[SMILES]", "[/SMILES]"]
  conf_tags: ["[CONFORMER]", "[/CONFORMER]"]

generation:
  max_completion_length: 1000
  temperature: 1.0
  do_sample: true

grpo:
  output_dir: "./outputs"
  learning_rate: 0.0001
  batch_size: 2
  num_generations: 16
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

The project uses:
- Black for code formatting
- Flake8 for linting
- MyPy for type checking

To run all checks:
```bash
black .
flake8
mypy .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{molgen3d2024,
  author = {Your Name},
  title = {3DMolGen: 3D Molecular Generation using Language Models},
  year = {2024},
  url = {https://github.com/yourusername/molgen3D}
}
```
