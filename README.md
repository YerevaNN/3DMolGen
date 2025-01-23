# 3DMOLGEN

## Overview

This project is focused on building a system with functionalities such as data processing, model training, evaluation, and utility functions. The structure of the project is organized into different modules:

- **vq_vae/**: The main module for the VQ-VAE (Vector Quantized Variational Autoencoder) implementation.
- **data_processing/**: Scripts for processing and preparing data for VQ-VAE model training.
- **train/**: Contains code related to training the VQ-VAE model.
- **evaluation/**: Includes scripts for evaluating both the VQ-VAE model and the generated molecules by main LLM.
- **scripts/**: Utility scripts for different operations.
- **tests/**: Unit tests for ensuring code quality.
- **utils/**: Helper functions, utility scripts and simple data converters.

## Installation

Create a conda environment and install the required packages using the following commands:

```bash
conda create -n 3dmolgen python=3.11
conda activate 3dmolgen
pip install -r requirements.txt

conda install -c conda-forge rdkit openbabel
```

## Usage and Contributing

We use `black` formatter for code formatting. Please make sure to install the pre-commit hooks by running the following command:

```bash
chmod +x scripts/install_hooks.sh
./scripts/install_hooks.sh
```

## Running the tests

To run the tests, execute the following command:

```bash
pytest tests/
# or
pytest tests/path_to_test_file.py
```
