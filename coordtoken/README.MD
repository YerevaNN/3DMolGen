# CoordToken

This project implements coordinate tokenization using Finite Scalar Quantization (FSQ) for molecular data.

## FSQ Implementation

The FSQ is currently implemented in the iFSQ version by default. If you want to switch to the standard FSQ:

1. In `fsq.py`, comment out lines 126 and 127.
2. Uncomment line 128.

This changes the activation from `torch.sigmoid` to `torch.tanh` in the symmetry-preserving bound.

## Installation

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. (Optional) Set up a conda environment or adjust paths as needed.

## Usage

- Training: Adjust and run `submit_train.sh` or `python submit_train.py`.
- Evaluation: Adjust and run `python submit_eval.py --ckpt <checkpoint_path>`.

## Downloads

Authenticate with Hugging Face:

```bash
hf auth login
```

Download the tokenizer checkpoint into the repo root:

```bash
hf download FilyaGeikyan/CoordToken tokenizer.ckpt --repo-type model --local-dir .
```

Download the training dataset:

```bash
hf download FilyaGeikyan/CoordToken-data merged_train.csv.zst --repo-type dataset --local-dir .
```

Run evaluation with the downloaded checkpoint:

```bash
python submit_eval.py --ckpt tokenizer.ckpt
```

## Notes

- Adjust file paths (e.g., data directories, logs) for your environment.
- Ensure CUDA/GPU availability for training.
