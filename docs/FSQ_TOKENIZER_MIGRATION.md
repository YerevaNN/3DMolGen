# Implemented atom/register interface

Format version: `atom12_reg1_prefix_v1`. Selected checkpoint: the immutable epoch-2 artifact linked in [the handoff](../COLLEAGUE_HANDOFF.md).

## Contract

For N atoms, serialize N+1 codes: a leading molecular code, then the graph string with one code immediately after each atom. For example:

```text
[SMILES]O=CO[/SMILES][CONFORMER]<317>[C]<42>(=[O]<901>)[O]<0>[/CONFORMER]
```

The leading code and atom codes are each ordinary added Qwen tokens `<0>` ... `<4095>`. They share IDs/vocabulary; no register marker is added. The leading code is a trained next-token target. Structural tokens remain graph context and receive no geometry code. A graph's exact atom order remains the order in the encoded string.

Inside the neural model, the register is appended after the padded graph width, not prepended. The adapter translates between this dense layout and the leading text code. Missing/extra/out-of-range codes fail. Code ID zero is valid; absent punctuation geometry becomes a zero vector, not quantized ID zero. Checkpoints load strictly with config, vocabulary and SHA-256 checks.

The model and quantizer are copied from the selected run's frozen source (only the model import changes). Centering, masks, atom order, graph identities and register placement match training. CPU fp32 is supported; GPU inference supports fp32 or bf16. Quantization remains fp32 internally. Numerical precision/batch kernels may affect codes close to quantization boundaries; keep the recorded preprocessing precision fixed across a dataset.

`decode_cartesian_v2` now sanitizes/assigns input SMILES stereochemistry while retaining explicit H atom positions. The old `coords_rmsd` helper no longer aligns/mutates its input molecule in place; its aligned comparison semantics are otherwise unchanged. Published tokenizer reconstruction metrics use a separate unaligned evaluator. None of these fixes recovers source chemistry lost through historical decorative atom brackets.

## Coordinate CSV to Qwen records

Use Python 3.10 and the repository environment (see `docs/CURRENT_TRAINING_HANDOFF.md`). The selected codec was checked with PyTorch 2.6.0, RDKit 2025.09.3 and Transformers 4.57.6; packed-loader checks additionally used the installed TorchTitan environment. `einops` is a runtime dependency. The new assets are already committed at:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
TOK="$PWD/src/molgen3D/training/tokenizers/Qwen3_tokenizer_coordtoken_atom12_reg1"
```

`build_coordtoken_qwen_tokenizer.py` reproducibly builds these assets from the existing Qwen FSQ tokenizer. Geometry token IDs are unchanged, but their backend `special` flags are cleared: generation with `skip_special_tokens=True` must retain geometry codes. Existing legacy assets are untouched.

For a new output directory, convert each existing split independently:

```bash
python -m molgen3D.data_processing.preprocess_coordtoken_csv   --input /path/to/merged_train.csv --output /path/to/new_encoded_train   --split train --tokenizer "$TOK" --device cuda --precision bf16   --batch-size 64 --num-shards 1 --shard-index 0 --seq-len 4096
```

For many independent GPU workers, submit `scripts/run_coordtoken_csv.sbatch INPUT OUTPUT SPLIT TOKENIZER_DIR NUM_SHARDS` as an array `0..NUM_SHARDS-1`, without a concurrency cap. This is inference preprocessing, one GPU per shard, not a change to model-training batch/GPU count. Choose the shard count for the allocation/time limit and use the same settings for every shard. Set the Python environment/PYTHONPATH before submission. No array has been submitted by this code publication.

Shards cover disjoint row-start byte ranges, preserve source columns and byte offsets, and write version/checkpoint metadata per record. Quoted single-line CSV is supported; embedded newlines are rejected. Input CSVs must be immutable. Unsupported/nonfinite/over-context records stop the shard; they are not dropped or truncated. Partial/failed outputs are not usable training data. Use a fresh directory after a failed attempt. Source stat checks are not full-input hashing; final-release SHA evidence is separate.

After all shards finish, verify counts and bytes:

```bash
python -m molgen3D.data_processing.assemble_coordtoken /path/to/new_encoded_train   --expected-rows 189792028
```

Use the actual source count for other corpora/splits. This creates `coordtoken_manifest.json` only after complete shard coverage, matching provenance, row counts and output hashes pass. Train/val/test are never re-split here. The packed loader shuffles row indices using its configured seed and epoch handling, keeping each molecular unit intact. Do not externally rewrite/shuffle JSONLs after assembly without rebuilding the manifest.

## Qwen training and export

Generate a separate run config; do not reuse old geometry-trained checkpoints as an unchanged resume. `--steps` and batch are explicit because the atom-only stream has a different token budget. The example is configuration syntax, not a tuned new Qwen recommendation:

```bash
python scripts/configure_coordtoken_qwen.py --size 06b   --data /path/to/new_encoded_train --tokenizer "$TOK"   --base-model /path/to/prepared/Qwen3-0.6B-Base   --output /path/to/new_coordtoken_qwen.toml --steps 10000   --global-batch 32 --local-batch 2 --gpus 8 --seq-len 4096 --lr 0.0008

TRAIN_TOML=/path/to/new_coordtoken_qwen.toml sbatch --nodes=1 --gres=gpu:8   scripts/launch_torchtitan_qwen3.sh
```

`--size 17b` selects the corresponding 1.7B template and requires its matching prepared base model. Global batch must be divisible by local batch times GPU count. Eight GPUs are one-node data parallelism here; do not use the launcher's historical three-node default with an eight-GPU config. The script preserves optimizer/scheduler implementation, requires a verified train manifest, checks tokenizer/checkpoint identity, and refuses truncation. Count actual packed-token consumption before choosing a full-duration run. Hash verification reads all encoded shard bytes at startup.

The existing DCP-to-HF exporter, `molgen3D.training.pretraining.helpers.convert_qwen3_dcp_to_hf`, now copies the versioned tokenizer assets and records the coordinate format/checkpoint/tokenizer hashes in the exported Qwen config. It derives these from the training config. New-format inference refuses an unmarked/legacy Qwen checkpoint or mismatched tokenizer. Never add metadata to an old trained model to bypass this guard.

```bash
python -m molgen3D.evaluation.inference --device local --test_set distinct   --coordtoken-format atom12_reg1_prefix_v1   --model_path /path/to/exported_new_qwen   --tokenizer_path /path/to/exported_new_qwen   --test_data_path /path/to/existing_prompt_pickle   --results_path /path/to/new_results --run_name coordtoken_atom12_reg1
```

The frozen coordinate checkpoint defaults to the selected shared artifact; `--coordtoken-checkpoint` can point to a copy with the same hash. Use validation prompts for implementation checks; official test results must not drive tuning. Generated strings are parsed under the strict format and decoded directly by the new model. Current ICL prompt pickles must be regenerated with the same new format if used; old ICL geometry examples are not interchangeable.

## Checks and limits

[Validation receipt](coordtoken_integration_validation.json) records the completed checks. They include exact CPU parity with frozen source; dense/serialized decode parity and padding; explicit H/stereo; all 4,096 LM code tokens; strict data/checkpoint failures; CSV shard accounting; actual packed targets; Qwen tiny-model train/save/resume; and constrained generation. The latter verifies the interface, not learned generation quality. A separate sample verifies the first 16 retained training rows without official test access.

Full-corpus encoding, production GPU/bf16 parity, multi-GPU Qwen training, and downstream scientific benchmarks have not been completed by these checks. Legacy interpolation/refinement/BindGPT paths remain legacy until separately migrated and tested. This prevents old results or caches from silently acquiring new meanings.
