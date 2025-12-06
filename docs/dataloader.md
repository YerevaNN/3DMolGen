# MolGen3D Packed Dataloader Overview

This document explains how `JsonlTaggedPackedDataset` (the dataloader exposed via `build_dataloader`) works, how to use it, and which requirements/design choices shaped its architecture.

## Way of Working
1. **Loading tokenizer** – instantiate `AutoTokenizer.from_pretrained(..., use_fast=True, fix_mistral_regex=True)` to support both LLaMA 3 and Qwen3 fast tokenizers when no tokenizer is provided. In that path the dataloader ensures `<|endoftext|>` exists via `ensure_tokenizer_pad_token()`, sets it as the pad token, and reuses it for both separators and padding. When a tokenizer object is supplied by the caller (e.g., LLaMA), the dataloader uses *its* `pad_token_id` as the separator/pad token; if the tokenizer has no pad token id, initialization fails fast.
2. **Streaming JSONL units** – read one entry per line, validate canonical/embedded SMILES (minimum embedded length `min_emb_len=16`), build the unit string via `build_unit(...)`, and tokenize without special tokens before appending to sequences. Units that tokenize to more than `seq_len - 1` tokens are truncated in-place (preserving the prefix) so they can still be packed.
3. **Packing sequences** – append each `[SMILES]…[/SMILES][CONFORMER]…[/CONFORMER]` unit followed by the separator token (the tokenizer’s pad token; `<|endoftext|>` when auto-built), and only pad the tail with that same token when no more units fit in the remaining context. Labels are shifted left by one token (next-token prediction) and any suffix without a real successor—including the final position and all padding—is set to `ignore_index=-100`. There is no BOS/EOS—every sample starts with a fresh unit, and `seq_len` remains configurable (default 2048).
4. **Buffered shuffle + lookahead** – maintain a shuffle buffer (default 4096 entries) per worker, refill it deterministically per epoch/seed/rank, and maintain a lookahead buffer of up to `lookahead_limit` (default 100) pending units. The packer selects the unit that best fills the remaining space (minimizes wasted space) from the lookahead, ensuring no unit is split.
5. **Distributed determinism + resumability** – global indices are sharded via `(global_idx % world_size) == rank`, state dict exposes epoch/RNG state/pair cursor/buffer/pending units/current sequence tokens, and TorchTitan can restore the exact packing position later.
6. **Monitoring** – `_PreviewLogger` logs the first two decoded sequences per rank, while `count_tokens.py` provides aggregated diagnostics and statistics.

## Way of Using the Dataloader
- `build_dataloader(...)` returns a `TitanStatefulDataLoader` wrapping `JsonlTaggedPackedDataset`. Batches come back as `({"input": input_ids}, labels)`; use `inputs["input"]` when accessing tensors.
- To inspect the loader, run `count_tokens.py` (estimates dataset-level token counts, padding, and packing efficiency).
- For TorchTitan training, rely on the `state_dict`/`load_state_dict` hooks to checkpoint the sampler state (epoch, RNG state, buffer/pending units, current sequence).

## Diagnostic Tools

### `count_tokens.py`
A dataset analysis tool that estimates token counts and packing efficiency across the entire dataset. It:
- Counts total lines and bytes across all JSONL files.
- Samples each train file (capped by `--sample-lines`) through the production dataloader, aggregates per-file averages (items/sample, pad/sample, valid-line ratio), and extrapolates totals for the training split with a byte-size sanity check.
- Exhausts the validation split to report exact samples/tokens/padding.
- Supports multiple tokenizers simultaneously for comparison.
- Useful for planning training runs, estimating storage requirements, and comparing tokenizer efficiency.

**Usage:**
```bash
python src/molgen3D/training/pretraining/dataprocessing/count_tokens.py
python src/molgen3D/training/pretraining/dataprocessing/count_tokens.py --tokenizers qwen3_0.6b_origin qwen3_0.6b_custom --sample-lines 20000
```

## Requirements & Design Choices
- **Fast tokenizer compatibility**: relies on `AutoTokenizer(use_fast=True, fix_mistral_regex=True)` with dynamic `<|endoftext|>` configuration via `ensure_tokenizer_pad_token()` so it works for both Qwen3 and LLaMA 3.x without hard-coded IDs.
- **Atomic units**: never split `[SMILES]…[/SMILES][CONFORMER]…[/CONFORMER]` units; when a tokenized unit exceeds `seq_len - 1` tokens, truncate it in-place (logging once per rank) so the remaining prefix still participates in training. Set `truncate_overflow_units=False` to revert to the old skip-only behavior.
- **Padding & labels**: sequences are padded to `seq_len` with the separator token (tokenizer pad token; `<|endoftext|>` when auto-built), labels are shifted left by one token, and only the suffix whose targets fall outside real data (final slot + padding) is set to `ignore_index=-100`, keeping separators between real units loss-relevant.
- **Buffered shuffle + limited lookahead**: use a moderate shuffle buffer (default 4096) and maintain up to `lookahead_limit` (default 100) pending units, selecting the unit that best fills the remaining space (minimizes wasted space) before finalizing the current sequence to maximize context utilization.
- **Determinism & resumability**: shard indices deterministically before buffering, reseed RNG per epoch, and serialize enough state (epoch, RNG state, cursor, buffer, pending units, current sequence tokens) for TorchTitan checkpoints.
- **Preview logging**: each rank logs its first two decoded sequences to help diagnose data/configuration issues without impacting throughput.
- **Tools alignment**: `smoke_test_dataloader.py` and `count_tokens.py` operate on the dict batch format `({"input": input_ids}, labels)`, report tokenizer choices, pad usage, sampled file info, and use the loader to extrapolate dataset-level statistics.

Use this guide whenever you need to inspect, debug, or extend how the MolGen3D dataloader produces packed training sequences. Ensure all future changes honor these requirements/design choices.
