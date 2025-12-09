#!/usr/bin/env python
"""
Inspect a TorchTitan Qwen3-0.6B DCP checkpoint on CPU.

Assumptions (hard-coded for your setup):

  BASE_VOCAB    = 151_669  # original Qwen3 tokenizer vocab size
  PADDED_VOCAB  = 151_936  # embedding rows (multiple of 128)
  NUM_NEW_TOKENS= 4        # custom tokens

Usage:
  python scripts/inspect_qwen3_dcp.py /path/to/step-200
  python scripts/inspect_qwen3_dcp.py /path/to/step-200/__0_0.distcp
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.distributed.checkpoint as dcp

# ---- Qwen3 vocab constants (hard-coded for your setup) ----

BASE_VOCAB: int = 151_669
PADDED_VOCAB: int = 151_936
NUM_NEW_TOKENS: int = 4

EXTRA_START: int = BASE_VOCAB
EXTRA_END: int = BASE_VOCAB + NUM_NEW_TOKENS  # 151673

# ---- Candidate tensor keys ----

EMBED_WEIGHT_KEYS: Tuple[str, ...] = (
    "model.embed_tokens.weight",
    "embed_tokens.weight",
    "model.tok_embeddings.weight",
    "tok_embeddings.weight",
    "model.input_embeddings.weight",
    "input_embeddings.weight",
)

HEAD_WEIGHT_KEYS: Tuple[str, ...] = (
    "lm_head.weight",
    "model.lm_head.weight",
    "output.weight",
    "model.output.weight",
)


def _find_tensor_key_from_metadata(
    tensor_metadata: Dict[str, dcp.TensorStorageMetadata],
    candidates: Tuple[str, ...],
) -> Optional[str]:
    """Find a tensor name in DCP metadata, trying exact match first, then suffix match."""
    # Exact key match first
    for k in candidates:
        if k in tensor_metadata:
            return k

    # Fallback: suffix match (e.g. "...embed_tokens.weight")
    suffixes = tuple(k.split(".", 1)[-1] for k in candidates)
    for name in tensor_metadata.keys():
        for suf in suffixes:
            if name.endswith(suf):
                return name
    return None


def load_embed_and_head_from_dcp(
    ckpt_dir: Path,
) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
    """
    CPU-only: reconstruct full embedding + lm_head tensors from a DCP checkpoint.

    ckpt_dir should be the step directory, e.g. .../step-200, NOT "__0_0.distcp".
    """
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")

    print(f"[INFO] Using DCP checkpoint directory: {ckpt_dir}")

    reader = dcp.FileSystemReader(str(ckpt_dir))
    metadata = reader.read_metadata()
    tmeta = metadata.state_dict_metadata  # dict[name -> TensorStorageMetadata]

    # Locate embedding and head keys based on metadata
    embed_key = _find_tensor_key_from_metadata(tmeta, EMBED_WEIGHT_KEYS)
    head_key = _find_tensor_key_from_metadata(tmeta, HEAD_WEIGHT_KEYS)
    if embed_key is None:
        raise RuntimeError("Could not find embedding tensor in DCP metadata.")
    if head_key is None:
        raise RuntimeError("Could not find LM head tensor in DCP metadata.")

    print(f"[INFO] Resolved embedding key: {embed_key}")
    print(f"[INFO] Resolved LM head key : {head_key}")

    embed_md = tmeta[embed_key]
    head_md = tmeta[head_key]

    # Allocate CPU tensors with the correct shape/dtype
    embed = torch.empty(
        embed_md.size,
        dtype=embed_md.properties.dtype,
        device="cpu",
    )
    head = torch.empty(
        head_md.size,
        dtype=head_md.properties.dtype,
        device="cpu",
    )

    # Build a minimal state_dict for DCP to fill
    state_dict = {
        embed_key: embed,
        head_key: head,
    }

    print("[INFO] Loading tensors from DCP shards into CPU state_dict ...")
    dcp.load(state_dict, storage_reader=reader)
    print("[INFO] Load complete.")

    return state_dict[embed_key], state_dict[head_key], embed_key, head_key


def sanity_check_embeddings(embed: torch.Tensor, head: torch.Tensor) -> None:
    """
    Sanity checks for a "normal" Qwen3 checkpoint with padded vocab + 4 custom tokens.

    - Embedding / head shapes & equality (tied head).
    - No NaN/Inf.
    - Extra 4 rows exist, are finite, and non-zero.
    - Basic row-usage snapshot for base / extra / tail padded blocks.
    """
    print("\n=== Embedding / LM head stats ===")
    print(f"Embedding shape: {tuple(embed.shape)}")
    print(f"LM head shape : {tuple(head.shape)}")

    emb_vocab, emb_dim = embed.shape
    head_vocab, head_dim = head.shape

    # Shape checks
    if emb_vocab != PADDED_VOCAB:
        print(
            f"[ERROR] emb_vocab ({emb_vocab}) != PADDED_VOCAB ({PADDED_VOCAB}); "
            "this violates the Qwen3 padded vocab assumption."
        )
    else:
        print(f"[CHECK] emb_vocab matches padded vocab: {emb_vocab}")

    if emb_vocab != head_vocab or emb_dim != head_dim:
        print(
            f"[ERROR] Embedding / LM head shape mismatch: "
            f"embed={tuple(embed.shape)}, head={tuple(head.shape)}"
        )
    else:
        print("[CHECK] Embedding and LM head have identical shapes.")

    # Tied head check
    tied = torch.equal(embed, head)
    print(f"[CHECK] Embedding and LM head weights tied (bitwise): {tied}")
    if not tied:
        # Not fatal for inspection, but this would violate your recipe.
        print("[WARN] LM head is not exactly tied to embeddings; check your training/init code.")

    # Global finiteness
    if not torch.isfinite(embed).all():
        print("[ERROR] Embedding contains non-finite values (NaN / Inf).")
    else:
        print("[CHECK] All embedding weights are finite.")

    # Slice blocks
    base_slice = embed[0:BASE_VOCAB]
    extra_slice = embed[EXTRA_START:EXTRA_END]
    tail_slice = embed[EXTRA_END:PADDED_VOCAB]

    print("\n=== Vocab layout (hard-coded expectations) ===")
    print(f"Base vocab rows      : [0, {BASE_VOCAB})  -> shape {tuple(base_slice.shape)}")
    print(f"Custom token rows    : [{EXTRA_START}, {EXTRA_END}) -> shape {tuple(extra_slice.shape)}")
    print(f"Tail padded rows     : [{EXTRA_END}, {PADDED_VOCAB}) -> shape {tuple(tail_slice.shape)}")

    # Extra rows finiteness & nonzero check
    print("\n=== Extra-row sanity checks ===")
    if extra_slice.numel() == 0:
        print("[ERROR] Extra slice is empty; NUM_NEW_TOKENS or BASE_VOCAB is wrong.")
    else:
        if not torch.isfinite(extra_slice).all():
            print("[ERROR] Extra embedding rows contain non-finite values (NaN / Inf).")
        else:
            print("[CHECK] Extra embedding rows are finite.")

        num_nonzero = torch.count_nonzero(extra_slice).item()
        if num_nonzero == 0:
            print(
                "[WARN] All extra embedding rows are zero; they may not have been "
                "initialized or updated as expected."
            )
        else:
            frac = num_nonzero / extra_slice.numel() * 100.0
            print(
                f"[CHECK] Extra embedding rows have {num_nonzero} non-zero elements "
                f"({frac:.2f}% of entries)."
            )

    # Row-usage snapshot using L2 norms
    print("\n=== Row-usage snapshot (L2 norms) ===")
    with torch.no_grad():
        base_norms = base_slice.norm(dim=1)
        extra_norms = extra_slice.norm(dim=1) if extra_slice.numel() > 0 else torch.tensor([])
        tail_norms = tail_slice.norm(dim=1) if tail_slice.numel() > 0 else torch.tensor([])

    def summarize_block(name: str, norms: torch.Tensor) -> None:
        if norms.numel() == 0:
            print(f"{name}: <empty>")
            return
        num_rows = norms.shape[0]
        num_zero = int((norms == 0).sum().item())
        min_norm = float(norms.min().item())
        max_norm = float(norms.max().item())
        mean_norm = float(norms.mean().item())
        std_norm = float(norms.std(unbiased=False).item())
        print(
            f"{name}: rows={num_rows}, zero-rows={num_zero}, "
            f"min_norm={min_norm:.3e}, max_norm={max_norm:.3e}, "
            f"mean_norm={mean_norm:.3e}, std_norm={std_norm:.3e}"
        )

    summarize_block("Base block   ", base_norms)
    summarize_block("Extra block  ", extra_norms)
    summarize_block("Tail padded  ", tail_norms)


def run_tiny_forward(
    embed: torch.Tensor,
    head: torch.Tensor,
    vocab_limit: int = 1024,
    seq_len: int = 8,
    batch_size: int = 2,
) -> None:
    """Build a tiny Embedding+Linear model on CPU and run a forward pass."""
    emb_vocab, emb_dim = embed.shape
    print("\n=== Tiny CPU forward pass ===")
    vocab_limit = min(vocab_limit, emb_vocab)
    print(f"[INFO] Using vocab limit {vocab_limit} (<= {emb_vocab})")

    if vocab_limit <= 0:
        print("[WARN] vocab_limit <= 0; skipping forward.")
        return

    emb_layer = nn.Embedding(emb_vocab, emb_dim)
    head_layer = nn.Linear(emb_dim, emb_vocab, bias=False)

    with torch.no_grad():
        emb_layer.weight.copy_(embed)
        head_layer.weight.copy_(head)

    input_ids = torch.randint(
        low=0,
        high=vocab_limit,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device="cpu",
    )
    print(f"[INFO] input_ids shape: {tuple(input_ids.shape)}")

    with torch.no_grad():
        hidden = emb_layer(input_ids)        # [B, T, D]
        logits = head_layer(hidden)          # [B, T, V]

    print(f"[INFO] hidden shape: {tuple(hidden.shape)}")
    print(f"[INFO] logits shape: {tuple(logits.shape)}")
    print("[OK] Forward pass completed successfully on CPU.")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Qwen3-0.6B embeddings/head from a TorchTitan DCP checkpoint (CPU-only)."
    )
    parser.add_argument(
        "ckpt_path",
        type=str,
        help=(
            "Path to step directory (e.g. .../step-200) or a __0_0.distcp shard; "
            "the script will normalize to the step directory."
        ),
    )
    args = parser.parse_args()

    p = Path(args.ckpt_path)
    if p.is_file() and p.name.endswith(".distcp"):
        ckpt_dir = p.parent
    else:
        ckpt_dir = p

    embed, head, embed_key, head_key = load_embed_and_head_from_dcp(ckpt_dir)

    print("\n=== Keys used ===")
    print(f"Embedding key: {embed_key}")
    print(f"LM head key : {head_key}")

    sanity_check_embeddings(embed, head)
    run_tiny_forward(embed, head, vocab_limit=1024)


if __name__ == "__main__":
    main()