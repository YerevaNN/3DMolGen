#!/usr/bin/env python
"""
Inspect a TorchTitan Qwen3-0.6B DCP checkpoint on CPU.

Usage:
  python scripts/inspect_qwen3_dcp.py /path/to/step-200
  python scripts/inspect_qwen3_dcp.py --ckpt_path /path/to/step-200
  python scripts/inspect_qwen3_dcp.py /path/to/step-200/__0_0.distcp
  python scripts/inspect_qwen3_dcp.py /path/to/step-200 --tokenizer-path /path/to/tokenizer

Example (this project):
  python scripts/inspect_qwen3_dcp.py \
    --ckpt_path /home/chem-project/checkpoints/qwen3_06b/260122-0843-3a41-qwen3_06b_pre_4e_8e-4_binned_grouped/step-34000 \
    --tokenizer-path /home/chem-project/mb-3dmolgen/3DMolGen/src/molgen3D/training/tokenizers/Qwen3_tokenizer_binned
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.distributed.checkpoint as dcp
from transformers import AutoTokenizer

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


def _maybe_get_config_json(ckpt_dir: Path) -> Optional[Dict[str, object]]:
    candidates = [ckpt_dir / "config.json", ckpt_dir.parent / "config.json"]
    for path in candidates:
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse config.json at {path}: {exc}") from exc
    return None


def _nested_get(config: Dict[str, object], *keys: str) -> Optional[object]:
    cursor: object = config
    for key in keys:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


def _looks_like_tokenizer_dir(path: Path) -> bool:
    if not path.exists():
        return False
    expected_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "sentencepiece.bpe.model",
    ]
    return any((path / name).exists() for name in expected_files)


def resolve_tokenizer_path(ckpt_dir: Path, tokenizer_path: Optional[str]) -> Path:
    if tokenizer_path:
        resolved = Path(tokenizer_path)
        if not _looks_like_tokenizer_dir(resolved):
            raise FileNotFoundError(f"Tokenizer path does not look valid: {resolved}")
        return resolved

    config = _maybe_get_config_json(ckpt_dir)
    if config is None:
        raise RuntimeError(
            "No tokenizer path provided and config.json was not found in the "
            "checkpoint directory or its parent."
        )

    candidate_values = [
        _nested_get(config, "model", "tokenizer_path"),
        _nested_get(config, "model", "hf_assets_path"),
        _nested_get(config, "molgen_data", "tokenizer_override"),
    ]
    for value in candidate_values:
        if isinstance(value, str):
            candidate = Path(value)
            if _looks_like_tokenizer_dir(candidate):
                return candidate

    raise RuntimeError(
        "Unable to infer tokenizer path from config.json. "
        "Pass --tokenizer-path explicitly."
    )


def load_tokenizer(tokenizer_path: Path):
    return AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        local_files_only=True,
    )


def get_tokenizer_vocab_info(tokenizer) -> Tuple[int, int, int]:
    total_vocab = len(tokenizer)
    base_vocab = getattr(tokenizer, "vocab_size", None)
    added_vocab_count = None
    if hasattr(tokenizer, "get_added_vocab"):
        added_vocab_count = len(tokenizer.get_added_vocab())

    if base_vocab is None:
        base_vocab = total_vocab - (added_vocab_count or 0)

    if base_vocab > total_vocab:
        base_vocab = total_vocab

    num_new_tokens = added_vocab_count if added_vocab_count is not None else max(total_vocab - base_vocab, 0)
    return base_vocab, num_new_tokens, total_vocab


def sanity_check_embeddings(
    embed: torch.Tensor,
    head: torch.Tensor,
    base_vocab: int,
    num_new_tokens: int,
    total_vocab: int,
    tokenizer_path: Path,
) -> None:
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

    print("\n=== Tokenizer info ===")
    print(f"Tokenizer path      : {tokenizer_path}")
    print(f"Tokenizer base vocab: {base_vocab}")
    print(f"Tokenizer new tokens: {num_new_tokens}")
    print(f"Tokenizer total     : {total_vocab}")

    extra_start = base_vocab
    extra_end = base_vocab + num_new_tokens
    tag_token_count = min(4, num_new_tokens)

    if total_vocab > emb_vocab:
        print(
            f"[ERROR] Tokenizer total vocab ({total_vocab}) exceeds embedding rows "
            f"({emb_vocab})."
        )
    else:
        print(f"[CHECK] Embedding rows cover tokenizer vocab: {emb_vocab} >= {total_vocab}")

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
    base_end = min(base_vocab, emb_vocab)
    extra_end = min(extra_end, emb_vocab)
    base_slice = embed[0:base_end]
    extra_slice = embed[base_end:extra_end]
    tag_end = min(base_end + tag_token_count, extra_end)
    tag_slice = embed[base_end:tag_end]
    new_extra_slice = embed[tag_end:extra_end]
    tail_slice = embed[extra_end:emb_vocab]

    print("\n=== Vocab layout (tokenizer-based) ===")
    print(f"Base vocab rows      : [0, {base_vocab})  -> shape {tuple(base_slice.shape)}")
    print(
        "Tag token rows       : "
        f"[{base_vocab}, {base_vocab + tag_token_count}) -> shape {tuple(tag_slice.shape)}"
    )
    print(
        "Other new token rows : "
        f"[{base_vocab + tag_token_count}, {base_vocab + num_new_tokens}) -> shape {tuple(new_extra_slice.shape)}"
    )
    print(f"Tail padded rows     : [{extra_end}, {emb_vocab}) -> shape {tuple(tail_slice.shape)}")

    # Extra rows finiteness & nonzero check
    print("\n=== Extra-row sanity checks ===")
    if num_new_tokens == 0:
        print("[INFO] No added tokens detected; skipping extra-row checks.")
    elif extra_slice.numel() == 0:
        print("[ERROR] Extra slice is empty; check tokenizer vocab alignment.")
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
        tag_norms = tag_slice.norm(dim=1) if tag_slice.numel() > 0 else torch.tensor([])
        new_extra_norms = new_extra_slice.norm(dim=1) if new_extra_slice.numel() > 0 else torch.tensor([])
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
    summarize_block("Tag tokens  ", tag_norms)
    summarize_block("Other new   ", new_extra_norms)
    summarize_block("Extra total ", extra_norms)
    summarize_block("Tail padded ", tail_norms)


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
        nargs="?",
        type=str,
        help=(
            "Path to step directory (e.g. .../step-200) or a __0_0.distcp shard; "
            "the script will normalize to the step directory."
        ),
    )
    parser.add_argument(
        "--ckpt_path",
        dest="ckpt_path_flag",
        type=str,
        default=None,
        help="Same as positional ckpt_path; provided for convenience.",
    )
    parser.add_argument(
        "--tokenizer-path",
        "--tokenizer_path",
        dest="tokenizer_path",
        type=str,
        default=None,
        help="Path to the tokenizer directory; if omitted, inferred from config.json.",
    )
    args = parser.parse_args()

    ckpt_path = args.ckpt_path_flag or args.ckpt_path
    if ckpt_path is None:
        raise SystemExit("Missing ckpt_path. Provide a positional path or --ckpt_path.")

    p = Path(ckpt_path)
    if p.is_file() and p.name.endswith(".distcp"):
        ckpt_dir = p.parent
    else:
        ckpt_dir = p

    embed, head, embed_key, head_key = load_embed_and_head_from_dcp(ckpt_dir)
    tokenizer_path = resolve_tokenizer_path(ckpt_dir, args.tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    base_vocab, num_new_tokens, total_vocab = get_tokenizer_vocab_info(tokenizer)

    print("\n=== Keys used ===")
    print(f"Embedding key: {embed_key}")
    print(f"LM head key : {head_key}")

    sanity_check_embeddings(
        embed,
        head,
        base_vocab=base_vocab,
        num_new_tokens=num_new_tokens,
        total_vocab=total_vocab,
        tokenizer_path=tokenizer_path,
    )
    run_tiny_forward(embed, head, vocab_limit=1024)


if __name__ == "__main__":
    main()