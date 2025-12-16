"""
Constrained Conformer Generation - Qwen-specific v5.0 (Vectorized ALLOWLIST)

Fully vectorized version of QwenAllowlistLogitsProcessor (v4.3).
Key optimizations:
1. Pre-stack all template data into batch tensors in __init__
2. No Python loops in __call__ - all operations use batch indexing
3. Pre-compute combined masks for different position types

Performance improvements:
- Eliminates per-batch-element Python loop overhead
- Uses vectorized gather/scatter operations
- Pre-computes all mask combinations to avoid repeated OR operations

CRITICAL: Use with qwen3 sampling config (temp=0.7, top_p=0.8, top_k=20)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from loguru import logger
from transformers import LogitsProcessor, PreTrainedTokenizer

# Import shared utilities from v4.3
from molgen3D.evaluation.qwen_allowlist_logit_processor import (
    COORD_PLACEHOLDER,
    COORD_VALID_CHARS,
    PrecomputedTemplate,
    _build_allowed_mask,
    _build_blocked_mask,
    _get_allowed_coord_token_ids,
    _get_comma_dash_token_ids,
    _get_comma_only_token_ids,
    build_precomputed_template,
    build_reference_skeleton,
    build_templates_for_batch,
)


class QwenVectorizedAllowlistLogitsProcessor(LogitsProcessor):
    """
    Fully vectorized allowlist logit processor - no Python loops in __call__.

    This is a drop-in replacement for QwenAllowlistLogitsProcessor with
    identical behavior but better performance on batched generation.

    At each position:
    - If is_free[pos]: allow ONLY the valid coordinate tokens
    - If is_first_free[pos]: additionally block comma tokens
    - If block_comma_dash[pos]: additionally block comma and dash tokens
    - Else: force ref_ids[pos] (copy exact token)

    v5.0: Fully vectorized with pre-computed stacked templates and combined masks.
    """

    VERSION = "v5.0_vectorized_allowlist"
    CONFIG = {
        "approach": "vectorized_strict_allowlist_smart_blocking",
        "coord_placeholder": COORD_PLACEHOLDER,
        "lookahead_range": 2,
        "allowed_chars": "0123456789.,-",
        "first_free_blocks": "comma_only",
        "last_free_blocks": "comma_and_dash",
        "description": "Fully vectorized allowlist with smart position blocking",
        "recommended_sampling": "qwen3 (temp=0.7, top_p=0.8, top_k=20)",
    }

    def __init__(
        self,
        templates: Sequence[PrecomputedTemplate],
        prompt_lengths: Sequence[int],
        *,
        tokenizer: PreTrainedTokenizer,
        eos_token_id: int | None = None,
    ):
        if len(templates) != len(prompt_lengths):
            raise ValueError("QwenVectorizedLP: templates and prompt_lengths must match")

        batch_size = len(templates)
        max_seq_len = max(t.seq_len for t in templates)

        # Pre-stack ALL template data into batch tensors
        self.ref_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        self.is_free = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
        self.is_first_free = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
        self.block_comma_dash = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
        self.seq_lens = torch.tensor([t.seq_len for t in templates], dtype=torch.long)
        # NOTE: prompt_lengths from attention_mask.sum() is NON-PADDED, but input_ids.shape[1]
        # is PADDED. We capture the actual initial cur_len on first __call__ instead.
        self._batch_size = batch_size

        for b, t in enumerate(templates):
            slen = t.seq_len
            self.ref_ids[b, :slen] = t.ref_ids
            self.is_free[b, :slen] = t.is_free
            if t.is_first_free is not None:
                self.is_first_free[b, :slen] = t.is_first_free
            if t.block_comma_dash is not None:
                self.block_comma_dash[b, :slen] = t.block_comma_dash

        # Store token sets for mask building
        self.allowed_ids = _get_allowed_coord_token_ids(tokenizer)
        self.comma_dash_ids = _get_comma_dash_token_ids(tokenizer)
        self.comma_only_ids = _get_comma_only_token_ids(tokenizer)
        self.eos_token_id = eos_token_id

        # Masks will be initialized on first __call__ when we know the device
        self._initialized = False
        # Capture initial cur_len on first call (like v4.3's _prev_lens pattern)
        self._initial_len: int | None = None

    def _init_masks(self, vocab_size: int, device: torch.device):
        """One-time setup of masks on the correct device."""
        # Build base masks
        allowed_mask = _build_allowed_mask(self.allowed_ids, vocab_size, device)
        comma_dash_mask = _build_blocked_mask(self.comma_dash_ids, vocab_size, device)
        comma_only_mask = _build_blocked_mask(self.comma_only_ids, vocab_size, device)

        # Pre-compute combined masks for each FREE position type
        # These are masks of tokens to BLOCK (set to -inf)
        self.free_base_mask = ~allowed_mask  # Block all non-coord tokens
        self.free_first_mask = self.free_base_mask | comma_only_mask  # Also block comma
        self.free_last_mask = self.free_base_mask | comma_dash_mask  # Also block comma+dash
        self.free_first_last_mask = self.free_first_mask | comma_dash_mask  # Both restrictions

        # Move stacked templates to device
        self.ref_ids = self.ref_ids.to(device)
        self.is_free = self.is_free.to(device)
        self.is_first_free = self.is_first_free.to(device)
        self.block_comma_dash = self.block_comma_dash.to(device)
        self.seq_lens = self.seq_lens.to(device)

        self._initialized = True

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size, cur_len = input_ids.shape
        vocab_size = scores.shape[1]
        device = scores.device

        if not self._initialized:
            self._init_masks(vocab_size, device)

        # Capture initial length on first call (like v4.3's _prev_lens pattern)
        # This is the PADDED length, which is the same for all sequences in the batch
        if self._initial_len is None:
            self._initial_len = cur_len

        # ===== FULLY VECTORIZED - NO PYTHON LOOPS =====

        # Compute position (same for all batch elements since they're padded to same length)
        pos_scalar = cur_len - self._initial_len
        # Broadcast to batch dimension for vectorized comparisons
        pos = torch.full((batch_size,), pos_scalar, dtype=torch.long, device=device)
        pos_clamped = pos.clamp(0, self.ref_ids.shape[1] - 1)

        # Gather per-position data using batch indexing
        batch_idx = torch.arange(batch_size, device=device)
        ref_ids_at_pos = self.ref_ids[batch_idx, pos_clamped]  # [batch]
        is_free_at_pos = self.is_free[batch_idx, pos_clamped]  # [batch]
        is_first_at_pos = self.is_first_free[batch_idx, pos_clamped]  # [batch]
        block_cd_at_pos = self.block_comma_dash[batch_idx, pos_clamped]  # [batch]

        # Compute condition masks for each batch element
        past_end = pos >= self.seq_lens  # [batch] - past expected length
        is_copy = ~is_free_at_pos & ~past_end  # [batch] - should copy ref token
        is_free = is_free_at_pos & ~past_end  # [batch] - FREE position

        # Sub-classify FREE positions into 4 types based on restrictions
        is_free_first = is_free & is_first_at_pos & ~block_cd_at_pos  # First only
        is_free_last = is_free & ~is_first_at_pos & block_cd_at_pos  # Last only
        is_free_first_last = is_free & is_first_at_pos & block_cd_at_pos  # Both
        is_free_base = is_free & ~is_first_at_pos & ~block_cd_at_pos  # Neither

        # ===== Apply constraints using batch operations =====

        # 1. PAST_END: force EOS token
        if past_end.any():
            scores[past_end] = float("-inf")
            if self.eos_token_id is not None:
                scores[past_end, self.eos_token_id] = 0.0

        # 2. COPY: force exact reference token
        if is_copy.any():
            scores[is_copy] = float("-inf")
            copy_batch_idx = batch_idx[is_copy]
            copy_tokens = ref_ids_at_pos[is_copy]
            scores[copy_batch_idx, copy_tokens] = 0.0

        # 3. FREE variants: apply appropriate blocking mask
        if is_free_base.any():
            scores[is_free_base] = scores[is_free_base].masked_fill(
                self.free_base_mask, float("-inf")
            )
        if is_free_first.any():
            scores[is_free_first] = scores[is_free_first].masked_fill(
                self.free_first_mask, float("-inf")
            )
        if is_free_last.any():
            scores[is_free_last] = scores[is_free_last].masked_fill(
                self.free_last_mask, float("-inf")
            )
        if is_free_first_last.any():
            scores[is_free_first_last] = scores[is_free_first_last].masked_fill(
                self.free_first_last_mask, float("-inf")
            )

        return scores


# Re-export helper functions for convenience
__all__ = [
    "QwenVectorizedAllowlistLogitsProcessor",
    "PrecomputedTemplate",
    "build_precomputed_template",
    "build_templates_for_batch",
    "build_reference_skeleton",
    "COORD_PLACEHOLDER",
    "COORD_VALID_CHARS",
]
