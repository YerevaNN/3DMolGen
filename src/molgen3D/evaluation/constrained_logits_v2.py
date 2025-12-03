"""
Constrained Conformer Generation - v2 Pre-computed Mask

Version 2: Position-based pre-computed mask for fast constrained generation.

Key differences from v1.1 state machine:
- Pre-compute reference skeleton with placeholder coordinates
- Derive COPY/FREE mask from token positions (outside/inside <...>)
- Pure position-based lookup: O(1) per token per sequence
- No Python loops in hot path, no state machine

Assumptions:
- Coordinates use ~4 decimal places (0.0000,0.0000,0.0000)
- Placeholder tokenizes similarly to actual coordinates
- Model naturally generates appropriate coordinate content
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from rdkit import Chem
from transformers import LogitsProcessor, PreTrainedTokenizer

from molgen3D.data_processing.smiles_encoder_decoder import (
    _expected_plain_token,
    _format_atom_descriptor,
    _normalize_atom_descriptor,
    tokenize_smiles,
)


# Placeholder coordinate string (4 decimal places, typical format)
COORD_PLACEHOLDER = "0.0000,0.0000,0.0000"


@dataclass
class PrecomputedTemplate:
    """Pre-computed template for hybrid constraint enforcement."""
    copy_tokens: torch.LongTensor   # [num_copy] - COPY token IDs in order
    copy_is_lt: torch.BoolTensor    # [num_copy] - True if that token is '<'
    gt_token_id: int                # The '>' token ID for detecting coord block end
    num_copy: int                   # Number of COPY tokens


# Cache for tokenizer-specific blocked token IDs (computed once per tokenizer)
_BLOCKED_TOKEN_IDS_CACHE: dict = {}


def _get_blocked_token_ids(tokenizer: PreTrainedTokenizer) -> set:
    """
    Get the set of token IDs that should be blocked in FREE positions.
    """
    cache_key = id(tokenizer)
    if cache_key in _BLOCKED_TOKEN_IDS_CACHE:
        return _BLOCKED_TOKEN_IDS_CACHE[cache_key]

    blocked_ids = set()

    # Special tag tokens to block in coordinate blocks
    special_tags = [
        "[CONFORMER]", "[/CONFORMER]",
        "[SMILES]", "[/SMILES]",
        "<|begin_of_text|>", "<|end_of_text|>",
    ]

    for tag in special_tags:
        try:
            tokens = tokenizer.encode(tag, add_special_tokens=False)
            blocked_ids.update(tokens)
        except:
            pass

    # Also block EOS/BOS/PAD if they exist
    if tokenizer.eos_token_id is not None:
        blocked_ids.add(tokenizer.eos_token_id)
    if tokenizer.bos_token_id is not None:
        blocked_ids.add(tokenizer.bos_token_id)
    if tokenizer.pad_token_id is not None:
        blocked_ids.add(tokenizer.pad_token_id)

    _BLOCKED_TOKEN_IDS_CACHE[cache_key] = blocked_ids
    return blocked_ids


def _build_blocked_mask(blocked_ids: set, vocab_size: int, device: torch.device) -> torch.BoolTensor:
    """Build boolean mask from blocked token IDs for given vocab size."""
    blocked_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for tok_id in blocked_ids:
        if 0 <= tok_id < vocab_size:
            blocked_mask[tok_id] = True
    return blocked_mask


def _canonicalize_smiles(smiles: str) -> tuple[str, Chem.Mol, list[int]]:
    """Return canonical smiles, RDKit mol without Hs, and atom order used in SMILES output."""
    import ast

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    mol_no_h = Chem.RemoveHs(mol)
    smiles_can = Chem.MolToSmiles(
        mol_no_h,
        canonical=True,
        isomericSmiles=True,
        allHsExplicit=False,
        allBondsExplicit=False,
    )
    if not mol_no_h.HasProp("_smilesAtomOutputOrder"):
        raise ValueError("Mol is missing _smilesAtomOutputOrder after MolToSmiles.")

    atom_order_raw = mol_no_h.GetProp("_smilesAtomOutputOrder")
    atom_order = list(map(int, ast.literal_eval(atom_order_raw)))
    return smiles_can, mol_no_h, atom_order


def build_reference_skeleton(smiles: str) -> str:
    """
    Build enriched skeleton string with placeholder coordinates.

    Example: "CC=O" -> "[C]<0.0000,0.0000,0.0000>[C]<0.0000,0.0000,0.0000>=[O]<0.0000,0.0000,0.0000>"
    """
    smiles_can, mol_no_h, atom_order = _canonicalize_smiles(smiles)

    expected_atom_tokens = [
        _expected_plain_token(mol_no_h.GetAtomWithIdx(idx)) for idx in atom_order
    ]
    tokens = tokenize_smiles(smiles_can, expected_atom_tokens=expected_atom_tokens)

    out_parts = []
    atom_idx_in_smiles = 0

    for tok in tokens:
        if tok["type"] == "atom":
            if atom_idx_in_smiles >= len(atom_order):
                raise ValueError("Atom tokens exceed atom order mapping.")

            rd_idx = atom_order[atom_idx_in_smiles]
            atom_descriptor = tok["text"]
            if atom_descriptor[0] != "[":
                atom_descriptor = _format_atom_descriptor(mol_no_h.GetAtomWithIdx(rd_idx))
            atom_descriptor = _normalize_atom_descriptor(atom_descriptor)

            # Use placeholder coordinates
            out_parts.append(f"{atom_descriptor}<{COORD_PLACEHOLDER}>")
            atom_idx_in_smiles += 1
        else:
            # Non-atom SMILES tokens (bonds, branches, ring digits, dots)
            out_parts.append(tok["text"])

    return "".join(out_parts)


def build_precomputed_template(
    smiles: str,
    tokenizer: PreTrainedTokenizer,
    *,
    include_tags: bool = True,
) -> PrecomputedTemplate:
    """
    Build pre-computed template for hybrid constraint enforcement.

    Hybrid approach:
    - Extract only COPY tokens (structural tokens, '<', '>')
    - Track which COPY tokens are '<' (to know when to enter FREE mode)
    - Store '>' token ID for detecting when to exit FREE mode

    This handles variable-length coordinates correctly because we don't
    pre-allocate positions for coordinate content.
    """
    # Build skeleton
    skeleton = build_reference_skeleton(smiles)

    # Wrap with tags
    if include_tags:
        ref_str = f"[CONFORMER]{skeleton}[/CONFORMER]"
    else:
        ref_str = skeleton

    # Tokenize with offset mapping to get character positions
    encoding = tokenizer.encode_plus(
        ref_str,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ref_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"]  # List of (start_char, end_char)

    # Step 1: Build character-level FREE/COPY mask (array of booleans)
    char_is_free = []
    in_coord = False
    for char in ref_str:
        if char == '<':
            char_is_free.append(False)  # < itself is COPY
            in_coord = True
        elif char == '>':
            char_is_free.append(False)  # > itself is COPY
            in_coord = False
        else:
            char_is_free.append(in_coord)

    # Step 2: Extract COPY tokens and identify '<' tokens
    copy_tokens = []
    copy_is_lt = []

    for tok_idx, (start_char, end_char) in enumerate(offset_mapping):
        if start_char >= end_char:
            # Empty token range - treat as COPY
            copy_tokens.append(ref_ids[tok_idx])
            copy_is_lt.append(False)
        else:
            token_char_mask = char_is_free[start_char:end_char]
            # Token is COPY if ANY of its characters are COPY
            if not all(token_char_mask):
                copy_tokens.append(ref_ids[tok_idx])
                # Check if this token contains '<'
                token_text = ref_str[start_char:end_char]
                copy_is_lt.append('<' in token_text)

    # Get '>' token ID
    gt_tokens = tokenizer.encode(">", add_special_tokens=False)
    if len(gt_tokens) != 1:
        raise ValueError(f"Expected '>' to be a single token, got {gt_tokens}")
    gt_token_id = gt_tokens[0]

    return PrecomputedTemplate(
        copy_tokens=torch.tensor(copy_tokens, dtype=torch.long),
        copy_is_lt=torch.tensor(copy_is_lt, dtype=torch.bool),
        gt_token_id=gt_token_id,
        num_copy=len(copy_tokens),
    )


class ConformerConstraintLogitsProcessorV2(LogitsProcessor):
    """
    V2 Hybrid: Pre-computed structural tokens with dynamic coordinate handling.

    Hybrid approach that handles variable-length coordinates:
    - Force structural tokens (COPY) in order
    - When '<' is forced, switch to FREE mode
    - In FREE mode, model generates coordinates freely (only special tokens blocked)
    - When model generates '>', switch back to COPY mode

    This avoids the token alignment issues of pure position-based masking.
    """

    VERSION = "v2_hybrid"
    CONFIG = {
        "approach": "hybrid_precomputed_with_state",
        "coord_handling": "free_until_gt_detected",
        "description": "Pre-computed COPY tokens, FREE mode for coordinates until > detected"
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
            raise ValueError("templates and prompt_lengths must have the same length.")

        self.templates = templates
        self.prompt_lengths = list(prompt_lengths)
        self.eos_token_id = eos_token_id

        # Store blocked token IDs - mask will be built lazily with correct vocab size
        self.blocked_ids = _get_blocked_token_ids(tokenizer)
        self.blocked_mask = None
        self._device = None

        # Per-sequence state for hybrid approach
        batch_size = len(templates)
        self._copy_idx: List[int] = [0] * batch_size  # Current index in copy_tokens
        self._in_free: List[bool] = [False] * batch_size  # Are we in FREE mode?
        self._prev_lens: List[int | None] = [None] * batch_size  # For tracking new tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, cur_len = input_ids.shape
        device = scores.device
        vocab_size = scores.shape[1]

        # Build/rebuild blocked_mask if needed
        if self.blocked_mask is None or self._device != device:
            self.blocked_mask = _build_blocked_mask(self.blocked_ids, vocab_size, device)
            for template in self.templates:
                template.copy_tokens = template.copy_tokens.to(device)
                template.copy_is_lt = template.copy_is_lt.to(device)
            self._device = device

        if batch_size != len(self.templates):
            raise ValueError("Batch size does not match number of templates.")

        for b in range(batch_size):
            template = self.templates[b]

            # Initialize on first call
            if self._prev_lens[b] is None:
                self._prev_lens[b] = cur_len

            # Process previously generated token (if any) to update state
            elif cur_len > self._prev_lens[b]:
                # Get the token that was just generated
                prev_token = input_ids[b, cur_len - 1].item()

                if self._in_free[b]:
                    # In FREE mode - check if model generated '>'
                    if prev_token == template.gt_token_id:
                        # Exit FREE mode, skip the '>' in copy_tokens
                        self._in_free[b] = False
                        self._copy_idx[b] += 1
                else:
                    # In COPY mode - we just forced a token
                    # Check if it was '<' to enter FREE mode
                    idx = self._copy_idx[b]
                    if idx < template.num_copy and template.copy_is_lt[idx]:
                        self._in_free[b] = True
                    # Advance to next copy token
                    self._copy_idx[b] += 1

                self._prev_lens[b] = cur_len

            # Decide scores for next token
            copy_idx = self._copy_idx[b]

            if copy_idx >= template.num_copy:
                # All COPY tokens consumed - force EOS
                if self.eos_token_id is not None:
                    scores[b, :] = float('-inf')
                    scores[b, self.eos_token_id] = 0.0
            elif self._in_free[b]:
                # FREE mode - block only special tokens, let model generate coordinates
                scores[b, self.blocked_mask] = float('-inf')
            else:
                # COPY mode - force exact token
                scores[b, :] = float('-inf')
                scores[b, template.copy_tokens[copy_idx]] = 0.0

        return scores


def build_templates_for_batch(
    smiles_list: Sequence[str],
    tokenizer: PreTrainedTokenizer,
) -> List[PrecomputedTemplate]:
    """Convenience helper to build templates for a batch of SMILES strings."""
    return [build_precomputed_template(smi, tokenizer) for smi in smiles_list]


def debug_template(smiles: str, tokenizer: PreTrainedTokenizer) -> None:
    """Debug function to print template details for a SMILES string."""
    skeleton = build_reference_skeleton(smiles)
    ref_str = f"[CONFORMER]{skeleton}[/CONFORMER]"

    print(f"SMILES: {smiles}")
    print(f"Reference string: {ref_str}")
    print(f"Reference string length: {len(ref_str)} chars")
    print()

    encoding = tokenizer.encode_plus(
        ref_str,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ref_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"]

    print(f"Number of tokens: {len(ref_ids)}")
    print(f"Offset mapping present: {offset_mapping is not None}")
    print()

    # Build is_free and show details
    is_free = []
    in_coord = False

    print("Token details:")
    print("-" * 80)
    for tok_idx, (start_char, end_char) in enumerate(offset_mapping):
        token_text = ref_str[start_char:end_char]
        tok_id = ref_ids[tok_idx]

        if "<" in token_text:
            is_free.append(False)
            in_coord = True
            marker = "< (COPY, enter coord)"
        elif ">" in token_text:
            is_free.append(False)
            in_coord = False
            marker = "> (COPY, exit coord)"
        else:
            is_free.append(in_coord)
            marker = "FREE" if in_coord else "COPY"

        print(f"  [{tok_idx:3d}] id={tok_id:6d} chars=[{start_char:3d}:{end_char:3d}] "
              f"text={repr(token_text):20s} -> {marker}")

    print("-" * 80)
    num_free = sum(is_free)
    print(f"Total FREE positions: {num_free} / {len(is_free)}")
    print(f"is_free: {is_free[:20]}..." if len(is_free) > 20 else f"is_free: {is_free}")
