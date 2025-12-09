"""
Constrained Conformer Generation - Qwen-specific Allowlist Approach

This version uses an ALLOWLIST for FREE positions instead of a blocklist.
The Qwen tokenizer has single-character digit tokens, so we can strictly
enforce that only coordinate-valid tokens (0-9, ., ,, -) appear in coordinates.

Key differences from the generic version:
- FREE positions use allowlist (only permit coordinate tokens) instead of blocklist
- This prevents garbage like "SCSCSC..." from appearing in coordinates
- Works specifically with Qwen's tokenizer characteristics
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence

from loguru import logger
from rdkit import Chem
import torch
from transformers import LogitsProcessor, PreTrainedTokenizer

from molgen3D.data_processing.smiles_encoder_decoder import (
    _expected_plain_token,
    _format_atom_descriptor,
    _normalize_atom_descriptor,
    tokenize_smiles,
)


# Placeholder coordinate string (4 decimal places per spec)
COORD_PLACEHOLDER = "0.0000,0.0000,0.0000"

# Valid characters for coordinate content
COORD_VALID_CHARS = set("0123456789.,-")


@dataclass
class PrecomputedTemplate:
    """Pre-computed mask for position-based constraint enforcement."""
    ref_ids: torch.LongTensor       # [seq_len] - reference token IDs from skeleton
    is_free: torch.BoolTensor       # [seq_len] - True=FREE, False=COPY
    seq_len: int                    # expected sequence length
    block_comma_dash: torch.BoolTensor | None = None  # [seq_len] - True if comma/dash should be blocked


# Cache for allowed coordinate token IDs
_ALLOWED_COORD_TOKEN_IDS_CACHE: dict = {}

# Cache for comma/dash token IDs (for look-ahead blocking)
_COMMA_DASH_TOKEN_IDS_CACHE: dict = {}


def _get_allowed_coord_token_ids(tokenizer: PreTrainedTokenizer) -> set:
    """Get token IDs that are ALLOWED in FREE (coordinate) positions.

    Uses ALLOWLIST approach: only tokens consisting entirely of valid
    coordinate characters (0-9, ., ,, -) are permitted.

    This is much stricter than the blocklist approach and prevents
    garbage tokens like "SC", "VALUE", etc. from appearing.
    """
    cache_key = id(tokenizer)
    if cache_key in _ALLOWED_COORD_TOKEN_IDS_CACHE:
        return _ALLOWED_COORD_TOKEN_IDS_CACHE[cache_key]

    allowed_ids = set()
    vocab = tokenizer.get_vocab()

    for token_str, token_id in vocab.items():
        # Token is allowed if ALL characters are valid coordinate chars
        if token_str and all(c in COORD_VALID_CHARS for c in token_str):
            allowed_ids.add(token_id)

    logger.info(f"QwenLogitProcessor: Found {len(allowed_ids)} allowed coordinate tokens")
    _ALLOWED_COORD_TOKEN_IDS_CACHE[cache_key] = allowed_ids
    return allowed_ids


def _get_comma_dash_token_ids(tokenizer: PreTrainedTokenizer) -> set:
    """Get token IDs for comma and dash tokens (for look-ahead blocking).

    These are blocked in the last FREE position before a COPY `>` to prevent
    trailing punctuation patterns like `,>` or `->`.
    """
    cache_key = id(tokenizer)
    if cache_key in _COMMA_DASH_TOKEN_IDS_CACHE:
        return _COMMA_DASH_TOKEN_IDS_CACHE[cache_key]

    comma_dash_ids = set()
    vocab = tokenizer.get_vocab()

    # Find all tokens that are ONLY comma, dash, or combinations thereof
    for token_str, token_id in vocab.items():
        if token_str and all(c in ',-' for c in token_str):
            comma_dash_ids.add(token_id)

    logger.debug(f"QwenLogitProcessor: Found {len(comma_dash_ids)} comma/dash tokens for look-ahead blocking")
    _COMMA_DASH_TOKEN_IDS_CACHE[cache_key] = comma_dash_ids
    return comma_dash_ids


def _build_allowed_mask(allowed_ids: set, vocab_size: int, device: torch.device) -> torch.BoolTensor:
    """Build boolean mask where True=ALLOWED, False=BLOCKED.

    This is the inverse of the blocklist approach - we mark what IS allowed.
    """
    allowed_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for tok_id in allowed_ids:
        if 0 <= tok_id < vocab_size:
            allowed_mask[tok_id] = True
    return allowed_mask


def _build_blocked_mask(blocked_ids: set, vocab_size: int, device: torch.device) -> torch.BoolTensor:
    """Build boolean mask where True=BLOCKED, False=ALLOWED."""
    blocked_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for tok_id in blocked_ids:
        if 0 <= tok_id < vocab_size:
            blocked_mask[tok_id] = True
    return blocked_mask


def _canonicalize_smiles(smiles: str) -> tuple[str, Chem.Mol, list[int]]:
    """Return canonical smiles, mol, and atom order."""
    import ast
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"QwenLogitProcessor: Could not parse SMILES: {smiles}")
    mol_no_h = Chem.RemoveHs(mol)
    smiles_can = Chem.MolToSmiles(
        mol_no_h, canonical=True, isomericSmiles=True,
        allHsExplicit=False, allBondsExplicit=False,
    )
    if not mol_no_h.HasProp("_smilesAtomOutputOrder"):
        raise ValueError("QwenLogitProcessor: Mol missing _smilesAtomOutputOrder")
    atom_order = list(map(int, ast.literal_eval(mol_no_h.GetProp("_smilesAtomOutputOrder"))))
    return smiles_can, mol_no_h, atom_order


def build_reference_skeleton(smiles: str) -> str:
    """Build enriched skeleton with placeholder coordinates."""
    smiles_can, mol_no_h, atom_order = _canonicalize_smiles(smiles)
    expected_atom_tokens = [
        _expected_plain_token(mol_no_h.GetAtomWithIdx(idx)) for idx in atom_order
    ]
    tokens = tokenize_smiles(smiles_can, expected_atom_tokens=expected_atom_tokens)

    out_parts = []
    atom_idx = 0
    for tok in tokens:
        if tok["type"] == "atom":
            rd_idx = atom_order[atom_idx]
            atom_descriptor = tok["text"]
            if atom_descriptor[0] != "[":
                atom_descriptor = _format_atom_descriptor(mol_no_h.GetAtomWithIdx(rd_idx))
            atom_descriptor = _normalize_atom_descriptor(atom_descriptor)
            out_parts.append(f"{atom_descriptor}<{COORD_PLACEHOLDER}>")
            atom_idx += 1
        else:
            out_parts.append(tok["text"])
    return "".join(out_parts)


def build_precomputed_template(
    smiles: str,
    tokenizer: PreTrainedTokenizer,
    *,
    include_tags: bool = True,
) -> PrecomputedTemplate:
    """
    Build pre-computed template per spec:
    1. Build reference skeleton
    2. Tokenize to get ref_ids
    3. Derive is_free mask using character positions
    """
    skeleton = build_reference_skeleton(smiles)
    ref_str = f"[CONFORMER]{skeleton}[/CONFORMER]" if include_tags else skeleton

    encoding = tokenizer.encode_plus(
        ref_str,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    ref_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"]

    # Build character-level mask
    char_is_free = []
    in_coord = False
    for char in ref_str:
        if char == '<':
            char_is_free.append(False)
            in_coord = True
        elif char == '>':
            char_is_free.append(False)
            in_coord = False
        else:
            char_is_free.append(in_coord)

    # Convert to token-level: token is FREE only if ALL its chars are FREE
    is_free = []
    for start_char, end_char in offset_mapping:
        if start_char >= end_char:
            is_free.append(False)
        else:
            token_chars = char_is_free[start_char:end_char]
            is_free.append(all(token_chars))

    # Build look-ahead blocking mask
    block_comma_dash = []
    for pos in range(len(is_free)):
        should_block = False
        if is_free[pos]:
            if pos + 1 < len(is_free) and not is_free[pos + 1]:
                next_token_str = tokenizer.decode([ref_ids[pos + 1]])
                if '>' in next_token_str:
                    should_block = True
        block_comma_dash.append(should_block)

    return PrecomputedTemplate(
        ref_ids=torch.tensor(ref_ids, dtype=torch.long),
        is_free=torch.tensor(is_free, dtype=torch.bool),
        seq_len=len(ref_ids),
        block_comma_dash=torch.tensor(block_comma_dash, dtype=torch.bool),
    )


class QwenConformerConstraintLogitsProcessor(LogitsProcessor):
    """
    Qwen-specific position-based constraint processor using ALLOWLIST approach.

    Key difference from generic version:
    - FREE positions only allow coordinate-valid tokens (0-9, ., ,, -)
    - This prevents garbage tokens from appearing in coordinates
    - Much stricter than the blocklist approach

    At each position:
    - If is_free[pos]: only allow coordinate tokens (ALLOWLIST)
    - Else: force ref_ids[pos] (COPY)
    """

    VERSION = "v3.0_qwen_allowlist"
    CONFIG = {
        "approach": "qwen_allowlist_based",
        "coord_placeholder": COORD_PLACEHOLDER,
        "description": "Qwen-specific allowlist approach for FREE positions",
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
            raise ValueError("QwenLogitProcessor: templates and prompt_lengths must match")

        self.templates = templates
        self.prompt_lengths = list(prompt_lengths)
        self.eos_token_id = eos_token_id
        self.allowed_coord_ids = _get_allowed_coord_token_ids(tokenizer)
        self.comma_dash_ids = _get_comma_dash_token_ids(tokenizer)
        self.allowed_mask = None  # ALLOWLIST mask for FREE positions
        self.comma_dash_mask = None
        self._device = None
        self._prev_lens: List[int | None] = [None] * len(templates)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, cur_len = input_ids.shape
        device = scores.device
        vocab_size = scores.shape[1]

        # Build masks lazily
        if self.allowed_mask is None or self._device != device:
            self.allowed_mask = _build_allowed_mask(self.allowed_coord_ids, vocab_size, device)
            self.comma_dash_mask = _build_blocked_mask(self.comma_dash_ids, vocab_size, device)
            for template in self.templates:
                template.ref_ids = template.ref_ids.to(device)
                template.is_free = template.is_free.to(device)
                if template.block_comma_dash is not None:
                    template.block_comma_dash = template.block_comma_dash.to(device)
            self._device = device

        for b in range(batch_size):
            template = self.templates[b]

            if self._prev_lens[b] is None:
                self._prev_lens[b] = cur_len

            pos = cur_len - self._prev_lens[b]

            if pos >= template.seq_len:
                # Done - force EOS
                if self.eos_token_id is not None:
                    scores[b, :] = float('-inf')
                    scores[b, self.eos_token_id] = 0.0
            elif template.is_free[pos]:
                # FREE position - ALLOWLIST approach: only permit coordinate tokens
                # Block everything NOT in the allowlist
                scores[b, ~self.allowed_mask] = float('-inf')
                # Look-ahead blocking for comma/dash
                if template.block_comma_dash is not None and template.block_comma_dash[pos]:
                    scores[b, self.comma_dash_mask] = float('-inf')
            else:
                # COPY position - force exact token
                scores[b, :] = float('-inf')
                scores[b, template.ref_ids[pos]] = 0.0

        return scores


def build_templates_for_batch(
    smiles_list: Sequence[str],
    tokenizer: PreTrainedTokenizer,
) -> List[PrecomputedTemplate]:
    """Build templates for a batch of SMILES strings."""
    return [build_precomputed_template(smi, tokenizer) for smi in smiles_list]
