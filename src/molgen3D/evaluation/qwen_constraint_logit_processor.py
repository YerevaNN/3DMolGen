"""
Constrained Conformer Generation - Qwen-specific v4.0 (Targeted Blocking)

Simple position-based pre-computed mask with TARGETED blocking:
1. Build reference skeleton with placeholder coordinates
2. Tokenize once to get ref_ids
3. Derive is_free mask (True for tokens inside <...>)
4. Position-based lookup: if is_free[pos], block structural tokens; else force ref_ids[pos]

v4.0: TARGETED BLOCKING - blocks SMILES structural chars that can leak into coordinates
Key changes from v3.9:
- Re-added []@+ to blocklist (v3.9 failures showed these leaking into coords)
  - `@@` chirality markers caused parse errors
  - `+]` charge/bracket combinations leaked in
  - `[]` brackets should never appear inside coordinates
- Keep () unblocked (parentheses can appear in SMILES but not in coord tokens)
- Shorter placeholder: "0.000,0.000,0.000" (3 decimals, ~18 tokens)
- LOOKAHEAD_RANGE: 2 (shorter placeholder = less drift)

CRITICAL: Use with qwen3 sampling config (temp=0.7, top_p=0.8, top_k=20)
- Qwen3 official docs: DO NOT use greedy decoding
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


# Placeholder coordinate string (3 decimal places - matches median coord length better)
# Analysis showed: median coord is 21 chars, 4-decimal placeholder was 23 chars (9.3% match)
# 3-decimal placeholder is ~19 chars, closer to median
COORD_PLACEHOLDER = "0.000,0.000,0.000"

# Valid characters for coordinate content (ALLOWLIST approach)
COORD_VALID_CHARS = set("0123456789.,-")

# Cache for tokenizer encode_plus results (optimization for repeated skeletons)
# Key: (ref_str, tokenizer_id) -> (input_ids_tuple, offset_mapping_tuple)
_TOKENIZER_ENCODING_CACHE: dict = {}


def _cached_encode_plus(
    ref_str: str,
    tokenizer: PreTrainedTokenizer,
    *,
    use_cache: bool = True,
) -> tuple[list[int], list[tuple[int, int]]]:
    """
    Encode a reference string with optional caching.

    Caching reduces redundant tokenization for repeated skeletons.
    Expected gain: 10-20% on preprocessing for batches with repeated molecules.

    Args:
        ref_str: The reference skeleton string to encode
        tokenizer: HuggingFace tokenizer
        use_cache: Whether to use caching (default: True)

    Returns:
        Tuple of (input_ids list, offset_mapping list)
    """
    if not use_cache:
        encoding = tokenizer.encode_plus(
            ref_str,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        return encoding["input_ids"], encoding["offset_mapping"]

    cache_key = (ref_str, id(tokenizer))
    if cache_key in _TOKENIZER_ENCODING_CACHE:
        # Return cached result (convert tuples back to lists for compatibility)
        input_ids_tuple, offset_tuple = _TOKENIZER_ENCODING_CACHE[cache_key]
        return list(input_ids_tuple), list(offset_tuple)

    # Encode and cache
    encoding = tokenizer.encode_plus(
        ref_str,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    # Store as immutable tuples
    _TOKENIZER_ENCODING_CACHE[cache_key] = (
        tuple(encoding["input_ids"]),
        tuple(tuple(x) for x in encoding["offset_mapping"]),
    )
    return encoding["input_ids"], encoding["offset_mapping"]


def clear_tokenizer_cache():
    """Clear the tokenizer encoding cache (useful for testing)."""
    _TOKENIZER_ENCODING_CACHE.clear()


def get_tokenizer_cache_stats() -> dict:
    """Get cache statistics for debugging."""
    return {
        "size": len(_TOKENIZER_ENCODING_CACHE),
        "memory_keys": sum(len(k[0]) for k in _TOKENIZER_ENCODING_CACHE.keys()),
    }


@dataclass
class PrecomputedTemplate:
    """Pre-computed mask for position-based constraint enforcement."""
    ref_ids: torch.LongTensor       # [seq_len] - reference token IDs from skeleton
    is_free: torch.BoolTensor       # [seq_len] - True=FREE, False=COPY
    seq_len: int                    # expected sequence length
    block_comma_dash: torch.BoolTensor | None = None  # [seq_len] - True if comma/dash should be blocked (look-ahead)


# Cache for blocked token IDs (BLOCKLIST approach - like generic LP)
_BLOCKED_TOKEN_IDS_CACHE: dict = {}

# Cache for comma/dash token IDs (for look-ahead blocking)
_COMMA_DASH_TOKEN_IDS_CACHE: dict = {}

# Cache for allowed coordinate token IDs (ALLOWLIST approach)
_ALLOWED_COORD_TOKEN_IDS_CACHE: dict = {}


def _get_blocked_token_ids(tokenizer: PreTrainedTokenizer) -> set:
    """Get token IDs that should be BLOCKED in FREE positions.

    BLOCKLIST approach (same as generic LP for performance):
    - Tokens containing '<' or '>' (prevent coordinate bracket leakage)
    - Special tags: [CONFORMER], [/CONFORMER], [SMILES], [/SMILES]
    - BOS, EOS, PAD tokens
    - Problematic coord tokens: '...', '..', '--', ',,', '.,', ',.', '.-', '-,'

    This blocks ~1,200 tokens instead of the ALLOWLIST's ~151,000 tokens.
    128x fewer writes per FREE position = massive performance gain.
    """
    cache_key = id(tokenizer)
    if cache_key in _BLOCKED_TOKEN_IDS_CACHE:
        return _BLOCKED_TOKEN_IDS_CACHE[cache_key]

    blocked_ids = set()
    vocab = tokenizer.get_vocab()

    # 1. Block tokens containing structural characters
    # TARGETED: Block chars that should NEVER appear in coordinates
    # v4.0: Re-added []@+ after v3.9 failures showed these leaking
    structural_chars = set('<>[]@+')
    for token_str, token_id in vocab.items():
        if any(c in structural_chars for c in token_str):
            blocked_ids.add(token_id)

    # 2. Block special tags
    special_tags = ["[CONFORMER]", "[/CONFORMER]", "[SMILES]", "[/SMILES]"]
    for tag in special_tags:
        try:
            tokens = tokenizer.encode(tag, add_special_tokens=False)
            blocked_ids.update(tokens)
        except:
            pass

    # 3. Block BOS, EOS, PAD
    if tokenizer.eos_token_id is not None:
        blocked_ids.add(tokenizer.eos_token_id)
    if tokenizer.bos_token_id is not None:
        blocked_ids.add(tokenizer.bos_token_id)
    if tokenizer.pad_token_id is not None:
        blocked_ids.add(tokenizer.pad_token_id)

    # 4. Block problematic multi-char coord tokens (learned from v3.2-v3.4 failures)
    # These cause parse errors like '1.234...,>' or missing commas
    for token_str, token_id in vocab.items():
        if token_str and all(c in COORD_VALID_CHARS for c in token_str):
            # Block if it has repeated punctuation or invalid pairs
            if not _is_valid_coord_token(token_str):
                blocked_ids.add(token_id)

    logger.info(f"QwenLogitProcessor: Blocking {len(blocked_ids)} tokens (BLOCKLIST approach)")
    _BLOCKED_TOKEN_IDS_CACHE[cache_key] = blocked_ids
    return blocked_ids


def _is_valid_coord_token(token_str: str) -> bool:
    """Check if a token string is valid for coordinate positions.

    Rules:
    1. All characters must be valid coord chars (0-9, ., ,, -)
    2. Single-char tokens are always valid
    3. Multi-char tokens are valid ONLY if they don't contain:
       - Repeated punctuation (.., --, ,,, ...)
       - Invalid pairs: ., (dot-comma), ,. (comma-dot), .- (dot-dash)

    Valid multi-char examples: ',−' (separator before negative), '−.' (negative decimal)
    Invalid multi-char examples: '...', '..', '--', '.,', ',.', '.-', ',,'
    """
    if not token_str:
        return False

    # All chars must be valid coord chars
    if not all(c in COORD_VALID_CHARS for c in token_str):
        return False

    # Single char tokens are always valid
    if len(token_str) == 1:
        return True

    # Multi-char token validation
    # Invalid pairs that should never appear in coordinates
    # .,  = dot then comma (trailing decimal)
    # ,.  = comma then dot (missing leading digit)
    # .-  = dot then dash (decimal then negative)
    # -,  = dash then comma (negative sign then separator - invalid)
    INVALID_PAIRS = {'.,', ',.', '.-', '-,'}

    for i in range(len(token_str) - 1):
        c1, c2 = token_str[i], token_str[i + 1]
        pair = c1 + c2

        # Block repeated punctuation (not digits)
        if c1 == c2 and c1 in '.,-':
            return False

        # Block invalid pairs
        if pair in INVALID_PAIRS:
            return False

    return True


def _get_allowed_coord_token_ids(tokenizer: PreTrainedTokenizer) -> set:
    """Get token IDs that are ALLOWED in FREE (coordinate) positions.

    Uses SMART ALLOWLIST approach:
    - Single-char tokens (0-9, ., ,, -): always allowed
    - Multi-char tokens: allowed only if they pass _is_valid_coord_token()
      - ',-' (4999): ALLOWED - separator before negative coord
      - '-.' (14523): ALLOWED - negative decimal
      - '...' (1112): BLOCKED - causes parse errors
      - '--' (313): BLOCKED - invalid double dash
      - '.,', ',.' etc: BLOCKED - invalid patterns
    """
    cache_key = id(tokenizer)
    if cache_key in _ALLOWED_COORD_TOKEN_IDS_CACHE:
        return _ALLOWED_COORD_TOKEN_IDS_CACHE[cache_key]

    allowed_ids = set()
    vocab = tokenizer.get_vocab()

    for token_str, token_id in vocab.items():
        if _is_valid_coord_token(token_str):
            allowed_ids.add(token_id)

    logger.info(f"QwenLogitProcessor: Found {len(allowed_ids)} allowed coordinate tokens")
    _ALLOWED_COORD_TOKEN_IDS_CACHE[cache_key] = allowed_ids
    return allowed_ids


def _get_comma_dash_token_ids(tokenizer: PreTrainedTokenizer) -> set:
    """Get token IDs for comma/dash tokens and problematic punct patterns (for look-ahead blocking).

    These are blocked in the last N FREE positions before a COPY `>` to prevent
    trailing punctuation patterns like `,>`, `->`, `,0>`, `.,>` etc.

    Blocked token categories:
    1. Pure comma/dash: `,`, `-`, `,-`, `--`, etc.
    2. Merged punct tokens: `.,`, `,.` (can cause `.,>` patterns)
    """
    cache_key = id(tokenizer)
    if cache_key in _COMMA_DASH_TOKEN_IDS_CACHE:
        return _COMMA_DASH_TOKEN_IDS_CACHE[cache_key]

    comma_dash_ids = set()
    vocab = tokenizer.get_vocab()

    # Find all tokens that are ONLY comma, dash, or combinations thereof
    # We want to block: `,`, `-`, `,-`, `--`, etc.
    for token_str, token_id in vocab.items():
        # Token consists only of comma and/or dash characters
        if token_str and all(c in ',-' for c in token_str):
            comma_dash_ids.add(token_id)

    # Also block merged punct tokens that can cause issues
    # These tokens end a coordinate badly when followed by `>`
    merged_punct_patterns = {'.,', ',.'}
    for token_str, token_id in vocab.items():
        if token_str in merged_punct_patterns:
            comma_dash_ids.add(token_id)

    logger.debug(f"QwenLogitProcessor: Found {len(comma_dash_ids)} comma/dash/punct tokens for look-ahead blocking")
    _COMMA_DASH_TOKEN_IDS_CACHE[cache_key] = comma_dash_ids
    return comma_dash_ids


def _build_allowed_mask(allowed_ids: set, vocab_size: int, device: torch.device) -> torch.BoolTensor:
    """Build boolean mask where True=ALLOWED, False=BLOCKED.

    This is the inverse of the blocklist approach - we mark what IS allowed.

    Returns:
        allowed_mask: torch.BoolTensor of [vocab_size] such that True=ALLOWED, False=BLOCKED
    """
    allowed_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for tok_id in allowed_ids:
        if 0 <= tok_id < vocab_size:
            allowed_mask[tok_id] = True
    return allowed_mask


def _build_blocked_mask(blocked_ids: set, vocab_size: int, device: torch.device) -> torch.BoolTensor:
    """Build boolean mask where True=BLOCKED, False=ALLOWED.

    Returns:
        blocked_mask: torch.BoolTensor of [vocab_size] such that True=BLOCKED, False=ALLOWED
    """
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

    # iterate through tokens and build the skeleton string that is in order of the atoms in the SMILES string
    out_parts = []  # list of strings (atom descriptors and coordinate placeholders)
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
    use_tokenizer_cache: bool = False,
) -> PrecomputedTemplate:
    """
    Build pre-computed template per spec:
    1. Build reference skeleton
    2. Tokenize to get ref_ids
    3. Derive is_free mask using character positions

    Args:
        smiles: Input SMILES string
        tokenizer: HuggingFace tokenizer
        include_tags: Whether to include [CONFORMER] tags (default: True)
        use_tokenizer_cache: Enable tokenizer caching for repeated skeletons (default: False)
    """
    # skeleton looks like this: "[C]<0.0000,0.0000,0.0000>[C]<0.0000,0.0000,0.0000>=[O]<0.0000,0.0000,0.0000>"
    skeleton = build_reference_skeleton(smiles)
    ref_str = f"[CONFORMER]{skeleton}[/CONFORMER]" if include_tags else skeleton

    # Tokenize with offset mapping (optionally cached)
    ref_ids, offset_mapping = _cached_encode_plus(
        ref_str, tokenizer, use_cache=use_tokenizer_cache
    )

    # Build character-level mask (array of booleans) (not token-level because we want to know if the entire token is FREE or not)
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
    is_free = []  # list of booleans for each *token*, True=FREE, False=COPY
    for start_char, end_char in offset_mapping:  # iterate through the offset mapping and build the is_free array
        if start_char >= end_char:  # empty token range - treat as COPY
            is_free.append(False)
        else:  # non-empty token range - check if all characters are FREE
            # Slice/Splice the character mask for this token's span
            token_chars = char_is_free[start_char:end_char]
            # Token is FREE only if every character in span is FREE.
            is_free.append(all(token_chars))

    # Build look-ahead blocking mask
    # Block comma/dash at last N FREE positions before a COPY `>` token
    # This handles position drift where generated coords use fewer tokens than placeholder
    # v3.9: Reduced to 2 (shorter placeholder = less drift = shorter lookahead needed)
    LOOKAHEAD_RANGE = 2  # Block comma/dash up to 2 positions before `>`
    block_comma_dash = []
    for pos in range(len(is_free)):
        should_block = False
        if is_free[pos]:  # Current position is FREE
            # Look ahead up to LOOKAHEAD_RANGE positions for a COPY `>`
            for offset in range(1, LOOKAHEAD_RANGE + 1):
                next_pos = pos + offset
                if next_pos >= len(is_free):
                    break
                if not is_free[next_pos]:
                    # Found a COPY position - check if it contains '>'
                    next_token_str = tokenizer.decode([ref_ids[next_pos]])
                    if '>' in next_token_str:
                        should_block = True
                    # Stop looking (hit a COPY position)
                    break
                # If is_free[next_pos] is True, continue looking ahead
        block_comma_dash.append(should_block)

    return PrecomputedTemplate(
        ref_ids=torch.tensor(ref_ids, dtype=torch.long),  # list of token IDs for the reference string
        is_free=torch.tensor(is_free, dtype=torch.bool),  # list of booleans for each *token*, True=FREE, False=COPY
        seq_len=len(ref_ids),  # expected sequence length (number of tokens in the reference string)
        block_comma_dash=torch.tensor(block_comma_dash, dtype=torch.bool),  # look-ahead blocking mask
    )


class QwenConformerConstraintLogitsProcessor(LogitsProcessor):
    """
    Simple position-based pre-computed mask with TARGETED BLOCKLIST approach.

    At each position:
    - If is_free[pos]: block structural tokens (<>[]@+) and problematic patterns
    - Else: force ref_ids[pos]

    v4.0: Targeted blocking of SMILES chars that leak into coordinates.
    Use with qwen3 sampling config for best results.
    """

    VERSION = "v4.0_targeted_blocking"
    CONFIG = {
        "approach": "targeted_blocklist",
        "coord_placeholder": COORD_PLACEHOLDER,
        "lookahead_range": 2,
        "structural_chars": "<>[]@+",
        "description": "Targeted blocking of SMILES chars that leak into coords, 3-decimal placeholder",
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
            raise ValueError("QwenLogitProcessor: templates and prompt_lengths must match")

        self.templates = templates
        self.prompt_lengths = list(prompt_lengths)
        self.eos_token_id = eos_token_id
        # BLOCKLIST approach (like generic LP) - much faster than ALLOWLIST
        self.blocked_ids = _get_blocked_token_ids(tokenizer)
        self.comma_dash_ids = _get_comma_dash_token_ids(tokenizer)
        # Pre-computed masks (built lazily on first call)
        self.blocked_mask = None  # Boolean mask for blocked tokens
        self.comma_dash_mask = None  # Boolean mask for comma/dash (look-ahead blocking)
        self._device = None
        self._prev_lens: List[int | None] = [None] * len(templates)

    # This is the main function that is called by the model to apply the constraints.
    # it is invoked once per decoding step by huggingface `generate` loop (via LogitsProcessor hooks).
    # for each step, it processes every batch element:
    # compute current position `pos` relative to initial prompt length.
    # apply corresponding masking to that row of `scores` tensor.
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, cur_len = input_ids.shape
        device = scores.device
        vocab_size = scores.shape[1]

        # Build blocked mask lazily (like generic LP)
        if self.blocked_mask is None or self._device != device:
            self.blocked_mask = _build_blocked_mask(self.blocked_ids, vocab_size, device)
            self.comma_dash_mask = _build_blocked_mask(self.comma_dash_ids, vocab_size, device)
            # Move templates to correct device
            for template in self.templates:
                template.ref_ids = template.ref_ids.to(device)
                template.is_free = template.is_free.to(device)
                if template.block_comma_dash is not None:
                    template.block_comma_dash = template.block_comma_dash.to(device)
            self._device = device

        # Iterate through the batch and apply the constraints.
        for b in range(batch_size):
            template = self.templates[b]

            # Initialize prev_len on first call (handles padding)
            if self._prev_lens[b] is None:
                self._prev_lens[b] = cur_len

            # Position = number of tokens generated so far
            pos = cur_len - self._prev_lens[b]

            if pos >= template.seq_len:
                # Done - force EOS
                if self.eos_token_id is not None:
                    scores[b, :] = float('-inf')
                    scores[b, self.eos_token_id] = 0.0
            elif template.is_free[pos]:
                # FREE position - BLOCKLIST approach (like generic LP)
                # Block only structural tokens (~1,200) instead of all non-coord tokens (~151k)
                # This is 128x fewer writes = MASSIVE performance gain
                scores[b, self.blocked_mask] = float('-inf')
                # Look-ahead blocking: also block comma/dash at positions near '>'
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
    *,
    use_tokenizer_cache: bool = False,
) -> List[PrecomputedTemplate]:
    """Build templates for a batch of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        tokenizer: HuggingFace tokenizer
        use_tokenizer_cache: Enable tokenizer caching for repeated skeletons (default: False)
    """
    return [
        build_precomputed_template(smi, tokenizer, use_tokenizer_cache=use_tokenizer_cache)
        for smi in smiles_list
    ]
