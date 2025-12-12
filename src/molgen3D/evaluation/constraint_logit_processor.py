"""
Constrained Conformer Generation - v2 Pre-computed Mask (Simple)

Simple position-based pre-computed mask per the original spec:
1. Build reference skeleton with placeholder coordinates
2. Tokenize once to get ref_ids
3. Derive is_free mask (True for tokens inside <...>)
4. Position-based lookup: if is_free[pos], allow free; else force ref_ids[pos]

This is the simplest implementation of the spec without state machine complexity.
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


# Cache for blocked token IDs
_BLOCKED_TOKEN_IDS_CACHE: dict = {}

# Cache for comma/dash token IDs (for look-ahead blocking)
_COMMA_DASH_TOKEN_IDS_CACHE: dict = {}

def _get_blocked_token_ids(tokenizer: PreTrainedTokenizer) -> set:
    """Get token IDs that should be blocked in FREE positions. (Recall that FREE positions are the ones inside <...>)

    Why block some tokens in FREE positions?
    - Some tokens containing '<' or '>' (to prevent premature coordinate closing)
    - Special tags like [CONFORMER], [/CONFORMER], etc.
    - BOS, EOS, PAD tokens
    """
    cache_key = id(tokenizer)
    if cache_key in _BLOCKED_TOKEN_IDS_CACHE:
        return _BLOCKED_TOKEN_IDS_CACHE[cache_key]

    blocked_ids = set()

    # Block tokens that contain '<' or '>' to prevent coordinate bracket leakage.
    # This is critical: tokens like '>(' (2284), '>[' (31868), etc. would corrupt structure
    vocab = tokenizer.get_vocab()
    for token_str, token_id in vocab.items():
        if '<' in token_str or '>' in token_str:
            blocked_ids.add(token_id)

    # Also block other special tags (of which some may already be covered by above)
    special_tags = [
        "[CONFORMER]", "[/CONFORMER]",
        "[SMILES]", "[/SMILES]",
    ]
    for tag in special_tags:
        try:
            tokens = tokenizer.encode(tag, add_special_tokens=False)
            blocked_ids.update(tokens)
        except:
            logger.warning(f"LogitProcessor: Could not encode special tag {tag}")
            pass

    # Also block EOS, BOS, PAD tokens if they exist
    if tokenizer.eos_token_id is not None:
        blocked_ids.add(tokenizer.eos_token_id)
    if tokenizer.bos_token_id is not None:
        blocked_ids.add(tokenizer.bos_token_id)
    if tokenizer.pad_token_id is not None:
        blocked_ids.add(tokenizer.pad_token_id)

    _BLOCKED_TOKEN_IDS_CACHE[cache_key] = blocked_ids
    return blocked_ids


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
    # We want to block: `,`, `-`, `,-`, `--`, etc.
    # But NOT: `1,` or `-1` (those contain digits and are valid coord content)
    for token_str, token_id in vocab.items():
        # Token consists only of comma and/or dash characters
        if token_str and all(c in ',-' for c in token_str):
            comma_dash_ids.add(token_id)

    logger.debug(f"LogitProcessor: Found {len(comma_dash_ids)} comma/dash tokens for look-ahead blocking")
    _COMMA_DASH_TOKEN_IDS_CACHE[cache_key] = comma_dash_ids
    return comma_dash_ids


def _build_blocked_mask(blocked_ids: set, vocab_size: int, device: torch.device) -> torch.BoolTensor:
    """Build boolean mask (like a bitmap/bitmask) from blocked token IDs.
    Why build a boolean mask?
    - It is more efficient to block a via something like a set of tokens than to iterate through a set of tokens.
    - The dimensions of vocabulary size gives us O(1) lookup time for blocked tokens. 
    - We also get vectorized write through this, that is why `scores[b, self.blocked_mask] = float('-inf')` is efficient (it happens in 1 kernel).
    - Vocab size is small too, which makes this approach reasonable.
    - Should be built once per device as well.
    - You only pay cost of O (blocked_ids) to build the blocked mask once.

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
        raise ValueError(f"LogitProcessorV2PrecomputeMask: Could not parse SMILES: {smiles}")
    mol_no_h = Chem.RemoveHs(mol)
    smiles_can = Chem.MolToSmiles(
        mol_no_h, canonical=True, isomericSmiles=True,
        allHsExplicit=False, allBondsExplicit=False,
    )
    if not mol_no_h.HasProp("_smilesAtomOutputOrder"):
        raise ValueError("LogitProcessorV2PrecomputeMask: Mol missing _smilesAtomOutputOrder")
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
    is_free = [] # list of booleans for each *token*, True=FREE, False=COPY
    for start_char, end_char in offset_mapping: # iterate through the offset mapping and build the is_free array
        if start_char >= end_char: # empty token range - treat as COPY
            is_free.append(False)
        else: # non-empty token range - check if all characters are FREE
            # Slice/Splice the character mask for this token's span
            token_chars = char_is_free[start_char:end_char]
            # Token is FREE only if every character in span is FREE.
            is_free.append(all(token_chars))

    # Build look-ahead blocking mask: block comma/dash when NEXT position is COPY and contains '>'
    # This prevents trailing punctuation patterns like `,>` or `->`
    block_comma_dash = []
    for pos in range(len(is_free)):
        should_block = False
        if is_free[pos]:  # Current position is FREE
            # Check if next position is COPY and contains '>'
            if pos + 1 < len(is_free) and not is_free[pos + 1]:
                # Decode the next token to check if it contains '>'
                next_token_str = tokenizer.decode([ref_ids[pos + 1]])
                if '>' in next_token_str:
                    should_block = True
        block_comma_dash.append(should_block)

    return PrecomputedTemplate(
        # TODO: Look into replacing `ref_ids` with `encoding["input_ids"]` in function return block and removing the unused variable above
        ref_ids=torch.tensor(ref_ids, dtype=torch.long),  # list of token IDs for the reference string (in order of reference skeleton)
        is_free=torch.tensor(is_free, dtype=torch.bool),  # list of booleans for each *token*, True=FREE, False=COPY
        seq_len=len(ref_ids),  # expected sequence length (number of tokens in the reference string)
        block_comma_dash=torch.tensor(block_comma_dash, dtype=torch.bool),  # look-ahead blocking mask
    )


class ConformerConstraintLogitsProcessor(LogitsProcessor):
    """
    Simple position-based pre-computed mask per spec.

    At each position:
    - If is_free[pos]: allow free generation (block only special tokens)
    - Else: force ref_ids[pos]
    """

    VERSION = "v2.1_precompute_mask_lookahead"
    CONFIG = {
        "approach": "pure_position_based_with_lookahead",
        "coord_placeholder": COORD_PLACEHOLDER,
        "description": "Position-based COPY/FREE mask with angle bracket blocking and look-ahead comma/dash blocking",
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
            raise ValueError("LogitProcessor: templates and prompt_lengths must match")

        self.templates = templates
        self.prompt_lengths = list(prompt_lengths)
        self.eos_token_id = eos_token_id
        self.blocked_ids = _get_blocked_token_ids(tokenizer)
        self.comma_dash_ids = _get_comma_dash_token_ids(tokenizer)
        self.blocked_mask = None
        self.comma_dash_mask = None  # Look-ahead blocking mask for comma/dash tokens
        self._device = None
        self._prev_lens: List[int | None] = [None] * len(templates)

    # This is the main function that is called by the model to apply the constraints.
    # it is invoked once per decoding step by huggingface `generate` loop (via LogitsProcessor hooks).
    # for each step, it processes every batch element :
    # compute current position `pos` relative to initial prompt length.
    # apply corresponding masking to that row of `scores` tensor.
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, cur_len = input_ids.shape 
        device = scores.device
        vocab_size = scores.shape[1] 

        # Build blocked mask lazily with correct vocab size
        # If no mask exists or the device has changed, build the mask.
        if self.blocked_mask is None or self._device != device:
            self.blocked_mask = _build_blocked_mask(self.blocked_ids, vocab_size, device)
            self.comma_dash_mask = _build_blocked_mask(self.comma_dash_ids, vocab_size, device)
            # Move templates to correct device because;
            # - They are built on the CPU and we want to move them to the GPU.
            # - Indexing becomes GPU-local (torch tensor of booleans, etc).
            for template in self.templates:
                template.ref_ids = template.ref_ids.to(device)
                template.is_free = template.is_free.to(device)
                if template.block_comma_dash is not None:
                    template.block_comma_dash = template.block_comma_dash.to(device)
            self._device = device

        # Iterate through the batch and apply the constraints.
        # Each item has a separate instance of the template (because the input prompt lengths are different).
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
                # FREE position - block only special tokens
                # We are in a coordinate block, so we need to block all tokens that are not part of the coordinate block.
                scores[b, self.blocked_mask] = float('-inf')
                # Look-ahead blocking: if next position is COPY '>', also block comma/dash
                # This prevents trailing punctuation patterns like `,>` or `->`
                if template.block_comma_dash is not None and template.block_comma_dash[pos]:
                    scores[b, self.comma_dash_mask] = float('-inf')
            else:
                # COPY position - force exact token
                # Core SMILES structure is preserved.
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
