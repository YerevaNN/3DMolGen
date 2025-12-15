"""
Constrained Conformer Generation - Qwen-specific v4.3 (ALLOWLIST + Smart Blocking)

Simple position-based pre-computed mask with STRICT ALLOWLIST approach:
1. Build reference skeleton with placeholder coordinates
2. Tokenize once to get ref_ids
3. Derive is_free mask (True for tokens inside <...>)
4. Position-based lookup: if is_free[pos], allow ONLY coord tokens; else force ref_ids[pos]

v4.3: ALLOWLIST with position-aware smart blocking
- ALLOWLIST (66 allowed tokens) - only 0-9, period, comma, minus
- Smart position blocking (prevents structural errors without hurting accuracy):
  - First FREE position after `<`: block comma only (minus OK for negative coords)
  - Last FREE position before `>`: block comma AND minus (no trailing punctuation)
  - LOOKAHEAD_RANGE: 2 (conservative, avoids v4.2's 1000 failures)
- Shorter placeholder: "0.000,0.000,0.000" (3 decimals, ~18 tokens)

CRITICAL: Use with qwen3 sampling config (temp=0.7, top_p=0.8, top_k=20)
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


COORD_PLACEHOLDER = "0.000,0.000,0.000"
COORD_VALID_CHARS = set("0123456789.,-")

_TOKENIZER_ENCODING_CACHE: dict = {}
_ALLOWED_COORD_TOKEN_IDS_CACHE: dict = {}
_COMMA_DASH_TOKEN_IDS_CACHE: dict = {}


def _cached_encode_plus(
    ref_str: str,
    tokenizer: PreTrainedTokenizer,
    *,
    use_cache: bool = True,
) -> tuple[list[int], list[tuple[int, int]]]:
    if not use_cache:
        encoding = tokenizer.encode_plus(
            ref_str,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        return encoding["input_ids"], encoding["offset_mapping"]

    cache_key = (ref_str, id(tokenizer))
    if cache_key in _TOKENIZER_ENCODING_CACHE:
        input_ids_tuple, offset_tuple = _TOKENIZER_ENCODING_CACHE[cache_key]
        return list(input_ids_tuple), list(offset_tuple)

    encoding = tokenizer.encode_plus(
        ref_str,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    _TOKENIZER_ENCODING_CACHE[cache_key] = (
        tuple(encoding["input_ids"]),
        tuple(tuple(x) for x in encoding["offset_mapping"]),
    )
    return encoding["input_ids"], encoding["offset_mapping"]


def clear_tokenizer_cache():
    _TOKENIZER_ENCODING_CACHE.clear()


@dataclass
class PrecomputedTemplate:
    ref_ids: torch.LongTensor
    is_free: torch.BoolTensor
    seq_len: int
    block_comma_dash: torch.BoolTensor | None = None
    is_first_free: torch.BoolTensor | None = None


def _is_valid_coord_token(token_str: str) -> bool:
    if not token_str:
        return False
    if not all(c in COORD_VALID_CHARS for c in token_str):
        return False
    if len(token_str) == 1:
        return True
    INVALID_PAIRS = {'.,', ',.', '.-', '-,'}
    for i in range(len(token_str) - 1):
        c1, c2 = token_str[i], token_str[i + 1]
        pair = c1 + c2
        if c1 == c2 and c1 in '.,-':
            return False
        if pair in INVALID_PAIRS:
            return False
    return True


def _get_allowed_coord_token_ids(tokenizer: PreTrainedTokenizer) -> set:
    cache_key = id(tokenizer)
    if cache_key in _ALLOWED_COORD_TOKEN_IDS_CACHE:
        return _ALLOWED_COORD_TOKEN_IDS_CACHE[cache_key]

    allowed_ids = set()
    vocab = tokenizer.get_vocab()

    for token_str, token_id in vocab.items():
        if _is_valid_coord_token(token_str):
            allowed_ids.add(token_id)

    logger.info(f"QwenAllowlistLP: Found {len(allowed_ids)} allowed coordinate tokens")
    _ALLOWED_COORD_TOKEN_IDS_CACHE[cache_key] = allowed_ids
    return allowed_ids


def _get_comma_dash_token_ids(tokenizer: PreTrainedTokenizer) -> set:
    cache_key = id(tokenizer)
    if cache_key in _COMMA_DASH_TOKEN_IDS_CACHE:
        return _COMMA_DASH_TOKEN_IDS_CACHE[cache_key]

    comma_dash_ids = set()
    vocab = tokenizer.get_vocab()

    for token_str, token_id in vocab.items():
        if token_str and all(c in ',-' for c in token_str):
            comma_dash_ids.add(token_id)

    merged_punct_patterns = {'.,', ',.'}
    for token_str, token_id in vocab.items():
        if token_str in merged_punct_patterns:
            comma_dash_ids.add(token_id)

    logger.debug(f"QwenAllowlistLP: Found {len(comma_dash_ids)} comma/dash tokens for look-ahead blocking")
    _COMMA_DASH_TOKEN_IDS_CACHE[cache_key] = comma_dash_ids
    return comma_dash_ids


_COMMA_ONLY_TOKEN_IDS_CACHE: dict = {}


def _get_comma_only_token_ids(tokenizer: PreTrainedTokenizer) -> set:
    cache_key = id(tokenizer)
    if cache_key in _COMMA_ONLY_TOKEN_IDS_CACHE:
        return _COMMA_ONLY_TOKEN_IDS_CACHE[cache_key]

    comma_only_ids = set()
    vocab = tokenizer.get_vocab()

    for token_str, token_id in vocab.items():
        if token_str and all(c == ',' for c in token_str):
            comma_only_ids.add(token_id)

    logger.debug(f"QwenAllowlistLP: Found {len(comma_only_ids)} comma-only tokens for first-position blocking")
    _COMMA_ONLY_TOKEN_IDS_CACHE[cache_key] = comma_only_ids
    return comma_only_ids


def _build_allowed_mask(allowed_ids: set, vocab_size: int, device: torch.device) -> torch.BoolTensor:
    allowed_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for tok_id in allowed_ids:
        if 0 <= tok_id < vocab_size:
            allowed_mask[tok_id] = True
    return allowed_mask


def _build_blocked_mask(blocked_ids: set, vocab_size: int, device: torch.device) -> torch.BoolTensor:
    blocked_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for tok_id in blocked_ids:
        if 0 <= tok_id < vocab_size:
            blocked_mask[tok_id] = True
    return blocked_mask


def _canonicalize_smiles(smiles: str) -> tuple[str, Chem.Mol, list[int]]:
    import ast
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"QwenAllowlistLP: Could not parse SMILES: {smiles}")
    mol_no_h = Chem.RemoveHs(mol)
    smiles_can = Chem.MolToSmiles(
        mol_no_h, canonical=True, isomericSmiles=True,
        allHsExplicit=False, allBondsExplicit=False,
    )
    if not mol_no_h.HasProp("_smilesAtomOutputOrder"):
        raise ValueError("QwenAllowlistLP: Mol missing _smilesAtomOutputOrder")
    atom_order = list(map(int, ast.literal_eval(mol_no_h.GetProp("_smilesAtomOutputOrder"))))
    return smiles_can, mol_no_h, atom_order


def build_reference_skeleton(smiles: str) -> str:
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
    use_tokenizer_cache: bool = False,
) -> PrecomputedTemplate:
    skeleton = build_reference_skeleton(smiles)
    ref_str = f"[CONFORMER]{skeleton}[/CONFORMER]" if include_tags else skeleton

    ref_ids, offset_mapping = _cached_encode_plus(
        ref_str, tokenizer, use_cache=use_tokenizer_cache
    )

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

    is_free = []
    for start_char, end_char in offset_mapping:
        if start_char >= end_char:
            is_free.append(False)
        else:
            token_chars = char_is_free[start_char:end_char]
            is_free.append(all(token_chars))

    LOOKAHEAD_RANGE = 2
    block_comma_dash = []
    is_first_free = []

    for pos in range(len(is_free)):
        should_block_last = False
        first_free = False

        if is_free[pos]:
            if pos == 0 or not is_free[pos - 1]:
                first_free = True
            for offset in range(1, LOOKAHEAD_RANGE + 1):
                next_pos = pos + offset
                if next_pos >= len(is_free):
                    break
                if not is_free[next_pos]:
                    next_token_str = tokenizer.decode([ref_ids[next_pos]])
                    if '>' in next_token_str:
                        should_block_last = True
                    break

        block_comma_dash.append(should_block_last)
        is_first_free.append(first_free)

    return PrecomputedTemplate(
        ref_ids=torch.tensor(ref_ids, dtype=torch.long),
        is_free=torch.tensor(is_free, dtype=torch.bool),
        seq_len=len(ref_ids),
        block_comma_dash=torch.tensor(block_comma_dash, dtype=torch.bool),
        is_first_free=torch.tensor(is_first_free, dtype=torch.bool),
    )


class QwenAllowlistLogitsProcessor(LogitsProcessor):
    """
    Strict ALLOWLIST approach - ONLY coordinate tokens (0-9, ., ,, -) in FREE positions.

    At each position:
    - If is_free[pos]: allow ONLY the 66 valid coordinate tokens
    - Else: force ref_ids[pos]

    v4.3: Smart position blocking - first FREE blocks comma, last FREE blocks comma+dash.
    """

    VERSION = "v4.3_allowlist_smart"
    CONFIG = {
        "approach": "strict_allowlist_smart_blocking",
        "coord_placeholder": COORD_PLACEHOLDER,
        "lookahead_range": 2,
        "allowed_chars": "0123456789.,-",
        "first_free_blocks": "comma_only",
        "last_free_blocks": "comma_and_dash",
        "description": "Strict allowlist with smart position blocking to prevent structural errors",
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
            raise ValueError("QwenAllowlistLP: templates and prompt_lengths must match")

        self.templates = templates
        self.prompt_lengths = list(prompt_lengths)
        self.eos_token_id = eos_token_id
        self.allowed_ids = _get_allowed_coord_token_ids(tokenizer)
        self.comma_dash_ids = _get_comma_dash_token_ids(tokenizer)
        self.comma_only_ids = _get_comma_only_token_ids(tokenizer)
        self.allowed_mask = None
        self.comma_dash_mask = None
        self.comma_only_mask = None
        self._device = None
        self._prev_lens: List[int | None] = [None] * len(templates)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, cur_len = input_ids.shape
        device = scores.device
        vocab_size = scores.shape[1]

        if self.allowed_mask is None or self._device != device:
            self.allowed_mask = _build_allowed_mask(self.allowed_ids, vocab_size, device)
            self.comma_dash_mask = _build_blocked_mask(self.comma_dash_ids, vocab_size, device)
            self.comma_only_mask = _build_blocked_mask(self.comma_only_ids, vocab_size, device)
            for template in self.templates:
                template.ref_ids = template.ref_ids.to(device)
                template.is_free = template.is_free.to(device)
                if template.block_comma_dash is not None:
                    template.block_comma_dash = template.block_comma_dash.to(device)
                if template.is_first_free is not None:
                    template.is_first_free = template.is_first_free.to(device)
            self._device = device

        for b in range(batch_size):
            template = self.templates[b]

            if self._prev_lens[b] is None:
                self._prev_lens[b] = cur_len

            pos = cur_len - self._prev_lens[b]

            if pos >= template.seq_len:
                if self.eos_token_id is not None:
                    scores[b, :] = float('-inf')
                    scores[b, self.eos_token_id] = 0.0
            elif template.is_free[pos]:
                scores[b, ~self.allowed_mask] = float('-inf')
                if template.is_first_free is not None and template.is_first_free[pos]:
                    scores[b, self.comma_only_mask] = float('-inf')
                if template.block_comma_dash is not None and template.block_comma_dash[pos]:
                    scores[b, self.comma_dash_mask] = float('-inf')
            else:
                scores[b, :] = float('-inf')
                scores[b, template.ref_ids[pos]] = 0.0

        return scores


def build_templates_for_batch(
    smiles_list: Sequence[str],
    tokenizer: PreTrainedTokenizer,
    *,
    use_tokenizer_cache: bool = False,
) -> List[PrecomputedTemplate]:
    return [
        build_precomputed_template(smi, tokenizer, use_tokenizer_cache=use_tokenizer_cache)
        for smi in smiles_list
    ]
