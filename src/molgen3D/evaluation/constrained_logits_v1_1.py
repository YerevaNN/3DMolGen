"""
Constrained Conformer Generation - v1.1 Baseline

Version 1.1: Structural constraints with minimal robustness fixes.

Based on v1 (structural only), but adds:
1. Block special tags ([CONFORMER], [/CONFORMER], [SMILES], [/SMILES]) in coord blocks
   - These can NEVER be valid inside <x,y,z>
2. Max coord tokens safety limit - force > after N tokens
   - Prevents infinite loops if model generates very long garbage

IMPORTANT: Does NOT use "smart > detection" (requiring preceding digit).
That approach caused infinite > loops when combined with max_coord_tokens.
Simple > detection is more robust.
"""
from __future__ import annotations

from collections import deque
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


@dataclass
class Segment:
    """Represents one contiguous chunk of expected output."""
    kind: str  # "fixed" or "coord"
    tokens: List[int]  # token ids for fixed segments; empty for coord
    label: str  # debugging label (e.g., "[C]", "=")


@dataclass
class SequenceTemplate:
    """Holds the ordered constraint segments for a single prompt."""
    segments: List[Segment]
    start_coord_tokens: List[int]  # tokenizer ids for "<"
    end_coord_tokens: List[int]  # tokenizer ids for ">"
    # v1.1 additions
    blocked_in_coords: set  # token ids that are NEVER valid in coords (special tags)


@dataclass
class _SeqState:
    """Mutable tracking of generation progress for one sequence."""
    prompt_len: int  # number of non-pad tokens in the prompt
    prev_len: int | None = None  # total length seen last call (including padding)
    seg_idx: int = 0  # which segment we are in
    seg_offset: int = 0  # index within the current fixed segment
    coord_window: deque | None = None  # rolling window to detect '>' in coord blocks
    coord_prefix_offset: int = 0  # number of '<' tokens already matched in coord block
    done: bool = False  # template fully consumed
    # v1.1 addition
    coord_token_count: int = 0  # tokens generated in current coord block (after '<')


# Cache for tokenizer-specific blocked token sets (computed once per tokenizer)
_BLOCKED_TOKENS_CACHE: dict = {}


def _get_blocked_tokens(tokenizer: PreTrainedTokenizer) -> set:
    """
    Get tokens that should be blocked in coordinate blocks.

    These are special tags that can NEVER be valid inside <x,y,z>.
    """
    cache_key = id(tokenizer)
    if cache_key in _BLOCKED_TOKENS_CACHE:
        return _BLOCKED_TOKENS_CACHE[cache_key]

    # Special tag tokens to block in coordinate blocks
    special_tags = [
        "[CONFORMER]", "[/CONFORMER]",
        "[SMILES]", "[/SMILES]",
        "<|begin_of_text|>", "<|end_of_text|>",
    ]
    blocked = set()
    for tag in special_tags:
        try:
            tokens = tokenizer.encode(tag, add_special_tokens=False)
            blocked.update(tokens)
        except:
            pass

    # Also block EOS/BOS if they exist
    if tokenizer.eos_token_id is not None:
        blocked.add(tokenizer.eos_token_id)
    if tokenizer.bos_token_id is not None:
        blocked.add(tokenizer.bos_token_id)

    _BLOCKED_TOKENS_CACHE[cache_key] = blocked
    return blocked


def _canonicalize_smiles(smiles: str) -> tuple[str, Chem.Mol, list[int]]:
    """Return canonical smiles, RDKit mol without Hs, and atom order used in SMILES output."""
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
    import ast
    atom_order = list(map(int, ast.literal_eval(atom_order_raw)))
    return smiles_can, mol_no_h, atom_order


def build_sequence_template(
    smiles: str,
    tokenizer: PreTrainedTokenizer,
    *,
    include_tags: bool = True,
) -> SequenceTemplate:
    """
    Build the ordered constraint segments for a given SMILES.

    - Every SMILES token (atoms, bonds, branches, ring digits) is emitted as a fixed segment.
    - After each atom, a coordinate block segment is inserted where logits are UNCONSTRAINED
      (except for blocked special tokens).
    - By default we also prepend `[CONFORMER]` and append `[/CONFORMER]` as fixed segments.
    """
    smiles_can, mol_no_h, atom_order = _canonicalize_smiles(smiles)

    expected_atom_tokens = [
        _expected_plain_token(mol_no_h.GetAtomWithIdx(idx)) for idx in atom_order
    ]
    tokens = tokenize_smiles(smiles_can, expected_atom_tokens=expected_atom_tokens)

    segments: List[Segment] = []
    if include_tags:
        segments.append(
            Segment(
                kind="fixed",
                tokens=tokenizer.encode("[CONFORMER]", add_special_tokens=False),
                label="[CONFORMER]",
            )
        )

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

            segments.append(
                Segment(
                    kind="fixed",
                    tokens=tokenizer.encode(atom_descriptor, add_special_tokens=False),
                    label=atom_descriptor,
                )
            )
            segments.append(Segment(kind="coord", tokens=[], label="<coords>"))
            atom_idx_in_smiles += 1
        else:
            # non-atom SMILES tokens (bonds, branches, ring digits, dots)
            segments.append(
                Segment(
                    kind="fixed",
                    tokens=tokenizer.encode(tok["text"], add_special_tokens=False),
                    label=tok["text"],
                )
            )

    if include_tags:
        segments.append(
            Segment(
                kind="fixed",
                tokens=tokenizer.encode("[/CONFORMER]", add_special_tokens=False),
                label="[/CONFORMER]",
            )
        )

    start_coord_tokens = tokenizer.encode("<", add_special_tokens=False)
    end_coord_tokens = tokenizer.encode(">", add_special_tokens=False)

    # v1.1: Get blocked tokens for coord blocks
    blocked_in_coords = _get_blocked_tokens(tokenizer)

    return SequenceTemplate(
        segments=segments,
        start_coord_tokens=start_coord_tokens,
        end_coord_tokens=end_coord_tokens,
        blocked_in_coords=blocked_in_coords,
    )


class ConformerConstraintLogitsProcessorV1_1(LogitsProcessor):
    """
    V1.1 Baseline: Structural constraints with minimal coord protection.

    Forces exact SMILES structure (atoms, bonds, rings, branches) while applying
    minimal constraints to coordinate blocks:
    - Blocks special tags ([CONFORMER], [/CONFORMER], etc.) in coord blocks
    - Max coord tokens safety limit (forces > after N tokens)

    Uses SIMPLE > detection (any > ends coord block).
    Does NOT use "smart > detection" (that caused infinite > loops).
    """

    VERSION = "v1.1_structural_with_protection"
    CONFIG = {
        "approach": "structural_with_minimal_coord_protection",
        "coord_constraints": "block_special_tags_only",
        "smart_gt_detection": False,  # Simple > detection
        "max_coord_tokens": 100,
        "description": "Forces SMILES structure, blocks special tags in coords, simple > detection"
    }

    # Default maximum tokens in a coordinate block before forcing >
    # Note: Valid coords only need ~12-15 tokens, but model may generate
    # garbage first. Higher limit lets model "find its way" to valid coords.
    DEFAULT_MAX_COORD_TOKENS = 100

    def __init__(
        self,
        templates: Sequence[SequenceTemplate],
        prompt_lengths: Sequence[int],
        *,
        tokenizer: PreTrainedTokenizer | None = None,
        eos_token_id: int | None = None,
        max_coord_tokens: int | None = None,
    ):
        if len(templates) != len(prompt_lengths):
            raise ValueError("templates and prompt_lengths must have the same length.")
        self.templates = templates
        self.states: List[_SeqState] = []

        # EOS token to force after completing all segments
        self.eos_token_id = eos_token_id

        # v1.1: Max tokens in a coordinate block
        self.max_coord_tokens = max_coord_tokens or self.DEFAULT_MAX_COORD_TOKENS

        for prompt_len in prompt_lengths:
            self.states.append(
                _SeqState(
                    prompt_len=prompt_len,
                    coord_window=None,
                )
            )

    def _advance_state(self, state: _SeqState, template: SequenceTemplate, new_tokens: Sequence[int]) -> None:
        """Consume newly generated tokens to advance the constraint state."""
        for tok in new_tokens:
            if state.done:
                return
            if state.seg_idx >= len(template.segments):
                state.done = True
                return

            seg = template.segments[state.seg_idx]
            if seg.kind == "fixed":
                # If mismatch occurs, stop advancing; masking will prevent further deviation.
                expected = seg.tokens[state.seg_offset]
                if tok != expected:
                    return
                state.seg_offset += 1
                if state.seg_offset >= len(seg.tokens):
                    state.seg_idx += 1
                    state.seg_offset = 0
            else:  # coord
                prefix = template.start_coord_tokens
                if state.coord_prefix_offset < len(prefix):
                    expected = prefix[state.coord_prefix_offset]
                    if tok != expected:
                        return
                    state.coord_prefix_offset += 1
                    if state.coord_prefix_offset == len(prefix):
                        state.coord_window = deque(maxlen=len(template.end_coord_tokens))
                        state.coord_token_count = 0
                    continue

                if state.coord_window is None:
                    state.coord_window = deque(maxlen=len(template.end_coord_tokens))
                    state.coord_token_count = 0

                state.coord_window.append(tok)
                state.coord_token_count += 1

                # Simple > detection: any > ends coord block
                if (
                    len(state.coord_window) == len(template.end_coord_tokens)
                    and list(state.coord_window) == template.end_coord_tokens
                ):
                    # End of coord block detected
                    state.seg_idx += 1
                    state.seg_offset = 0
                    state.coord_window.clear()
                    state.coord_window = None
                    state.coord_prefix_offset = 0
                    state.coord_token_count = 0

        if state.seg_idx >= len(template.segments):
            state.done = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, cur_len = input_ids.shape
        if batch_size != len(self.states):
            raise ValueError("Batch size does not match number of tracked states.")

        for b in range(batch_size):
            state = self.states[b]
            template = self.templates[b]

            # On first call, align prev_len to the padded input length
            if state.prev_len is None:
                state.prev_len = cur_len

            # Advance using any new tokens since last call
            if cur_len > state.prev_len:
                new_tokens = input_ids[b, state.prev_len:cur_len].tolist()
                self._advance_state(state, template, new_tokens)
                state.prev_len = cur_len

            if state.done or state.seg_idx >= len(template.segments):
                # Template fully consumed - force EOS to stop generation
                if self.eos_token_id is not None:
                    scores[b, :] = -torch.inf
                    scores[b, self.eos_token_id] = 0.0
                continue

            seg = template.segments[state.seg_idx]
            if seg.kind == "fixed":
                # Fixed segment: force exact token
                # (100% probability for correct, 0% for all others)
                expected = seg.tokens[state.seg_offset]
                scores[b, :] = -torch.inf
                scores[b, expected] = 0.0
            else:
                # Coordinate segment
                prefix = template.start_coord_tokens
                suffix = template.end_coord_tokens
                if state.coord_prefix_offset < len(prefix):
                    # Still emitting the '<' prefix - force it
                    expected = prefix[state.coord_prefix_offset]
                    scores[b, :] = -torch.inf
                    scores[b, expected] = 0.0
                else:
                    # Inside coordinate block <x,y,z>
                    # v1.1: Check if we've hit max_coord_tokens limit
                    if state.coord_token_count >= self.max_coord_tokens:
                        # Force > to end the coordinate block
                        scores[b, :] = -torch.inf
                        for t in suffix:
                            scores[b, t] = 0.0
                    else:
                        # v1.1: Block special tokens in coord blocks
                        # These can NEVER be valid inside <x,y,z>
                        for blocked_id in template.blocked_in_coords:
                            scores[b, blocked_id] = -torch.inf
                        # Otherwise let model generate freely

        return scores


def build_templates_for_batch(smiles_list: Sequence[str], tokenizer: PreTrainedTokenizer) -> List[SequenceTemplate]:
    """Convenience helper to build templates for a batch of SMILES strings."""
    return [build_sequence_template(smi, tokenizer) for smi in smiles_list]
