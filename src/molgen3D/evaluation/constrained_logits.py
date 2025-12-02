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


@dataclass
class _SeqState:
    """Mutable tracking of generation progress for one sequence."""

    prompt_len: int  # number of non-pad tokens in the prompt
    prev_len: int | None = None  # total length seen last call (including padding)
    seg_idx: int = 0  # which segment we are in
    seg_offset: int = 0  # index within the current fixed segment
    coord_window: deque | None = None  # rolling window to detect '>' in coord blocks
    coord_prefix_offset: int = 0  # number of '<' tokens already matched in coord block
    coord_token_count: int = 0  # tokens generated after '<' inside current coord block
    done: bool = False  # template fully consumed


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
    # The property is a stringified list, e.g., "[0, 1, 2, ...]"
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
    - After each atom, a coordinate block segment is inserted where logits are left unconstrained
      until the first occurrence of the tokenizer-encoded '>' token.
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
    return SequenceTemplate(
        segments=segments,
        start_coord_tokens=start_coord_tokens,
        end_coord_tokens=end_coord_tokens,
    )


def _build_forbidden_ids(tokenizer: PreTrainedTokenizer, max_token_length: int = 8) -> set[int]:
    """
    Collect token ids that should never appear inside coordinate spans.

    V11 minimal approach: Simple blocklist without complex allowlist.
    Block tokens that:
    1. Contain structural fragments (brackets, tags)
    2. Are special tokens (BOS/EOS)
    3. Are very long (likely to be garbage like "120157094632576653")
    4. Are repeated punctuation (e.g., ",,,,", "-----")
    """
    vocab = tokenizer.get_vocab()
    forbidden = set()

    # Structural pollution
    bad_fragments = [
        "[",
        "]",
        "SMILES",
        "CONFORMER",
        "<|end_of_text|>",
        "<|begin_of_text|>",
    ]

    for tok, idx in vocab.items():
        # Check for bad fragments
        if any(frag in tok for frag in bad_fragments):
            forbidden.add(idx)
            continue

        # Remove SentencePiece boundary marker for length check
        core = tok.replace("▁", "")

        # Block very long tokens (likely garbage integers)
        if len(core) > max_token_length:
            forbidden.add(idx)
            continue

        # Block repeated punctuation (e.g., ",,,,", "-----", "......")
        if len(core) > 2 and len(set(core)) == 1 and core[0] in ",-+.":
            forbidden.add(idx)
            continue

    return forbidden


class ConformerConstraintLogitsProcessor(LogitsProcessor):
    """
    Masks logits to force a predetermined token sequence (non-coordinate SMILES structure),
    while leaving coordinate blocks minimally constrained.

    V11 approach: Simple blocklist for coordinates, trust model to generate valid patterns.
    """

    # Class-level version tracking
    VERSION = "v11"
    CONFIG = {
        "approach": "minimal_blocklist",
        "max_coord_tokens": 20,  # Increased from 6 to give model more room
        "coord_forbidden_enabled": True,
        "coord_allowlist_enabled": False,
        "max_token_length": 8,
        "blocks_repeated_punctuation": True,
    }

    def __init__(
        self,
        templates: Sequence[SequenceTemplate],
        prompt_lengths: Sequence[int],
        *,
        tokenizer: PreTrainedTokenizer | None = None,
        max_coord_tokens: int | None = None,
    ):
        if len(templates) != len(prompt_lengths):
            raise ValueError("templates and prompt_lengths must have the same length.")
        self.templates = templates
        self.states: List[_SeqState] = []
        self.max_coord_tokens = max_coord_tokens or self.CONFIG["max_coord_tokens"]

        # V11: Only use forbidden set, no allowlist
        self.coord_forbidden: set[int] = set()
        if tokenizer is not None:
            self.coord_forbidden = _build_forbidden_ids(
                tokenizer,
                max_token_length=self.CONFIG["max_token_length"]
            )

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
                state.coord_window.append(tok)
                state.coord_token_count += 1
                if (
                    len(state.coord_window) == len(template.end_coord_tokens)
                    and list(state.coord_window) == template.end_coord_tokens
                ):
                    # end of coord block detected
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
            # On first call, align prev_len to the padded input length to avoid counting pads as generated.
            if state.prev_len is None:
                state.prev_len = cur_len
            # Advance using any new tokens since last call
            if cur_len > state.prev_len:
                new_tokens = input_ids[b, state.prev_len:cur_len].tolist()
                self._advance_state(state, template, new_tokens)
                state.prev_len = cur_len

            if state.done or state.seg_idx >= len(template.segments):
                continue

            seg = template.segments[state.seg_idx]
            if seg.kind == "fixed":
                # Fixed segment: force exact token
                expected = seg.tokens[state.seg_offset]
                scores[b, :] = -torch.inf
                scores[b, expected] = 0.0
            else:
                # Coordinate segment
                prefix = template.start_coord_tokens
                if state.coord_prefix_offset < len(prefix):
                    # Still emitting the '<' prefix
                    expected = prefix[state.coord_prefix_offset]
                    scores[b, :] = -torch.inf
                    scores[b, expected] = 0.0
                else:
                    # Inside coordinate block <x,y,z>
                    # V11 minimal approach: Only use forbidden blocklist
                    end_ids = template.end_coord_tokens

                    if state.coord_token_count >= self.max_coord_tokens:
                        # Safety limit reached, force close bracket
                        scores[b, :] = -torch.inf
                        scores[b, end_ids] = 0.0
                    elif self.coord_forbidden:
                        # Block known-bad tokens, allow everything else
                        scores[b, list(self.coord_forbidden)] = -torch.inf
                    # Otherwise leave logits unchanged - trust the model

        return scores


def build_templates_for_batch(smiles_list: Sequence[str], tokenizer: PreTrainedTokenizer) -> List[SequenceTemplate]:
    """Convenience helper to build templates for a batch of SMILES strings."""
    return [build_sequence_template(smi, tokenizer) for smi in smiles_list]
