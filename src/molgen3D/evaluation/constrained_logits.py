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
    end_coord_tokens: List[int]  # tokenizer ids for ">"


@dataclass
class _SeqState:
    """Mutable tracking of generation progress for one sequence."""

    prompt_len: int  # number of non-pad tokens in the prompt
    prev_len: int | None = None  # total length seen last call (including padding)
    seg_idx: int = 0  # which segment we are in
    seg_offset: int = 0  # index within the current fixed segment
    coord_window: deque | None = None  # rolling window to detect '>' in coord blocks
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

    end_coord_tokens = tokenizer.encode(">", add_special_tokens=False)
    return SequenceTemplate(segments=segments, end_coord_tokens=end_coord_tokens)


class ConformerConstraintLogitsProcessor(LogitsProcessor):
    """
    Masks logits to force a predetermined token sequence (non-coordinate SMILES structure),
    while leaving coordinate blocks unconstrained.
    """

    def __init__(self, templates: Sequence[SequenceTemplate], prompt_lengths: Sequence[int]):
        if len(templates) != len(prompt_lengths):
            raise ValueError("templates and prompt_lengths must have the same length.")
        self.templates = templates
        self.states: List[_SeqState] = []
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
                if state.coord_window is None:
                    state.coord_window = deque(maxlen=len(template.end_coord_tokens))
                state.coord_window.append(tok)
                if (
                    len(state.coord_window) == len(template.end_coord_tokens)
                    and list(state.coord_window) == template.end_coord_tokens
                ):
                    # end of coord block detected
                    state.seg_idx += 1
                    state.seg_offset = 0
                    state.coord_window.clear()
                    state.coord_window = None

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
                expected = seg.tokens[state.seg_offset]
                scores[b, :] = -torch.inf
                scores[b, expected] = 0.0
            # coord block -> leave logits unchanged

        return scores


def build_templates_for_batch(smiles_list: Sequence[str], tokenizer: PreTrainedTokenizer) -> List[SequenceTemplate]:
    """Convenience helper to build templates for a batch of SMILES strings."""
    return [build_sequence_template(smi, tokenizer) for smi in smiles_list]
