"""Turn a generated token sequence into a molecule with a conformer.

Both arms share every step except the geometry decode, and every rejection is
attributed a reason from the closed :class:`FailureReason` vocabulary. Nothing is
optimized, energy-ranked or RMSD-filtered here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

from molgen3D.data_processing.smiles_encoder_decoder import strip_smiles
from molgen3D.experiments.bindgpt_fsq.sequences import (
    Arm,
    FailureReason,
    Layout,
    SequenceFormatError,
    parse_coord_stream,
    parse_coord_suffix,
    parse_fsq_stream,
    parse_fsq_suffix,
    rebracket,
    split_sequence,
)

RDLogger.DisableLog("rdApp.*")


@dataclass
class DecodedGeneration:
    """One parsed attempt, successful or not."""

    tokens: Sequence[str]
    arm: str
    prompt_smiles: Optional[str] = None
    left_smiles: Optional[str] = None
    right_smiles: Optional[str] = None
    canonical_smiles: Optional[str] = None
    coordinates: Optional[np.ndarray] = None
    mol: Optional[Chem.Mol] = None
    graph_consistent: Optional[bool] = None
    matches_prompt: Optional[bool] = None
    connected: Optional[bool] = None
    failure_reason: Optional[str] = None
    flags: Dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.failure_reason is None and self.mol is not None


def _canonical(smiles: str, isomeric: bool = True) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric, canonical=True)


def decode_generation(
    tokens: Sequence[str],
    arm: Arm,
    fsq_tokenizer=None,
    prompt_smiles: Optional[str] = None,
    hit_length_limit: bool = False,
    layout: str = Layout.SUFFIX.value,
) -> DecodedGeneration:
    """Parse one generated sequence into a molecule with attached coordinates.

    Under the suffix layout the graph is *derived* from the generated SMILES
    rather than transmitted, so a graph/SMILES disagreement is structurally
    impossible and that failure mode disappears. What remains is a wrong number
    of geometry tokens, which is reported as ``atom_count_mismatch``.
    """
    result = DecodedGeneration(tokens=list(tokens), arm=arm.value, prompt_smiles=prompt_smiles)

    try:
        smiles_tokens, stream = split_sequence(tokens)
    except SequenceFormatError as exc:
        result.failure_reason = (
            FailureReason.SEQUENCE_LENGTH_LIMIT.value
            if hit_length_limit
            else exc.reason.value
        )
        return result

    left_text = "".join(smiles_tokens)
    result.left_smiles = left_text
    # RDKit parses the empty string into an empty molecule rather than failing,
    # so an empty left-hand SMILES has to be rejected explicitly.
    left_canonical = _canonical(left_text) if left_text else None
    if not left_canonical:
        result.failure_reason = FailureReason.SMILES_PARSE_FAILURE.value
        return result

    try:
        if layout == Layout.SUFFIX.value:
            # The decoder's token sequence comes from the SMILES, exactly as it
            # did when the example was built.
            graph_tokens = rebracket(left_text)
            if arm is Arm.COORD:
                coordinates = parse_coord_suffix(stream, graph_tokens)
            else:
                codes = parse_fsq_suffix(stream, graph_tokens)
                if fsq_tokenizer is None:
                    raise SequenceFormatError(
                        FailureReason.COORDINATE_DECODE_FAILURE,
                        "the FSQ arm needs the frozen tokenizer to decode",
                    )
                coordinates = fsq_tokenizer.decode(graph_tokens, codes)
        elif arm is Arm.COORD:
            graph_tokens, coordinates = parse_coord_stream(stream)
        else:
            graph_tokens, codes = parse_fsq_stream(stream)
            if fsq_tokenizer is None:
                raise SequenceFormatError(
                    FailureReason.COORDINATE_DECODE_FAILURE,
                    "the FSQ arm needs the frozen tokenizer to decode",
                )
            coordinates = fsq_tokenizer.decode(graph_tokens, codes)
    except SequenceFormatError as exc:
        result.failure_reason = exc.reason.value
        return result
    except Exception as exc:  # noqa: BLE001 - attributed, never dropped
        result.failure_reason = FailureReason.COORDINATE_DECODE_FAILURE.value
        result.flags["decode_error"] = str(exc)[:200]
        return result

    right_text = (
        left_text if layout == Layout.SUFFIX.value
        else strip_smiles("".join(graph_tokens))
    )
    result.right_smiles = right_text
    right_mol = Chem.MolFromSmiles(right_text, sanitize=False)
    if right_mol is None:
        result.failure_reason = FailureReason.SMILES_PARSE_FAILURE.value
        return result
    try:
        Chem.SanitizeMol(right_mol)
    except Exception:  # noqa: BLE001
        result.failure_reason = FailureReason.SANITIZATION_FAILURE.value
        return result

    if right_mol.GetNumAtoms() != coordinates.shape[0]:
        result.failure_reason = FailureReason.ATOM_COUNT_MISMATCH.value
        result.flags["atoms"] = right_mol.GetNumAtoms()
        result.flags["coordinates"] = int(coordinates.shape[0])
        return result

    if not np.isfinite(coordinates).all():
        result.failure_reason = FailureReason.NONFINITE_COORDINATE.value
        return result

    right_canonical = Chem.MolToSmiles(right_mol, isomericSmiles=True, canonical=True)
    result.canonical_smiles = right_canonical
    result.graph_consistent = right_canonical == left_canonical
    result.flags["left_canonical"] = left_canonical
    if not result.graph_consistent:
        result.failure_reason = FailureReason.GRAPH_MISMATCH.value
        return result

    fragments = Chem.GetMolFrags(right_mol)
    result.connected = len(fragments) == 1

    conformer = Chem.Conformer(right_mol.GetNumAtoms())
    for index, (x, y, z) in enumerate(coordinates):
        conformer.SetAtomPosition(index, Point3D(float(x), float(y), float(z)))
    molecule = Chem.Mol(right_mol)
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)

    result.coordinates = coordinates
    result.mol = molecule

    if prompt_smiles is not None:
        prompt_canonical = _canonical(prompt_smiles)
        result.matches_prompt = prompt_canonical == right_canonical
        if not result.matches_prompt:
            result.failure_reason = FailureReason.GRAPH_MISMATCH.value
    return result


def record(result: DecodedGeneration, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Flatten one attempt into a storable record, raw text always kept."""
    payload: Dict[str, object] = {
        "arm": result.arm,
        "raw_tokens": " ".join(result.tokens),
        "prompt_smiles": result.prompt_smiles,
        "left_smiles": result.left_smiles,
        "right_smiles": result.right_smiles,
        "parsed_canonical_smiles": result.canonical_smiles,
        "graph_consistent": result.graph_consistent,
        "matches_prompt": result.matches_prompt,
        "connected": result.connected,
        "failure_reason": result.failure_reason,
        "coordinates": (
            [[round(float(value), 4) for value in row] for row in result.coordinates]
            if result.coordinates is not None
            else None
        ),
        "flags": result.flags,
    }
    if extra:
        payload.update(extra)
    return payload
