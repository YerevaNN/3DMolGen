"""The paired sequence format shared by both arms of the experiment.

Every training example is one molecule-conformer pair spelled as

    <LIGAND> {canonical isomeric SMILES} <XYZ> {geometry stream} <EOS>

The left-hand side is the plain canonical isomeric SMILES in BindGPT's
element-level tokens. The right-hand side repeats the molecular graph -- in the
frozen FSQ tokenizer's bracketed-atom vocabulary, because that vocabulary is
what the FSQ encoder and decoder were trained on -- with the geometry
interleaved in each arm's native position:

* ``coord``: six BindGPT scalar coordinate tokens after every *atom-bearing*
  graph token; non-atom graph tokens carry nothing.
* ``fsq``: one FSQ code token after *every* graph token, atom or not, which is
  the representation the frozen tokenizer actually emits.

Repeating the graph on the right is a deliberate adaptation of the published
BindGPT layout, which has a coordinate-only suffix: the FSQ codes cannot be
lifted out of the graph without destroying the representation, so instead both
arms pay the same graph-repetition cost. Delimiters, graph vocabulary and
graph-token loss treatment are therefore identical across arms, and only the
geometry tokens differ.

Token sequences are handled as lists of token strings throughout and stored
space-joined; no token contains a space, so the round trip is exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np

from molgen3D.data_processing.smiles_encoder_decoder_fsq import (
    VOCAB as FSQ_GRAPH_VOCAB,
    tokenize_enriched,
)
from molgen3D.experiments.bindgpt_fsq import coord_tokens

LIGAND = "<LIGAND>"
XYZ = "<XYZ>"
EOS = "<EOS>"
PAD = "<PAD>"
UNK = "<UNK>"

SPECIAL_TOKENS = (EOS, UNK, PAD, XYZ, LIGAND)

#: Graph tokens that carry a coordinate triplet (the bracketed atom tokens).
ATOM_GRAPH_TOKENS = tuple(
    token for token in FSQ_GRAPH_VOCAB if token.startswith("[")
)
_ATOM_GRAPH_SET = frozenset(ATOM_GRAPH_TOKENS)
_GRAPH_SET = frozenset(FSQ_GRAPH_VOCAB)


class Layout(str, Enum):
    """Where the geometry sits relative to the molecular graph.

    ``SUFFIX`` is published BindGPT's layout: the SMILES, the separator, then
    geometry only. It works for FSQ codes too, because the graph token sequence
    the decoder needs is a pure syntactic function of the SMILES -- see
    :func:`rebracket` -- so the codes can be emitted bare and re-paired with
    their tokens at decode time.

    ``INTERLEAVED`` repeats the graph after the separator with geometry woven
    into it. Retained only to reproduce the first run, which used it on the
    mistaken belief that FSQ codes could not be placed in a suffix.
    """

    SUFFIX = "suffix"
    INTERLEAVED = "interleaved"


class Arm(str, Enum):
    """The two experimental conditions."""

    COORD = "coord"
    FSQ = "fsq"


class FailureReason(str, Enum):
    """Closed vocabulary for parse and decode failures."""

    MISSING_SPECIAL_TOKEN = "missing_special_token"
    SMILES_PARSE_FAILURE = "smiles_parse_failure"
    SANITIZATION_FAILURE = "sanitization_failure"
    DISCONNECTED = "disconnected"
    COORDINATE_DECODE_FAILURE = "coordinate_decode_failure"
    ATOM_COUNT_MISMATCH = "atom_count_mismatch"
    GRAPH_MISMATCH = "graph_mismatch"
    NONFINITE_COORDINATE = "nonfinite_coordinate"
    SEQUENCE_LENGTH_LIMIT = "sequence_length_limit"
    OTHER = "other"


class SequenceFormatError(ValueError):
    """Raised when a sequence cannot be built or parsed."""

    def __init__(self, reason: FailureReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class GraphToken:
    """One token of the repeated right-hand graph."""

    text: str
    coords: Optional[Tuple[float, float, float]] = None

    @property
    def is_atom(self) -> bool:
        return self.text in _ATOM_GRAPH_SET


#: Atoms the enriched format brackets without an explicit hydrogen count.
_ORGANIC_SUBSET = frozenset("CNOSPFIB") | frozenset("cnosp")
_TWO_LETTER = ("Cl", "Br")


def rebracket(smiles: str) -> List[str]:
    """Return the FSQ graph token sequence for a plain SMILES.

    A pure string operation and the exact inverse of the enriched format's
    un-bracketing: organic-subset atoms gain brackets, already-decorated atoms
    (``[C@@H]``, ``[nH]``, ``[O-]``) pass through, and structural characters are
    single tokens. Verified to reproduce the enriched graph tokens exactly on
    4,994 of 4,994 connected validation molecules, which is what makes the
    suffix layout possible: the decoder's token sequence never has to be
    transmitted, only derived.
    """
    tokens: List[str] = []
    index = 0
    while index < len(smiles):
        char = smiles[index]
        if char == "[":
            close = smiles.find("]", index)
            if close < 0:
                raise SequenceFormatError(
                    FailureReason.SMILES_PARSE_FAILURE,
                    f"unclosed bracket atom in {smiles!r}",
                )
            tokens.append(smiles[index : close + 1])
            index = close + 1
        elif smiles[index : index + 2] in _TWO_LETTER:
            tokens.append(f"[{smiles[index:index + 2]}]")
            index += 2
        elif char in _ORGANIC_SUBSET:
            tokens.append(f"[{char}]")
            index += 1
        elif char == "%":
            tokens.append(smiles[index : index + 3])
            index += 3
        else:
            tokens.append(char)
            index += 1

    unknown = [token for token in tokens if token not in _GRAPH_SET]
    if unknown:
        raise SequenceFormatError(
            FailureReason.GRAPH_MISMATCH,
            f"re-bracketing produced tokens outside the graph vocabulary: {unknown[:6]}",
        )
    return tokens


def unbracket(graph_tokens: Sequence[str]) -> str:
    """Return the plain SMILES for a graph token sequence (inverse of rebracket)."""
    out: List[str] = []
    for token in graph_tokens:
        if token.startswith("[") and token.endswith("]") and len(token) == 3:
            out.append(token[1])
        elif token in ("[Cl]", "[Br]"):
            out.append(token[1:-1])
        else:
            out.append(token)
    return "".join(out)


def parse_enriched(enriched_text: str) -> List[GraphToken]:
    """Parse a decimal enriched string into graph tokens with coordinates."""
    parsed: List[GraphToken] = []
    for token in tokenize_enriched(enriched_text):
        if token["type"] == "atom_with_coords":
            coords = tuple(float(value) for value in token["coords"])  # type: ignore[arg-type]
            parsed.append(GraphToken(str(token["atom_desc"]), coords))
        else:
            parsed.append(GraphToken(str(token["text"])))
    return parsed


def graph_token_texts(tokens: Sequence[GraphToken]) -> List[str]:
    """Return the bare graph token strings, dropping geometry."""
    return [token.text for token in tokens]


def atom_coordinates(tokens: Sequence[GraphToken]) -> np.ndarray:
    """Return the ``(n_atoms, 3)`` coordinate array in graph order."""
    coords = [token.coords for token in tokens if token.is_atom]
    if any(value is None for value in coords):
        raise SequenceFormatError(
            FailureReason.COORDINATE_DECODE_FAILURE,
            "atom-bearing graph token without coordinates",
        )
    return np.asarray(coords, dtype=np.float64).reshape(-1, 3)


# --------------------------------------------------------------------------- #
# Geometry streams
# --------------------------------------------------------------------------- #


def build_coord_stream(tokens: Sequence[GraphToken]) -> List[str]:
    """Interleave BindGPT scalar coordinate tokens into the repeated graph."""
    stream: List[str] = []
    for token in tokens:
        stream.append(token.text)
        if not token.is_atom:
            continue
        if token.coords is None:
            raise SequenceFormatError(
                FailureReason.COORDINATE_DECODE_FAILURE,
                f"atom token {token.text!r} has no coordinates",
            )
        stream.extend(coord_tokens.triplet_to_tokens(token.coords))
    return stream


def parse_coord_stream(stream: Sequence[str]) -> Tuple[List[str], np.ndarray]:
    """Invert :func:`build_coord_stream`.

    Returns the graph token strings and the ``(n_atoms, 3)`` coordinates.
    """
    graph: List[str] = []
    coords: List[Tuple[float, float, float]] = []
    index = 0
    total = len(stream)
    while index < total:
        token = stream[index]
        if token not in _GRAPH_SET:
            raise SequenceFormatError(
                FailureReason.GRAPH_MISMATCH,
                f"unexpected token {token!r} where a graph token was required",
            )
        graph.append(token)
        index += 1
        if token not in _ATOM_GRAPH_SET:
            continue
        end = index + coord_tokens.TOKENS_PER_ATOM
        if end > total:
            raise SequenceFormatError(
                FailureReason.COORDINATE_DECODE_FAILURE,
                f"truncated coordinate triplet after {token!r}",
            )
        try:
            coords.append(coord_tokens.tokens_to_triplet(stream[index:end]))
        except ValueError as exc:
            raise SequenceFormatError(
                FailureReason.COORDINATE_DECODE_FAILURE, str(exc)
            ) from exc
        index = end

    if not graph:
        raise SequenceFormatError(
            FailureReason.COORDINATE_DECODE_FAILURE, "empty geometry stream"
        )
    return graph, np.asarray(coords, dtype=np.float64).reshape(-1, 3)


def fsq_code_token(code: int) -> str:
    """Return the LM token spelling FSQ code ``code``."""
    return f"<{int(code)}>"


def build_fsq_stream(graph: Sequence[str], codes: Sequence[int]) -> List[str]:
    """Interleave one FSQ code token after every graph token."""
    if len(graph) != len(codes):
        raise SequenceFormatError(
            FailureReason.ATOM_COUNT_MISMATCH,
            f"{len(graph)} graph tokens but {len(codes)} FSQ codes",
        )
    stream: List[str] = []
    for token, code in zip(graph, codes):
        stream.append(token)
        stream.append(fsq_code_token(code))
    return stream


def parse_fsq_stream(stream: Sequence[str]) -> Tuple[List[str], np.ndarray]:
    """Invert :func:`build_fsq_stream`."""
    if len(stream) % 2 != 0:
        raise SequenceFormatError(
            FailureReason.COORDINATE_DECODE_FAILURE,
            "FSQ stream has an odd token count, so a code is missing",
        )
    graph: List[str] = []
    codes: List[int] = []
    for position in range(0, len(stream), 2):
        token, code_token = stream[position], stream[position + 1]
        if token not in _GRAPH_SET:
            raise SequenceFormatError(
                FailureReason.GRAPH_MISMATCH,
                f"unexpected token {token!r} where a graph token was required",
            )
        if not (
            len(code_token) > 2
            and code_token[0] == "<"
            and code_token[-1] == ">"
            and code_token[1:-1].isdigit()
        ):
            raise SequenceFormatError(
                FailureReason.COORDINATE_DECODE_FAILURE,
                f"unexpected token {code_token!r} where an FSQ code was required",
            )
        graph.append(token)
        codes.append(int(code_token[1:-1]))

    if not graph:
        raise SequenceFormatError(
            FailureReason.COORDINATE_DECODE_FAILURE, "empty geometry stream"
        )
    return graph, np.asarray(codes, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Whole sequences
# --------------------------------------------------------------------------- #


def build_coord_suffix(tokens: Sequence[GraphToken]) -> List[str]:
    """Published BindGPT's layout: coordinate tokens only, in atom order."""
    suffix: List[str] = []
    for token in tokens:
        if not token.is_atom:
            continue
        if token.coords is None:
            raise SequenceFormatError(
                FailureReason.COORDINATE_DECODE_FAILURE,
                f"atom token {token.text!r} has no coordinates",
            )
        suffix.extend(coord_tokens.triplet_to_tokens(token.coords))
    return suffix


def parse_coord_suffix(
    suffix: Sequence[str], graph: Sequence[str]
) -> np.ndarray:
    """Invert :func:`build_coord_suffix` given the graph derived from the SMILES."""
    expected_atoms = sum(1 for token in graph if token in _ATOM_GRAPH_SET)
    if len(suffix) != expected_atoms * coord_tokens.TOKENS_PER_ATOM:
        raise SequenceFormatError(
            FailureReason.ATOM_COUNT_MISMATCH,
            f"{len(suffix)} coordinate tokens for {expected_atoms} atoms "
            f"(expected {expected_atoms * coord_tokens.TOKENS_PER_ATOM})",
        )
    coords = []
    for start in range(0, len(suffix), coord_tokens.TOKENS_PER_ATOM):
        try:
            coords.append(
                coord_tokens.tokens_to_triplet(
                    suffix[start : start + coord_tokens.TOKENS_PER_ATOM]
                )
            )
        except ValueError as exc:
            raise SequenceFormatError(
                FailureReason.COORDINATE_DECODE_FAILURE, str(exc)
            ) from exc
    return np.asarray(coords, dtype=np.float64).reshape(-1, 3)


def build_fsq_suffix(graph: Sequence[str], codes: Sequence[int]) -> List[str]:
    """Published BindGPT's layout applied to FSQ: bare codes, in graph order.

    One code per graph token, atom or not -- the tokenizer's native output --
    with no graph repetition. The decoder's token sequence is recovered from the
    SMILES by :func:`rebracket`.
    """
    if len(graph) != len(codes):
        raise SequenceFormatError(
            FailureReason.ATOM_COUNT_MISMATCH,
            f"{len(graph)} graph tokens but {len(codes)} FSQ codes",
        )
    return [fsq_code_token(code) for code in codes]


def parse_fsq_suffix(
    suffix: Sequence[str], graph: Sequence[str]
) -> np.ndarray:
    """Invert :func:`build_fsq_suffix` given the graph derived from the SMILES."""
    if len(suffix) != len(graph):
        raise SequenceFormatError(
            FailureReason.ATOM_COUNT_MISMATCH,
            f"{len(suffix)} FSQ codes for {len(graph)} graph tokens",
        )
    codes: List[int] = []
    for token in suffix:
        if not (
            len(token) > 2
            and token[0] == "<"
            and token[-1] == ">"
            and token[1:-1].isdigit()
        ):
            raise SequenceFormatError(
                FailureReason.COORDINATE_DECODE_FAILURE,
                f"unexpected token {token!r} where an FSQ code was required",
            )
        codes.append(int(token[1:-1]))
    return np.asarray(codes, dtype=np.int64)


def build_sequence(
    smiles_tokens: Sequence[str], stream_tokens: Sequence[str]
) -> List[str]:
    """Assemble ``<LIGAND> smiles <XYZ> stream <EOS>``."""
    return [LIGAND, *smiles_tokens, XYZ, *stream_tokens, EOS]


def split_sequence(tokens: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split a generated or stored sequence into its SMILES and stream halves.

    Requires exactly one ``<LIGAND>`` and one ``<XYZ>`` in that order. A missing
    trailing ``<EOS>`` is tolerated so that truncated generations can still be
    attributed a failure reason downstream.
    """
    body = [token for token in tokens if token not in (PAD,)]
    if body and body[-1] == EOS:
        body = body[:-1]

    if body.count(LIGAND) != 1 or body.count(XYZ) != 1:
        raise SequenceFormatError(
            FailureReason.MISSING_SPECIAL_TOKEN,
            f"expected exactly one {LIGAND} and one {XYZ}",
        )
    ligand_at = body.index(LIGAND)
    xyz_at = body.index(XYZ)
    if ligand_at != 0 or xyz_at < ligand_at:
        raise SequenceFormatError(
            FailureReason.MISSING_SPECIAL_TOKEN,
            f"{LIGAND} must open the sequence and precede {XYZ}",
        )
    if EOS in body:
        raise SequenceFormatError(
            FailureReason.MISSING_SPECIAL_TOKEN, f"unexpected interior {EOS}"
        )
    return list(body[1:xyz_at]), list(body[xyz_at + 1 :])


def render(tokens: Sequence[str]) -> str:
    """Space-join a token list for storage."""
    return " ".join(tokens)


def load(text: str) -> List[str]:
    """Invert :func:`render`."""
    return text.split(" ") if text else []
