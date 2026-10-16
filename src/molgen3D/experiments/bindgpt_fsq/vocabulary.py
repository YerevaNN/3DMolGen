"""The shared union vocabulary for both arms.

Both arms are trained on one vocabulary so that embedding shapes, output-head
shapes and total parameter counts are identical and the only difference between
the two models is which geometry tokens actually occur in their data. During
generation an arm-specific mask forbids the other arm's geometry tokens.

Ordering is fixed and provenance-preserving:

===============  ==================================================
block            contents
===============  ==================================================
specials         ``<PAD> <EOS> <UNK> <LIGAND> <XYZ>``
smiles           BindGPT element-level SMILES tokens, public order,
                 plus ``.`` for disconnected SFT rows
graph            FSQ bracketed-atom graph tokens not already present
coordinates      BindGPT integer tokens then fraction tokens; the
                 integers are already present in the SMILES block
fsq_codes        ``<0>`` .. ``<4095>``
===============  ==================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

from molgen3D.data_processing.smiles_encoder_decoder_fsq import (
    VOCAB as FSQ_GRAPH_VOCAB,
)
from molgen3D.experiments.bindgpt_fsq import coord_tokens
from molgen3D.experiments.bindgpt_fsq.sequences import (
    EOS,
    LIGAND,
    PAD,
    UNK,
    XYZ,
    Arm,
    fsq_code_token,
)

#: Number of FSQ codes in the pinned checkpoint (``levels=[2]*12``).
FSQ_CODEBOOK_SIZE = 4096

SPECIAL_ORDER = (PAD, EOS, UNK, LIGAND, XYZ)
SMILES_COMPATIBILITY_TOKENS = (".",)


def _with_smiles_compatibility(tokens: Sequence[str]) -> List[str]:
    """Add tokens required by the existing SFT data but absent upstream."""
    result = list(tokens)
    result.extend(token for token in SMILES_COMPATIBILITY_TOKENS if token not in result)
    return result

def load_smiles_tokens(path: str | Path | None = None) -> List[str]:
    """Return the BindGPT SMILES tokens from the pinned artifact.

    The list is derived from the public tokenizer by
    :func:`~molgen3D.experiments.bindgpt_fsq.public_tokenizer.derive_smiles_tokens`
    and frozen to JSON by ``prepare-tokenizers``, so nothing here is copied by
    hand and no run needs the network.
    """
    from molgen3D.experiments.bindgpt_fsq import config

    path = Path(path or config.SMILES_TOKENS_JSON)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run "
            "`python -m molgen3D.experiments.bindgpt_fsq.cli prepare-tokenizers` first"
        )
    payload = json.loads(path.read_text())
    return _with_smiles_compatibility(payload["smiles_tokens"])


@dataclass(frozen=True)
class Vocabulary:
    """An ordered union vocabulary with arm-specific generation masks."""

    tokens: Sequence[str]
    blocks: Dict[str, Sequence[str]]

    def __post_init__(self) -> None:
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("union vocabulary contains duplicates")

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def token_to_id(self) -> Dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def ids(self, tokens: Iterable[str]) -> List[int]:
        table = self.token_to_id
        unknown = table[UNK]
        return [table.get(token, unknown) for token in tokens]

    def block_ids(self, name: str) -> List[int]:
        table = self.token_to_id
        return [table[token] for token in self.blocks[name]]

    def forbidden_ids(self, arm: Arm) -> List[int]:
        """Return ids the given arm must never emit.

        The coord arm is denied the FSQ codebook. The FSQ arm is denied the
        *fraction* tokens only: integer-part tokens are shared with the SMILES
        block, because the public tokenizer merges adjacent ring-closure digits
        into them, and a coordinate scalar always costs an integer *and* a
        fraction token -- so forbidding fractions already makes scalar
        coordinates unspellable.
        """
        blocked = "fsq_codes" if arm is Arm.COORD else "coordinate_fractions"
        allowed = set(self.blocks["smiles"]) | set(self.blocks["graph"])
        blocked_tokens = set(self.blocks[blocked])
        return [
            index
            for index, token in enumerate(self.tokens)
            if token in blocked_tokens and token not in allowed
        ]

    def to_json(self) -> str:
        return json.dumps(
            {"tokens": list(self.tokens), "blocks": {k: list(v) for k, v in self.blocks.items()}},
            indent=1,
        )


def build_vocabulary(smiles_tokens: Sequence[str] | None = None) -> Vocabulary:
    """Construct the deterministic union vocabulary."""
    smiles_tokens = (
        _with_smiles_compatibility(smiles_tokens)
        if smiles_tokens is not None
        else load_smiles_tokens()
    )
    seen: Set[str] = set()
    tokens: List[str] = []

    def extend(items: Iterable[str]) -> List[str]:
        block: List[str] = []
        for token in items:
            block.append(token)
            if token not in seen:
                seen.add(token)
                tokens.append(token)
        return block

    blocks = {
        "specials": extend(SPECIAL_ORDER),
        "smiles": extend(smiles_tokens),
        "graph": extend(FSQ_GRAPH_VOCAB),
        "coordinate_integers": extend(coord_tokens.integer_tokens()),
        "coordinate_fractions": extend(coord_tokens.fraction_tokens()),
        "fsq_codes": extend(fsq_code_token(code) for code in range(FSQ_CODEBOOK_SIZE)),
    }
    return Vocabulary(tokens=tuple(tokens), blocks=blocks)


def save_hf_tokenizer(vocabulary: Vocabulary, destination: str | Path) -> Path:
    """Write a ``PreTrainedTokenizerFast`` for ``vocabulary``.

    A ``WordLevel`` model with whitespace pre-tokenization is used so that our
    space-joined token lists encode exactly, with no greedy-matching ambiguity
    between, say, the SMILES token ``"1"`` and the integer coordinate token
    ``"1"`` -- they are the same token by construction.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    backend = Tokenizer(
        models.WordLevel(vocab=vocabulary.token_to_id, unk_token=UNK)
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    fast = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token=PAD,
        eos_token=EOS,
        unk_token=UNK,
        bos_token=LIGAND,
        additional_special_tokens=[XYZ],
        model_max_length=2048,
    )
    fast.save_pretrained(str(destination))
    (destination / "union_vocabulary.json").write_text(vocabulary.to_json())
    return destination
