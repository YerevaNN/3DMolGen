"""Read-only access to the public BindGPT tokenizer files.

Only tokenizer and config files are used; no model weights are ever loaded. The
files are downloaded once into the experiment directory and pinned by SHA-256,
so training and evaluation never depend on the network.

The public vocabulary is the source of truth for two things:

* the exact coordinate token spelling (see :mod:`.coord_tokens`); and
* the element-level SMILES token set used on the left-hand side of every
  sequence.

:func:`derive_smiles_tokens` recovers the latter by removing the blocks that are
not SMILES: specials, fraction tokens, multi-digit and negative integer tokens,
and the ``_XXX`` pocket-residue tokens. The ten single digits are kept because
SMILES ring-closure bonds need them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ID = "insilicomedicine/bindgpt_finetuned"

#: Files pulled from the model card. Weights are deliberately absent.
TOKENIZER_FILES = (
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

#: Architecture fields the plan pins for both arms.
EXPECTED_ARCHITECTURE = {
    "model_type": "gpt_neox",
    "num_hidden_layers": 15,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "max_position_embeddings": 2048,
    "tie_word_embeddings": False,
}


@dataclass(frozen=True)
class PublicTokenizerFiles:
    """A pinned local copy of the public tokenizer."""

    directory: Path

    def path(self, name: str) -> Path:
        return self.directory / name

    def config(self) -> Dict[str, object]:
        return json.loads(self.path("config.json").read_text())

    def added_tokens(self) -> List[Tuple[int, str]]:
        payload = json.loads(self.path("tokenizer.json").read_text())
        return sorted(
            (entry["id"], entry["content"]) for entry in payload["added_tokens"]
        )

    def hashes(self) -> Dict[str, str]:
        return {name: sha256_file(self.path(name)) for name in TOKENIZER_FILES}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download(destination: str | Path) -> PublicTokenizerFiles:
    """Fetch the tokenizer and config files (never the weights)."""
    from huggingface_hub import hf_hub_download

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        local = hf_hub_download(REPO_ID, name, local_dir=str(destination))
        if Path(local) != destination / name:  # pragma: no cover - hub layout
            Path(local).replace(destination / name)
    return PublicTokenizerFiles(destination)


def verify_architecture(files: PublicTokenizerFiles) -> Dict[str, object]:
    """Assert the public config still matches the pinned architecture."""
    config = files.config()
    mismatched = {
        key: (config.get(key), value)
        for key, value in EXPECTED_ARCHITECTURE.items()
        if config.get(key) != value
    }
    if mismatched:
        raise ValueError(f"public BindGPT config drifted: {mismatched}")
    return config


def derive_smiles_tokens(files: PublicTokenizerFiles) -> List[str]:
    """Return the tokens BindGPT can spell a SMILES with, in public order.

    Specials, fraction tokens and ``_XXX`` pocket-residue tokens are removed;
    everything else is kept, *including* the integer-part coordinate tokens. That
    is not an accident of the vocabulary but how the public tokenizer actually
    behaves: its added-token trie is longest-match, so two adjacent ring-closure
    digits in ``...c2ccccc21)...`` are merged into the single integer token
    ``"21"``, and a ring bond written ``c2-c3`` can merge ``-3``. Excluding the
    integers would make our tokenization disagree with theirs, so they are shared
    between the SMILES block and the coordinate block instead, and the
    arm-specific mask forbids fraction tokens rather than integers.
    """
    tokens: List[str] = []
    for _, content in files.added_tokens():
        if content.startswith("<") and content.endswith(">"):
            continue  # special
        if content.startswith("."):
            continue  # fraction token
        if content.startswith("_"):
            continue  # pocket residue
        tokens.append(content)
    return tokens


def verify_coordinate_tokens(files: PublicTokenizerFiles) -> None:
    """Assert our coordinate spelling matches the public vocabulary exactly."""
    from molgen3D.experiments.bindgpt_fsq import coord_tokens

    public = {content for _, content in files.added_tokens()}
    missing = [token for token in coord_tokens.coordinate_tokens() if token not in public]
    if missing:
        raise ValueError(
            f"{len(missing)} coordinate tokens absent from the public vocabulary, "
            f"first few: {missing[:8]}"
        )


def load_reference_tokenizer(files: PublicTokenizerFiles):
    """Load the public tokenizer itself, for round-trip assertions."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(files.directory), local_files_only=True)


def reference_tokens(tokenizer, text: str) -> List[str]:
    """Return the public tokenizer's token strings for ``text``.

    ``encode`` prepends an out-of-vocabulary id and appends ``<EOS>``; both are
    stripped so the result is comparable with our own tokenization.
    """
    ids = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    if tokens and tokens[0] is None:
        tokens = tokens[1:]
    while tokens and tokens[-1] == "<EOS>":
        tokens = tokens[:-1]
    return [token for token in tokens if token is not None]


def greedy_tokenize(text: str, vocabulary: Sequence[str]) -> List[str]:
    """Longest-match tokenization of ``text`` over ``vocabulary``.

    This reproduces the public tokenizer's added-token matching for SMILES,
    which is element-level rather than character-level: ``Cl`` and ``Br`` are
    single tokens.
    """
    table = set(vocabulary)
    longest = max(len(token) for token in table)
    tokens: List[str] = []
    position = 0
    while position < len(text):
        for width in range(min(longest, len(text) - position), 0, -1):
            candidate = text[position : position + width]
            if candidate in table:
                tokens.append(candidate)
                position += width
                break
        else:
            raise ValueError(
                f"character {text[position]!r} at {position} is outside the "
                f"BindGPT SMILES vocabulary (input: {text!r})"
            )
    return tokens
