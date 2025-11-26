import json
from pathlib import Path

import torch
import pytest

from molgen3D.training.pretraining.dataprocessing import dataloader as dataloader_module
from molgen3D.training.pretraining.dataprocessing.dataloader import (
    JsonlTaggedPackedDataset,
)


class _DummyTokenizer:
    """
    Lightweight AutoTokenizer stand-in for unit tests. Assigns incremental IDs
    to unseen characters and exposes <|endoftext|> as a pad/separator token.
    """

    def __init__(self):
        self._next_id = 10
        self._token_map = {
            "<|endoftext|>": 3,
            "[": 4,
            "]": 5,
            "/": 6,
        }
        self.pad_token = "<|endoftext|>"
        self.pad_token_id = self._token_map["<|endoftext|>"]

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def convert_tokens_to_ids(self, token):
        return self._token_map.get(token)

    def add_special_tokens(self, mapping):
        pad_token = mapping.get("pad_token")
        if pad_token:
            self.pad_token = pad_token
            if pad_token not in self._token_map:
                self._token_map[pad_token] = self._next_id
                self._next_id += 1
            self.pad_token_id = self._token_map[pad_token]
        for token in mapping.get("additional_special_tokens", []):
            if token not in self._token_map:
                self._token_map[token] = self._next_id
                self._next_id += 1

    def encode(self, text: str, add_special_tokens: bool = False):
        ids = []
        for ch in text:
            if ch not in self._token_map:
                self._token_map[ch] = self._next_id
                self._next_id += 1
            ids.append(self._token_map[ch])
        return ids

    def decode(self, tokens, skip_special_tokens: bool = False):
        inv = {v: k for k, v in self._token_map.items()}
        chars = []
        for tok in tokens:
            tok_id = int(tok)
            chars.append(inv.get(tok_id, f"<{tok_id}>"))
        return "".join(chars)


def _patch_tokenizer(monkeypatch):
    monkeypatch.setattr(dataloader_module, "AutoTokenizer", _DummyTokenizer)


def _write_jsonl(path: Path, count: int = 2) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for idx in range(count):
            obj = {
                "canonical_smiles": f"C{idx}",
                "embedded_smiles": f"[H]C{idx}[H]",
            }
            fp.write(json.dumps(obj) + "\n")


def _build_dataset(tmp_path: Path):
    data_file = tmp_path / "mini.jsonl"
    _write_jsonl(data_file, count=2)
    return JsonlTaggedPackedDataset(
        train_path=str(data_file),
        tokenizer_path="unused",
        seq_len=128,
        min_emb_len=1,
        shuffle_lines=False,
        infinite=False,
        seed=123,
        world_size=1,
        rank=0,
        lookahead_limit=4,
    )


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    _patch_tokenizer(monkeypatch)
    return _build_dataset(tmp_path)


def test_dataset_performs_single_label_shift(dataset):
    """Invariant: only the dataloader performs label shifting."""
    first_sample = next(iter(dataset))
    inputs = first_sample[0]["input"]
    labels = first_sample[1]

    ignore_mask = labels == dataset.ignore_index
    assert torch.all(labels[-1:] == dataset.ignore_index)

    # Compare every supervised label to the next input token.
    valid_positions = (~ignore_mask)[:-1]
    assert valid_positions.any()
    assert torch.equal(labels[:-1][valid_positions], inputs[1:][valid_positions])

    # Ensure the unshifted pattern does NOT hold.
    assert not torch.equal(labels[~ignore_mask], inputs[~ignore_mask])


def test_ignore_index_is_suffix_only(dataset):
    """Invariant: ignore_index is only applied to the suffix without successors."""
    first_sample = next(iter(dataset))
    labels = first_sample[1]

    ignore_mask = labels == dataset.ignore_index
    non_ignore_indices = torch.nonzero(~ignore_mask, as_tuple=True)[0]
    assert non_ignore_indices.numel() > 0

    expected_span = torch.arange(
        non_ignore_indices[0], non_ignore_indices[-1] + 1, device=labels.device
    )
    assert torch.equal(non_ignore_indices, expected_span)

    suffix = torch.arange(non_ignore_indices[-1] + 1, labels.numel(), device=labels.device)
    if suffix.numel() > 0:
        assert torch.all(labels[suffix] == dataset.ignore_index)
