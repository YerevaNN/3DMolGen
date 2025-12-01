import json
import re
from pathlib import Path
from typing import List

import pytest
import torch

from molgen3D.training.pretraining.dataprocessing import dataloader as dataloader_module
from molgen3D.training.pretraining.dataprocessing.dataloader import (
    JsonlTaggedPackedDataset,
)
from molgen3D.training.pretraining.dataprocessing.text_processing import (
    ChunkPacker,
    build_unit,
    is_valid_unit,
)

SMILES_PATTERN = re.compile(r"\[SMILES\](.*?)\[/SMILES\]", re.DOTALL)


class _DummyTokenizer:
    """Fast AutoTokenizer stand-in so tests never download real assets."""

    def __init__(self):
        self.pad_token = "<|endoftext|>"
        self.pad_token_id = 0
        self._next_id = 3
        self._map = {self.pad_token: self.pad_token_id}

    @classmethod
    def from_pretrained(cls, *_, **__):
        return cls()

    def convert_tokens_to_ids(self, token):
        return self._map.get(token)

    def add_special_tokens(self, mapping):
        pad = mapping.get("pad_token")
        if pad:
            self.pad_token = pad
            self._map.setdefault(pad, self.pad_token_id)
            self.pad_token_id = self._map[pad]
        for token in mapping.get("additional_special_tokens", []):
            if token not in self._map:
                self._map[token] = self._next_id
                self._next_id += 1

    def encode(self, text: str, add_special_tokens: bool = False):
        ids = []
        for ch in text:
            if ch not in self._map:
                self._map[ch] = self._next_id
                self._next_id += 1
            ids.append(self._map[ch])
        return ids

    def decode(self, tokens, skip_special_tokens: bool = False):
        inv = {v: k for k, v in self._map.items()}
        return "".join(inv.get(int(tok), "?") for tok in tokens)


@pytest.fixture(autouse=True)
def patch_tokenizer(monkeypatch):
    monkeypatch.setattr(dataloader_module, "AutoTokenizer", _DummyTokenizer)


def _write_jsonl(path: Path, count: int = 4) -> List[str]:
    canonical_entries = []
    with path.open("w", encoding="utf-8") as fp:
        for idx in range(count):
            canonical = f"C{idx}"
            embedded = f"[H]C{idx}[H]"
            json.dump({"canonical_smiles": canonical, "embedded_smiles": embedded}, fp)
            fp.write("\n")
            canonical_entries.append(canonical)
    return canonical_entries


def _make_dataset(
    data_file: Path,
    *,
    seq_len: int = 256,
    world_size: int = 1,
    rank: int = 0,
    infinite: bool = False,
    **overrides,
):
    return JsonlTaggedPackedDataset(
        train_path=str(data_file),
        tokenizer_path="unused",
        seq_len=seq_len,
        min_emb_len=1,
        shuffle_lines=False,
        infinite=infinite,
        seed=1234,
        world_size=world_size,
        rank=rank,
        lookahead_limit=4,
        **overrides,
    )


def test_dataloader_interface_compliance(tmp_path):
    """Dataloader returns TorchTitan-compatible tensors with shifted labels."""
    data_file = tmp_path / "train.jsonl"
    _write_jsonl(data_file, count=4)
    seq_len = 128
    ds = _make_dataset(data_file, seq_len=seq_len)
    features, labels = next(iter(ds))
    inputs = features["input"]
    assert isinstance(inputs, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert inputs.dtype == torch.long
    assert labels.dtype == torch.long
    assert inputs.shape == labels.shape == (seq_len,)

    shifted = torch.empty_like(labels)
    shifted[:-1] = inputs[1:]
    shifted[-1] = -100
    non_ignore = labels != -100
    assert torch.equal(labels[non_ignore], shifted[non_ignore])
    assert labels[-1] == -100


def test_rank_sharding_and_determinism(tmp_path):
    """World-size/rank arguments should deterministically shard samples."""
    data_file = tmp_path / "train.jsonl"
    canonical = _write_jsonl(data_file, count=12)
    world_size = 3

    seen_per_rank = []
    all_seen = []
    for rank in range(world_size):
        ds = _make_dataset(data_file, world_size=world_size, rank=rank)
        rank_smiles = []
        for features, _ in ds:
            rank_smiles.extend(_extract_smiles(features["input"], ds.tk))
        seen_per_rank.append(rank_smiles)
        all_seen.extend(rank_smiles)

    assert set(all_seen) == set(canonical)
    assert all(rank_smiles for rank_smiles in seen_per_rank)


def test_state_dict_resume_round_trip(tmp_path):
    """JsonlTaggedPackedDataset exposes resumable state for TorchTitan checkpoints."""
    data_file = tmp_path / "train.jsonl"
    _write_jsonl(data_file, count=6)
    ds = _make_dataset(data_file)
    iterator = iter(ds)
    first_batch = next(iterator)
    state = ds.state_dict()

    resumed = _make_dataset(data_file)
    resumed.load_state_dict(state)
    second_batch = next(iter(resumed))

    assert not torch.equal(first_batch[0]["input"], second_batch[0]["input"])


def test_build_unit_helpers_and_validation():
    """Text-processing helpers should build valid units respecting constraints."""
    canonical = "CCO"
    embedded = "[H]CCO[H]" * 2
    unit = build_unit(canonical, embedded)
    assert "[SMILES]" in unit and embedded in unit
    assert is_valid_unit(canonical, embedded, min_emb_len=4)
    assert not is_valid_unit("", embedded, min_emb_len=4)


def test_chunk_packer_emits_shifted_pairs():
    """ChunkPacker yields consecutive tokens suitable for next-token training."""
    packer = ChunkPacker(seq_len=4, bos_id=1, eos_id=2)
    tokens = list(range(10))
    packer.try_add_unit(tokens)
    blocks = list(packer.yield_blocks())
    assert blocks, "Expected packer to emit at least one block"
    for inp, target in blocks:
        assert torch.equal(target[:-1], inp[1:])


def test_dataloader_truncates_monster_units(tmp_path):
    """Units longer than seq_len - 1 should be truncated instead of dropped."""
    data_file = tmp_path / "train.jsonl"
    canonical = "C"
    embedded = "[H]" + "C" * 5000
    with data_file.open("w", encoding="utf-8") as fp:
        json.dump({"canonical_smiles": canonical, "embedded_smiles": embedded}, fp)
        fp.write("\n")
    seq_len = 64
    ds = _make_dataset(data_file, seq_len=seq_len)
    ds._ensure_tokenizer_ready()
    fps = [open(str(data_file), "rb")]
    try:
        unit = ds._read_unit_from_pair(fps, (0, 0))
    finally:
        for f in fps:
            f.close()
    assert unit is not None
    full_unit = build_unit(canonical, embedded)
    tokens = ds.tk.encode(full_unit, add_special_tokens=False)
    assert len(tokens) > ds.max_unit_tokens
    assert unit.tokens == tokens[: ds.max_unit_tokens]


def test_dataloader_can_disable_truncation(tmp_path):
    """Truncation behavior can be disabled for strict filtering."""
    data_file = tmp_path / "train.jsonl"
    canonical = "C"
    embedded = "[H]" + "C" * 5000
    with data_file.open("w", encoding="utf-8") as fp:
        json.dump({"canonical_smiles": canonical, "embedded_smiles": embedded}, fp)
        fp.write("\n")
    ds = _make_dataset(data_file, seq_len=64, truncate_overflow_units=False)
    ds._ensure_tokenizer_ready()
    fps = [open(str(data_file), "rb")]
    try:
        unit = ds._read_unit_from_pair(fps, (0, 0))
    finally:
        for f in fps:
            f.close()
    assert unit is None
def _extract_smiles(tensor: torch.Tensor, tokenizer: _DummyTokenizer) -> List[str]:
    decoded = tokenizer.decode(tensor.tolist(), skip_special_tokens=False)
    return [match.strip() for match in SMILES_PATTERN.findall(decoded)]
