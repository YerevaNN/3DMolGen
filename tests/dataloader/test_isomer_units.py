import json
from pathlib import Path
from typing import List

import pytest
from molgen3D.training.pretraining.dataprocessing import dataloader as dataloader_module
from molgen3D.training.pretraining.dataprocessing.dataloader import (
    IsomerUnit,
    JsonlTaggedPackedDataset,
    _encode_isomer_chunks,
    pack_units,
    serialize_isomer_unit,
)


class _DummyTokenizer:
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


@pytest.fixture(autouse=True)
def patch_tokenizer(monkeypatch):
    monkeypatch.setattr(dataloader_module, "AutoTokenizer", _DummyTokenizer)


def test_packer_delimiter_and_end_pad():
    units = [[1, 2], [3, 4]]
    samples = list(pack_units(units, pad_id=0, ctx_len=6, ignore_index=-100))
    assert len(samples) == 1
    features, labels = samples[0]
    input_ids = features["input"].tolist()
    attention = features["attention_mask"].tolist()

    assert input_ids == [1, 2, 0, 3, 4, 0]
    assert attention == [1, 1, 1, 1, 1, 0]
    assert labels[1].item() == 0  # delimiter pad is learned
    assert labels[4].item() == -100  # points to end padding


def test_packer_non_fitting_next_unit():
    units = [[1, 2, 3, 4], [5, 6]]
    samples = list(pack_units(units, pad_id=0, ctx_len=5, ignore_index=-100))
    assert len(samples) == 2

    first_ids = samples[0][0]["input"].tolist()
    first_attention = samples[0][0]["attention_mask"].tolist()
    assert first_ids == [1, 2, 3, 4, 0]
    assert first_attention == [1, 1, 1, 1, 0]

    second_ids = samples[1][0]["input"].tolist()
    assert second_ids[:2] == [5, 6]
    assert second_ids[2:] == [0, 0, 0]


def test_serialize_truncates_long_unit():
    unit = IsomerUnit(isomeric_smiles="C", conf_embedded_strings=["A" * 20])
    tokenizer = _DummyTokenizer()
    unit_ids = serialize_isomer_unit(unit, tokenizer, ctx_len=5)
    assert len(unit_ids) == 5

    samples = list(pack_units([unit_ids], pad_id=0, ctx_len=5, ignore_index=-100))
    assert len(samples) == 1
    features, labels = samples[0]
    assert features["attention_mask"].tolist() == [1, 1, 1, 1, 1]
    assert labels[-1].item() == -100


def test_zero_conformer_isomer_skipped(tmp_path: Path):
    data_file = tmp_path / "data.jsonl"
    payload = {
        "geom_id": "geom0",
        "isomers": {
            "C": [],
            "CC": [{"embedded_smiles": "[H]CC[H]"}],
        },
    }
    data_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    ds = JsonlTaggedPackedDataset(
        train_path=str(data_file),
        tokenizer_path="unused",
        seq_len=32,
        min_emb_len=1,
        shuffle_lines=False,
        infinite=False,
        seed=123,
        world_size=1,
        rank=0,
        lookahead_limit=4,
        serialization_mode="isomer_units",
    )
    ds._ensure_tokenizer_ready()
    fps = [open(str(data_file), "rb")]
    try:
        units = ds._read_isomer_units_from_pair(fps, (0, 0))
    finally:
        for f in fps:
            f.close()
    assert len(units) == 1


def _starts_with(tokens: List[int], prefix: List[int]) -> bool:
    return tokens[: len(prefix)] == prefix


def test_chunks_start_with_smiles_prefix():
    tokenizer = _DummyTokenizer()
    smiles = "C"
    confs = ["AAAA", "BBBB", "CCCC"]
    ctx_len = 64
    chunks = _encode_isomer_chunks(smiles, confs, tokenizer, ctx_len)
    prefix = tokenizer.encode("[SMILES]", add_special_tokens=False)
    assert chunks, "Expected at least one chunk"
    assert all(_starts_with(chunk, prefix) for chunk in chunks)


def test_chunk_flushes_and_repeats_smiles_when_conformer_overflows():
    tokenizer = _DummyTokenizer()
    smiles = "C"
    conf1 = "A" * 5
    conf2 = "B" * 5
    prefix = tokenizer.encode("[SMILES]", add_special_tokens=False)
    conf_tokens = tokenizer.encode("[CONFORMER]X[/CONFORMER]", add_special_tokens=False)
    ctx_len = len(prefix) + len(conf_tokens) + 2
    chunks = _encode_isomer_chunks(smiles, [conf1, conf2], tokenizer, ctx_len)
    assert len(chunks) >= 2
    assert _starts_with(chunks[0], prefix)
    assert _starts_with(chunks[1], prefix)


def test_oversized_conformer_truncates_when_alone():
    tokenizer = _DummyTokenizer()
    smiles = "C"
    conf = "Z" * 200
    prefix = tokenizer.encode("[SMILES]", add_special_tokens=False)
    ctx_len = len(prefix) + 4
    chunks = _encode_isomer_chunks(smiles, [conf], tokenizer, ctx_len)
    assert len(chunks) == 1
    assert len(chunks[0]) == ctx_len


def test_smiles_truncates_when_longer_than_ctx_len():
    tokenizer = _DummyTokenizer()
    smiles = "C" * 200
    ctx_len = 12
    chunks = _encode_isomer_chunks(smiles, ["A"], tokenizer, ctx_len)
    assert len(chunks) == 1
    assert len(chunks[0]) == ctx_len
