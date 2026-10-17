"""Actual packed-loader validation for the versioned atom/register stream."""

import json
from pathlib import Path
import pytest
from transformers import AutoTokenizer
from molgen3D.data_processing.coordtoken import FORMAT
from molgen3D.data_processing.coordtoken_contract import load_contract
from molgen3D.training.pretraining.dataprocessing.dataloader import JsonlTaggedPackedDataset

TOK = (
    Path(__file__).resolve().parents[2]
    / "src/molgen3D/training/tokenizers/Qwen3_tokenizer_coordtoken_atom12_reg1"
)


def make(tmp_path, record, seq_len=128):
    f = tmp_path / "train.jsonl"
    f.write_text(json.dumps(record) + "\n")
    return JsonlTaggedPackedDataset(
        train_path=str(f),
        tokenizer_path=str(TOK),
        seq_len=seq_len,
        min_emb_len=0,
        shuffle_lines=False,
        infinite=False,
        world_size=1,
        rank=0,
        preview_enabled=False,
    )


def record():
    return {
        "canonical_smiles": "CO",
        "embedded_smiles": "<317>[C]<42>[O]<0>",
        "coordtoken_format": FORMAT,
        "coordtoken_sha256": load_contract(TOK)["checkpoint_sha256"],
    }


def test_geometry_targets_survive_loader(tmp_path):
    ds = make(tmp_path, record())
    features, labels = next(iter(ds))
    tok = AutoTokenizer.from_pretrained(TOK, local_files_only=True)
    ids = features["input"].tolist()
    for code in [317, 42, 0]:
        token = tok.convert_tokens_to_ids(f"<{code}>")
        i = ids.index(token)
        assert i > 0 and int(labels[i - 1]) == token


@pytest.mark.parametrize(
    "kind", ["missing_version", "wrong_checkpoint", "missing_register", "truncation"]
)
def test_reject_invalid_versioned_records(tmp_path, kind):
    r = record()
    if kind == "missing_version":
        r.pop("coordtoken_format")
    if kind == "wrong_checkpoint":
        r["coordtoken_sha256"] = "wrong"
    if kind == "missing_register":
        r["embedded_smiles"] = "[C]<42>[O]<0>"
    with pytest.raises(ValueError):
        list(make(tmp_path, r, seq_len=8 if kind == "truncation" else 128))
