"""Fail closed on mixed coordinate-tokenizer versions in Qwen data and runs."""

import json
from pathlib import Path
from .coordtoken import FORMAT, parse, sha256


def load_contract(tokenizer_path):
    path = Path(tokenizer_path) / "coordtoken_contract.json"
    if not path.exists():
        return None
    c = json.loads(path.read_text())
    if c["format"] != FORMAT or sha256(path.parent / "tokenizer.json") != c["tokenizer_sha256"]:
        raise ValueError("LM tokenizer contract/hash mismatch")
    return c


def validate_record(record, contract):
    if (
        record.get("coordtoken_format") != contract["format"]
        or record.get("coordtoken_sha256") != contract["checkpoint_sha256"]
    ):
        raise ValueError("Wrong/missing CoordToken record version or checkpoint")
    if not record.get("canonical_smiles"):
        raise ValueError("Missing conditioning SMILES")
    return parse(record.get("embedded_smiles", ""))


def validate_training_data(directory, tokenizer_path, seq_len):
    contract = load_contract(tokenizer_path)
    if contract is None:
        raise ValueError("Expected versioned CoordToken LM tokenizer")
    root = Path(directory)
    m = json.loads((root / "coordtoken_manifest.json").read_text())
    if m["status"] != "verified" or m["split"] != "train":
        raise ValueError("Expected assembled training split")
    for k in ["format", "checkpoint_sha256", "tokenizer_sha256", "geometry_token_ids"]:
        if m[k] != contract[k]:
            raise ValueError("Encoded data/tokenizer mismatch: " + k)
    if m["max_unit_tokens_with_separator"] > seq_len:
        raise ValueError("Context would truncate a molecule")
    if {p.name for p in root.glob("*.jsonl")} != {p["path"] for p in m["files"]}:
        raise ValueError("Training file population changed")
    for entry in m["files"]:
        if sha256(root / entry["path"]) != entry["sha256"]:
            raise ValueError("Training shard changed")
    return m


def attach_model_contract(model_config, tokenizer_path, output_directory):
    """Carry the training tokenizer contract into an exported Qwen checkpoint."""
    import shutil

    c = load_contract(tokenizer_path)
    if c is None:
        return
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    for key in ["format", "checkpoint_sha256", "tokenizer_sha256"]:
        setattr(model_config, "coordtoken_" + key, c[key])
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "coordtoken_contract.json",
    ]:
        source = Path(tokenizer_path) / name
        if source.is_file() and source.resolve() != (out / name).resolve():
            shutil.copy2(source, out / name)


def validate_model_contract(model_config, contract):
    for key in ["format", "checkpoint_sha256", "tokenizer_sha256"]:
        if getattr(model_config, "coordtoken_" + key, None) != contract[key]:
            raise ValueError("Qwen checkpoint has missing/mismatched CoordToken provenance: " + key)
