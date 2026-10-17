"""Preserve Qwen token IDs but make geometry codes ordinary, retained tokens."""

import argparse, json, shutil
from pathlib import Path
from transformers import AutoTokenizer
from molgen3D.data_processing.coordtoken import FORMAT, SELECTED_SHA256, sha256
from molgen3D.data_processing.preprocess_coordtoken_csv import validate_tokenizer


def build(source, output):
    source = Path(source)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for p in source.iterdir():
        if p.is_file():
            shutil.copy2(p, output / p.name)
    codes = {f"<{i}>" for i in range(4096)}
    path = output / "tokenizer.json"
    j = json.loads(path.read_text())
    changed = 0
    for entry in j["added_tokens"]:
        if entry["content"] in codes:
            entry["special"] = False
            changed += 1
    if changed != 4096:
        raise ValueError("Expected all4096 existing code IDs")
    path.write_text(json.dumps(j, ensure_ascii=False, separators=(",", ":")) + "\n")
    path = output / "tokenizer_config.json"
    j = json.loads(path.read_text())
    for entry in j.get("added_tokens_decoder", {}).values():
        if entry["content"] in codes:
            entry["special"] = False
    j["additional_special_tokens"] = [
        x
        for x in j.get("additional_special_tokens", [])
        if (x if isinstance(x, str) else x["content"]) not in codes
    ]
    path.write_text(json.dumps(j, indent=2, ensure_ascii=False) + "\n")
    old = AutoTokenizer.from_pretrained(source, local_files_only=True)
    new = AutoTokenizer.from_pretrained(output, local_files_only=True)
    ids = validate_tokenizer(new)
    if old.get_vocab() != new.get_vocab():
        raise ValueError("Token IDs changed")
    (output / "coordtoken_contract.json").write_text(
        json.dumps(
            {
                "format": FORMAT,
                "checkpoint_sha256": SELECTED_SHA256,
                "geometry_token_ids": ids,
                "tokenizer_sha256": sha256(output / "tokenizer.json"),
                "molecular_code_position": "first code inside each CONFORMER span; same vocabulary as atom codes",
                "source_tokenizer_sha256": sha256(source / "tokenizer.json"),
                "vocab_size": len(new),
                "no_extra_register_marker": True,
            },
            indent=2,
        )
        + "\n"
    )
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(build(a.source, a.output))
