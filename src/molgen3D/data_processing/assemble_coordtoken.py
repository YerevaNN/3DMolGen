"""Validate complete CSV conversion before permitting Qwen training."""

import argparse, json
from pathlib import Path
from .coordtoken import FORMAT, sha256
from .preprocess_coordtoken_csv import atomic_json


def assemble(directory, expected_rows):
    root = Path(directory).resolve()
    target = root / "coordtoken_manifest.json"
    if target.exists():
        raise FileExistsError(target)
    receipts = sorted(root.glob("part-*.json"))
    parts = [json.loads(p.read_text()) for p in receipts]
    if not parts:
        raise ValueError("No completed shards")
    n = parts[0]["num_shards"]
    if len(parts) != n or sorted(p["shard_index"] for p in parts) != list(range(n)):
        raise ValueError("Missing/duplicate shards")
    common = [
        "format",
        "checkpoint_sha256",
        "tokenizer_sha256",
        "precision",
        "source",
        "source_bytes",
        "source_mtime_ns",
        "split",
        "num_shards",
        "seq_len",
        "geometry_token_ids",
    ]
    files = []
    for i, p in enumerate(parts):
        if p["status"] != "complete" or p["failures"] != 0 or p["format"] != FORMAT:
            raise ValueError("Incomplete/invalid shard")
        if any(p[k] != parts[0][k] for k in common):
            raise ValueError("Inconsistent shard provenance")
        if p["byte_range"] != [p["source_bytes"] * i // n, p["source_bytes"] * (i + 1) // n]:
            raise ValueError("Shard coverage differs")
        f = root / f"part-{i:05d}.jsonl"
        if sha256(f) != p["output_sha256"]:
            raise ValueError("Output changed")
        with f.open("rb") as stream:
            rows = sum(1 for _ in stream)
        if rows != p["rows"]:
            raise ValueError("Shard row count differs")
        files.append({"path": f.name, "sha256": p["output_sha256"], "rows": rows})
    if {p.name for p in root.glob("*.jsonl")} != {f["path"] for f in files}:
        raise ValueError("Unexpected JSONL files")
    if sum(p["rows"] for p in parts) != expected_rows:
        raise ValueError("Expected source row count differs; do not silently drop rows")
    result = {
        **{k: parts[0][k] for k in common},
        "status": "verified",
        "rows": expected_rows,
        "files": files,
        "lm_tokens_with_separator": sum(p["lm_tokens_with_separator"] for p in parts),
        "max_unit_tokens_with_separator": max(p["max_unit_tokens_with_separator"] for p in parts),
        "training_steps": "Set explicitly for the chosen LM batch/context; packed padding and shuffling affect epoch accounting.",
    }
    atomic_json(target, result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    p.add_argument("--expected-rows", type=int, required=True)
    a = p.parse_args()
    print(json.dumps(assemble(a.directory, a.expected_rows)))


if __name__ == "__main__":
    main()
