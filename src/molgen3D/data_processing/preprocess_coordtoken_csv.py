"""Strict, independently sharded conversion of existing coordinate CSV splits.

No new split or shuffle. Each row is assigned by its physical starting byte.
Incomplete shards are not published. Source identity/provenance columns survive.
"""

import argparse, csv, functools, json, os
from pathlib import Path
from rdkit import Chem
from transformers import AutoTokenizer
from .coordtoken import (
    CoordTokenCodec,
    FORMAT,
    SELECTED_CHECKPOINT,
    SELECTED_SHA256,
    parse,
    graph_mol,
    sha256,
)
from molgen3D.training.pretraining.dataprocessing.text_processing import build_unit


def validate_tokenizer(tokenizer):
    ids = []
    for code in range(4096):
        token = f"<{code}>"
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded, skip_special_tokens=False) != token:
            raise ValueError("Geometry code is not one reversible LM token: " + token)
        if (
            encoded[0] in tokenizer.all_special_ids
            or tokenizer.decode(encoded, skip_special_tokens=True) != token
        ):
            raise ValueError("Geometry codes must survive skip_special_tokens=True")
        ids.append(encoded[0])
    if len(set(ids)) != 4096:
        raise ValueError("Geometry IDs collide")
    sample = "<317>[C]<42>(=[O]<901>)[O]<0>"
    if (
        tokenizer.decode(
            tokenizer.encode(sample, add_special_tokens=False), skip_special_tokens=True
        )
        != sample
    ):
        raise ValueError("Contextual code round trip failed")
    return ids


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def run(args):
    if (
        args.num_shards < 1
        or not 0 <= args.shard_index < args.num_shards
        or args.batch_size < 1
        or args.seq_len < 2
    ):
        raise ValueError("Invalid shard or batch size")
    source = Path(args.input).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"part-{args.shard_index:05d}"
    target = out / (stem + ".jsonl")
    receipt = out / (stem + ".json")
    if target.exists() or receipt.exists():
        raise FileExistsError(
            "Use a new output directory or shard; completed outputs are immutable"
        )
    lock = out / (stem + ".lock")
    tmp = out / (stem + ".jsonl.partial")
    codec = CoordTokenCodec(
        args.checkpoint,
        expected_sha256=args.expected_sha256,
        device=args.device,
        precision=args.precision,
    )
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    code_ids = validate_tokenizer(tok)
    token_file = Path(args.tokenizer) / "tokenizer.json"
    token_sha = sha256(token_file)
    st = source.stat()
    start = st.st_size * args.shard_index // args.num_shards
    end = st.st_size * (args.shard_index + 1) // args.num_shards
    count = tokens_total = max_tokens = 0
    first_offset = last_offset = None
    pending = []

    @functools.lru_cache(maxsize=100000)
    def canonical(graph):
        # Zero dummy codes permit structural validation without decoding geometry.
        return Chem.MolToSmiles(graph_mol(parse(graph)), canonical=True, isomericSmiles=True)

    def flush(dest):
        nonlocal count, tokens_total, max_tokens
        if not pending:
            return
        texts = [row["enriched_text"] for offset, row in pending]
        encoded = codec.encode_batch(texts)
        for (offset, row), stream in zip(pending, encoded):
            s = parse(stream)
            dummy = "<0>" + "".join(
                t + ("<0>" if t.startswith("[") else "") for t in s.graph_tokens
            )
            smiles = canonical(dummy)
            unit = build_unit(smiles, stream)
            n = len(tok.encode(unit, add_special_tokens=False)) + 1
            if n > args.seq_len:
                raise ValueError(
                    f"Row at byte{offset} exceeds LM context ({n}>{args.seq_len}); truncation is forbidden"
                )
            record = {
                **row,
                "canonical_smiles": smiles,
                "embedded_smiles": stream,
                "coordtoken_format": FORMAT,
                "coordtoken_sha256": codec.checkpoint_sha256,
                "source_csv_offset": offset,
                "split": args.split,
            }
            record.pop("enriched_text", None)
            dest.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
            tokens_total += n
            max_tokens = max(max_tokens, n)
        pending.clear()

    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    # Another worker may have finished while this worker loaded its model.
    # Recheck under the lock before opening a temporary output for writing.
    if target.exists() or receipt.exists():
        lock.unlink()
        raise FileExistsError("Shard completed while this worker initialized")
    try:
        with source.open("rb") as f, tmp.open("w") as dest:
            header = next(csv.reader([f.readline().decode("utf-8")], strict=True))
            header_end = f.tell()
            if len(set(header)) != len(header) or "enriched_text" not in header:
                raise ValueError("Invalid CSV schema")
            if start > header_end:
                f.seek(start - 1)
                if f.read(1) != b"\n":
                    f.readline()
            else:
                f.seek(header_end)
            while f.tell() < end:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                fields = next(csv.reader([line.decode("utf-8")], strict=True))
                if len(fields) != len(header):
                    raise ValueError(f"Invalid/multiline CSV record at{offset}")
                row = dict(zip(header, fields))
                pending.append((offset, row))
                first_offset = offset if first_offset is None else first_offset
                last_offset = offset
                if len(pending) >= args.batch_size:
                    flush(dest)
            flush(dest)
        now = source.stat()
        if (now.st_size, now.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
            raise ValueError("Input changed during conversion")
        info = {
            "status": "complete",
            "format": FORMAT,
            "checkpoint_sha256": codec.checkpoint_sha256,
            "tokenizer_sha256": token_sha,
            "geometry_token_ids": code_ids,
            "precision": args.precision,
            "source": str(source),
            "source_bytes": st.st_size,
            "source_mtime_ns": st.st_mtime_ns,
            "split": args.split,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "byte_range": [start, end],
            "first_row_offset": first_offset,
            "last_row_offset": last_offset,
            "rows": count,
            "lm_tokens_with_separator": tokens_total,
            "max_unit_tokens_with_separator": max_tokens,
            "seq_len": args.seq_len,
            "output_sha256": sha256(tmp),
            "failures": 0,
        }
        tmp.replace(target)
        atomic_json(receipt, info)
    except BaseException as exc:
        atomic_json(
            out / (stem + ".failure.json"),
            {
                "error": repr(exc),
                "published": False,
                "last_row_offset": last_offset,
                "rows_written_to_partial": count,
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--checkpoint", default=SELECTED_CHECKPOINT)
    p.add_argument("--expected-sha256", default=SELECTED_SHA256)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    print(json.dumps(run(p.parse_args())))


if __name__ == "__main__":
    main()
