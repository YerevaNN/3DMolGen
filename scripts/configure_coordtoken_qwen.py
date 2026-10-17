"""Create a Qwen training TOML only from an assembled versioned dataset.

Steps must be chosen explicitly: old FSQ token budgets are not transferable.
"""

import argparse, json, math
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib
from molgen3D.data_processing.coordtoken_contract import validate_training_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", choices=["06b", "17b"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--global-batch", type=int, required=True)
    p.add_argument("--local-batch", type=int, default=2)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--lr", type=float, default=8e-4)
    a = p.parse_args()
    if min(a.steps, a.local_batch, a.global_batch, a.gpus, a.seq_len) <= 0 or a.global_batch % (
        a.local_batch * a.gpus
    ):
        raise ValueError("Invalid step/batch/GPU configuration")
    if not math.isfinite(a.lr) or a.lr <= 0:
        raise ValueError("Learning rate must be positive and finite")
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    root = Path(__file__).resolve().parents[1]
    template = root / f"src/molgen3D/config/pretrain/qwen3_{a.size}_fsq_bigdata_from_base.toml"
    m = validate_training_data(a.data, a.tokenizer, a.seq_len)
    c = tomllib.loads(template.read_text())
    base = str(Path(a.base_model).resolve())
    tok = str(Path(a.tokenizer).resolve())
    data = str(Path(a.data).resolve())
    if not (Path(base) / "config.json").is_file():
        raise FileNotFoundError("Provide a prepared HuggingFace Qwen base model directory")
    c["job"]["description"] = f"qwen3_{a.size}_coordtoken_atom12_reg1"
    c["molgen_run"].update(init_mode="hf_pretrain", tokenizer_tag=tok, base_model_tag=base)
    c["model"]["hf_assets_path"] = tok
    c["training"].update(
        steps=a.steps,
        local_batch_size=a.local_batch,
        global_batch_size=a.global_batch,
        seq_len=a.seq_len,
        dataset_path=data,
    )
    d = c["molgen_data"]
    d.pop("tokenizer_key", None)
    d.update(train_path=data, tokenizer_override=tok, min_emb_len=0, serialization_mode="pairs")
    c["optimizer"]["lr"] = a.lr
    c["wsds_scheduler"].update(checkpoints=[a.steps], warmup_steps=min(500, max(1, a.steps // 20)))
    c["checkpoint"].update(initial_load_path=base, interval=max(1, a.steps // 5))
    c["parallelism"].update(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=a.gpus,
        tensor_parallel_degree=1,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        for section, values in c.items():
            f.write(f"[{section}]\n")
            for k, v in values.items():
                f.write(f"{k} = {json.dumps(v)}\n")
            f.write("\n")
    out.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "data_manifest": str(Path(data) / "coordtoken_manifest.json"),
                "checkpoint_sha256": m["checkpoint_sha256"],
                "format": m["format"],
                "rows": m["rows"],
                "lm_tokens_with_separator": m["lm_tokens_with_separator"],
                "steps": a.steps,
                "global_batch": a.global_batch,
                "gpus": a.gpus,
                "scope": "Launch configuration, not a completed or tuned Qwen training run.",
            },
            indent=2,
        )
        + "\n"
    )
    print(out)


if __name__ == "__main__":
    main()
