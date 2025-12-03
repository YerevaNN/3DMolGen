#!/usr/bin/env python3
"""
Minimal dataset token counting script for MolGen3D.

The script samples a configurable number of JSONL entries from one file per dataset
using the production dataloader, reports how many batches and tokens that sample produced,
extrapolates totals for the whole split, and verifies the extrapolation using file sizes.
"""

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer

from molgen3D.config.paths import get_data_path, get_tokenizer_path
from molgen3D.training.pretraining.dataprocessing.dataloader import build_dataloader


def list_jsonl_files(directory: str) -> List[Path]:
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(root.glob("*.jsonl"))


def _ensure_dataset_tokenizer(dataset) -> None:
    """Ensure the dataset has initialized its tokenizer so pad/sep IDs exist."""
    if dataset is None:
        return
    if hasattr(dataset, "_ensure_tokenizer_ready"):
        dataset._ensure_tokenizer_ready()
    elif hasattr(dataset, "tk"):
        _ = dataset.tk


def count_lines_and_bytes(files: List[Path]) -> Tuple[int, int]:
    total_lines = 0
    total_bytes = 0
    for file in files:
        try:
            with file.open("rb") as fh:
                total_lines += sum(1 for _ in fh)
            total_bytes += file.stat().st_size
        except OSError:
            continue
    return total_lines, total_bytes


def bytes_for_lines(file_path: Path, lines: int) -> Optional[int]:
    if not file_path.is_file() or lines <= 0:
        return None
    try:
        with file_path.open("rb") as fh:
            total = 0
            for i, chunk in enumerate(fh, 1):
                total += len(chunk)
                if i >= lines:
                    break
        return total
    except OSError:
        return None


def sample_dataloader(
    file_path: Path,
    tokenizer_path: str,
    tokenizer,
    seq_len: int,
    max_items: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Optional[Dict[str, float]]:
    loader = build_dataloader(
        train_path=str(file_path),
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle_lines=shuffle,
        infinite=False,
        seed=seed,
        min_emb_len=0,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=None,
        world_size=1,
        rank=0,
    )

    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None

    _ensure_dataset_tokenizer(dataset)

    pad_id = getattr(dataset, "pad_id", None)
    sep_id = getattr(dataset, "sep_id", None)
    if pad_id is None or sep_id is None:
        raise RuntimeError("MolGen dataset missing pad/sep token IDs.")

    items = 0
    samples = 0
    batches = 0
    pad_total = 0
    tokens_produced = 0

    for batch in loader:
        batches += 1
        inputs = batch[0]
        inputs = inputs["input"] if isinstance(inputs, dict) else inputs
        bsz = inputs.size(0)
        tokens_produced += bsz * seq_len

        for idx in range(bsz):
            sample = inputs[idx]
            pad_count = 0
            for token in reversed(sample.tolist()):
                if token == pad_id:
                    pad_count += 1
                else:
                    break
            sep_count = int((sample == sep_id).sum().item()) - pad_count
            sep_count = max(sep_count, 0)

            items += sep_count
            pad_total += pad_count
            samples += 1

            if items >= max_items:
                break
        if items >= max_items:
            break

    if samples == 0:
        return None

    avg_items_per_sample = items / samples
    avg_pad_per_sample = pad_total / samples
    effective_tokens = tokens_produced - pad_total

    return {
        "lines_consumed": float(items),
        "samples": float(samples),
        "batches": float(batches),
        "tokens_produced": float(tokens_produced),
        "effective_tokens": float(effective_tokens),
        "avg_pad_per_sample": float(avg_pad_per_sample),
        "avg_items_per_sample": float(avg_items_per_sample),
    }


def verify_bytes(sample_bytes: Optional[int], sample_lines: int, total_lines: int, total_bytes: int) -> Optional[Dict[str, float]]:
    if not sample_bytes or sample_lines == 0 or total_lines == 0 or total_bytes == 0:
        return None
    factor = total_lines / sample_lines
    expected_bytes = sample_bytes * factor
    diff = abs(expected_bytes - total_bytes)
    diff_pct = (diff / total_bytes) * 100 if total_bytes else 0.0
    return {
        "expected_bytes": expected_bytes,
        "actual_bytes": float(total_bytes),
        "difference": diff,
        "difference_pct": diff_pct,
    }


def summarize_dataset(
    name: str,
    directory: str,
    tokenizer_aliases: List[str],
    tokenizer_map: Dict[str, Tuple[str, AutoTokenizer]],
    seq_len: int,
    sample_lines: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Optional[Dict[str, Any]]:
    files = list_jsonl_files(directory)
    if not files:
        return None

    total_lines, total_bytes = count_lines_and_bytes(files)
    chosen_file = random.choice(files)
    with chosen_file.open("rb") as fh:
        file_line_count = sum(1 for _ in fh)
    target_lines = min(sample_lines, file_line_count)

    dataset_summary: Dict[str, Any] = {
        "name": name,
        "path": directory,
        "total_lines": total_lines,
        "total_bytes": total_bytes,
        "sample_file": chosen_file.name,
        "sample_file_lines": file_line_count,
        "target_lines": target_lines,
        "tokenizers": {},
    }

    for alias in tokenizer_aliases:
        tokenizer_path, tokenizer = tokenizer_map[alias]
        stats = sample_dataloader(
            chosen_file,
            tokenizer_path,
            tokenizer,
            seq_len,
            max_items=target_lines,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )
        if not stats:
            dataset_summary["tokenizers"][alias] = None
            continue

        avg_items = stats["avg_items_per_sample"]
        estimated_samples = int(total_lines / avg_items) if avg_items else 0
        estimated_batches = math.ceil(estimated_samples / batch_size) if batch_size else 0
        estimated_tokens = estimated_samples * seq_len
        estimated_pad = int(stats["avg_pad_per_sample"] * estimated_samples)
        estimated_effective = estimated_tokens - estimated_pad

        sample_bytes = bytes_for_lines(chosen_file, target_lines)
        verification = verify_bytes(sample_bytes, target_lines, total_lines, total_bytes)

        dataset_summary["tokenizers"][alias] = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "lines_target": target_lines,
            "lines_consumed": int(stats["lines_consumed"]),
            "batches": int(stats["batches"]),
            "tokens_produced": int(stats["tokens_produced"]),
            "effective_tokens": int(stats["effective_tokens"]),
            "avg_items_per_sample": stats["avg_items_per_sample"],
            "avg_pad_per_sample": stats["avg_pad_per_sample"],
            "estimated_samples": estimated_samples,
            "estimated_batches": estimated_batches,
            "estimated_tokens": estimated_tokens,
            "estimated_pad": estimated_pad,
            "estimated_effective": estimated_effective,
            "verification": verification,
        }

    return dataset_summary


def print_train_report(summary: Optional[Dict[str, Any]]) -> None:
    if not summary:
        print("SKIP dataset: no data collected.")
        return

    name = summary.get("name", "dataset").upper()
    print(f"\n{name} DATASET")
    print(f"  path: {summary.get('path')}")
    print(f"  sample source: {summary.get('sample_file')} ({summary.get('sample_file_lines', 0)} lines)")
    print(f"  total lines: {summary.get('total_lines', 0):,}, total bytes: {summary.get('total_bytes', 0):,}")

    for alias, stats in (summary.get("tokenizers") or {}).items():
        print(f"\n  Tokenizer: {alias}")
        if stats is None:
            print("    no samples collected")
            continue
        verification = stats.get("verification")
        print(f"    batch_size: {stats.get('batch_size')}, seq_len: {stats.get('seq_len')}")
        print(f"    sampled lines target: {stats.get('lines_target', 0):,}, consumed: {stats.get('lines_consumed', 0):,}")
        print(f"    dataloader batches: {stats.get('batches', 0):,}")
        print(f"    tokens produced: {stats.get('tokens_produced', 0):,}, effective: {stats.get('effective_tokens', 0):,}")
        print(f"    avg items/sample: {stats.get('avg_items_per_sample', 0):.2f}, avg pad/sample: {stats.get('avg_pad_per_sample', 0):.2f}")
        print(f"    estimated total samples: {stats.get('estimated_samples', 0):,}, batches: {stats.get('estimated_batches', 0):,}")
        print(f"    estimated tokens: {stats.get('estimated_tokens', 0):,}, pad tokens: {stats.get('estimated_pad', 0):,}, effective: {stats.get('estimated_effective', 0):,}")
        if verification:
            print(f"    verification: expected bytes {verification['expected_bytes']:,}, actual {verification['actual_bytes']:,}, diff {verification['difference']:.0f} ({verification['difference_pct']:.2f}%)")
        else:
            print("    verification: insufficient byte data")


def exhaust_validation_dataset(
    directory: str,
    tokenizer_aliases: List[str],
    tokenizer_map: Dict[str, Tuple[str, AutoTokenizer]],
    seq_len: int,
    batch_size: int,
    num_workers: int,
) -> Optional[Dict[str, Any]]:
    files = list_jsonl_files(directory)
    if not files:
        return None

    total_lines, total_bytes = count_lines_and_bytes(files)
    result: Dict[str, Any] = {
        "path": directory,
        "total_lines": total_lines,
        "total_bytes": total_bytes,
        "tokenizers": {},
    }

    for alias in tokenizer_aliases:
        tokenizer_path, tokenizer = tokenizer_map[alias]
        loader = build_dataloader(
            train_path=directory,
            tokenizer_path=tokenizer_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            shuffle_lines=False,
            infinite=False,
            seed=0,
            min_emb_len=0,
            drop_last=False,
            persistent_workers=False,
            prefetch_factor=None,
            world_size=1,
            rank=0,
        )

        dataset = getattr(loader, "dataset", None)
        _ensure_dataset_tokenizer(dataset)
        pad_id = getattr(dataset, "pad_id", None)
        sep_id = getattr(dataset, "sep_id", None)
        first_sample_text = None
        sample_count = 0
        batch_count = 0
        token_count = 0
        pad_total = 0

        for inputs, _ in loader:
            input_ids = inputs["input"] if isinstance(inputs, dict) else inputs
            if input_ids.numel() == 0:
                continue
            if first_sample_text is None:
                try:
                    first_sample_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
                except Exception:
                    first_sample_text = "<decode_error>"

            batch_count += 1
            sample_count += input_ids.size(0)
            token_count += input_ids.numel()
            if pad_id is not None:
                pad_total += int((input_ids == pad_id).sum().item())

        effective_tokens = token_count - pad_total
        utilization = (effective_tokens / token_count * 100) if token_count else 0.0

        result["tokenizers"][alias] = {
            "samples": sample_count,
            "batches": batch_count,
            "token_count": token_count,
            "pad_count": pad_total,
            "effective_tokens": effective_tokens,
            "utilization": utilization,
            "first_sample_text": first_sample_text,
            "pad_id": pad_id,
            "sep_id": sep_id,
        }

    return result


def print_validation_report(summary: Optional[Dict[str, Any]]) -> None:
    if not summary:
        print("\nSKIP validation dataset (no files).")
        return

    print("\nVALIDATION DATASET")
    print(f"  path: {summary['path']}")
    print(f"  total lines: {summary['total_lines']:,}, total bytes: {summary['total_bytes']:,}")

    for alias, stats in summary.get("tokenizers", {}).items():
        print(f"\n  Tokenizer: {alias}")
        if stats.get("samples", 0) == 0:
            print("    no samples processed")
            continue
        print(f"    samples: {stats['samples']:,}")
        print(f"    tokens: {stats['token_count']:,}, pad tokens: {stats['pad_count']:,}")
        print(f"    effective tokens: {stats['effective_tokens']:,} ({stats['utilization']:.2f}% utilization)")
        print(f"    pad/sep id: {stats.get('pad_id')} / {stats.get('sep_id')}")
        if stats.get("first_sample_text"):
            print(f"    decoded first sample: {stats['first_sample_text']!r}")


def print_overall_summary(
    train_summary: Optional[Dict[str, Any]],
    validation_summary: Optional[Dict[str, Any]],
    elapsed: float,
) -> None:
    print("\n" + "=" * 70)
    print("RUN SUMMARY")
    print("=" * 70)
    print(f"Total runtime: {elapsed:.2f}s")

    if train_summary:
        print("\nTRAIN ESTIMATES OVERVIEW")
        print(f"  Dataset: {train_summary.get('path')} | Lines: {train_summary.get('total_lines'):,}")
        for alias, stats in (train_summary.get("tokenizers") or {}).items():
            if not stats:
                print(f"  {alias}: no stats")
                continue
            print(
                f"  {alias}: est. samples={stats['estimated_samples']:,}, "
                f"est. tokens={stats['estimated_tokens']:,}, pad={stats['estimated_pad']:,}, "
                f"effective={stats['estimated_effective']:,}"
            )

    if validation_summary:
        print("\nVALIDATION COUNTS OVERVIEW")
        print(f"  Dataset: {validation_summary.get('path')} | Lines: {validation_summary.get('total_lines'):,}")
        for alias, stats in (validation_summary.get("tokenizers") or {}).items():
            if stats.get("samples", 0) == 0:
                print(f"  {alias}: no samples processed")
                continue
            print(
                f"  {alias}: samples={stats['samples']:,}, "
                f"tokens={stats['token_count']:,}, pad={stats['pad_count']:,}, "
                f"effective={stats['effective_tokens']:,} ({stats['utilization']:.2f}%)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplified MolGen3D token counting tool")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--sample-lines", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tokenizers", nargs="+", default=["qwen3_0.6b_origin", "qwen3_0.6b_custom"])
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--shuffle", action="store_true", help="Sample random lines via dataloader shuffle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument("--validation-num-workers", type=int, default=0)

    args = parser.parse_args()

    train_path = str(get_data_path("conformers_train"))
    valid_path = str(get_data_path("conformers_valid"))

    tokenizer_map: Dict[str, Tuple[str, AutoTokenizer]] = {}
    for alias in args.tokenizers:
        tok_path = str(get_tokenizer_path(alias))
        tokenizer_map[alias] = (tok_path, AutoTokenizer.from_pretrained(tok_path, use_fast=True, fix_mistral_regex=True))

    random.seed(args.seed)

    print("MolGen3D dataset token counting")
    print(f"seq_len: {args.seq_len}, sample_lines: {args.sample_lines}, batch_size: {args.batch_size}")

    start_time = time.time()

    train_summary = summarize_dataset(
        "train",
        train_path,
        args.tokenizers,
        tokenizer_map,
        seq_len=args.seq_len,
        sample_lines=args.sample_lines,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    validation_summary = None
    if not args.skip_validation:
        validation_summary = exhaust_validation_dataset(
            valid_path,
            args.tokenizers,
            tokenizer_map,
            seq_len=args.seq_len,
            batch_size=args.validation_batch_size,
            num_workers=args.validation_num_workers,
        )

    print_train_report(train_summary)
    if not args.skip_validation:
        print_validation_report(validation_summary)

    elapsed = time.time() - start_time
    print_overall_summary(train_summary, validation_summary, elapsed)


if __name__ == "__main__":
    main()
