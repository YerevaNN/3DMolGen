#!/usr/bin/env python3
"""
Minimal dataset token counting script for MolGen3D.

The script samples a configurable number of JSONL entries from one file per dataset
using the production dataloader, reports how many batches and tokens that sample produced,
extrapolates totals for the whole split, and verifies the extrapolation using file sizes.
"""

import argparse
import json
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


def _extract_inputs(batch):
    """
    Normalize dataloader output into a tensor of input_ids.

    Supports:
      - inputs
      - (inputs, target)
      - {"input": tensor, ...}
    """
    if isinstance(batch, (tuple, list)):
        inputs = batch[0]
    else:
        inputs = batch
    return inputs["input"] if isinstance(inputs, dict) else inputs


def count_lines_and_bytes(files: List[Path]) -> Tuple[int, int, List[Dict[str, Any]]]:
    total_lines = 0
    total_bytes = 0
    file_stats: List[Dict[str, Any]] = []
    for file in files:
        try:
            with file.open("rb") as fh:
                line_count = sum(1 for _ in fh)
            byte_count = file.stat().st_size
            total_lines += line_count
            total_bytes += byte_count
            file_stats.append({"path": str(file), "lines": line_count, "bytes": byte_count})
        except OSError:
            continue
    return total_lines, total_bytes, file_stats


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
    target_lines: int,
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
    # This script assumes pad tokens only appear at the end and that pad_id == sep_id,
    # so it interprets "sep_count minus trailing pads" as the number of items.
    if pad_id != sep_id:
        raise RuntimeError(
            f"This token counting script assumes pad_id == sep_id, "
            f"but got pad_id={pad_id}, sep_id={sep_id}. "
            "Update the counting logic before using it with distinct pad/sep tokens."
        )

    items = 0
    samples = 0
    batches = 0
    pad_total = 0

    for batch in loader:
        batches += 1
        inputs = _extract_inputs(batch)
        bsz = inputs.size(0)

        for idx in range(bsz):
            sample = inputs[idx]
            pad_count = 0
            for token in reversed(sample.tolist()):
                if token == pad_id:
                    pad_count += 1
                else:
                    break
            sep_total = int((sample == sep_id).sum().item())
            # Given pad_id == sep_id, trailing pad tokens are also counted as seps; subtract them.
            sep_count = max(sep_total - pad_count, 0)

            items += sep_count
            pad_total += pad_count
            samples += 1

            if items >= target_lines:
                break
        if items >= target_lines:
            break

    if samples == 0:
        return None

    tokens_produced = samples * seq_len
    avg_items_per_sample = items / samples
    avg_pad_per_sample = pad_total / samples
    effective_tokens = tokens_produced - pad_total

    return {
        "lines_target": float(target_lines),
        "lines_consumed": float(items),
        "samples": float(samples),
        "batches": float(batches),
        "tokens_produced": float(tokens_produced),
        "effective_tokens": float(effective_tokens),
        "avg_pad_per_sample": float(avg_pad_per_sample),
        "avg_items_per_sample": float(avg_items_per_sample),
        "pad_id": float(pad_id),
        "sep_id": float(sep_id),
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

    total_lines, total_bytes, file_stats = count_lines_and_bytes(files)

    dataset_summary: Dict[str, Any] = {
        "name": name,
        "path": directory,
        "total_lines": total_lines,
        "total_bytes": total_bytes,
        "tokenizers": {},
    }

    for alias in tokenizer_aliases:
        tokenizer_path, tokenizer = tokenizer_map[alias]
        sum_samples = 0.0
        sum_items = 0.0
        sum_pad = 0.0
        sum_batches = 0.0
        sum_tokens_produced = 0.0
        sum_effective = 0.0
        sum_target_lines = 0.0
        sum_sample_bytes = 0.0

        for file_path in files:
            file_line_count = next((f["lines"] for f in file_stats if f["path"] == str(file_path)), None)
            if file_line_count is None:
                try:
                    with file_path.open("rb") as fh:
                        file_line_count = sum(1 for _ in fh)
                except OSError:
                    continue
            target_lines = min(sample_lines, file_line_count)

            stats = sample_dataloader(
                file_path,
                tokenizer_path,
                tokenizer,
                seq_len,
                target_lines=target_lines,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
            )
            if not stats:
                continue

            sum_samples += stats["samples"]
            sum_items += stats["lines_consumed"]
            sum_pad += stats["avg_pad_per_sample"] * stats["samples"]
            sum_batches += stats["batches"]
            sum_tokens_produced += stats["tokens_produced"]
            sum_effective += stats["effective_tokens"]
            sum_target_lines += stats["lines_target"]

            sample_bytes = bytes_for_lines(file_path, int(target_lines))
            if sample_bytes:
                sum_sample_bytes += sample_bytes

        if sum_samples == 0 or sum_target_lines == 0:
            dataset_summary["tokenizers"][alias] = None
            continue

        avg_items_per_sample = sum_items / sum_samples
        avg_pad_per_sample = sum_pad / sum_samples
        valid_ratio = sum_items / sum_target_lines  # fraction of sampled lines that produced usable items

        estimated_valid_lines = total_lines * valid_ratio
        estimated_samples = int(estimated_valid_lines / avg_items_per_sample) if avg_items_per_sample else 0
        estimated_batches = math.ceil(estimated_samples / batch_size) if batch_size else 0
        estimated_tokens = estimated_samples * seq_len
        estimated_pad = int(avg_pad_per_sample * estimated_samples)
        estimated_effective = estimated_tokens - estimated_pad

        # Byte-size sanity check: scale sampled bytes to total lines.
        verification = verify_bytes(
            sum_sample_bytes if sum_sample_bytes else None,
            int(sum_target_lines),
            total_lines,
            total_bytes,
        )

        dataset_summary["tokenizers"][alias] = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "sampled_files": len(files),
            "lines_target_total": int(sum_target_lines),
            "lines_consumed_total": int(sum_items),
            "sample_bytes_total": int(sum_sample_bytes),
            "batches_sampled": int(sum_batches),
            "tokens_produced_sampled": int(sum_tokens_produced),
            "effective_tokens_sampled": int(sum_effective),
            "avg_items_per_sample": float(avg_items_per_sample),
            "avg_pad_per_sample": float(avg_pad_per_sample),
            "valid_ratio": float(valid_ratio),
            "estimated_valid_lines": int(estimated_valid_lines),
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
    print(f"  total lines: {summary.get('total_lines', 0):,}")

    for alias, stats in (summary.get("tokenizers") or {}).items():
        print(f"\n  Tokenizer: {alias}")
        if stats is None:
            print("    no samples collected")
            continue
        print(
            "    estimates: units≈{estimated_valid_lines:,}, "
            "samples≈{estimated_samples:,}, tokens≈{estimated_tokens:,}, "
            "pad≈{estimated_pad:,}, effective≈{estimated_effective:,}".format(
                estimated_valid_lines=stats.get("estimated_valid_lines", 0),
                estimated_samples=stats.get("estimated_samples", 0),
                estimated_tokens=stats.get("estimated_tokens", 0),
                estimated_pad=stats.get("estimated_pad", 0),
                estimated_effective=stats.get("estimated_effective", 0),
            )
        )


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

    total_lines, total_bytes, _ = count_lines_and_bytes(files)
    result: Dict[str, Any] = {
        "path": directory,
        "total_lines": total_lines,
        "total_bytes": total_bytes,
        "tokenizers": {},
    }

    for alias in tokenizer_aliases:
        tokenizer_path, tokenizer = tokenizer_map[alias]
        # For PyTorch DataLoader compatibility: prefetch_factor must be an int when workers > 0.
        prefetch_factor = None if num_workers == 0 else 2
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
            prefetch_factor=prefetch_factor,
            world_size=1,
            rank=0,
        )

        dataset = getattr(loader, "dataset", None)
        _ensure_dataset_tokenizer(dataset)
        pad_id = getattr(dataset, "pad_id", None)
        sep_id = getattr(dataset, "sep_id", None)
        sample_count = 0
        batch_count = 0
        token_count = 0
        pad_total = 0

        for batch in loader:
            input_ids = _extract_inputs(batch)
            if input_ids.numel() == 0:
                continue

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
    print(f"  total lines: {summary.get('total_lines', 0):,}")

    for alias, stats in summary.get("tokenizers", {}).items():
        print(f"\n  Tokenizer: {alias}")
        if stats.get("samples", 0) == 0:
            print("    no samples processed")
            continue
        print(
            f"    samples={stats['samples']:,}, tokens={stats['token_count']:,}, "
            f"pad={stats['pad_count']:,}, effective={stats['effective_tokens']:,} "
            f"({stats['utilization']:.2f}% util)"
        )


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
        train_path = train_summary.get("path")
        sample_file = train_summary.get("sample_file")
        if train_path:
            print(f"  Dataset path: {train_path}")
        if sample_file:
            sample_path = f"{train_path}/{sample_file}" if train_path else sample_file
            print(f"  Sample file: {sample_path}")
        print(f"  Total lines: {train_summary.get('total_lines', 0):,}")
        for alias, stats in (train_summary.get("tokenizers") or {}).items():
            if not stats:
                print(f"  {alias}: no stats")
                continue
            print(
                f"  {alias}: units≈{stats['estimated_valid_lines']:,}, "
                f"samples≈{stats['estimated_samples']:,}, "
                f"tokens≈{stats['estimated_tokens']:,}, pad≈{stats['estimated_pad']:,}, "
                f"effective≈{stats['estimated_effective']:,}"
            )

    if validation_summary:
        print("\nVALIDATION COUNTS OVERVIEW")
        valid_path = validation_summary.get("path")
        if valid_path:
            print(f"  Dataset path: {valid_path}")
        print(f"  Total lines: {validation_summary.get('total_lines', 0):,}")
        for alias, stats in (validation_summary.get("tokenizers") or {}).items():
            if stats.get("samples", 0) == 0:
                print(f"  {alias}: no samples processed")
                continue
            print(
                f"  {alias}: samples={stats['samples']:,}, "
                f"tokens={stats['token_count']:,}, pad={stats['pad_count']:,}, "
                f"effective={stats['effective_tokens']:,} ({stats['utilization']:.2f}%)"
            )


def dump_json_summary(
    train_summary: Optional[Dict[str, Any]],
    validation_summary: Optional[Dict[str, Any]],
    elapsed: float,
) -> None:
    """
    Emit a machine-readable summary so callers can parse results without scraping stdout.
    """
    payload = {
        "train": train_summary,
        "validation": validation_summary,
        "elapsed_seconds": elapsed,
    }
    try:
        print("\nJSON_SUMMARY_START")
        print(json.dumps(payload, indent=2, sort_keys=True))
        print("JSON_SUMMARY_END")
    except Exception:
        # Fallback: avoid crashing the script if serialization fails on unexpected fields.
        pass


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
    dump_json_summary(train_summary, validation_summary, elapsed)


if __name__ == "__main__":
    main()
