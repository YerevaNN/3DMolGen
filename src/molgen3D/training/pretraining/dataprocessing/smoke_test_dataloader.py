#!/usr/bin/env python3
"""
Smoke test for molecular conformer dataloader.
Validates data loading, tokenization, and packing behavior.
"""

import argparse
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
from transformers import AutoTokenizer

from molgen3D.config.paths import get_data_path, get_tokenizer_path  # noqa: E402
from molgen3D.training.pretraining.dataprocessing.dataloader import build_dataloader, JsonlTaggedPackedDataset  # noqa: E402


def validate_tokenizer(tokenizer_path):
    """Load and validate tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, fix_mistral_regex=True)
    return tokenizer


def analyze_sample(sample_tokens: List[int], sep_id: int, pad_id: int, tokenizer) -> Dict[str, Any]:
    """Analyze a single sample and return statistics."""
    if not sample_tokens:
        return {"items": 0, "complete": False, "sep_count": 0, "token_count": 0, "text_length": 0, "tokens": [], "text": ""}

    # Count padding tokens at the end (they use the same ID as separators)
    # Padding only appears at the tail, so count consecutive pad_id from the end
    padding_count = 0
    for token in reversed(sample_tokens):
        if token == pad_id:
            padding_count += 1
        else:
            break
    
    # Count all occurrences of sep_id/pad_id, then subtract padding
    total_sep_pad = sum(1 for token in sample_tokens if token == sep_id)
    sep_count = total_sep_pad - padding_count
    
    # A sample is complete if it ends with a separator (not padding)
    # If there's no padding, check if last token is separator
    # If there is padding, check if the token before padding is separator
    if padding_count == 0:
        ends_with_sep = (sample_tokens[-1] == sep_id) if sample_tokens else False
    else:
        # Token before padding should be separator for complete sequence
        ends_with_sep = (len(sample_tokens) > padding_count and sample_tokens[-(padding_count + 1)] == sep_id)

    # Each separator represents an item boundary, so separator count = number of items
    items_in_sample = sep_count

    try:
        full_text = tokenizer.decode(sample_tokens, skip_special_tokens=False)
    except Exception:
        full_text = "<decode_error>"

    return {
        "items": items_in_sample,
        "complete": ends_with_sep,
        "sep_count": sep_count,
        "token_count": len(sample_tokens),
        "text": full_text,
        "tokens": sample_tokens
    }


def display_sample_analysis(sample_idx: int, stats: Dict[str, Any]) -> None:
    """Display analysis results for a single sample."""
    status = "COMPLETE" if stats["complete"] else "CUT-OFF"

    print(f"\n--- Sample {sample_idx} ({status}) ---")
    print(f"Items: {stats['items']} | Tokens: {stats['token_count']}")

    if not stats["complete"]:
        print("WARNING: Contains cut-off item (continues in next sample)")

    # Only show the first sample in full
    if sample_idx == 0:
        print(f"\nDecoded text: {stats['text']!r}")


def run_smoke_test(args, tokenizer):
    """Run the smoke test with comprehensive statistics."""
    start_time = time.time()

    # Build the dataloader
    print("Building dataloader...")
    loader = build_dataloader(
        train_path=args.train_path,
        tokenizer_path=args.tokenizer_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle_lines=True,
    )

    # Get separator and pad IDs from dataset (they're the same token: <|endoftext|>)
    dataset = loader.dataset
    sep_id = getattr(dataset, "sep_id", None)
    pad_id = getattr(dataset, "pad_id", None)
    
    if sep_id is None or pad_id is None:
        raise RuntimeError("Dataset missing separator/pad token IDs.")

    # Display configuration
    print("\n" + "="*60)
    print("MOLECULAR CONFORMER DATALOADER SMOKE TEST")
    print("="*60)
    print("Definitions: Item = dataset line (SMILES+conformer), Sample = 2048-token sequence, Batch = sample collection")
    print("="*60)
    print(f"Train paths: {args.train_path}")
    print(f"Tokenizer alias: {args.tokenizer_name}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Separator/Pad token ID: {sep_id} (<|endoftext|>)")
    print(f"Sequence length: {args.seq_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.num_workers}")
    print(f"Max batches: {args.max_batches}")
    print("="*60)

    # Statistics collection
    stats = {
        "batches_processed": 0,
        "samples_inspected": 0,
        "complete_samples": 0,
        "items_per_sample": [],
        "tokens_per_sample": [],
        "pad_counts": [],
        "seq_len": args.seq_len
    }

    print("\nTesting batches...")

    for batch_idx, (inputs, labels) in enumerate(loader):
        batch_start_time = time.time()
        input_ids = inputs["input"] if isinstance(inputs, dict) else inputs
        assert input_ids.shape == labels.shape, "Input and label shapes must match"

        batch_size, seq_len = input_ids.shape
        batch_num = batch_idx + 1

        print(f"\n--- Batch {batch_num}/{args.max_batches} ({batch_size} samples) ---")

        # Analyze samples from this batch
        batch_items = []
        batch_tokens = []
        batch_complete = 0
        samples_to_show = min(batch_size, 4)

        if pad_id is not None:
            pad_counts_batch = ((input_ids == pad_id).sum(dim=1)).tolist()
        else:
            pad_counts_batch = [0] * batch_size
        stats["pad_counts"].extend(pad_counts_batch)

        for sample_idx in range(samples_to_show):
            sample_tensor = input_ids[sample_idx]
            sample_tokens = sample_tensor.tolist()
            sample_stats = analyze_sample(sample_tokens, sep_id, pad_id, tokenizer)

            batch_items.append(sample_stats["items"])
            batch_tokens.append(sample_stats["token_count"])
            stats["items_per_sample"].append(sample_stats["items"])
            stats["tokens_per_sample"].append(sample_stats["token_count"])

            if sample_stats["complete"]:
                batch_complete += 1
                stats["complete_samples"] += 1

            stats["samples_inspected"] += 1
            display_sample_analysis(sample_idx, sample_stats)

        # Simple batch progress indicator
        print(f"✓ Batch {batch_num} processed")

        stats["batches_processed"] += 1
        if stats["batches_processed"] >= args.max_batches:
            break

    # Count validation samples
    try:
        val_count = count_validation_samples(args.tokenizer_path)
    except Exception as e:
        print(f"Warning: Could not count validation samples: {e}")
        val_count = None

    # Final comprehensive statistics
    display_comprehensive_stats(stats, start_time, val_count)

    avg_pad = sum(stats["pad_counts"]) / len(stats["pad_counts"]) if stats["pad_counts"] else 0.0
    print(f"\nTokenizer path used: {args.tokenizer_path}")
    print(f"Avg pads per sample: {avg_pad:.2f}")


def count_validation_samples(tokenizer_path: str) -> int:
    """Count all samples in the validation set in finite mode."""
    print("\n" + "="*60)
    print("COUNTING VALIDATION SET SAMPLES")
    print("="*60)

    start_time = time.time()
    val_path = str(get_data_path("conformers_valid"))

    print(f"Validation path: {val_path}")

    # Create finite dataloader (no infinite looping, no shuffling)
    dataloader = build_dataloader(
        train_path=val_path,
        tokenizer_path=tokenizer_path,
        seq_len=2048,  # Standard sequence length
        batch_size=1,   # Process one sample at a time for accurate counting
        num_workers=0,  # No multiprocessing for counting
        shuffle_lines=False,  # Deterministic for counting
        infinite=False,  # Finite mode - process each sample exactly once
    )

    print("Iterating through validation set (finite mode, no shuffling)...")

    sample_count = 0
    batch_count = 0

    try:
        for inputs, targets in dataloader:
            input_tensor = inputs["input"] if isinstance(inputs, dict) else inputs
            batch_size = input_tensor.shape[0]
            sample_count += batch_size
            batch_count += 1

            if batch_count % 100 == 0:  # Progress update every 100 batches
                elapsed = time.time() - start_time
                print(f"Processed {batch_count} batches, {sample_count} samples in {elapsed:.2f}s")

    except StopIteration:
        pass  # Normal end of dataset

    elapsed = time.time() - start_time

    print(f"\nValidation set counting complete:")
    print(f"   Total batches: {batch_count}")
    print(f"   Total samples: {sample_count}")
    print(f"   Time taken: {elapsed:.2f}s")
    print(f"   Samples per second: {sample_count/elapsed:.1f}" if elapsed > 0 else "   Samples per second: N/A")

    print("="*60)

    return sample_count


def display_comprehensive_stats(stats: Dict[str, Any], start_time: float, val_count: Optional[int] = None) -> None:
    """Display comprehensive statistics from the smoke test."""
    if not stats["samples_inspected"]:
        print("\nERROR: No samples were processed!")
        return

    total_time = time.time() - start_time

    print("\n" + "="*50)
    print("SMOKE TEST RESULTS")
    print("="*50)

    # Basic counts
    print(f"Runtime: {total_time:.2f}s")
    print(f"Batches: {stats['batches_processed']}")
    print(f"Samples: {stats['samples_inspected']}")

    # Validation set information
    if val_count is not None:
        print(f"Validation samples: {val_count}")

    # Item statistics (simplified)
    items_data = stats["items_per_sample"]
    if items_data:
        avg_items = sum(items_data) / len(items_data)
        print(f"Avg items per sample: {avg_items:.1f}")

    # Token efficiency (simplified)
    tokens_data = stats["tokens_per_sample"]
    if tokens_data:
        avg_tokens = sum(tokens_data) / len(tokens_data)
        efficiency = avg_tokens / stats["seq_len"] * 100
        print(f"Token utilization: {efficiency:.1f}%")

    print("\nSmoke test completed successfully!")
    print("="*50)


def main():
    """Main entry point for the smoke test."""
    # Set up distributed environment
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Molecular Conformer Dataloader Smoke Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python smoke_test_dataloader.py                           # Use defaults
  python smoke_test_dataloader.py --max_batches 2         # Test fewer batches
  python smoke_test_dataloader.py --batch_size 8          # Larger batches
        """
    )

    parser.add_argument(
        "--train_path", nargs="+",
        default=[str(get_data_path("conformers_train"))],
        help="Path(s)/globs to .jsonl files (default: conformers_train from paths.yaml)"
    )
    parser.add_argument(
        "--tokenizer_path",
        default="",
        help="Explicit HuggingFace tokenizer path (overrides --tokenizer_name)"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="qwen3_0.6b_origin",
        help="Alias to resolve via paths.yaml (default: qwen3_0.6b_origin, can be set to llama3_chem_v1 or qwen3_06b_custom)"
    )
    parser.add_argument(
        "--seq_len", type=int, default=2048,
        help="Maximum sequence length in tokens"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--max_batches", type=int, default=4,
        help="Maximum number of batches to test"
    )

    args = parser.parse_args()

    # Validate and load tokenizer
    tokenizer_path = args.tokenizer_path or str(get_tokenizer_path(args.tokenizer_name))
    args.tokenizer_path = tokenizer_path

    try:
        tokenizer = validate_tokenizer(tokenizer_path)
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer: {e}")
        return 1

    # Run the smoke test
    try:
        run_smoke_test(args, tokenizer)
        return 0
    except Exception as e:
        print(f"ERROR: Smoke test failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
