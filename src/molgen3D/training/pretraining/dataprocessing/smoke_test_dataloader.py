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
    """Load and validate tokenizer has required special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    if bos_id is None or eos_id is None:
        raise RuntimeError("Tokenizer must define bos_token_id and eos_token_id.")

    return tokenizer, bos_id, eos_id


def analyze_sample(sample_tokens: List[int], eos_id: int, tokenizer) -> Dict[str, Any]:
    """Analyze a single sample and return statistics."""
    if not sample_tokens:
        return {"items": 0, "complete": False, "eos_count": 0, "token_count": 0, "text_length": 0, "tokens": [], "text": ""}

    eos_count = sum(1 for token in sample_tokens if token == eos_id)
    ends_with_eos = sample_tokens[-1] == eos_id

    # Each EOS represents an item (molecular conformer), so EOS count = number of items
    items_in_sample = eos_count

    try:
        full_text = tokenizer.decode(sample_tokens)
    except Exception:
        full_text = "<decode_error>"

    return {
        "items": items_in_sample,
        "complete": ends_with_eos,
        "eos_count": eos_count,
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


def run_smoke_test(args, tokenizer, bos_id, eos_id):
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

    # Display configuration
    print("\n" + "="*60)
    print("MOLECULAR CONFORMER DATALOADER SMOKE TEST")
    print("="*60)
    print("Definitions: Item = dataset line (SMILES+conformer), Sample = 2048-token sequence, Batch = sample collection")
    print("="*60)
    print(f"Train paths: {args.train_path}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Special tokens: BOS={bos_id}, EOS={eos_id}")
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
        "seq_len": args.seq_len
    }

    print("\nTesting batches...")

    for batch_idx, (inputs, labels) in enumerate(loader):
        batch_start_time = time.time()
        assert inputs.shape == labels.shape, "Input and label shapes must match"

        batch_size, seq_len = inputs.shape
        batch_num = batch_idx + 1

        print(f"\n--- Batch {batch_num}/{args.max_batches} ({batch_size} samples) ---")

        # Analyze samples from this batch
        batch_items = []
        batch_tokens = []
        batch_complete = 0
        samples_to_show = min(batch_size, 4)

        for sample_idx in range(samples_to_show):
            sample_tokens = inputs[sample_idx].tolist()
            sample_stats = analyze_sample(sample_tokens, eos_id, tokenizer)

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


def count_validation_samples(tokenizer_path: str) -> int:
    """Count all samples in the validation set in finite mode."""
    print("\n" + "="*60)
    print("COUNTING VALIDATION SET SAMPLES")
    print("="*60)

    start_time = time.time()
    val_path = str(get_data_path("conformers_valid"))

    print(f"Validation path: {val_path}")

    # Create dataset in finite mode (no infinite looping)
    dataset = JsonlTaggedPackedDataset(
        train_path=val_path,
        tokenizer_path=tokenizer_path,
        seq_len=2048,  # Standard sequence length
        shuffle_lines=False,  # Deterministic for counting
        infinite=False  # Finite mode - process each sample exactly once
    )

    print("Iterating through validation set (finite mode)...")

    sample_count = 0
    batch_count = 0

    try:
        for inputs, targets in dataset:
            batch_size = inputs.shape[0]
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
        default=str(get_tokenizer_path("llama3_chem_v1")),
        help="HuggingFace tokenizer path (default: llama3_chem_v1 from paths.yaml)"
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
    try:
        tokenizer, bos_id, eos_id = validate_tokenizer(args.tokenizer_path)
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer: {e}")
        return 1

    # Run the smoke test
    try:
        run_smoke_test(args, tokenizer, bos_id, eos_id)
        return 0
    except Exception as e:
        print(f"ERROR: Smoke test failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())