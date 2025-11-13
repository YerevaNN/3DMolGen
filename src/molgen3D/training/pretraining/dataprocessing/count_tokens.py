#!/usr/bin/env python3
"""
Dataset Token Counting and Statistics Tool for MolGen3D

This script analyzes molecular conformer datasets to provide comprehensive statistics
about token counts, sample packing efficiency, and data distribution.
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

# Fix module resolution when running this file directly
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent.parent  # Go up to repo root
if sys.path and sys.path[0] == str(_script_dir):
    sys.path.pop(0)
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Import our modules
from transformers import AutoTokenizer
from molgen3D.config.paths import get_data_path, get_tokenizer_path
from molgen3D.training.pretraining.dataprocessing.text_processing import build_unit, is_valid_unit
from molgen3D.training.pretraining.dataprocessing.dataloader import build_dataloader
from molgen3D.training.pretraining.dataprocessing.utils import expand_paths


def debug_examine_files():
    """Debug function to examine the first few lines of dataset files."""
    # Use our config paths
    train_path = str(get_data_path("conformers_train"))
    valid_path = str(get_data_path("conformers_valid"))

    print("DEBUG: Examining train dataset files...")
    if os.path.exists(train_path):
        jsonl_files = sorted(glob.glob(os.path.join(train_path, "*.jsonl")))
        print(f"Found {len(jsonl_files)} JSONL files in train: {jsonl_files[:3]}...")

        if jsonl_files:
            print(f"\nExamining first file: {jsonl_files[0]}")
            with open(jsonl_files[0], 'r') as f:
                for i, line in enumerate(f, 1):
                    if i > 5:  # Just first 5 lines
                        break
                    print(f"Line {i}: {repr(line[:150])}")
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            print(f"  Parsed as JSON: keys = {list(data.keys()) if isinstance(data, dict) else 'not dict'}")
                            if isinstance(data, dict):
                                for key, value in data.items():
                                    if isinstance(value, str):
                                        print(f"    {key}: {repr(value[:80])}{'...' if len(value) > 80 else ''}")
                                    else:
                                        print(f"    {key}: {type(value)} (len={len(value) if hasattr(value, '__len__') else 'N/A'})")
                        except json.JSONDecodeError as e:
                            print(f"  ERROR: Not JSON: {e}")
                    print()
    else:
        print(f"ERROR: Train path does not exist: {train_path}")

    print("\n" + "="*60)
    print("DEBUG: Examining validation dataset files...")

    if os.path.exists(valid_path):
        jsonl_files = sorted(glob.glob(os.path.join(valid_path, "*.jsonl")))
        print(f"Found {len(jsonl_files)} JSONL files in valid: {jsonl_files[:3]}...")

        if jsonl_files:
            print(f"\nExamining first file: {jsonl_files[0]}")
            with open(jsonl_files[0], 'r') as f:
                for i, line in enumerate(f, 1):
                    if i > 5:  # Just first 5 lines
                        break
                    print(f"Line {i}: {repr(line[:150])}")
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            print(f"  Parsed as JSON: keys = {list(data.keys()) if isinstance(data, dict) else 'not dict'}")
                            if isinstance(data, dict):
                                for key, value in data.items():
                                    if isinstance(value, str):
                                        print(f"    {key}: {repr(value[:80])}{'...' if len(value) > 80 else ''}")
                                    else:
                                        print(f"    {key}: {type(value)} (len={len(value) if hasattr(value, '__len__') else 'N/A'})")
                        except json.JSONDecodeError as e:
                            print(f"  ERROR: Not JSON: {e}")
                    print()
    else:
        print(f"ERROR: Valid path does not exist: {valid_path}")


def count_lines_in_jsonl_files(directory):
    """Count total lines across all .jsonl files in directory."""
    if not directory or not os.path.exists(directory):
        return 0, 0

    jsonl_files = sorted(glob.glob(os.path.join(directory, "*.jsonl")))
    if not jsonl_files:
        return 0, 0

    total_lines = 0
    for file in jsonl_files:
        with open(file, 'r') as f:
            lines = sum(1 for _ in f)
            total_lines += lines

    return total_lines, len(jsonl_files)


def sample_items_and_create_samples(dataset_path: str, tokenizer_path: str, seq_len: int, num_items: int = 10000) -> Optional[Dict[str, Any]]:
    """Load items from dataset and pack them into samples (2048-token sequences) using actual dataloader logic."""
    try:
        # Load tokenizer for processing
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

        # First, get total items and build file list
        if not os.path.exists(dataset_path):
            return None

        jsonl_files = sorted(glob.glob(os.path.join(dataset_path, "*.jsonl")))
        if not jsonl_files:
            return None

        # Build cumulative item counts per file
        import random
        random.seed(random.randint(0, 2**31 - 1))  # Random seed for this run

        file_item_counts = []
        total_items = 0

        for file_path in jsonl_files:
            with open(file_path, 'r') as f:
                count = sum(1 for _ in f)
                file_item_counts.append((file_path, count))
                total_items += count

        if total_items == 0:
            return None

        # Sample random item indices
        if num_items >= total_items:
            selected_indices = list(range(total_items))
        else:
            selected_indices = sorted(random.sample(range(total_items), num_items))

        print(f"  Loading {len(selected_indices)} items from {total_items:,} total items...")

        # Convert global indices to file-specific indices and collect all tokenized items
        all_tokenized_items = []
        items_processed = 0

        global_idx = 0
        selected_idx = 0

        for file_path, file_item_count in file_item_counts:
            if selected_idx >= len(selected_indices):
                break

            file_start_idx = global_idx
            file_end_idx = global_idx + file_item_count

            # Find indices that belong to this file
            file_indices = []
            while selected_idx < len(selected_indices) and selected_indices[selected_idx] < file_end_idx:
                file_indices.append(selected_indices[selected_idx] - file_start_idx)
                selected_idx += 1

            if not file_indices:
                global_idx = file_end_idx
                continue

            # Read the specific lines from this file
            with open(file_path, 'r') as f:
                lines = f.readlines()

            for local_idx in file_indices:
                if local_idx < len(lines):
                    line = lines[local_idx].strip()
                    if not line:
                        continue

                    try:
                        # Parse JSON line for item
                        data = json.loads(line)
                        if not isinstance(data, dict) or 'canonical_smiles' not in data or 'embedded_smiles' not in data:
                            continue

                        canonical = data['canonical_smiles']
                        embedded = data['embedded_smiles']

                        # Validate and build unit (item)
                        if not is_valid_unit(canonical, embedded):
                            continue

                        processed_text = build_unit(canonical, embedded)

                        # Use actual dataloader tokenization logic: EOS at end of each item
                        tokens = tokenizer.encode(processed_text, add_special_tokens=False) + [tokenizer.eos_token_id]

                        all_tokenized_items.append({
                            'tokens': tokens,
                            'canonical': canonical,
                            'embedded': embedded,
                            'processed_text': processed_text,
                        })

                        items_processed += 1

                    except (json.JSONDecodeError, KeyError, Exception):
                        continue

            global_idx = file_end_idx

        if items_processed == 0:
            return None

        # Now pack all items into samples (2048-token sequences) using ChunkPacker
        from molgen3D.training.pretraining.dataprocessing.text_processing import ChunkPacker

        packer = ChunkPacker(seq_len=seq_len, bos_id=tokenizer.bos_token_id, eos_id=tokenizer.eos_token_id)
        samples_created = []

        # Feed all tokenized items to the packer and collect samples as they become available
        print(f"  Packing {len(all_tokenized_items)} items with ChunkPacker...")
        for i, item in enumerate(all_tokenized_items):
            packer.try_add_unit(item['tokens'])

            # Collect any completed samples after adding this item
            for inp, lab in packer.yield_blocks():
                sample_tokens = inp.tolist()
                decoded_text = tokenizer.decode(sample_tokens)
                samples_created.append({
                    'tokens': sample_tokens,
                    'decoded_text': decoded_text,
                    'token_count': len(sample_tokens),  # Should be seq_len
                })

            if (i + 1) % 500 == 0:
                print(f"    Added {i+1} items, created {len(samples_created)} samples so far")

        print(f"  Total samples created: {len(samples_created)}")

        # Handle any remaining content in the buffer (partial sample)
        # In real dataloader, this would be carried to next batch, but for estimation we'll treat it as a partial sample
        if len(packer.buf) > 0:
            print(f"  Remaining buffer length: {len(packer.buf)}")
            # For estimation, we'll consider the remaining buffer as a partial sample
            # but only if it contains actual content (not just padding)
            remaining_tokens = packer.buf[:]
            if len(remaining_tokens) > 0:
                # Don't pad this one since it's partial - just include as-is for counting purposes
                decoded_text = tokenizer.decode(remaining_tokens)
                samples_created.append({
                    'tokens': remaining_tokens,
                    'decoded_text': decoded_text,
                    'token_count': len(remaining_tokens),
                })
                print(f"  Created 1 partial sample from remaining buffer ({len(remaining_tokens)} tokens)")

        # Calculate stats
        total_tokens_in_samples = len(samples_created) * seq_len
        avg_items_per_sample = items_processed / len(samples_created) if samples_created else 0

        return {
            'items_processed': items_processed,
            'samples_created': len(samples_created),
            'total_tokens_in_samples': total_tokens_in_samples,
            'avg_items_per_sample': avg_items_per_sample,
            'samples': samples_created[:2],  # Only keep first 2 samples for display
            'first_item': all_tokenized_items[0] if all_tokenized_items else None,  # For detailed token breakdown
        }

    except Exception as e:
        print(f"ERROR: Failed to process items: {e}")
        return None


def estimate_dataset(dataset_name, dataset_path, tokenizer_path, tokenizer, num_items, seq_len, is_validation=False):
    """Estimate tokens and samples by loading items and creating samples using dataloader logic."""
    dataset_type = "VALIDATION" if is_validation else "TRAIN"
    print(f"\n{'='*70}")
    print(f"{dataset_type} DATASET: {dataset_name}")
    print(f"{'='*70}")

    # Count total items in dataset
    if not dataset_path:
        print("ERROR: No dataset path provided")
        return None

    total_items, num_files = count_lines_in_jsonl_files(dataset_path)
    print(f"Dataset path: {dataset_path}")
    print(f"Total .jsonl files: {num_files}")
    print(f"Total items in dataset: {total_items:,}")

    if total_items == 0:
        print("ERROR: No items found in dataset")
        return None

    # Load items and create samples using dataloader logic
    sample_result = sample_items_and_create_samples(
        dataset_path,
        tokenizer_path,
        seq_len,
        num_items
    )

    if sample_result is None:
        print("ERROR: Failed to process items")
        return None

    items_processed = sample_result['items_processed']
    samples_created = sample_result['samples_created']
    total_tokens_in_samples = sample_result['total_tokens_in_samples']
    avg_items_per_sample = sample_result['avg_items_per_sample']
    samples = sample_result['samples']  # First 2 samples
    first_item = sample_result['first_item']

    print(f"\nFROM SAMPLE OF {items_processed} ITEMS:")
    print(f"  Samples created: {samples_created}")
    print(f"  Total tokens in samples: {total_tokens_in_samples:,}")
    print(f"  Avg items per sample: {avg_items_per_sample:.2f}")

    if items_processed == 0:
        print("ERROR: No items were successfully processed.")
        return None

    # Extrapolate to full dataset
    items_ratio = total_items / items_processed
    estimated_total_samples = int(samples_created * items_ratio)
    estimated_total_tokens = estimated_total_samples * seq_len

    print(f"\nEXTRAPOLATED TO FULL DATASET:")
    print(f"  Estimated total samples (seq_len={seq_len}): {estimated_total_samples:,}")
    print(f"  Estimated total tokens: {estimated_total_tokens:,}")

    # Print decoded version of first 2 samples
    if samples:
        print(f"\nSAMPLE ANALYSIS:")
        for i, sample in enumerate(samples, 1):
            print(f"\n--- Sample {i} ---")
            print(f"Tokens: {sample['token_count']}")
            print(f"Decoded: {sample['decoded_text']!r}")

    # Print full token-to-decoded breakdown of 1 item only
    if first_item:
        print(f"\nFIRST ITEM TOKEN BREAKDOWN:")
        print(f"SMILES: {first_item['canonical']}")
        print(f"Tokens: {len(first_item['tokens'])}")
        print(f"Decoded: {tokenizer.decode(first_item['tokens'])!r}")

        print(f"\nDetailed token breakdown:")
        for j, token_id in enumerate(first_item['tokens'][:50]):  # Show first 50 tokens
            token_text = tokenizer.decode([token_id])
            print(f"Token {j:3d}: {token_id:6d} -> {repr(token_text)}")
        if len(first_item['tokens']) > 50:
            print(f"... and {len(first_item['tokens']) - 50} more tokens")

    return {
        'total_items': total_items,
        'items_processed': items_processed,
        'samples_created': samples_created,
        'avg_items_per_sample': avg_items_per_sample,
        'estimated_total_samples': estimated_total_samples,
        'estimated_total_tokens': estimated_total_tokens,
    }


def main():
    """Main entry point for the dataset token counting tool."""
    # Special debug mode to examine files without dependencies
    if len(sys.argv) > 1 and sys.argv[1] == "--debug-files":
        debug_examine_files()
        return

    parser = argparse.ArgumentParser(
        description="MolGen3D Dataset Token Counting and Statistics Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python count_tokens.py                           # Analyze all datasets with defaults
  python count_tokens.py --sample-lines 500       # Use smaller sample size
  python count_tokens.py --seq-len 4096           # Different sequence length
  python count_tokens.py --skip-validation        # Skip validation analysis
  python count_tokens.py --debug-files            # Just examine file formats
        """
    )

    parser.add_argument(
        "--train-dataset-name",
        type=str,
        default="conformers_train",
        help="Name identifier for train dataset",
    )
    parser.add_argument(
        "--valid-dataset-name",
        type=str,
        default="conformers_valid",
        help="Name identifier for validation dataset",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length for tokenization",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=10000,
        help="Number of items to load from dataset for estimation",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation dataset analysis",
    )

    args = parser.parse_args()

    # Get dataset paths from our config
    train_path = str(get_data_path("conformers_train"))
    valid_path = str(get_data_path("conformers_valid"))

    print("="*70)
    print("MOLGEN3D DATASET TOKEN COUNTING TOOL")
    print("="*70)
    print(f"Loading {args.num_items} items from datasets to estimate sample/token counts")
    print(f"Sequence length: {args.seq_len} (samples = 2048-token sequences)")
    print("="*70)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer_path = str(get_tokenizer_path("llama3_chem_v1"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    start_time = time.time()

    # Analyze train dataset
    train_stats = estimate_dataset(
        args.train_dataset_name,
        train_path,
        tokenizer_path,
        tokenizer,
        args.num_items,
        args.seq_len,
        is_validation=False
    )

    # Analyze validation dataset
    valid_stats = None
    if not args.skip_validation:
        valid_stats = estimate_dataset(
            args.valid_dataset_name,
            valid_path,
            tokenizer_path,
            tokenizer,
            args.num_items,
            args.seq_len,
            is_validation=True
        )

    # Print final comprehensive summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total analysis time: {total_time:.2f}s")

    if train_stats:
        print(f"\nTRAIN DATASET ({args.train_dataset_name}):")
        print(f"  Total items: {train_stats['total_items']:,}")
        print(f"  Avg items per sample: {train_stats['avg_items_per_sample']:.2f}")
        print(f"  Estimated total tokens: {train_stats['estimated_total_tokens']:,}")
        print(f"  Estimated total samples: {train_stats['estimated_total_samples']:,}")
        print(f"  Estimated data size: ~{train_stats['estimated_total_tokens'] * 2 / (1024**3):.1f} GB (2 bytes/token)")
    else:
        print(f"\nTRAIN DATASET: Failed to estimate")

    if valid_stats:
        print(f"\nVALIDATION DATASET ({args.valid_dataset_name}):")
        print(f"  Total items: {valid_stats['total_items']:,}")
        print(f"  Avg items per sample: {valid_stats['avg_items_per_sample']:.2f}")
        print(f"  Estimated total tokens: {valid_stats['estimated_total_tokens']:,}")
        print(f"  Estimated total samples: {valid_stats['estimated_total_samples']:,}")
        print(f"  Estimated data size: ~{valid_stats['estimated_total_tokens'] * 2 / (1024**3):.1f} GB (2 bytes/token)")
    elif not args.skip_validation:
        print(f"\nVALIDATION DATASET: Failed to estimate")

    # Training recommendations
    if train_stats and valid_stats:
        train_samples = train_stats['estimated_total_samples']
        valid_samples = valid_stats['estimated_total_samples']

        print(f"\nTRAINING RECOMMENDATIONS:")
        print(f"  Train/Valid split: {train_samples/valid_samples:.1f}:1")
        print(f"  Gradient accumulation: Consider ~{max(1, int(1000000 // train_samples))} if batch_size=1")
        print(f"  Epochs for 100M tokens: ~{int(100000000 // train_stats['estimated_total_tokens'])}")

    print(f"\n{'='*70}")
    print("Analysis complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
