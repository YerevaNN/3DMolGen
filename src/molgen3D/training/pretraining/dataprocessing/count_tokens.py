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


def sample_random_n_lines(directory: str, tokenizer, num_lines: int = 1000) -> Optional[Dict[str, Any]]:
    """Randomly sample N lines from JSONL files and process them using our molgen3D pipeline."""
    if not directory or not os.path.exists(directory):
        return None

    jsonl_files = sorted(glob.glob(os.path.join(directory, "*.jsonl")))
    if not jsonl_files:
        return None

    import numpy as np
    rng = np.random.default_rng(42)

    # First, build a map of cumulative line counts per file
    file_line_counts = []
    total_lines = 0

    for file_path in jsonl_files:
        with open(file_path, 'r') as f:
            count = sum(1 for _ in f)
            file_line_counts.append((file_path, count))
            total_lines += count

    if total_lines == 0:
        return None

    # Randomly select line indices from the entire dataset
    if num_lines >= total_lines:
        # If we want more lines than available, take all lines
        selected_indices = list(range(total_lines))
    else:
        selected_indices = rng.choice(total_lines, size=num_lines, replace=False)
        selected_indices.sort()  # Sort for efficient file reading

    print(f"  Randomly sampling {len(selected_indices)} lines from {total_lines:,} total lines across {len(jsonl_files)} files...")

    lines_processed = 0
    total_chars = 0
    total_tokens = 0
    samples = []  # Store up to 3 samples with tokens and decoded text

    # Convert global indices to file-specific indices
    global_idx = 0
    selected_idx = 0

    for file_path, file_line_count in file_line_counts:
        if selected_idx >= len(selected_indices):
            break

        file_start_idx = global_idx
        file_end_idx = global_idx + file_line_count

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
                    # Parse JSON line for our conformer format
                    data = json.loads(line)
                    if not isinstance(data, dict) or 'canonical_smiles' not in data or 'embedded_smiles' not in data:
                        continue

                    canonical = data['canonical_smiles']
                    embedded = data['embedded_smiles']

                    # Validate and build unit using our processing
                    if not is_valid_unit(canonical, embedded):
                        continue

                    processed_text = build_unit(canonical, embedded)
                    # Add EOS token at the beginning (our dataloader does this)
                    tokens = [tokenizer.eos_token_id] + tokenizer.encode(processed_text, add_special_tokens=False)

                    # Store sample information (up to 3 samples for detailed analysis)
                    if len(samples) < 3:
                        decoded_text = tokenizer.decode(tokens)
                        samples.append({
                            'tokens': tokens,
                            'decoded_text': decoded_text,
                            'processed_text': processed_text,
                            'canonical': canonical,
                            'embedded': embedded
                        })

                    total_chars += len(processed_text)
                    total_tokens += len(tokens)
                    lines_processed += 1

                except (json.JSONDecodeError, KeyError, Exception) as e:
                    # Skip malformed lines
                    continue

        global_idx = file_end_idx

    return {
        'lines_processed': lines_processed,
        'total_chars': total_chars,
        'total_tokens': total_tokens,
        'samples': samples,  # List of up to 3 samples with detailed info
    }


def estimate_dataset(dataset_name, dataset_path, tokenizer, sample_lines, seq_len, is_validation=False):
    """Estimate tokens and samples by randomly sampling N lines and extrapolating."""
    dataset_type = "VALIDATION" if is_validation else "TRAIN"
    print(f"\n{'='*70}")
    print(f"{dataset_type} DATASET: {dataset_name}")
    print(f"{'='*70}")

    # Count total lines in dataset
    if not dataset_path:
        print("ERROR: No dataset path provided")
        return None

    total_lines, num_files = count_lines_in_jsonl_files(dataset_path)
    print(f"Dataset path: {dataset_path}")
    print(f"Total .jsonl files: {num_files}")
    print(f"Total lines in dataset: {total_lines:,}")

    if total_lines == 0:
        print("ERROR: No lines found in dataset")
        return None

    # Sample N random lines
    sample_result = sample_random_n_lines(
        dataset_path,
        tokenizer,
        sample_lines
    )

    if sample_result is None:
        print("ERROR: Failed to sample lines")
        return None

    lines_sampled = sample_result['lines_processed']
    chars_sampled = sample_result['total_chars']
    tokens_sampled = sample_result['total_tokens']
    samples = sample_result['samples']

    print(f"\nFROM RANDOM SAMPLE OF {lines_sampled} LINES:")
    print(f"  Characters in sampled lines: {chars_sampled:,}")
    print(f"  Tokens produced: {tokens_sampled:,}")
    print(f"  Samples this represents: {tokens_sampled // seq_len:,}")

    if lines_sampled == 0:
        print("ERROR: No lines were successfully processed.")
        return None

    print(f"  Avg chars per line: {chars_sampled / lines_sampled:.1f}")
    print(f"  Avg tokens per line: {tokens_sampled / lines_sampled:.1f}")

    # Calculate lines per sample
    lines_per_sample = lines_sampled / (tokens_sampled / seq_len)
    print(f"  Lines per sample: {lines_per_sample:.2f}")

    # Extrapolate to full dataset
    lines_ratio = total_lines / lines_sampled
    estimated_total_chars = chars_sampled * lines_ratio
    estimated_total_tokens = tokens_sampled * lines_ratio
    estimated_total_samples = int(estimated_total_tokens / seq_len)

    print(f"\nEXTRAPOLATED TO FULL DATASET:")
    print(f"  Estimated total characters: {int(estimated_total_chars):,}")
    print(f"  Estimated total tokens: {int(estimated_total_tokens):,}")
    print(f"  Estimated total samples (seq_len={seq_len}): {estimated_total_samples:,}")

    # Print sample analysis
    if samples:
        print(f"\nSAMPLE ANALYSIS:")
        for i, sample in enumerate(samples, 1):
            print(f"\n--- Sample {i} ---")
            print(f"SMILES: {sample['canonical']}")
            print(f"Conformer length: {len(sample['embedded'])} chars")

            tokens = sample['tokens']
            print(f"Tokens: {len(tokens)} (showing first 50): {tokens[:50]}{'...' if len(tokens) > 50 else ''}")

            decoded_text = sample['decoded_text']
            print(f"Decoded (first 200 chars): {decoded_text[:200]}{'...' if len(decoded_text) > 200 else ''}")

            # Verify EOS token placement
            eos_positions = [j for j, t in enumerate(tokens) if t == tokenizer.eos_token_id]
            print(f"EOS positions: {eos_positions}")

            # Show detailed token breakdown for first sample only
            if i == 1:  # Only for the first sample
                print(f"\nDETAILED TOKEN BREAKDOWN (first 100 tokens):")
                for j, token_id in enumerate(tokens[:100]):
                    try:
                        token_text = tokenizer.decode([token_id])
                        print(f"Token {j:3d}: {token_id:6d} -> {repr(token_text)}")
                    except Exception as e:
                        print(f"Token {j:3d}: {token_id:6d} -> <error: {e}>")

    return {
        'total_lines': total_lines,
        'lines_sampled': lines_sampled,
        'chars_sampled': chars_sampled,
        'tokens_sampled': tokens_sampled,
        'lines_per_sample': lines_per_sample,
        'estimated_total_chars': int(estimated_total_chars),
        'estimated_total_tokens': int(estimated_total_tokens),
        'estimated_total_samples': estimated_total_samples,
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
        "--sample-lines",
        type=int,
        default=1000,
        help="Number of lines to randomly sample for estimation",
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
    print(f"Randomly sampling {args.sample_lines} lines to estimate full datasets")
    print(f"Sequence length: {args.seq_len}")
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
        tokenizer,
        args.sample_lines,
        args.seq_len,
        is_validation=False
    )

    # Analyze validation dataset
    valid_stats = None
    if not args.skip_validation:
        valid_stats = estimate_dataset(
            args.valid_dataset_name,
            valid_path,
            tokenizer,
            args.sample_lines,
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
        print(f"  Total lines: {train_stats['total_lines']:,}")
        print(f"  Lines per sample: {train_stats['lines_per_sample']:.2f}")
        print(f"  Estimated total tokens: {train_stats['estimated_total_tokens']:,}")
        print(f"  Estimated total samples: {train_stats['estimated_total_samples']:,}")
        print(f"  Estimated data size: ~{train_stats['estimated_total_tokens'] * 2 / (1024**3):.1f} GB (2 bytes/token)")
    else:
        print(f"\nTRAIN DATASET: Failed to estimate")

    if valid_stats:
        print(f"\nVALIDATION DATASET ({args.valid_dataset_name}):")
        print(f"  Total lines: {valid_stats['total_lines']:,}")
        print(f"  Lines per sample: {valid_stats['lines_per_sample']:.2f}")
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
