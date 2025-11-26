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
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Sequence, Tuple, Union


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
from molgen3D.training.pretraining.dataprocessing.dataloader import build_dataloader


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
    files_with_counts = list_jsonl_files_with_counts(directory)
    total_lines = sum(count for _, count in files_with_counts)
    return total_lines, len(files_with_counts)


def list_jsonl_files_with_counts(directory: str) -> List[Tuple[str, int]]:
    """Return all .jsonl files under `directory` and their line counts."""
    if not directory or not os.path.exists(directory):
        return []
    jsonl_files = sorted(glob.glob(os.path.join(directory, "*.jsonl")))
    out: List[Tuple[str, int]] = []
    for file in jsonl_files:
        with open(file, 'r') as f:
            count = sum(1 for _ in f)
        out.append((file, count))
    return out


def get_file_sizes(files_with_counts: List[Tuple[str, int]]) -> Dict[str, int]:
    """Get file sizes in bytes for all files."""
    file_sizes = {}
    for file_path, _ in files_with_counts:
        if os.path.exists(file_path):
            file_sizes[file_path] = os.path.getsize(file_path)
    return file_sizes


def get_bytes_for_lines(file_path: str, num_lines: int) -> Optional[int]:
    """
    Get the byte size of the first num_lines in a file.
    
    Args:
        file_path: Path to the file
        num_lines: Number of lines to read
    
    Returns:
        Byte size of the first num_lines, or None if file doesn't exist
    """
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'rb') as f:
            bytes_count = 0
            lines_read = 0
            for line in f:
                bytes_count += len(line)
                lines_read += 1
                if lines_read >= num_lines:
                    break
            return bytes_count
    except Exception:
        return None


def verify_estimate_with_file_sizes(
    sampled_file: str,
    sampled_count: int,
    files_with_counts: List[Tuple[str, int]],
    total_items: int,
    threshold_percent: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """
    Verify the line count estimate using file sizes.
    
    Gets the byte size of sampled lines, calculates extrapolation factor,
    and checks if total bytes matches sampled_bytes * extrapolation_factor.
    
    Args:
        sampled_file: Path to the sampled file
        sampled_count: Number of lines that were sampled
        files_with_counts: List of (file_path, line_count) tuples
        total_items: Total line count from counting
        threshold_percent: Maximum allowed difference percentage (default: 10%)
    
    Returns:
        Dictionary with verification results, or None if verification failed
    """
    if not os.path.exists(sampled_file) or sampled_count == 0 or total_items == 0:
        return None
    
    # Get byte size of the sampled lines
    sampled_bytes = get_bytes_for_lines(sampled_file, sampled_count)
    if sampled_bytes is None:
        return None
    
    # Calculate extrapolation factor
    extrapolation_factor = total_items / sampled_count
    
    # Calculate expected total bytes
    expected_total_bytes = sampled_bytes * extrapolation_factor
    
    # Get actual total size of all files
    file_sizes = get_file_sizes(files_with_counts)
    actual_total_bytes = sum(file_sizes.get(file_path, 0) for file_path, _ in files_with_counts)
    
    if actual_total_bytes == 0:
        return None
    
    # Calculate difference
    difference = abs(actual_total_bytes - expected_total_bytes)
    difference_percent = (difference / actual_total_bytes * 100) if actual_total_bytes > 0 else 0.0
    
    # Check if within threshold
    is_within_threshold = difference_percent <= threshold_percent
    
    return {
        'sampled_bytes': sampled_bytes,
        'sampled_line_count': sampled_count,
        'extrapolation_factor': extrapolation_factor,
        'expected_total_bytes': expected_total_bytes,
        'actual_total_bytes': actual_total_bytes,
        'difference': difference,
        'difference_percent': difference_percent,
        'threshold_percent': threshold_percent,
        'is_within_threshold': is_within_threshold,
    }


def collect_samples_from_dataloader(
    train_path: Union[str, Sequence[str]],
    tokenizer_path: str,
    tokenizer: AutoTokenizer,
    seq_len: int,
    max_items: int,
    batch_size: int = 1,
    shuffle_lines: bool = False,
    seed: int = 0,
) -> Optional[Dict[str, Any]]:
    """Run the dataloader until at least `max_items` units are observed."""
    loader = build_dataloader(
        train_path=train_path,
        tokenizer_path=tokenizer_path,
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle_lines=shuffle_lines,
        infinite=False,
        seed=seed,
        min_emb_len=16,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=None,
        world_size=1,
        rank=0,
    )

    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None
    pad_id = getattr(dataset, "pad_id", None)
    sep_id = getattr(dataset, "sep_id", None)
    if pad_id is None or sep_id is None:
        raise RuntimeError("Dataset missing pad/separator token IDs.")

    pad_total = 0
    items_total = 0
    samples_processed = 0
    stop = False

    for batch_idx, batch in enumerate(loader):
        inputs, _ = batch
        input_tensor = inputs["input"] if isinstance(inputs, dict) else inputs
        bsz = input_tensor.size(0)
        for sample_idx in range(bsz):
            sample = input_tensor[sample_idx]
            
            # Count padding tokens at the end (they use the same ID as separators)
            padding_count = 0
            for token in reversed(sample.tolist()):
                if token == pad_id:
                    padding_count += 1
                else:
                    break
            
            # Count all sep_id/pad_id tokens, then subtract padding to get separator count
            total_sep_pad = int((sample == sep_id).sum().item())
            sep_count = total_sep_pad - padding_count
            pad_count = padding_count
            
            items_total += sep_count
            pad_total += pad_count
            samples_processed += 1
            if items_total >= max_items:
                stop = True
                break
        if stop:
            break

    if samples_processed == 0:
        return None

    avg_items_per_sample = items_total / samples_processed if samples_processed else 0.0
    avg_pad_per_sample = pad_total / samples_processed if samples_processed else 0.0

    return {
        "train_path": train_path,
        "samples_processed": samples_processed,
        "items_total": items_total,
        "avg_items_per_sample": avg_items_per_sample,
        "avg_pad_per_sample": avg_pad_per_sample,
        "tokenizer_path": tokenizer_path,
        "pad_total": pad_total,
        "seq_len": seq_len,
    }



def estimate_dataset(
    dataset_name,
    dataset_path,
    tokenizer_path,
    tokenizer,
    num_items,
    seq_len,
    is_validation=False,
    sampled_file: Optional[str] = None,
    sampled_count: Optional[int] = None,
    sample_seed: int = 0,
    files_with_counts: Optional[List[Tuple[str, int]]] = None,
    print_overview: bool = True,
):
    """Estimate tokens and samples by loading items and creating samples using dataloader logic."""
    dataset_type = "VALIDATION" if is_validation else "TRAIN"
    if print_overview:
        print(f"\n{'='*70}")
        print(f"{dataset_type} DATASET: {dataset_name}")
        print(f"{'='*70}")

    if not dataset_path:
        print("ERROR: No dataset path provided")
        return None

    files_with_counts = files_with_counts or list_jsonl_files_with_counts(dataset_path)
    total_items = sum(count for _, count in files_with_counts)
    num_files = len(files_with_counts)
    if print_overview:
        print(f"Dataset path: {dataset_path}")
        print(f"Total .jsonl files: {num_files}")
        print(f"Total items in dataset: {total_items:,}")
        if files_with_counts:
            print("\nDataset files:")
            for file_path, count in files_with_counts:
                print(f"  {file_path}: {count} lines")

    if total_items == 0:
        print("ERROR: No items found in dataset")
        return None

    if sampled_file and sampled_count is not None:
        selected_file, selected_count = sampled_file, sampled_count
        print(f"\nUsing pre-selected sampling file: {selected_file}")
    else:
        selected_file, selected_count = random.choice(files_with_counts)
        print(f"\nSampling from {selected_file} ({selected_count} lines)")

    print(f"\nSampling from {selected_file} ({selected_count} lines)")
    stats = collect_samples_from_dataloader(
        selected_file,
        tokenizer_path,
        tokenizer,
        seq_len,
        max_items=min(num_items, selected_count),
        batch_size=1,
        shuffle_lines=False,
        seed=sample_seed,
    )

    if stats is None:
        print("ERROR: Failed to sample sequences via dataloader.")
        return None

    samples_processed = stats['samples_processed']
    avg_items_per_sample = stats['avg_items_per_sample']
    avg_pad_per_sample = stats['avg_pad_per_sample']

    if samples_processed == 0:
        print("ERROR: No samples were analyzed.")
        return None

    estimated_total_samples = int(total_items / avg_items_per_sample) if avg_items_per_sample else 0
    estimated_total_tokens = estimated_total_samples * seq_len
    estimated_pad_tokens = int(avg_pad_per_sample * estimated_total_samples)
    estimated_effective_tokens = max(0, estimated_total_tokens - estimated_pad_tokens)
    
    # Calculate pad percentage metrics
    avg_pad_percentage = (avg_pad_per_sample / seq_len * 100) if seq_len > 0 else 0.0
    estimated_pad_percentage = (estimated_pad_tokens / estimated_total_tokens * 100) if estimated_total_tokens > 0 else 0.0

    # Verify estimate using file sizes
    verification = verify_estimate_with_file_sizes(
        selected_file,
        selected_count,
        files_with_counts,
        total_items,
        threshold_percent=10.0,
    )

    # Organized stats printing
    print(f"\n{'─'*70}")
    print(f"Sampling Statistics:")
    print(f"  Samples analyzed: {samples_processed:,}")
    print(f"  Sampling file: {os.path.basename(selected_file)}")
    print(f"  Avg items/sample: {avg_items_per_sample:.2f}")
    print(f"  Avg pad tokens/sample: {avg_pad_per_sample:.2f} ({avg_pad_percentage:.2f}% of seq_len)")
    print(f"\nExtrapolated Dataset Statistics:")
    print(f"  Total items: {total_items:,}")
    print(f"  Estimated samples (seq_len={seq_len}): {estimated_total_samples:,}")
    print(f"  Estimated total tokens: {estimated_total_tokens:,}")
    print(f"  Estimated pad tokens: {estimated_pad_tokens:,} ({estimated_pad_percentage:.2f}% of total)")
    print(f"  Estimated effective tokens: {estimated_effective_tokens:,}")
    print(f"  Tokenizer path: {stats['tokenizer_path']}")
    
    # Verification section
    if verification:
        print(f"\n{'─'*70}")
        print(f"Verification (File Size Based):")
        print(f"  Sampled lines: {verification['sampled_line_count']:,}")
        print(f"  Sampled bytes: {verification['sampled_bytes']:,} bytes ({verification['sampled_bytes'] / (1024**2):.2f} MB)")
        print(f"  Extrapolation factor: {verification['extrapolation_factor']:.2f}x")
        print(f"  Expected total bytes: {verification['expected_total_bytes']:,.0f} bytes ({verification['expected_total_bytes'] / (1024**3):.2f} GB)")
        print(f"  Actual total bytes: {verification['actual_total_bytes']:,} bytes ({verification['actual_total_bytes'] / (1024**3):.2f} GB)")
        print(f"  Difference: {verification['difference']:,.0f} bytes ({verification['difference_percent']:.2f}%)")
        if verification['is_within_threshold']:
            print(f"  ✓ Verification PASSED (within {verification['threshold_percent']:.1f}% threshold)")
        else:
            print(f"  ⚠️  Verification WARNING (exceeds {verification['threshold_percent']:.1f}% threshold)")
    else:
        print(f"\n{'─'*70}")
        print(f"Verification (File Size Based):")
        print(f"  ⚠️  Could not perform verification (file size data unavailable)")

    return {
        'total_items': total_items,
        'samples_processed': samples_processed,
        'avg_items_per_sample': avg_items_per_sample,
        'avg_pad_per_sample': avg_pad_per_sample,
        'avg_pad_percentage': avg_pad_percentage,
        'estimated_total_samples': estimated_total_samples,
        'estimated_total_tokens': estimated_total_tokens,
        'estimated_pad_tokens': estimated_pad_tokens,
        'estimated_pad_percentage': estimated_pad_percentage,
        'estimated_effective_tokens': estimated_effective_tokens,
        'tokenizer_path': stats['tokenizer_path'],
        'sampled_file': selected_file,
        'verification': verification,
    }


def analyze_dataset(
    dataset_name: str,
    dataset_path: str,
    tokenizer_map: Dict[str, Tuple[str, AutoTokenizer]],
    num_items: int,
    seq_len: int,
    is_validation: bool,
):
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    files_with_counts = list_jsonl_files_with_counts(dataset_path)
    if not files_with_counts:
        print(f"ERROR: No files found for dataset {dataset_name} at {dataset_path}")
        return results
    selected_file, selected_count = random.choice(files_with_counts)
    sample_seed = random.randint(0, 2**31 - 1)

    first = True
    for alias, (tokenizer_path, tokenizer) in tokenizer_map.items():
        print(f"\n>>> Tokenizer: {alias}")
        stats = estimate_dataset(
            dataset_name,
            dataset_path,
            tokenizer_path,
            tokenizer,
            num_items,
            seq_len,
            is_validation=is_validation,
            sampled_file=selected_file,
            sampled_count=selected_count,
            sample_seed=sample_seed,
            files_with_counts=files_with_counts,
            print_overview=first,
        )
        results[alias] = stats
        first = False
    return results


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
        "--tokenizer-aliases",
        nargs="+",
        default=["qwen3_0.6b_origin", "qwen3_0.6b_custom"],
        help="Tokenizer aliases to evaluate (default: qwen3_0.6b_origin qwen3_0.6b_custom)",
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
    print(f"Sequence length: {args.seq_len} (samples = {args.seq_len}-token sequences)")
    print(f"Tokenizer aliases: {', '.join(args.tokenizer_aliases)}")
    print("="*70)

    tokenizer_map: Dict[str, Tuple[str, AutoTokenizer]] = {}
    print("\nLoading tokenizers:")
    loaded_paths = {}  # Track paths to detect duplicates
    for alias in args.tokenizer_aliases:
        tok_path = str(get_tokenizer_path(alias))
        abs_tok_path = os.path.abspath(tok_path)
        
        # Check for duplicate paths
        if abs_tok_path in loaded_paths.values():
            print(f"  WARNING: {alias} has the same path as {[k for k, v in loaded_paths.items() if v == abs_tok_path][0]}")
            print(f"           Path: {abs_tok_path}")
        
        # Load tokenizer fresh each time
        tok = AutoTokenizer.from_pretrained(tok_path, use_fast=True, fix_mistral_regex=True, local_files_only=False)
        vocab_size = len(tok)  # Use len() to get actual vocab size from the tokenizer
        tokenizer_map[alias] = (tok_path, tok)
        loaded_paths[alias] = abs_tok_path
        
        # Print with both relative and absolute path for clarity
        print(f"  - {alias}: vocab_size={vocab_size} (path: {abs_tok_path})")
        
        # Verify the tokenizer actually loaded from the expected path
        if hasattr(tok, 'name_or_path'):
            actual_path = tok.name_or_path
            if actual_path != tok_path and actual_path != abs_tok_path:
                print(f"    NOTE: Tokenizer reports path as: {actual_path}")

    start_time = time.time()

    # Analyze train dataset
    train_stats = analyze_dataset(
        args.train_dataset_name,
        train_path,
        tokenizer_map,
        args.num_items,
        args.seq_len,
        is_validation=False,
    )

    # Analyze validation dataset
    valid_stats: Union[None, Dict[str, Optional[Dict[str, Any]]]] = None
    if not args.skip_validation:
        valid_stats = analyze_dataset(
            args.valid_dataset_name,
            valid_path,
            tokenizer_map,
            args.num_items,
            args.seq_len,
            is_validation=True,
        )
    else:
        valid_stats = {}

    # Print final comprehensive summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total analysis time: {total_time:.2f}s")

    if train_stats:
        print(f"\n{'='*70}")
        print(f"TRAIN DATASET ({args.train_dataset_name})")
        print(f"{'='*70}")
        for alias, stats in train_stats.items():
            if not stats:
                continue
            print(f"\nTokenizer: {alias}")
            print(f"  Path: {stats['tokenizer_path']}")
            print(f"  Sampled file: {os.path.basename(stats['sampled_file'])}")
            print(f"\n  Dataset Statistics:")
            print(f"    Total items: {stats['total_items']:,}")
            print(f"    Avg items/sample: {stats['avg_items_per_sample']:.2f}")
            avg_pad_pct = stats.get('avg_pad_percentage', (stats['avg_pad_per_sample'] / args.seq_len * 100) if args.seq_len > 0 else 0.0)
            print(f"    Avg pad tokens/sample: {stats['avg_pad_per_sample']:.2f} ({avg_pad_pct:.2f}% of seq_len)")
            print(f"\n  Estimated Totals:")
            print(f"    Samples: {stats['estimated_total_samples']:,}")
            print(f"    Total tokens: {stats['estimated_total_tokens']:,}")
            est_pad_pct = stats.get('estimated_pad_percentage', (stats['estimated_pad_tokens'] / stats['estimated_total_tokens'] * 100) if stats['estimated_total_tokens'] > 0 else 0.0)
            print(f"    Pad tokens: {stats['estimated_pad_tokens']:,} ({est_pad_pct:.2f}% of total)")
            print(f"    Effective tokens: {stats['estimated_effective_tokens']:,}")
            print(f"    Data size: ~{stats['estimated_total_tokens'] * 2 / (1024**3):.1f} GB (2 bytes/token)")
    else:
        print(f"\nTRAIN DATASET: Failed to estimate")

    if valid_stats:
        print(f"\n{'='*70}")
        print(f"VALIDATION DATASET ({args.valid_dataset_name})")
        print(f"{'='*70}")
        for alias, stats in valid_stats.items():
            if not stats:
                continue
            print(f"\nTokenizer: {alias}")
            print(f"  Path: {stats['tokenizer_path']}")
            print(f"  Sampled file: {os.path.basename(stats['sampled_file'])}")
            print(f"\n  Dataset Statistics:")
            print(f"    Total items: {stats['total_items']:,}")
            print(f"    Avg items/sample: {stats['avg_items_per_sample']:.2f}")
            avg_pad_pct = stats.get('avg_pad_percentage', (stats['avg_pad_per_sample'] / args.seq_len * 100) if args.seq_len > 0 else 0.0)
            print(f"    Avg pad tokens/sample: {stats['avg_pad_per_sample']:.2f} ({avg_pad_pct:.2f}% of seq_len)")
            print(f"\n  Estimated Totals:")
            print(f"    Samples: {stats['estimated_total_samples']:,}")
            print(f"    Total tokens: {stats['estimated_total_tokens']:,}")
            est_pad_pct = stats.get('estimated_pad_percentage', (stats['estimated_pad_tokens'] / stats['estimated_total_tokens'] * 100) if stats['estimated_total_tokens'] > 0 else 0.0)
            print(f"    Pad tokens: {stats['estimated_pad_tokens']:,} ({est_pad_pct:.2f}% of total)")
            print(f"    Effective tokens: {stats['estimated_effective_tokens']:,}")
            print(f"    Data size: ~{stats['estimated_total_tokens'] * 2 / (1024**3):.1f} GB (2 bytes/token)")
    elif not args.skip_validation:
        print(f"\nVALIDATION DATASET: Failed to estimate")

    # Training recommendations
    primary_alias = args.tokenizer_aliases[0] if args.tokenizer_aliases else None
    if (
        primary_alias
        and isinstance(train_stats, dict)
        and isinstance(valid_stats, dict)
        and train_stats.get(primary_alias)
        and valid_stats.get(primary_alias)
    ):
        train_samples = train_stats[primary_alias]['estimated_total_samples']
        valid_samples = valid_stats[primary_alias]['estimated_total_samples']
        print(f"\nTRAINING RECOMMENDATIONS (based on {primary_alias}):")
        print(f"  Train/Valid split: {train_samples/valid_samples:.1f}:1")
        print(f"  Gradient accumulation: Consider ~{max(1, int(1000000 // train_samples))} if batch_size=1")
        print(f"  Epochs for 100M tokens: ~{int(100000000 // train_stats[primary_alias]['estimated_total_tokens'])}")

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
