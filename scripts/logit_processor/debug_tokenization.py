#!/usr/bin/env python
"""
Debug script to analyze tokenization differences between placeholder and actual coordinates.

This tests the hypothesis that the v2 precompute mask fails because actual coordinate
values tokenize into different numbers of tokens than the placeholder 0.0000,0.0000,0.0000.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter

from molgen3D.config.paths import get_tokenizer_path
from molgen3D.evaluation.constraint_logit_processor import (
    COORD_PLACEHOLDER,
    build_reference_skeleton,
    build_precomputed_template,
)
from transformers import AutoTokenizer


def load_tokenizer(name: str = "llama3_chem_v1"):
    """Load tokenizer."""
    path = get_tokenizer_path(name)
    return AutoTokenizer.from_pretrained(path)


def extract_coordinates(conformer_text: str) -> list[str]:
    """Extract coordinate strings from conformer text (content between < and >)."""
    pattern = r'<([^>]+)>'
    return re.findall(pattern, conformer_text)


def tokenize_and_count(text: str, tokenizer) -> tuple[list[int], int]:
    """Tokenize text and return (token_ids, count)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids, len(ids)


def analyze_coordinate_tokenization(tokenizer, coords: list[str], placeholder: str = COORD_PLACEHOLDER):
    """Analyze how real coordinates tokenize vs placeholder."""
    placeholder_ids, placeholder_count = tokenize_and_count(placeholder, tokenizer)

    print(f"\n=== Placeholder Tokenization ===")
    print(f"Placeholder: '{placeholder}'")
    print(f"Token count: {placeholder_count}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(placeholder_ids)}")

    diffs = []
    mismatches = []

    for coord in coords:
        coord_ids, coord_count = tokenize_and_count(coord, tokenizer)
        diff = coord_count - placeholder_count
        diffs.append(diff)

        if diff != 0:
            mismatches.append({
                "coord": coord,
                "count": coord_count,
                "diff": diff,
                "tokens": tokenizer.convert_ids_to_tokens(coord_ids)
            })

    print(f"\n=== Token Count Differences (coord - placeholder) ===")
    diff_counter = Counter(diffs)
    for diff, count in sorted(diff_counter.items()):
        pct = count / len(coords) * 100
        print(f"  Diff {diff:+d}: {count} occurrences ({pct:.1f}%)")

    print(f"\n=== Sample Mismatches (showing first 10) ===")
    for m in mismatches[:10]:
        print(f"  '{m['coord']}' -> {m['count']} tokens (diff: {m['diff']:+d})")
        print(f"    Tokens: {m['tokens']}")

    return diffs, mismatches


def analyze_failure(failure: dict, tokenizer):
    """Deep analysis of a single failure."""
    prompt_smiles = failure["prompt_smiles"]
    decoded_text = failure["decoded_text"]

    print(f"\n{'='*80}")
    print(f"Prompt SMILES: {prompt_smiles}")

    # Build reference skeleton and template
    try:
        skeleton = build_reference_skeleton(prompt_smiles)
        template = build_precomputed_template(prompt_smiles, tokenizer)
    except Exception as e:
        print(f"ERROR building template: {e}")
        return

    # Extract coordinates from decoded text
    conformer_start = decoded_text.find("[CONFORMER]")
    conformer_end = decoded_text.find("[/CONFORMER]")
    if conformer_start == -1 or conformer_end == -1:
        print("No conformer block found")
        return

    conformer_text = decoded_text[conformer_start + len("[CONFORMER]"):conformer_end]
    actual_coords = extract_coordinates(conformer_text)
    skeleton_coords = extract_coordinates(skeleton)

    print(f"\nReference skeleton (first 200 chars):")
    print(f"  {skeleton[:200]}...")

    print(f"\nActual conformer (first 200 chars):")
    print(f"  {conformer_text[:200]}...")

    print(f"\nCoordinate count: skeleton={len(skeleton_coords)}, actual={len(actual_coords)}")

    # Compare tokenizations
    ref_str = f"[CONFORMER]{skeleton}[/CONFORMER]"
    ref_ids = tokenizer.encode(ref_str, add_special_tokens=False)

    actual_str = decoded_text[conformer_start:conformer_end + len("[/CONFORMER]")]
    actual_ids = tokenizer.encode(actual_str, add_special_tokens=False)

    print(f"\nToken counts: ref={len(ref_ids)}, actual={len(actual_ids)}, diff={len(actual_ids) - len(ref_ids)}")

    # Find first divergence
    min_len = min(len(ref_ids), len(actual_ids))
    for i in range(min_len):
        if ref_ids[i] != actual_ids[i]:
            print(f"\nFirst divergence at position {i}:")
            context_start = max(0, i - 3)
            context_end = min(min_len, i + 5)

            ref_context = ref_ids[context_start:context_end]
            actual_context = actual_ids[context_start:context_end]

            print(f"  Ref tokens:    {tokenizer.convert_ids_to_tokens(ref_context)}")
            print(f"  Actual tokens: {tokenizer.convert_ids_to_tokens(actual_context)}")
            print(f"  Ref at {i}: {tokenizer.convert_ids_to_tokens([ref_ids[i]])}")
            print(f"  Actual at {i}: {tokenizer.convert_ids_to_tokens([actual_ids[i]])}")
            print(f"  is_free[{i}] = {template.is_free[i].item() if i < len(template.is_free) else 'OUT_OF_BOUNDS'}")
            break

    # Analyze coordinate tokenization differences
    if skeleton_coords and actual_coords:
        print(f"\n=== Coordinate Tokenization Analysis ===")
        for idx, (skel_coord, act_coord) in enumerate(zip(skeleton_coords[:5], actual_coords[:5])):
            skel_tok = tokenizer.encode(skel_coord, add_special_tokens=False)
            act_tok = tokenizer.encode(act_coord, add_special_tokens=False)
            diff = len(act_tok) - len(skel_tok)
            print(f"  Coord {idx}: skeleton '{skel_coord}' ({len(skel_tok)} toks) vs actual '{act_coord}' ({len(act_tok)} toks) -> diff={diff:+d}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("outputs/smoke/v2_precompute_simple.json"))
    parser.add_argument("--tokenizer", default="llama3_chem_v1")
    parser.add_argument("--num-failures", type=int, default=3)
    args = parser.parse_args()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer)

    # Load report
    print(f"Loading report from {args.report}...")
    report = json.loads(args.report.read_text())

    print(f"\n=== Report Summary ===")
    print(f"Version: {report['metadata']['version']}")
    print(f"Passed: {report['num_passed']}, Failed: {report['num_failed']}")
    print(f"Time: {report['time_taken']:.2f}s")

    # Analyze placeholder tokenization
    print("\n" + "="*80)
    print("PLACEHOLDER TOKENIZATION ANALYSIS")
    print("="*80)

    placeholder_ids, placeholder_count = tokenize_and_count(COORD_PLACEHOLDER, tokenizer)
    print(f"Placeholder: '{COORD_PLACEHOLDER}'")
    print(f"Token count: {placeholder_count}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(placeholder_ids)}")

    # Collect all coordinates from failures
    all_coords = []
    for f in report["failures"]:
        decoded = f["decoded_text"]
        coords = extract_coordinates(decoded)
        all_coords.extend(coords)

    print(f"\nTotal coordinates extracted from failures: {len(all_coords)}")

    # Analyze coordinate tokenization
    if all_coords:
        analyze_coordinate_tokenization(tokenizer, all_coords)

    # Deep analysis of failures
    print("\n" + "="*80)
    print("FAILURE DEEP ANALYSIS")
    print("="*80)

    for failure in report["failures"][:args.num_failures]:
        analyze_failure(failure, tokenizer)


if __name__ == "__main__":
    main()
