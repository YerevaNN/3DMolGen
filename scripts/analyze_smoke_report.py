#!/usr/bin/env python
"""Analyze constrained smoke test JSON reports."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Add src to path to import molgen3D modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from molgen3D.data_processing.smiles_encoder_decoder import strip_smiles
    STRIP_SMILES_AVAILABLE = True
except ImportError:
    STRIP_SMILES_AVAILABLE = False
    def strip_smiles(s):
        """Fallback: just remove <...> blocks."""
        return re.sub(r'<[^>]*>', '', s)


def extract_first_conformer(decoded_text: str) -> str | None:
    """Extract the first [CONFORMER]...[/CONFORMER] block from decoded text."""
    match = re.search(r'\[CONFORMER\](.*?)\[/CONFORMER\]', decoded_text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def validate_conformer_mapping(
    prompt_smiles: str,
    decoded_text: str
) -> tuple[bool, str]:
    """
    Validate that embedded conformer maps back to canonical SMILES.

    Returns: (is_valid, diagnostic_message)
    """
    conformer_block = extract_first_conformer(decoded_text)
    if not conformer_block:
        return False, "No conformer block found"

    try:
        # Strip coordinates to get SMILES representation
        stripped = strip_smiles(conformer_block)

        if stripped == prompt_smiles:
            return True, "Perfect match"
        else:
            return False, f"Mismatch: got '{stripped[:50]}...', expected '{prompt_smiles[:50]}...'"
    except Exception as e:
        return False, f"Error: {e}"


def analyze_report(report_path: Path) -> None:
    """Analyze a single smoke test report."""
    with open(report_path) as f:
        data = json.load(f)

    print(f"{'='*70}")
    print(f"Smoke Report: {report_path.name}")
    print(f"{'='*70}")

    # Metadata section
    if "metadata" in data:
        meta = data["metadata"]
        print(f"\n{'Version:':<25} {meta.get('version', 'unknown')}")
        print(f"{'Timestamp:':<25} {meta.get('timestamp', 'unknown')}")

        lp_config = meta.get("logit_processor_config", {})
        if lp_config:
            print(f"\nLogit Processor Config:")
            for k, v in lp_config.items():
                print(f"  {k:<30} {v}")

    # Summary stats
    total = data.get("sample_size", data.get("total", 0))
    passed = data.get("num_passed", 0)
    failed = data.get("num_failed", len(data.get("failures", [])))

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"{'Pass Rate:':<25} {passed}/{total} ({100*passed/max(total,1):.1f}%)")
    print(f"{'Failed:':<25} {failed}")

    # Issue breakdown
    issue_counts = Counter()
    for fail in data.get("failures", []):
        for issue in fail.get("issues", []):
            issue_counts[issue] += 1

    if issue_counts:
        print(f"\nFailure Breakdown:")
        for issue, count in issue_counts.most_common():
            print(f"  {issue:<40} {count:>4}")

    # Coordinate quality analysis
    print(f"\n{'='*70}")
    print(f"COORDINATE QUALITY")
    print(f"{'='*70}")
    coord_patterns = analyze_coordinates(data)
    for pattern, count in coord_patterns.most_common(10):
        print(f"  {pattern:<50} {count:>4}")

    # Conformer mapping validation
    print(f"\n{'='*70}")
    print(f"CONFORMER → SMILES MAPPING")
    print(f"{'='*70}")
    all_records = data.get("passes", []) + data.get("failures", [])
    mapping_valid = 0
    mapping_issues = Counter()
    for rec in all_records:
        is_valid, msg = validate_conformer_mapping(
            rec.get("prompt_smiles", ""),
            rec.get("decoded_text", "")
        )
        if is_valid:
            mapping_valid += 1
        else:
            mapping_issues[msg.split(":")[0]] += 1

    print(f"  Valid mappings:   {mapping_valid}/{total} ({100*mapping_valid/max(total,1):.1f}%)")
    if mapping_issues:
        print(f"  Mapping issues:")
        for issue, count in mapping_issues.most_common(5):
            print(f"    {issue:<45} {count:>4}")

    # Generation stopping check
    print(f"\n{'='*70}")
    print(f"GENERATION STOPPING")
    print(f"{'='*70}")
    samples_with_multiple_conformers = 0
    for rec in all_records:
        decoded = rec.get("decoded_text", "")
        conformer_count = decoded.count("[/CONFORMER]")
        if conformer_count > 1:
            samples_with_multiple_conformers += 1

    if samples_with_multiple_conformers > 0:
        print(f"  ⚠️  Samples with >1 [/CONFORMER]: {samples_with_multiple_conformers}/{total}")
        print(f"      (Generation didn't stop properly)")
    else:
        print(f"  ✓ All samples stopped correctly after [/CONFORMER]")

    # Special token pollution
    special_token_counts = count_special_tokens(data)
    if special_token_counts["samples_with_pollution"] > 0:
        print(f"\n{'='*70}")
        print(f"⚠️  SPECIAL TOKEN POLLUTION")
        print(f"{'='*70}")
        print(f"  Affected samples: {special_token_counts['samples_with_pollution']}/{total}")
        print(f"  Avg per sample:   {special_token_counts['avg_count']:.1f}")
    else:
        print(f"\n✓ No special token pollution detected")


def analyze_coordinates(data: dict) -> Counter:
    """Extract and classify coordinate patterns from decoded text."""
    patterns = Counter()
    all_records = data.get("passes", []) + data.get("failures", [])

    # Valid coordinate pattern: <float,float,float>
    valid_pattern = re.compile(r'^-?\d+(\.\d+)?,-?\d+(\.\d+)?,-?\d+(\.\d+)?$')

    for record in all_records:
        decoded = record.get("decoded_text", "")

        # Find all <...> blocks
        coords = re.findall(r'<([^>]+)>', decoded)

        for coord in coords:
            # Classify pattern
            if valid_pattern.match(coord):
                patterns["✓ valid: <float,float,float>"] += 1
            elif len(coord) > 50:
                patterns["✗ garbage: very long (>50 chars)"] += 1
            elif len(coord) > 30:
                patterns["✗ garbage: long (>30 chars)"] += 1
            elif ',' in coord:
                comma_count = coord.count(',')
                if comma_count == 2:
                    patterns["✗ malformed: 2 commas but invalid floats"] += 1
                elif comma_count > 2:
                    patterns[f"✗ malformed: {comma_count} commas (expected 2)"] += 1
                else:
                    patterns[f"✗ malformed: {comma_count} comma (expected 2)"] += 1
            else:
                patterns["✗ garbage: no commas"] += 1

    return patterns


def count_special_tokens(data: dict) -> dict:
    """Count special token pollution in decoded outputs."""
    total_special = 0
    samples_with = 0
    all_records = data.get("passes", []) + data.get("failures", [])

    for record in all_records:
        decoded = record.get("decoded_text", "")
        count = decoded.count("<|end_of_text|>") + decoded.count("<|begin_of_text|>")
        if count > 0:
            samples_with += 1
            total_special += count

    return {
        "samples_with_pollution": samples_with,
        "avg_count": total_special / max(samples_with, 1),
        "total_count": total_special,
    }


def compare_reports(report_paths: list[Path]) -> None:
    """Compare multiple smoke reports side-by-side."""
    print(f"{'='*90}")
    print(f"COMPARISON ACROSS VERSIONS")
    print(f"{'='*90}")
    print(f"{'Version':<12} {'Pass Rate':<15} {'Coord Valid%':<15} {'Main Issue':<40}")
    print("-" * 90)

    for path in sorted(report_paths):
        with open(path) as f:
            data = json.load(f)

        # Extract version
        version = data.get("metadata", {}).get("version", path.stem.replace("constrained_clean_64_", ""))

        # Pass rate
        total = data.get("sample_size", data.get("total", 0))
        passed = data.get("num_passed", 0)
        pass_rate = f"{passed}/{total}"

        # Coordinate validity
        coord_patterns = analyze_coordinates(data)
        valid_count = coord_patterns.get("✓ valid: <float,float,float>", 0)
        total_coords = sum(coord_patterns.values())
        coord_valid_pct = f"{100*valid_count/max(total_coords,1):.1f}%" if total_coords else "N/A"

        # Main issue
        issue_counts = Counter()
        for fail in data.get("failures", []):
            for issue in fail.get("issues", []):
                issue_counts[issue] += 1

        main_issue = issue_counts.most_common(1)[0] if issue_counts else ("none", 0)
        main_issue_str = f"{main_issue[0]}: {main_issue[1]}"

        print(f"{version:<12} {pass_rate:<15} {coord_valid_pct:<15} {main_issue_str:<40}")


def sample_failures(report_path: Path, n: int = 3) -> None:
    """Show sample failures with detailed decoded text."""
    with open(report_path) as f:
        data = json.load(f)

    failures = data.get("failures", [])
    if not failures:
        print("No failures to show.")
        return

    print(f"\n{'='*70}")
    print(f"SAMPLE FAILURES (showing first {min(n, len(failures))})")
    print(f"{'='*70}")

    for i, fail in enumerate(failures[:n], 1):
        print(f"\n--- Failure {i} ---")
        print(f"Prompt SMILES: {fail.get('prompt_smiles', 'N/A')}")
        print(f"Issues: {', '.join(fail.get('issues', []))}")
        decoded = fail.get("decoded_text", "")

        # Show first 500 chars of decoded text
        if len(decoded) > 500:
            print(f"Decoded (first 500 chars):\n{decoded[:500]}...")
        else:
            print(f"Decoded:\n{decoded}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze constrained conformer smoke test reports"
    )
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="JSON report files to analyze"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple reports side-by-side"
    )
    parser.add_argument(
        "--sample-failures",
        type=int,
        metavar="N",
        help="Show N sample failures with decoded text"
    )

    args = parser.parse_args()

    if args.compare:
        compare_reports(args.reports)
    elif args.sample_failures:
        for report in args.reports:
            sample_failures(report, n=args.sample_failures)
    else:
        for report in args.reports:
            analyze_report(report)
            print("\n")
