#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import argparse
from typing import List, Optional

def main() -> None:
    parser = argparse.ArgumentParser(description="Shuffle and combine all CSVs in the train folder into a single file.")
    parser.add_argument("--data-dir", type=Path, default=Path("/nfs/dgx/raid/chem/TokenizerData"), help="Base directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("/nfs/dgx/raid/chem/TokenizerData_combined"), help="Output directory.")
    args = parser.parse_args()

    train_dir: Path = args.data_dir / "train"
    output_dir: Path = args.output_dir
    output_file: Path = output_dir / "data_train.csv"

    # Check if shuf is available
    if subprocess.run(["which", "shuf"], capture_output=True).returncode != 0:
        print("Error: 'shuf' command not found. This script requires a Linux environment with 'shuf'.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Find all CSVs in train folder
    if not train_dir.exists():
        print(f"Train directory not found: {train_dir}")
        return

    all_files: List[Path] = sorted(list(train_dir.glob("*.csv")))

    if not all_files:
        print(f"No CSV files found in {train_dir}")
        return

    print(f"Found {len(all_files)} files in train folder. Combining (skipping headers)...")
    header: Optional[str] = None

    # Get header from the first file
    with open(all_files[0], 'r') as f:
        header = f.readline()

    if not header:
        print("Error: First file is empty or has no header.")
        return

    # 1. Prepare the command to combine files and skip headers
    # tail -q -n +2 skips the first line (header) of each file.
    file_list_str = " ".join([str(f) for f in all_files])

    # Calculate total size for pv progress bar
    total_bytes = sum(f.stat().st_size for f in all_files)

    print(f"Shuffling {len(all_files)} files (~{total_bytes / 1024**3:.1f} GB) from train folder.")
    print(f"Using 'pv' to track progress (Reading -> Shuffling -> Writing)...")

    # Write header first
    with open(output_file, 'w') as fout:
        fout.write(header)

    try:
        # Check if pv is available
        has_pv = subprocess.run(["which", "pv"], capture_output=True).returncode == 0

        if has_pv:
            # First pv tracks reading into shuf
            # Second pv tracks writing out of shuf
            # Note: shuf must read everything before it starts writing.
            cmd = (
                f"tail -q -n +2 {file_list_str} | "
                f"pv -N 'Reading' -s {total_bytes} | "
                f"shuf | "
                f"pv -N 'Writing' -s {total_bytes} >> {output_file}"
            )
        else:
            print("Warning: 'pv' not found, proceeding without progress bar.")
            cmd = f"tail -q -n +2 {file_list_str} | shuf >> {output_file}"

        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Shuffling failed: {e}")
        return

    print(f"\nDone! Shuffled train data saved to {output_file}")

    # Count total lines
    result = subprocess.run(["wc", "-l", str(output_file)], capture_output=True, text=True, check=True)
    total_lines: int = int(result.stdout.split()[0]) - 1 # Subtract header
    print(f"Total rows: {total_lines:,}")

if __name__ == "__main__":
    main()
