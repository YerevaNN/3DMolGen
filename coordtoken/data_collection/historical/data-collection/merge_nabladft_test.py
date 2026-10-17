#!/usr/bin/env python3
import csv
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/nfs/dgx/raid/chem/TokenizerData"))
    args = parser.parse_args()

    test_dir = args.data_dir / "test"
    files = [
        "nablaDFT_scaffolds.csv",
        "nablaDFT_structures.csv",
        "nablaDFT_conformations.csv"
    ]

    output_file = test_dir / "nablaDFT.csv"

    header_written = False
    total_rows = 0

    with open(output_file, 'w', newline='') as out_f:
        writer = None

        for filename in files:
            input_file = test_dir / filename
            if not input_file.exists():
                print(f"Warning: {input_file} not found, skipping")
                continue

            with open(input_file, 'r', newline='') as in_f:
                reader = csv.DictReader(in_f)
                if writer is None:
                    writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    header_written = True

                for row in reader:
                    writer.writerow(row)
                    total_rows += 1

            print(f"Merged {filename}: {total_rows:,} total rows so far")

    print(f"\nMerged {total_rows:,} rows into {output_file}")

if __name__ == "__main__":
    main()
