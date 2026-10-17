#!/usr/bin/env python3
import argparse
import gzip
import json
from pathlib import Path
from tqdm import tqdm


def iter_record_names(gz_path: Path):
    """Yield SDF record name (first line) for each molecule in a .sdf.gz file."""
    at_start = True
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if at_start:
                name = line.strip()
                if name:
                    yield name
                at_start = False
            if line.startswith("$$$$"):
                at_start = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf-dir", type=Path, required=True, help="Directory with *.sdf.gz")
    ap.add_argument("--out-dir", type=Path, required=True, help="Where to save ids.txt/meta.json")
    ap.add_argument("--prefix", type=str, default="ZINC", help="Keep IDs starting with this prefix ('' disables)")
    args = ap.parse_args()

    gz_files = sorted(args.sdf_dir.glob("*.sdf.gz"))
    if not gz_files:
        raise SystemExit(f"No *.sdf.gz found in {args.sdf_dir}")

    prefix = args.prefix if args.prefix != "" else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids_path = args.out_dir / "ids.txt"
    meta_path = args.out_dir / "meta.json"
    corrupted_path = Path("corrupted.txt")

    # reset outputs
    ids_path.unlink(missing_ok=True)
    corrupted_path.unlink(missing_ok=True)

    seen = set()
    unique_count = 0
    total_files = 0
    corrupted_files = 0

    with ids_path.open("w", encoding="utf-8") as ids_out:
        for gz in tqdm(gz_files, desc="Collecting unique IDs"):
            total_files += 1
            try:
                for name in iter_record_names(gz):
                    if prefix and not name.startswith(prefix):
                        continue
                    if name in seen:
                        continue
                    seen.add(name)
                    ids_out.write(name + "\n")
                    unique_count += 1
            except Exception as e:
                corrupted_files += 1
                with corrupted_path.open("a", encoding="utf-8") as log:
                    log.write(f"{gz.name}: {type(e).__name__}: {e}\n")

    meta = {
        "sdf_dir": str(args.sdf_dir),
        "prefix": args.prefix,
        "files_total": total_files,
        "files_corrupted": corrupted_files,
        "unique_ids": unique_count,
        "ids_file": str(ids_path),
        "corrupted_file": str(corrupted_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved {unique_count:,} unique IDs to: {ids_path}")
    print(f"Metadata written to: {meta_path}")
    if corrupted_files:
        print(f"Corrupted files logged to: {corrupted_path} ({corrupted_files} files)")


if __name__ == "__main__":
    main()
