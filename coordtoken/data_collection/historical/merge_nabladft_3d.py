#!/usr/bin/env python3
from pathlib import Path
import argparse
import shutil


def append_without_header(source_path: Path, dest_file) -> None:
    with source_path.open("r", encoding="utf-8", newline="") as src:
        header = src.readline()
        if not header:
            return
        shutil.copyfileobj(src, dest_file, length=1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/nfs/h100/raid/chem/3D_big_data/backup"))
    parser.add_argument("--dest-root", type=Path, default=Path("/nfs/h100/raid/chem/3D_big_data/grp_c"))
    parser.add_argument("--dataset", default="nablaDFT.csv")
    args = parser.parse_args()

    train_src = args.source_root / "train" / args.dataset
    val_src = args.source_root / "val" / args.dataset
    dest_dir = args.dest_root / "train"
    dest_path = dest_dir / args.dataset
    tmp_path = dest_dir / f"{args.dataset}.tmp"

    if not train_src.exists():
        raise FileNotFoundError(f"Missing train source: {train_src}")
    if not val_src.exists():
        raise FileNotFoundError(f"Missing val source: {val_src}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    with train_src.open("r", encoding="utf-8", newline="") as src, tmp_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)

    with tmp_path.open("a", encoding="utf-8", newline="") as dst:
        append_without_header(val_src, dst)

    tmp_path.replace(dest_path)
    print(f"Wrote merged file to {dest_path}")


if __name__ == "__main__":
    main()
