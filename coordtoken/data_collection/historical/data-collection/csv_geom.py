import csv
import json
import random
from pathlib import Path
from tqdm import tqdm

geom_dir = Path("/nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_processed/geom_cartesian_v3/processed_strings")
output_dir = Path("/nfs/dgx/raid/chem/TokenizerData")


def process_split(split: str, input_split: str) -> None:
    split_input_dir = geom_dir / input_split
    jsonl_files = sorted(split_input_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files in {split_input_dir}")

    all_records: list[tuple[int, int, str]] = []
    for file_idx, jsonl_path in enumerate(jsonl_files):
        with jsonl_path.open("r") as f:
            for line_idx, line in enumerate(tqdm(f, desc=f"Reading {jsonl_path.name}")):
                record = json.loads(line)
                all_records.append((file_idx, line_idx, record["embedded_smiles"]))

    print(f"{split}: Total records = {len(all_records):,}")
    random.shuffle(all_records)

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    output_csv = split_dir / "geom.csv"
    with output_csv.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name", "enriched_text"])
        for file_idx, line_idx, enriched in tqdm(all_records, desc=f"Writing {split}"):
            writer.writerow([f"geom_{input_split}_{file_idx}_{line_idx}", enriched])

    print(f"Saved {len(all_records):,} records to {output_csv}")


def main() -> None:
    random.seed(42)
    process_split("train", "train")
    process_split("val", "valid")
    process_split("test", "test")
    print("\nDone!")


if __name__ == "__main__":
    main()
