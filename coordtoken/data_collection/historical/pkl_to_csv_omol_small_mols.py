import pickle
import csv
import gc
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent / "data-collection"))
from utils import encode_cartesian_v2

if len(sys.argv) != 3:
    print("Usage: python pkl_to_csv_omol_small_mols.py <pkl_file> <output_csv>")
    print("Example: python pkl_to_csv_omol_small_mols.py /nfs/h100/raid/chem/Flowr/OMol25_small_mols_rdkit_mols.pkl /nfs/dgx/raid/chem/TokenizerData/train/OMol25_small_mols.csv")
    sys.exit(1)

pkl_path = Path(sys.argv[1])
output_csv = Path(sys.argv[2])

if not pkl_path.exists():
    print(f"Error: PKL file not found: {pkl_path}")
    sys.exit(1)

output_csv.parent.mkdir(parents=True, exist_ok=True)

print(f"Loading {pkl_path}...")
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

data_len = len(data)
print(f"Loaded {data_len} entries")

dataset_name = "OMol25_small_mols"

if data_len == 0:
    print("Error: PKL contains zero entries")
    sys.exit(1)

first_item = data[0]
if not isinstance(first_item, tuple) or len(first_item) < 2:
    raise ValueError(
        f"Unexpected item structure in OMol25_small_mols PKL: "
        f"type={type(first_item)}, repr={first_item!r}. "
        f"This script expects entries to be tuples where the first element "
        f"is an RDKit blob and the second element is a name."
    )

print(f"Detected tuple entries with length {len(first_item)}; "
      "using item[0] as blob and item[1] as name")

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['name', 'enriched_text'])

    processed = 0
    skipped = 0
    flush_interval = 1000

    for idx in tqdm(range(data_len), desc="Converting OMol25_small_mols to CSV"):
        try:
            item = data[idx]
            blob = item[0]
            name = item[1]
            mol = Chem.Mol(blob)
            row_name = name if name else f"{dataset_name}_{idx}"

            if mol is None:
                skipped += 1
                continue

            if mol.GetNumConformers() == 0:
                skipped += 1
                continue

            enriched_text = encode_cartesian_v2(mol, precision=4)
            writer.writerow([row_name, enriched_text])
            processed += 1

            if (idx + 1) % flush_interval == 0:
                csvfile.flush()
                gc.collect()

        except Exception as e:
            print(f"\nSkipped index {idx}: {type(e).__name__}: {e}")
            skipped += 1
            continue

print(f"\nDone!")
print(f"Processed: {processed:,}, Skipped: {skipped:,}")
print(f"Saved to {output_csv}")
