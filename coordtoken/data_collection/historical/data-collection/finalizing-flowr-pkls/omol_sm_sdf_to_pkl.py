import os
import pickle
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm

input_dirs = [
    Path("/nfs/h100/raid/chem/FlowrData/biomolecules"),
    Path("/nfs/h100/raid/chem/FlowrData/OMol25/biomolecules")
]
output_dir = Path("/nfs/h100/raid/chem/FlowrData/OMol25_small_mol")
output_file = output_dir / "molecules.pkl"

output_dir.mkdir(parents=True, exist_ok=True)

data = []
total_sdf_files = 0

for input_dir in input_dirs:
    if not input_dir.exists():
        print(f"Warning: {input_dir} does not exist, skipping")
        continue

    sdf_files = sorted(input_dir.glob("*.sdf"))

    if not sdf_files:
        print(f"Warning: No SDF files found in {input_dir}")
        continue

    total_sdf_files += len(sdf_files)

    for sdf_path in tqdm(sdf_files, desc=f"Processing {input_dir.name}"):
        filename_without_ext = sdf_path.stem

        supplier = Chem.SDMolSupplier(str(sdf_path))

        for mol in supplier:
            if mol is not None:
                blob = mol.ToBinary()
                data.append((blob, filename_without_ext)) # store blob, not mol
            else:
                tqdm.write(f"Warning: No molecule found in {sdf_path}")

if not data:
    raise ValueError("No valid molecules found in any SDF files")

with open(output_file, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved {len(data)} molecules to {output_file}")
print(f"Processed {total_sdf_files} SDF files from {len(input_dirs)} directories")
# Saved 5826501 molecules to /nfs/h100/raid/chem/FlowrData/OMol25_small_mol/molecules.pkl
