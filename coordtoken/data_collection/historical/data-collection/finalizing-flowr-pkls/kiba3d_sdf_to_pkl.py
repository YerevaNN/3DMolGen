import os
import pickle
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm

input_dir = Path("/nfs/h100/raid/chem/FlowrData/KIBA-3D/data")
output_dir = Path("/nfs/h100/raid/chem/Flowr")
output_file = output_dir / "KIBA-3D_rdkit_mols.pkl"

output_dir.mkdir(parents=True, exist_ok=True)

data = []

if not input_dir.exists():
    raise ValueError(f"Input directory does not exist: {input_dir}")

sdf_files = sorted(input_dir.glob("*.sdf"))

if not sdf_files:
    raise ValueError(f"No SDF files found in {input_dir}")

print(f"Found {len(sdf_files)} SDF files")

for sdf_path in tqdm(sdf_files, desc="Processing KIBA-3D"):
    filename_without_ext = sdf_path.stem

    supplier = Chem.SDMolSupplier(str(sdf_path))

    for mol in supplier:
        if mol is not None:
            blob = mol.ToBinary()
            data.append((blob, filename_without_ext))
        else:
            tqdm.write(f"Warning: No molecule found in {sdf_path}")

if not data:
    raise ValueError("No valid molecules found in any SDF files")

with open(output_file, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved {len(data)} molecules to {output_file}")
print(f"Processed {len(sdf_files)} SDF files")

# Saved 333670 molecules to /nfs/h100/raid/chem/Flowr/KIBA-3D_rdkit_mols.pkl
