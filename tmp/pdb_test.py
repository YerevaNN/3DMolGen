from pathlib import Path
from rdkit import Chem
from collections import defaultdict

ligand_dir = Path('/mnt/weka/vtarasov/CASF16/ligands/')

mol_files = defaultdict(list)
for f in sorted(ligand_dir.glob('*.mol2')):
    # e.g. 1a30_1a30_conf0.mol2 -> base = 1a30_1a30
    parts = f.stem.split('_conf')
    base = parts[0] if len(parts) > 1 else f.stem
    mol_files[base].append(f)

for base, files in sorted(mol_files.items()):
    # Load the first file to get the SMILES (all confs have same connectivity)
    ref_mol = Chem.MolFromMol2File(str(files[0]), sanitize=True, removeHs=True)
    if ref_mol is None:
        print(f"{base}: FAILED TO LOAD")
        continue
    smiles = Chem.MolToSmiles(ref_mol, isomericSmiles=True)
    # Count total conformers (each file contributes its conformers)
    n_confs = sum(
        Chem.MolFromMol2File(str(f), sanitize=True, removeHs=True).GetNumConformers()
        for f in files
        if Chem.MolFromMol2File(str(f), sanitize=True, removeHs=True) is not None
    )
    print(f"{base}: {n_confs} conformer(s)  SMILES: {smiles}")
