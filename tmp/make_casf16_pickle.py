import pickle
from pathlib import Path
from collections import defaultdict
from rdkit import Chem

ligand_dir = Path('/mnt/weka/vtarasov/CASF16/ligands_opt/')
out_path = Path('/home/vtarasov/code/3DMolGen/data/casf16_ligands_opt.pickle')

mol_files = defaultdict(list)
for f in sorted(ligand_dir.glob('*.mol2')):
    parts = f.stem.split('_conf')
    base = parts[0] if len(parts) > 1 else f.stem
    mol_files[base].append(f)

smiles_to_mols = defaultdict(list)
failed = []

for base, files in sorted(mol_files.items()):
    for f in sorted(files):
        mol = Chem.MolFromMol2File(str(f),  sanitize=True, removeHs=True)
        if mol is None:
            failed.append(str(f))
            continue
        if mol.GetNumConformers() == 0:
            failed.append(f"{f} (no conformers)")
            continue
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        smiles_to_mols[smiles].append(mol)

print(f"\n Loaded {sum(len(v) for v in smiles_to_mols.values())} mols across {len(smiles_to_mols)} unique SMILES")
if failed:
    print(f"Failed ({len(failed)}):")
    for f in failed:
        print(f"  {f}")

with open(out_path, 'wb') as fh:
    pickle.dump(dict(smiles_to_mols), fh)

print(f"Saved to {out_path}")
