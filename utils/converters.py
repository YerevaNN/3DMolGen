import tempfile

from openbabel import openbabel
from pathlib import Path
from rdkit import Chem


def obabel_convert(file_path: Path | str, obConversion: openbabel.OBConversion, canonicalize_smi: bool = True, canonicalize_atoms: bool = False):
    obMol = openbabel.OBMol()
    obConversion.ReadFile(obMol, file_path)
    with tempfile.NamedTemporaryFile() as tmp:
        obConversion.WriteFile(obMol, tmp.name)
        with open("./debug.mol", 'w') as f:
            obConversion.WriteFile(obMol, f.name)

        mol = Chem.MolFromMolFile(tmp.name, removeHs=False)
        # If molecule has no atoms, error out
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Failed to convert {file_path} to mol.")
        
        # This needs to be here for the _smilesAtomOutputOrder to be present
        Chem.MolToSmiles(mol)

         # Generate a canonical atom ordering
        if canonicalize_smi:
            mol = Chem.Mol(mol)
            Chem.SanitizeMol(mol)
            #This is a hack to get the _smilesAtomOutputOrder property
            Chem.MolToSmiles(mol)
            
        if canonicalize_atoms:
            canonical_order = [int(i) for i in mol.GetProp('_smilesAtomOutputOrder')[1:-1].split(',') if i != '']
            mol = Chem.RenumberAtoms(mol, canonical_order)

        # If there are no conformers, error out
        if mol.GetNumConformers() == 0 or not obMol.Has3D():
            raise ValueError(f"Failed to convert {file_path} to mol.")

    return mol


def convert_xyz_to_mol(file_path: Path | str, canonicalize_smi: bool = True, canonicalize_atoms: bool = False):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "mol")

    return obabel_convert(file_path, obConversion, canonicalize_smi, canonicalize_atoms)


def convert_xyz_to_sdf(file_path: Path | str, output_file: Path | str):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "sdf")

    obMol = openbabel.OBMol()
    obConversion.ReadFile(obMol, file_path)
    with open(output_file, 'w') as f:
        obConversion.WriteFile(obMol, f.name)
    


def convert_cif_to_mol(file_path: str, canonicalize_smi: bool = True, canonicalize_atoms: bool = False):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("cif", "mol")

    return obabel_convert(file_path, obConversion, canonicalize_smi, canonicalize_atoms)