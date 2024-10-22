import tempfile

from openbabel import openbabel
from pathlib import Path
from rdkit import Chem


def obabel_convert(file_path: Path | str, obConversion: openbabel.OBConversion, canonicalize: bool = False):
    obMol = openbabel.OBMol()
    obConversion.ReadFile(obMol, file_path)
    with tempfile.NamedTemporaryFile() as tmp:
        obConversion.WriteFile(obMol, tmp.name)
        mol = Chem.MolFromMolFile(tmp.name, removeHs=False)
        if canonicalize:
            mol = Chem.Mol(mol)
            Chem.SanitizeMol(mol)

        # If molecule has no atoms, error out
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Failed to convert {file_path} to mol.")

        # If there are no conformers, error out
        if mol.GetNumConformers() == 0 or not obMol.Has3D():
            raise ValueError(f"Failed to convert {file_path} to mol.")

    return mol


def convert_xyz_to_mol(file_path: Path | str, canonicalize: bool = False):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "mol")

    return obabel_convert(file_path, obConversion, canonicalize)


def convert_cif_to_mol(file_path: str, canonicalize: bool = False):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("cif", "mol")

    return obabel_convert(file_path, obConversion, canonicalize)