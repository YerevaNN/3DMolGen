from openbabel import openbabel
from pathlib import Path
from rdkit import Chem
import tempfile


def convert_xyz_to_mol(xyz_file: Path | str, mol_file: Path | str):
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("xyz", "mol")
    mol = openbabel.OBMol()
    conv.ReadFile(mol, str(xyz_file))
    conv.WriteFile(mol, str(mol_file))


def cif_to_mol(file_path: str):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("cif", "mol")

    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, file_path)
    with tempfile.NamedTemporaryFile() as tmp:
        obConversion.WriteFile(mol, tmp.name)
        mol = Chem.MolFromMolFile(tmp.name)

    return mol