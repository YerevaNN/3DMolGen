import openbabel
from pathlib import Path


def convert_xyz_to_mol(xyz_file: Path | str, mol_file: Path | str):
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("xyz", "mol")
    mol = openbabel.OBMol()
    conv.ReadFile(mol, str(xyz_file))
    conv.WriteFile(mol, str(mol_file))