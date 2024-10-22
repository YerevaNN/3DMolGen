import pytest
from rdkit import Chem 

from utils.converters import convert_cif_to_mol, convert_xyz_to_mol
from tests.fixtures.molecules import CANONICAL_ASPIRIN_SMILES


def test_basic_convert_cif_to_mol(cif_aspirin_molecule):
    CIF_PATH, gtMol = cif_aspirin_molecule

    gtMolWithHs = Chem.AddHs(gtMol)
    mol = convert_cif_to_mol(CIF_PATH, canonicalize=True)
    molWithoutHs = Chem.RemoveHs(mol)

    assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(gtMolWithHs)
    assert Chem.MolToSmiles(molWithoutHs) == CANONICAL_ASPIRIN_SMILES


def test_basic_convert_xyz_to_mol(xyz_aspirin_molecule):
    XYZ_PATH, gtMol = xyz_aspirin_molecule

    gtMolWithHs = Chem.AddHs(gtMol)
    mol = convert_xyz_to_mol(XYZ_PATH, canonicalize=True)
    molWithoutHs = Chem.RemoveHs(mol)

    assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(gtMolWithHs)
    assert Chem.MolToSmiles(molWithoutHs) == CANONICAL_ASPIRIN_SMILES


def test_basic_conversion_exception():
    with pytest.raises(ValueError):
        convert_cif_to_mol("non_existent_file.cif")