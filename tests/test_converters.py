import os
import pytest
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdMolAlign 

from utils.converters import convert_cif_to_mol, convert_xyz_to_mol
from tests.fixtures.molecules import CANONICAL_ASPIRIN_SMILES
from tests.utils.conversion_validators import validate_xyz_conversion


def test_basic_convert_cif_to_mol(cif_aspirin_molecule):
    CIF_PATH, gtMol = cif_aspirin_molecule

    gtMolWithHs = Chem.AddHs(gtMol)
    mol = convert_cif_to_mol(CIF_PATH)
    molWithoutHs = Chem.RemoveHs(mol)

    assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(gtMolWithHs)
    assert Chem.MolToSmiles(molWithoutHs) == CANONICAL_ASPIRIN_SMILES
    assert mol.GetNumAtoms() == 21


def test_basic_convert_xyz_to_mol(xyz_aspirin_molecule):
    XYZ_PATH, gtMol = xyz_aspirin_molecule

    gtMolWithHs = Chem.AddHs(gtMol)
    mol = convert_xyz_to_mol(XYZ_PATH)
    molWithoutHs = Chem.RemoveHs(mol)

    assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(gtMolWithHs)
    assert Chem.MolToSmiles(molWithoutHs) == CANONICAL_ASPIRIN_SMILES
    assert mol.GetNumAtoms() == 21

def test_conformer_coordinates(xyz_aspirin_molecule, cif_aspirin_molecule):
    XYZ_PATH, gtMol = xyz_aspirin_molecule
    CIF_PATH, gtMol = cif_aspirin_molecule

    mol_xyz = convert_xyz_to_mol(XYZ_PATH, canonicalize_smi=True, canonicalize_atoms=True)
    mol_cif = convert_cif_to_mol(CIF_PATH, canonicalize_smi=True, canonicalize_atoms=True)

    assert Chem.MolToSmiles(mol_xyz) == Chem.MolToSmiles(mol_cif) 

    # Align the conformers
    xyz_c = mol_xyz.GetConformer()
    cif_c = mol_cif.GetConformer()
    assert xyz_c.GetNumAtoms() == cif_c.GetNumAtoms()

    assert [a.GetAtomicNum() for a in mol_xyz.GetAtoms()] == [a.GetAtomicNum() for a in mol_cif.GetAtoms()]


def test_basic_conversion_exception():
    with pytest.raises(ValueError):
        convert_cif_to_mol("non_existent_file.cif")

@pytest.mark.skip(reason="This test is in progress")
def test_qm9_conversion():
    QM9_PATH = "./fixtures/dsgdb9nsd_000087.xyz"
    # List all files ending with .xyz
    xyz_files = [f for f in os.listdir(QM9_PATH) if f.endswith(".xyz") and "gdb" in f]
    inv = 0
    with_dot = 0
    for f in tqdm(xyz_files):
        try:
            mol = convert_xyz_to_mol(os.path.join(QM9_PATH, f))
        except:
            inv += 1
            continue

        assert mol is not None
        smi = Chem.MolToSmiles(mol)
        assert smi != ""
        with_dot += "." in smi

        inv += validate_xyz_conversion(os.path.join(QM9_PATH, f), mol)

    assert [inv, with_dot] == [0, 0]