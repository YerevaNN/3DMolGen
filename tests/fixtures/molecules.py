import pytest
from rdkit import Chem

_mol = Chem.Mol(Chem.MolFromSmiles("O=C(C)Oc1ccccc1C(=O)O"))
Chem.SanitizeMol(_mol)

CANONICAL_ASPIRIN_SMILES = Chem.MolToSmiles(_mol)


@pytest.fixture
def cif_aspirin_molecule():
    CIF_PATH = "./tests/fixtures/aspirin.cif"
    gtMol = Chem.MolFromSmiles(CANONICAL_ASPIRIN_SMILES)

    return CIF_PATH, gtMol


@pytest.fixture
def xyz_aspirin_molecule():
    XYZ_PATH = "./tests/fixtures/aspirin.xyz"
    gtMol = Chem.MolFromSmiles(CANONICAL_ASPIRIN_SMILES)

    return XYZ_PATH, gtMol
