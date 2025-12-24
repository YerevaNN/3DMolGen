import re

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from molgen3D.data_processing.smiles_encoder_decoder import encode_cartesian_binned


def _construct_mol_cn_all_atoms_on_edges() -> Chem.Mol:
    """
    Mirror the logic from `bins_playground.ipynb`:

    - Start from SMILES 'CN'
    - Add hydrogens and embed a 3D conformer
    - Set every atom position to (-21.0, 21.0, 21.0)
    """
    smiles = "CN"
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)

    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        conf.SetAtomPosition(i, (-21.0, 21.0, 21.0))

    return mol


def test_binning_indices_match_numpy_digitize_on_edges() -> None:
    """
    Use the same setup as `bins_playground.ipynb` and verify that the
    bin indices in the enriched string match what `numpy.digitize`
    would produce for coordinates exactly at the bin edges.
    """
    mol = _construct_mol_cn_all_atoms_on_edges()

    # Same configuration as in the notebook: default ranges [-21, 21] and bin_size=0.043
    bin_size = 0.043
    enriched, canon_smiles, bins, ranges = encode_cartesian_binned(mol, bin_size=bin_size)

    # Recreate the bin edges for each axis (encode_cartesian_binned uses get_bins_for_coords)
    bins_x = np.arange(ranges[0][0], ranges[0][1], bin_size)
    bins_y = np.arange(ranges[1][0], ranges[1][1], bin_size)
    bins_z = np.arange(ranges[2][0], ranges[2][1], bin_size)

    # Expected bin indices from numpy.digitize for the edge coordinates
    x_coord = -21.0
    yz_coord = 21.0

    expected_ix = int(np.digitize([x_coord], bins_x)[0])
    expected_iy = int(np.digitize([yz_coord], bins_y)[0])
    expected_iz = int(np.digitize([yz_coord], bins_z)[0])

    # Extract the indices from the enriched string: tokens like [C]<ix,iy,iz>
    atom_pattern = re.compile(r"\[[^\]]+\]<(\d+),(\d+),(\d+)>")
    matches = atom_pattern.findall(enriched)

    # We should have one block per heavy atom (C and N)
    assert len(matches) == 2, f"Unexpected enriched string format: {enriched}"

    for ix_txt, iy_txt, iz_txt in matches:
        ix = int(ix_txt)
        iy = int(iy_txt)
        iz = int(iz_txt)

        assert ix == expected_ix
        assert iy == expected_iy
        assert iz == expected_iz


def test_binning_expected_encoding_string_on_edges() -> None:
    """
    Using the same CN edge-case setup, build the exact expected enriched string
    (including zero-padding) and compare it directly to the encoder output.
    """
    mol = _construct_mol_cn_all_atoms_on_edges()

    bin_size = 0.043
    enriched, canon_smiles, bins, _ = encode_cartesian_binned(mol, bin_size=bin_size)

    # Compute expected indices using the same bins that the encoder used.
    x_coord = -21.0
    yz_coord = 21.0

    ix = int(np.digitize([x_coord], bins[0])[0])
    iy = int(np.digitize([yz_coord], bins[1])[0])
    iz = int(np.digitize([yz_coord], bins[2])[0])

    # Zero-padding width matches the encoder's logic: max(3, len(str(len(b))))
    digits = [max(3, len(str(len(b)))) for b in bins]
    ix_txt = f"{ix:0{digits[0]}d}"
    iy_txt = f"{iy:0{digits[1]}d}"
    iz_txt = f"{iz:0{digits[2]}d}"

    # For CN, canonical heavy-atom SMILES should be "CN",
    # so we expect [C]<...>[N]<...> in that order.
    assert canon_smiles == "CN"
    expected_enriched = f"[C]<{ix_txt},{iy_txt},{iz_txt}>[N]<{ix_txt},{iy_txt},{iz_txt}>"

    assert enriched == expected_enriched

