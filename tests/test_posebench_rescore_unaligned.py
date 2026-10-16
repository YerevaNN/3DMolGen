from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from posebench_rescore_unaligned import (  # noqa: E402
    coordinates,
    centroid,
    restore_centroid,
    score_pair,
    select_primary_fragment,
)


def translated_ethanol(shift: np.ndarray):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    conformer = mol.GetConformer()
    for atom_index in range(mol.GetNumAtoms()):
        point = np.asarray(conformer.GetAtomPosition(atom_index)) + shift
        conformer.SetAtomPosition(atom_index, point)
    return mol


def test_fragment_selection_does_not_mutate_reference():
    reference = translated_ethanol(np.zeros(3))
    prediction = translated_ethanol(np.array([4.0, -2.0, 1.0]))
    before = coordinates(reference)
    selected = select_primary_fragment(reference, (prediction,))
    assert selected is prediction
    np.testing.assert_array_equal(coordinates(reference), before)


def test_calc_rms_preserves_translation_while_kabsch_removes_it():
    reference = translated_ethanol(np.zeros(3))
    prediction = Chem.Mol(reference)
    conformer = prediction.GetConformer()
    shift = np.array([10.0, 0.0, 0.0])
    for atom_index in range(prediction.GetNumAtoms()):
        point = np.asarray(conformer.GetAtomPosition(atom_index)) + shift
        conformer.SetAtomPosition(atom_index, point)
    before = coordinates(reference)
    result = score_pair(reference, prediction)
    assert result["rmsd"] > 9.9
    assert result["kabsch_rmsd"] < 1e-5
    assert result["centroid_distance"] > 9.9
    np.testing.assert_array_equal(coordinates(reference), before)


def test_restore_centroid_translates_without_changing_shape():
    original = translated_ethanol(np.zeros(3))
    decoded = translated_ethanol(np.array([0.02, -0.01, 0.03]))
    target = centroid(original)
    before = coordinates(decoded)
    restored, correction = restore_centroid(decoded, target)
    np.testing.assert_allclose(centroid(restored), target, atol=1e-12)
    np.testing.assert_allclose(
        coordinates(restored) - coordinates(restored)[0],
        before - before[0],
        atol=1e-12,
    )
    assert correction > 0
