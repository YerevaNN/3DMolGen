import ast
import json
import os

import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from molgen3D.data_processing.smiles_encoder_decoder import (
    encode_cartesian_v2,
    decode_cartesian_v2,
    strip_smiles,
    coords_rmsd,
)
from molgen3D.data_processing.utils import filter_mols
from molgen3D.utils.utils import load_pkl
from collections import Counter


DRUGS_SUMM_PATH = os.environ.get(
    "GEOM_DRUGS_SUMM",
    "/nfs/ap/mnt/sxtn2/chem/GEOM_data/rdkit_folder/summary_drugs.json",
)
MAX_MOLS = int(os.environ.get("GEOM_MAX_MOLS", "1000"))
RMSD_TOL = 1e-4


def _iter_drugs_mols_from_summ(drugs_summ_path, max_mols=MAX_MOLS):
    with open(drugs_summ_path, "r") as fh:
        summary = json.load(fh)

    base_dir = os.path.dirname(drugs_summ_path)
    count = 0
    for _, entry in summary.items():
        if count >= max_mols:
            break

        rel_pickle = entry.get("pickle_path") or entry.get("path")
        if rel_pickle is None:
            continue
        pickle_path = rel_pickle
        if not os.path.isabs(rel_pickle):
            pickle_path = os.path.join(base_dir, rel_pickle)

        mol_object = load_pkl(pickle_path)
        failures: dict = {}
        mols = filter_mols(mol_object, failures=failures, max_confs=1)
        if not mols:
            continue

        geom_smiles = mol_object.get("smiles") or mol_object.get("geom_smiles")
        if geom_smiles is None:
            geom_smiles = Chem.MolToSmiles(mols[0], canonical=True, isomericSmiles=True)

        for mol in mols:
            if mol.GetNumConformers() == 0:
                continue
            yield geom_smiles, mol
            count += 1
            if count >= max_mols:
                break


def _renumber_to_canonical_order(mol: Chem.Mol) -> Chem.Mol:
    mol_copy = Chem.Mol(mol)
    Chem.MolToSmiles(
        mol_copy,
        canonical=True,
        isomericSmiles=True,
        allHsExplicit=False,
        allBondsExplicit=False,
    )
    if mol_copy.HasProp("_smilesAtomOutputOrder"):
        order = list(map(int, ast.literal_eval(mol_copy.GetProp("_smilesAtomOutputOrder"))))
        mol_copy = Chem.RenumberAtoms(mol_copy, order)
    return mol_copy


def _inchi_no_stereo(mol: Chem.Mol) -> str:
    return inchi.MolToInchi(
        Chem.RemoveHs(mol),
        options="-SNon",
    )


def test_large_scale_drugs_enriched_roundtrip():
    if not os.path.exists(DRUGS_SUMM_PATH):
        pytest.skip(
            f"drugs_summ file not found at {DRUGS_SUMM_PATH}; set GEOM_DRUGS_SUMM to enable this test."
        )

    processed = 0
    failures = 0
    max_fail_frac = float(os.environ.get("GEOM_MAX_FAIL_FRAC", "0.25"))

    for geom_smiles, mol in _iter_drugs_mols_from_summ(DRUGS_SUMM_PATH):
        try:
            mol_no_h = Chem.RemoveHs(mol)
            if mol_no_h.GetNumConformers() == 0:
                continue

            enriched, _ = encode_cartesian_v2(mol_no_h, precision=4)
            mol_rt = decode_cartesian_v2(enriched)

            enriched_again, _ = encode_cartesian_v2(mol_rt, precision=4)
            mol_rt_again = decode_cartesian_v2(enriched_again)

            if strip_smiles(enriched_again) != strip_smiles(enriched):
                failures += 1
                processed += 1
                continue

            mol_rt_canon = _renumber_to_canonical_order(mol_rt)
            mol_rt_again_canon = _renumber_to_canonical_order(mol_rt_again)
            r = coords_rmsd(mol_rt_canon, mol_rt_again_canon)
            if r >= RMSD_TOL:
                failures += 1
                processed += 1
                continue

        except Exception:
            failures += 1
        processed += 1

    if processed == 0:
        pytest.skip("No molecules could be loaded from drugs_summ.")

    fail_frac = failures / processed
    assert fail_frac <= max_fail_frac, f"Too many failures: {fail_frac:.4%}"
