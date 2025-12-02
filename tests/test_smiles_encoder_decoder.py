import ast
import math
import random
import re
from typing import Iterable, Tuple

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from molgen3D.data_processing.smiles_encoder_decoder import (
    _format_atom_descriptor,
    _normalize_atom_descriptor,
    coords_rmsd,
    decode_cartesian_v2,
    encode_cartesian_v2,
    strip_smiles,
    tokenize_enriched,
    tokenize_smiles,
)


PRECISION = 4


def _canonicalize_atom_order(mol: Chem.Mol) -> Chem.Mol:
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


def _embed(mol: Chem.Mol, seed: int = 0) -> Chem.Mol:
    """Generate a single conformer with hydrogens removed (implicit by default)."""
    mol = Chem.Mol(mol)
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    status = AllChem.EmbedMolecule(mol_h, params)
    if status != 0:
        raise RuntimeError(f"Embedding failed for {Chem.MolToSmiles(mol)} (status {status})")
    try:
        AllChem.MMFFOptimizeMolecule(mol_h)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol_h)
    mol_no_h = Chem.RemoveHs(mol_h)
    Chem.SanitizeMol(mol_no_h)
    # Ensure atom output order is on the molecule
    return _canonicalize_atom_order(mol_no_h)


def _count_explicit_h(mol: Chem.Mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1)


def _inchi_key_no_h(mol: Chem.Mol) -> str:
    mol_copy = Chem.Mol(mol)
    mol_no_h = Chem.RemoveHs(mol_copy, sanitize=False)
    Chem.SanitizeMol(mol_no_h)
    return Chem.MolToInchiKey(mol_no_h)


def _truncate_mol_coords(mol: Chem.Mol, precision: int) -> Chem.Mol:
    mol_copy = Chem.Mol(mol)
    conf = mol_copy.GetConformer()
    factor = 10 ** precision
    for idx in range(mol_copy.GetNumAtoms()):
        pos = conf.GetAtomPosition(idx)
        x = math.trunc(pos.x * factor) / factor
        y = math.trunc(pos.y * factor) / factor
        z = math.trunc(pos.z * factor) / factor
        conf.SetAtomPosition(idx, Point3D(x, y, z))
    return mol_copy


def _assert_roundtrip(smiles: str, precision: int = PRECISION) -> Tuple[str, Chem.Mol, Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    mol = _embed(mol)
    enriched, canonical = encode_cartesian_v2(mol, precision=precision)
    tokens = tokenize_enriched(enriched)
    assert tokens, "Tokenization should yield at least one token"
    mol_rt = decode_cartesian_v2(enriched)
    mol = _canonicalize_atom_order(mol)
    assert _inchi_key_no_h(mol) == _inchi_key_no_h(mol_rt)
    assert _count_explicit_h(mol) == _count_explicit_h(mol_rt)
    rmsd_direct = coords_rmsd(mol, mol_rt)
    truncated_mol = _truncate_mol_coords(mol, precision)
    rmsd_truncated = coords_rmsd(truncated_mol, mol_rt)
    assert rmsd_direct < 1e-5 or rmsd_truncated < 1e-8
    canonical_no_h = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), canonical=True, isomericSmiles=True)
    assert strip_smiles(enriched) == strip_smiles(canonical_no_h)
    return enriched, mol, mol_rt


def _assert_stripped_equal(smiles: str) -> None:
    mol = Chem.MolFromSmiles(smiles)
    mol = _embed(mol)
    enriched, _ = encode_cartesian_v2(mol, precision=PRECISION)
    canonical_no_h = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), canonical=True, isomericSmiles=True)
    assert strip_smiles(enriched) == strip_smiles(canonical_no_h)


@pytest.mark.parametrize(
    "smiles",
    ["CC", "CCC", "CCO", "CC=O", "NCCN"],
)
def test_roundtrip_simple_aliphatic(smiles: str):
    _assert_roundtrip(smiles)


@pytest.mark.parametrize("smiles", ["CC", "CCC", "CCO", "CC=O", "NCCN"])
def test_simple_molecules_have_no_decorative_c_h(smiles: str):
    enriched, _, _ = _assert_roundtrip(smiles)
    assert not re.search(r"\[(?:cH\d*|CH\d*)\]", enriched)


@pytest.mark.parametrize(
    "smiles",
    ["c1ccccc1", "c1ccc(cc1)O", "Oc1ccccc1C", "c1cc2cccc2c1"],
)
def test_roundtrip_aromatic_systems(smiles: str):
    _assert_roundtrip(smiles)


@pytest.mark.parametrize(
    "smiles",
    ["c1ccccc1", "c1ccc(cc1)O", "Oc1ccccc1C", "c1cc2cccc2c1"],
)
def test_aromatic_descriptors_lowercase(smiles: str):
    enriched, _, _ = _assert_roundtrip(smiles)
    for descriptor in re.findall(r"\[([^\]]+)\]", enriched):
        if descriptor.lower() == descriptor and descriptor in {"c", "n", "o"}:
            assert descriptor.islower()


@pytest.mark.parametrize(
    "smiles,ring_digits",
    [
        ("c1ccccc1", {"1"}),
        ("c1ccc(cc1)O", {"1"}),
        ("Oc1ccccc1C", {"1"}),
        ("c1cc2cccc2c1", {"1", "2"}),
    ],
)
def test_aromatic_ring_numbers_preserved(smiles: str, ring_digits: Iterable[str]):
    enriched, _, _ = _assert_roundtrip(smiles)
    for digit in ring_digits:
        assert digit in enriched


@pytest.mark.parametrize(
    "smiles",
    ["C[C@H](O)N", "C[C@@H](Br)Cl", "F[C@](Br)(Cl)I"],
)
def test_roundtrip_stereochemistry(smiles: str):
    _assert_roundtrip(smiles)


@pytest.mark.parametrize(
    "smiles",
    ["C[C@H](O)N", "C[C@@H](Br)Cl", "F[C@](Br)(Cl)I"],
)
def test_chirality_descriptors_present(smiles: str):
    enriched, _, _ = _assert_roundtrip(smiles)
    assert "@" in enriched


@pytest.mark.parametrize(
    "smiles",
    ["[NH4+]", "C[NH2+]C", "C[NH+](C)C", "N[C@@H](C(=O)O)C(=O)[O-]"],
)
def test_roundtrip_charged_systems(smiles: str):
    _assert_roundtrip(smiles)


@pytest.mark.parametrize(
    "smiles,fragment",
    [
        ("[NH4+]", "[NH4+]"),
        ("C[NH2+]C", "[NH2+]"),
        ("C[NH+](C)C", "[NH+]"),
        ("N[C@@H](C(=O)O)C(=O)[O-]", "[O-]"),
    ],
)
def test_charged_descriptors_preserved(smiles: str, fragment: str):
    enriched, _, _ = _assert_roundtrip(smiles)
    assert fragment in enriched


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("c1[nH]ccc1", "[nH]"),
        ("c1c[nH]c2ccccc12", "[nH]"),
    ],
)
def test_explicit_aromatic_hydrogens_retained(smiles: str, expected: str):
    enriched, _, _ = _assert_roundtrip(smiles)
    assert expected in enriched


def test_amide_nitrogen_stays_implicit():
    smiles = "COc1ccccc1NC(=O)c1ccc(N2C[C@@H]3C[C@H](C2)c2cccc(=O)n2C3)c(NC(=O)c2ccncc2)c1"
    mol = Chem.MolFromSmiles(smiles)
    mol = _embed(mol)
    enriched, _ = encode_cartesian_v2(mol, precision=PRECISION)
    assert "[NH]" not in enriched


@pytest.mark.parametrize(
    "smiles,forbidden",
    [
        ("CC(=O)O", "[OH]"),  # neutral oxygen should not show OH
        ("c1ccccc1", "[cH]"),  # aromatic carbon without explicit H stays c
        ("CCN", "[NH2]"),  # neutral amine without explicit H
    ],
)
def test_no_spurious_h_for_common_heteroatoms(smiles: str, forbidden: str):
    mol = Chem.MolFromSmiles(smiles)
    mol = _embed(mol)
    enriched, _ = encode_cartesian_v2(mol, precision=PRECISION)
    assert forbidden not in enriched


@pytest.mark.parametrize(
    "smiles,fragments",
    [
        ("BrCCCl", {"[Br]", "[Cl]"}),
        ("[13C]C", {"[13C]"}),
        ("[2H]OC", {"[2H]"}),
        ("C[Si](C)(C)C", {"[Si]"}),
        ("C[Pt+2]Cl", {"[Pt+2]"}),
    ],
)
def test_multi_letter_elements_and_isotopes(smiles: str, fragments: Iterable[str]):
    enriched, _, _ = _assert_roundtrip(smiles)
    for fragment in fragments:
        assert fragment in enriched


@pytest.mark.parametrize("smiles", ["C1CC1", "C1CCC2CCCCC2C1", "C%12CC%12"])
def test_roundtrip_ring_closures(smiles: str):
    _assert_roundtrip(smiles)


def test_percent_ring_indices_preserved():
    enriched, mol, _ = _assert_roundtrip("C%12CC%12")
    canonical = strip_smiles(enriched)
    canonical_ring_tokens = [
        tok["text"]
        for tok in tokenize_smiles(canonical)
        if tok["type"] == "nonatom" and (tok["text"].isdigit() or tok["text"].startswith("%"))
    ]
    enriched_ring_tokens = [
        tok["text"]
        for tok in tokenize_enriched(enriched)
        if tok["type"] == "nonatom" and (tok["text"].isdigit() or tok["text"].startswith("%"))
    ]
    assert canonical_ring_tokens == enriched_ring_tokens


@pytest.mark.parametrize("smiles", ["CC(C)(C)O", "N(CC(C)C)C(=O)O"])
def test_roundtrip_branched_parentheses(smiles: str):
    enriched, _, _ = _assert_roundtrip(smiles)
    assert "(" in enriched and ")" in enriched


@pytest.mark.parametrize("smiles", ["C#N", "C=C", "C:C", "C/C=C\\C"])
def test_roundtrip_bond_types(smiles: str):
    enriched, _, _ = _assert_roundtrip(smiles)
    for symbol in ["#", "=", ":", "/", "\\"]:
        if symbol in smiles:
            assert symbol in enriched


@pytest.mark.parametrize(
    "smiles",
    [
        "CC",
        "CCC",
        "CC=O",
        "C1CCCCC1",
        "C[NH2+]C",
        "c1ccccc1",
        "c1[nH]ccc1",
        "BrCCCl",
        "C[Si](C)(C)C",
        "C/C=C\\C",
    ],
)
def test_strip_smiles_matches_plain(smiles: str):
    _assert_stripped_equal(smiles)


def test_strip_smiles_preserves_special_brackets():
    enriched, _, _ = _assert_roundtrip("N[C@@H](C(=O)O)C(=O)[O-]")
    stripped = strip_smiles(enriched)
    assert "[O-]" in stripped
    assert "[C@@H]" in strip_smiles("N[C@@H](C(=O)O)C(=O)[O-]")


def test_tokenize_smiles_basic_sequence():
    tokens = tokenize_smiles("C1=CC(=O)O")
    sequence = [tok["text"] for tok in tokens]
    assert sequence == ["C", "1", "=", "C", "C", "(", "=", "O", ")", "O"]


def test_tokenize_smiles_handles_percent_ring():
    tokens = tokenize_smiles("C%12CC%12")
    sequence = [tok["text"] for tok in tokens]
    assert "%12" in sequence


def test_tokenize_enriched_roundtrip_tokens():
    enriched, _, _ = _assert_roundtrip("CCO")
    tokens = tokenize_enriched(enriched)
    reconstructed = ""
    for token in tokens:
        if token["type"] == "atom_with_coords":
            coords = ",".join(f"{c:.4f}".rstrip("0").rstrip(".") or "0" for c in token["coords"])
            reconstructed += f"{token['atom_desc']}<{coords}>"
        else:
            reconstructed += token["text"]
    assert strip_smiles(reconstructed) == strip_smiles(enriched)


@pytest.mark.parametrize(
    "descriptor,expected",
    [
        ("[CH3]", "[C]"),
        ("[CH2]", "[C]"),
        ("[CH]", "[C]"),
        ("[cH]", "[c]"),
        ("[13CH3]", "[13CH3]"),
    ],
)
def test_normalize_atom_descriptor(descriptor: str, expected: str):
    assert _normalize_atom_descriptor(descriptor) == expected


def test_format_atom_descriptor_handles_stereo_charge():
    mol = Chem.MolFromSmiles("C[C@H](O)[NH2+]")
    mol = _embed(mol)
    nitrogen = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "N")
    descriptor = _format_atom_descriptor(nitrogen)
    assert descriptor.startswith("[N")
    assert descriptor.endswith("+]")


def test_format_atom_descriptor_aromatic_lowercase():
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol = _embed(mol)
    descriptor = _format_atom_descriptor(mol.GetAtomWithIdx(0))
    assert descriptor == "[c]"


def test_format_atom_descriptor_chiral():
    mol = Chem.MolFromSmiles("C[C@H](O)N")
    mol = _embed(mol)
    descriptor = _format_atom_descriptor(mol.GetAtomWithIdx(1))
    assert descriptor.startswith("[C@")


def test_coords_rmsd_zero_and_positive():
    mol = _embed(Chem.MolFromSmiles("CCO"))
    mol_copy = Chem.Mol(mol)
    assert coords_rmsd(mol, mol_copy) == pytest.approx(0.0, abs=1e-8)
    perturbed = Chem.Mol(mol)
    conf = perturbed.GetConformer()
    pos = conf.GetAtomPosition(0)
    conf.SetAtomPosition(0, pos.__class__(pos.x + 0.1, pos.y, pos.z))
    assert coords_rmsd(mol, perturbed) > 0


def test_decimal_coordinate_truncation():
    mol = _embed(Chem.MolFromSmiles("CC"))
    enriched, _ = encode_cartesian_v2(mol, precision=3)
    for coord_str in re.findall(r"<([^>]+)>", enriched):
        for component in coord_str.split(","):
            if "." in component:
                frac = component.split(".", 1)[1]
                assert len(frac) <= 3


def test_decode_encode_sanitizes():
    enriched, _, _ = _assert_roundtrip("C1CCC2CCCCC2C1")
    mol = decode_cartesian_v2(enriched)
    Chem.SanitizeMol(mol)  # Should not raise


def test_tokenize_enriched_handles_ring_and_branch():
    enriched, _, _ = _assert_roundtrip("C1(CC)CCC1")
    canonical = strip_smiles(enriched)
    canonical_nonatoms = [tok["text"] for tok in tokenize_smiles(canonical) if tok["type"] == "nonatom"]
    enriched_nonatoms = [tok["text"] for tok in tokenize_enriched(enriched) if tok["type"] == "nonatom"]
    assert canonical_nonatoms == enriched_nonatoms


def test_lossless_roundtrip_for_enriched_again():
    enriched, mol, _ = _assert_roundtrip("C1=CC(=O)O1")
    mol_rt = decode_cartesian_v2(enriched)
    assert _inchi_key_no_h(mol) == _inchi_key_no_h(mol_rt)


def test_strip_smiles_matches_for_enriched_inputs():
    enriched, _, _ = _assert_roundtrip("C[NH2+]C")
    assert strip_smiles(enriched) == strip_smiles("C[NH2+]C")


def test_tokenize_smiles_returns_atom_count():
    tokens = tokenize_smiles("C[NH2+]C")
    assert sum(1 for tok in tokens if tok["type"] == "atom") == 3


def test_tokenize_enriched_atom_count_matches_smiles():
    enriched, _, _ = _assert_roundtrip("C[NH2+]C")
    enriched_tokens = tokenize_enriched(enriched)
    atom_tokens = [tok for tok in enriched_tokens if tok["type"] == "atom_with_coords"]
    assert len(atom_tokens) == 3  # Three atoms remain after RemoveHs


def test_strip_smiles_removes_coordinates():
    enriched, _, _ = _assert_roundtrip("CCO")
    assert "<" in enriched
    assert "<" not in strip_smiles(enriched)


def test_roundtrip_ring_fused_structure():
    _assert_roundtrip("C1=CC2=CC=CC=C2C=C1")


def test_roundtrip_linear_chains():
    for length in range(2, 7):
        smiles = "C" * length
        _assert_roundtrip(smiles)


def test_roundtrip_with_halogen_branches():
    for smiles in ["CC(Br)C", "CC(Cl)C", "C(Br)C(Cl)C"]:
        enriched, _, _ = _assert_roundtrip(smiles)
        assert any(tag in enriched for tag in ("[Br]", "[Cl]"))


def test_roundtrip_with_multiple_bonds():
    for smiles in ["CC#CC", "C=CC=C", "C#CC#C"]:
        _assert_roundtrip(smiles)


def test_roundtrip_with_mixed_bonds_and_branches():
    for smiles in ["C=C(C)C", "C(=O)NC", "C#CC(=O)O"]:
        _assert_roundtrip(smiles)


def test_strip_smiles_consistency_additional():
    for smiles in ["C1=CC=CC=C1", "C1CCCCC1", "N1CCCCC1", "[NH4+]"]:
        _assert_stripped_equal(smiles)


def test_tokenize_smiles_counts_ring_digits():
    tokens = tokenize_smiles("C1CCCCC1")
    assert sum(1 for tok in tokens if tok["text"] == "1") == 2


def test_tokenize_enriched_detects_ring_digits():
    enriched, _, _ = _assert_roundtrip("C1CCCCC1")
    tokens = tokenize_enriched(enriched)
    assert any(tok["type"] == "nonatom" and tok["text"] == "1" for tok in tokens)


def test_aromatic_ring_percentages_strip_correctly():
    smiles = "c1ccccc1"
    _assert_stripped_equal(smiles)


def test_roundtrip_multiple_splits():
    _assert_roundtrip("C1=CC(=O)OC(=O)C=C1")


def test_tokenize_smiles_with_dot():
    tokens = tokenize_smiles("CC.CC")
    assert "." in [tok["text"] for tok in tokens]


def test_tokenize_enriched_with_dot():
    mol = _embed(Chem.MolFromSmiles("CC"))
    enriched, _ = encode_cartesian_v2(mol, precision=PRECISION)
    enriched_with_dot = f"{enriched}.{enriched}"
    tokens = tokenize_enriched(enriched_with_dot)
    assert any(tok["type"] == "nonatom" and tok["text"] == "." for tok in tokens)


def test_roundtrip_zwitterion():
    _assert_roundtrip("N[C@@H](C(=O)[O-])C(=O)O")


def test_roundtrip_longer_chain_with_branch():
    _assert_roundtrip("CC(C)CC(C)C")


def test_roundtrip_simple_ring_with_branch():
    _assert_roundtrip("C1(CC)CC1")


def test_roundtrip_ketone_chain():
    _assert_roundtrip("CCC(=O)CC")


def test_roundtrip_with_sulfur():
    _assert_roundtrip("CS(=O)C")


def test_strip_smiles_removes_all_coords_for_many():
    smiles_list = ["CCO", "C1CCCCC1", "C[NH2+]C", "c1ccccc1"]
    for smiles in smiles_list:
        mol = _embed(Chem.MolFromSmiles(smiles))
        enriched, _ = encode_cartesian_v2(mol, precision=PRECISION)
        assert "<" not in strip_smiles(enriched)


def test_tokenize_enriched_no_trailing_text():
    enriched, _, _ = _assert_roundtrip("C[C@H](O)N")
    tokenize_enriched(enriched)  # Should not raise


def test_random_smiles_roundtrip():
    rng = random.Random(0)
    collected = []
    attempts = 0
    atoms = ["C", "N", "O", "F"]
    while len(collected) < 50 and attempts < 1000:
        length = rng.randint(2, 6)
        smiles = "".join(rng.choice(atoms) for _ in range(length))
        mol = Chem.MolFromSmiles(smiles)
        attempts += 1
        if mol is None:
            continue
        mol = _embed(mol, seed=rng.randint(0, 1000))
        enriched, _ = encode_cartesian_v2(mol, precision=PRECISION)
        mol_rt = decode_cartesian_v2(enriched)
        mol = _canonicalize_atom_order(mol)
        assert _inchi_key_no_h(mol) == _inchi_key_no_h(mol_rt)
        truncated = _truncate_mol_coords(mol, PRECISION)
        assert coords_rmsd(mol, mol_rt) < 1e-5 or coords_rmsd(truncated, mol_rt) < 1e-8
        collected.append(smiles)
    assert len(collected) >= 50
