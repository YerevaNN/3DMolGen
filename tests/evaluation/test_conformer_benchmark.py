from __future__ import annotations

import math

import pytest

Chem = pytest.importorskip("rdkit.Chem")
AllChem = pytest.importorskip("rdkit.Chem.AllChem")
Point3D = pytest.importorskip("rdkit.Geometry").Point3D

from molgen3D.evaluation.conformer_benchmark import (
    best_aligned_rmsd,
    bioactive_hit_at_k,
    chembl_coverage_fraction,
    cluster_distribution_metrics,
    cluster_entropy,
    energy_stats,
    greedy_cluster_metrics,
    greedy_cluster_count,
    rank_methods,
)


def mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=1) == 0
    return mol


def translate(mol: Chem.Mol, dx: float, dy: float = 0.0, dz: float = 0.0) -> Chem.Mol:
    copy = Chem.Mol(mol)
    conf = copy.GetConformer(0)
    for atom_idx in range(copy.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        conf.SetAtomPosition(atom_idx, Point3D(pos.x + dx, pos.y + dy, pos.z + dz))
    return copy


def test_greedy_cluster_count_respects_threshold(monkeypatch):
    base = mol_from_smiles("CC")
    close = translate(base, 0.2)
    far = translate(base, 4.0)

    def fake_rmsd(mol_a, mol_b):
        key_a = id(mol_a)
        key_b = id(mol_b)
        if key_a == key_b:
            return 0.0
        ids = {id(base): 0, id(close): 1, id(far): 2}
        pair = tuple(sorted((ids.get(key_a, -1), ids.get(key_b, -2))))
        return { (0, 1): 0.2, (0, 2): 2.5, (1, 2): 2.5 }[pair]

    monkeypatch.setattr(
        "molgen3D.evaluation.conformer_benchmark.heavy_rmsd_same_topology",
        fake_rmsd,
    )
    assert greedy_cluster_count([base, close, far], threshold=1.0) == 2
    assert greedy_cluster_count([base, close, far], threshold=0.05) == 3


def test_bioactive_hit_at_k():
    ref = mol_from_smiles("CC")
    near = translate(ref, 0.2)
    far = translate(ref, 3.0)
    hits = bioactive_hit_at_k([near, far], ref, thresholds=(0.5, 1.0, 2.0))
    assert hits["hit_0p5"] == 1.0
    assert hits["hit_2p0"] == 1.0
    assert hits["best_rmsd"] < 0.5


def test_best_aligned_rmsd_removes_hydrogens_and_aligns():
    ref = mol_from_smiles("CCO")
    shifted = translate(ref, 7.0, -3.0, 2.0)
    conf = shifted.GetConformer(0)
    for atom in shifted.GetAtoms():
        if atom.GetAtomicNum() == 1:
            conf.SetAtomPosition(atom.GetIdx(), Point3D(100.0 + atom.GetIdx(), -100.0, 50.0))

    assert best_aligned_rmsd(shifted, ref) == pytest.approx(0.0, abs=1e-6)


def test_chembl_coverage_fraction(monkeypatch):
    ref_a = mol_from_smiles("CC")
    ref_b = translate(ref_a, 4.0)
    gen_near = translate(ref_a, 0.3)
    gen_far = translate(ref_a, 8.0)

    def fake_rmsd(gen, ref):
        if gen is gen_near and ref is ref_a:
            return 0.3
        if gen is gen_near and ref is ref_b:
            return 2.0
        if gen is gen_far and ref is ref_a:
            return 8.0
        if gen is gen_far and ref is ref_b:
            return 4.0
        return 99.0

    monkeypatch.setattr(
        "molgen3D.evaluation.conformer_benchmark.best_aligned_rmsd",
        fake_rmsd,
    )
    coverage = chembl_coverage_fraction([gen_near, gen_far], [ref_a, ref_b], threshold=1.0)
    assert coverage == 0.5


def test_rank_methods_assigns_ranks():
    rows = [
        {"Method": "a", "clusters_per_100_1p0": 10.0, "hit_2p0": 0.2, "chembl_coverage_2p0": 0.3, "chembl_best_rmsd": 2.0},
        {"Method": "b", "clusters_per_100_1p0": 20.0, "hit_2p0": 0.8, "chembl_coverage_2p0": 0.1, "chembl_best_rmsd": 1.0},
    ]
    ranked, correlations = rank_methods(
        rows,
        method_key="Method",
        rank_columns={
            "clusters_per_100_1p0": False,
            "hit_2p0": False,
            "chembl_coverage_2p0": False,
            "chembl_best_rmsd": True,
        },
    )
    assert ranked[0]["Method"] == "a"
    assert ranked[0]["clusters_per_100_1p0_rank"] == 2.0
    assert ranked[1]["clusters_per_100_1p0_rank"] == 1.0
    assert "clusters_per_100_1p0_vs_hit_2p0" in correlations


def test_cluster_entropy_single_cluster_is_zero():
    mol = mol_from_smiles("CC")
    copy = translate(mol, 0.05)
    assert cluster_entropy([mol, copy], threshold=1.0) == 0.0


def test_cluster_distribution_metrics_balanced_vs_concentrated():
    balanced = cluster_distribution_metrics([0, 0, 1, 1])
    concentrated = cluster_distribution_metrics([0, 0, 0, 1])
    assert balanced["largest_cluster_fraction"] == pytest.approx(0.5)
    assert concentrated["largest_cluster_fraction"] == pytest.approx(0.75)
    assert balanced["effective_cluster_count"] > concentrated["effective_cluster_count"]


def test_greedy_cluster_metrics_requested_thresholds(monkeypatch):
    base = mol_from_smiles("CC")
    close = translate(base, 0.2)

    def fake_rmsd(mol_a, mol_b):
        return 0.2 if mol_a is not mol_b else 0.0

    monkeypatch.setattr(
        "molgen3D.evaluation.conformer_benchmark.heavy_rmsd_same_topology",
        fake_rmsd,
    )
    metrics = greedy_cluster_metrics([base, close], thresholds=(0.5, 1.0))
    assert metrics["greedy_clusters_0p5"] == pytest.approx(1.0)
    assert metrics["largest_cluster_fraction_1p0"] == pytest.approx(1.0)


def test_energy_stats_reads_post_min_energy():
    mol = mol_from_smiles("CC")
    mol.SetProp("minimization_applied", "true")
    mol.SetProp("post_min_energy", "12.5")
    stats = energy_stats([mol])
    assert stats["energy_count"] == pytest.approx(1.0)
    assert stats["energy_median"] == pytest.approx(12.5)
