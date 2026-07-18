from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

Chem = pytest.importorskip("rdkit.Chem")


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_casf16_ligands.py"
SPEC = importlib.util.spec_from_file_location("analyze_casf16_ligands", SCRIPT_PATH)
analyzer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return Chem.AddHs(mol)


def test_descriptor_from_mol_computes_core_fields():
    row = analyzer.descriptor_from_mol(
        mol_from_smiles("CCCC"),
        ligand_id="butane_conf0",
        source_file="butane.mol2",
    )

    assert row.ligand_id == "butane_conf0"
    assert row.source_file == "butane.mol2"
    assert row.canonical_smiles == "CCCC"
    assert row.rotatable_bonds == 1
    assert row.heavy_atoms == 4
    assert row.rotatable_bonds_per_heavy_atom == pytest.approx(0.25)
    assert row.status == "ok"


def test_build_stats_reports_buckets_duplicates_and_correlations():
    rows = [
        analyzer.descriptor_from_mol(mol_from_smiles("CC"), "ethane_a", "ethane_a.mol2"),
        analyzer.descriptor_from_mol(mol_from_smiles("CC"), "ethane_b", "ethane_b.mol2"),
        analyzer.descriptor_from_mol(mol_from_smiles("CCCC"), "butane", "butane.mol2"),
        analyzer.LigandDescriptor(
            ligand_id="bad",
            source_file="bad.mol2",
            canonical_smiles="",
            rotatable_bonds=None,
            heavy_atoms=None,
            rotatable_bonds_per_heavy_atom=None,
            status="failed",
            error="ValueError: no molecule",
        ),
    ]

    summary = analyzer.build_stats(rows, top_n=2)

    assert summary["total_files"] == 4
    assert summary["parsed_molecules"] == 3
    assert summary["parse_failures"] == 1
    assert summary["unique_canonical_smiles"] == 2
    assert summary["duplicate_structure_count"] == 1
    assert summary["duplicates"] == {"CC": ["ethane_a", "ethane_b"]}
    assert summary["rotatable_buckets"]["0"] == 2
    assert summary["rotatable_buckets"]["1-3"] == 1
    assert summary["heavy_atom_buckets"]["<=20"] == 3
    assert summary["rotatable_summary"]["median"] == 0
    assert summary["most_flexible"][0].ligand_id == "butane"
    assert summary["failures"][0].ligand_id == "bad"
    assert summary["pearson_heavy_vs_rotatable"] is not None
    assert summary["spearman_heavy_vs_rotatable"] is not None


def test_render_summary_includes_interesting_sections():
    rows = [
        analyzer.descriptor_from_mol(mol_from_smiles("CC"), "ethane_a", "ethane_a.mol2"),
        analyzer.descriptor_from_mol(mol_from_smiles("CC"), "ethane_b", "ethane_b.mol2"),
        analyzer.descriptor_from_mol(mol_from_smiles("CCCC"), "butane", "butane.mol2"),
    ]

    text = analyzer.render_summary(analyzer.build_stats(rows, top_n=1))

    assert "# CASF16 Ligand Descriptor Summary" in text
    assert "## Most Flexible Ligands" in text
    assert "## Duplicate Canonical SMILES" in text
    assert "ethane_a, ethane_b share `CC`" in text
