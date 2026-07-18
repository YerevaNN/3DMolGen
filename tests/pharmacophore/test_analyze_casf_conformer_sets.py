from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_casf_conformer_sets.py"
SPEC = importlib.util.spec_from_file_location("analyze_casf_conformer_sets", SCRIPT_PATH)
analyzer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def test_parse_method_metadata_handles_chembl_count():
    assert analyzer.parse_method_metadata("rdkit_random_raw_chembl_count") == (
        "rdkit_random",
        "raw",
        "chembl_count",
    )
    assert analyzer.parse_method_metadata("torsion_minimized_chembl_count") == (
        "torsion",
        "minimized",
        "chembl_count",
    )


def test_source_order_groups_fixed_dynamic_then_chembl_count():
    shuffled = pd.DataFrame(
        [
            {"mol_id": "lig_a", "generation_method": "torsion_minimized_chembl_count"},
            {"mol_id": "lig_a", "generation_method": "rdkit_random_raw_dynamic"},
            {"mol_id": "lig_a", "generation_method": "torsion_raw_fixed"},
            {"mol_id": "lig_a", "generation_method": "rdkit_random_minimized_fixed"},
            {"mol_id": "lig_a", "generation_method": "rdkit_random_raw_chembl_count"},
            {"mol_id": "lig_a", "generation_method": "rdkit_random_raw_fixed"},
        ]
    )
    summary = analyzer.build_generation_filter_summary(shuffled)

    assert list(summary["source"]) == [
        "rdkit_random_raw_fixed",
        "rdkit_random_minimized_fixed",
        "torsion_raw_fixed",
        "rdkit_random_raw_dynamic",
        "rdkit_random_raw_chembl_count",
        "torsion_minimized_chembl_count",
    ]
    assert summary[summary["source"] == "rdkit_random_raw_chembl_count"].iloc[0]["tier"] == "chembl_count"


def test_validate_metrics_completeness_requires_all_methods_and_chembl():
    rows = [
        {
            "mol_id": "lig_a",
            "source": source,
            "family": analyzer.source_metadata(source)[0],
            "stage": analyzer.source_metadata(source)[1],
            "tier": analyzer.source_metadata(source)[2],
            "conformer_count": 1 if source != "chembl3d_gt_pb" else 0,
            "chembl_load_status": "ok" if source in {"chembl3d_gt", "chembl3d_gt_pb"} else "",
            "pb_check_fail_counts_json": "{}",
            "pb_check_fail_rates_json": "{}",
            "casf_best_rmsd": 0.5,
            "casf_median_rmsd": 0.5,
            "casf_hit_0p25": 0.0,
            "casf_hit_0p5": 1.0,
            "casf_hit_0p75": 1.0,
            "casf_hit_2p0": 1.0,
            "casf_opt_best_rmsd": 0.6,
            "casf_opt_median_rmsd": 0.6,
            "casf_opt_hit_0p25": 0.0,
            "casf_opt_hit_0p5": 0.0,
            "casf_opt_hit_0p75": 1.0,
            "casf_opt_hit_2p0": 1.0,
        }
        for source in tuple(analyzer.METHODS) + analyzer.REFERENCE_SOURCES
    ]
    metrics = pd.DataFrame(rows)
    chembl_map = pd.DataFrame([{"ligand_id": "lig_a"}])

    analyzer.validate_metrics_completeness(metrics, ["lig_a"], chembl_map)

    incomplete = metrics[metrics["source"] != analyzer.METHODS[-1]]
    with pytest.raises(RuntimeError, match="Missing per-ligand metric"):
        analyzer.validate_metrics_completeness(incomplete, ["lig_a"], chembl_map)


def test_posebusters_result_uses_reference_and_keeps_all_check_rows(monkeypatch):
    predicted = object()
    reference = object()
    captured = {}

    def fake_posebusters_geometry_passes(mols, reference_mol, max_workers, energy_num_threads):
        captured["mols"] = mols
        captured["reference_mol"] = reference_mol
        captured["max_workers"] = max_workers
        captured["energy_num_threads"] = energy_num_threads
        return analyzer.PoseBustersResult(
            [False],
            [{"Energy ratio": False, "Bond lengths": True}],
        )

    monkeypatch.setattr(analyzer, "posebusters_geometry_passes", fake_posebusters_geometry_passes)

    result = analyzer.posebusters_result([predicted], reference)

    assert result.passes == [False]
    assert result.check_rows == [{"Energy ratio": False, "Bond lengths": True}]
    assert captured == {
        "mols": [predicted],
        "reference_mol": reference,
        "max_workers": 1,
        "energy_num_threads": 1,
    }


def test_build_generation_filter_summary_columns():
    manifest = pd.DataFrame(
        [
            {
                "mol_id": "lig_a",
                "generation_method": "rdkit_random_raw_fixed",
                "set_tier": "fixed",
                "num_target_confs": 100,
                "rotatable_bonds": 2,
                "generated_candidates": 100,
                "finite_rejected": 1,
                "clash_rejected": 20,
                "bond_rejected": 0,
                "stereo_rejected": 0,
                "rmsd_rejected": 0,
                "pre_clash_passed": 80,
                "minimization_input_confs": 0,
                "minimization_failed": 0,
                "pb_input_confs": 100,
                "pb_pass_confs": 90,
                "pb_fail_confs": 10,
                "kept_confs": 90,
                "minimization_invalid": 0,
                "pb_check_fail_counts_json": '{"Energy ratio":2}',
            },
            {
                "mol_id": "lig_a",
                "generation_method": "rdkit_random_raw_dynamic",
                "set_tier": "dynamic",
                "num_target_confs": 40,
                "rotatable_bonds": 2,
                "generated_candidates": 100,
                "finite_rejected": 1,
                "clash_rejected": 20,
                "bond_rejected": 0,
                "stereo_rejected": 0,
                "rmsd_rejected": 0,
                "pre_clash_passed": 40,
                "minimization_input_confs": 0,
                "minimization_failed": 0,
                "pb_input_confs": 40,
                "pb_pass_confs": 38,
                "pb_fail_confs": 2,
                "kept_confs": 38,
                "minimization_invalid": 0,
                "pb_check_fail_counts_json": '{"Energy ratio":1}',
            },
            {
                "mol_id": "lig_a",
                "generation_method": "rdkit_random_minimized_fixed",
                "set_tier": "fixed",
                "num_target_confs": 100,
                "rotatable_bonds": 2,
                "generated_candidates": 0,
                "finite_rejected": 0,
                "clash_rejected": 0,
                "bond_rejected": 0,
                "stereo_rejected": 0,
                "rmsd_rejected": 0,
                "pre_clash_passed": 0,
                "minimization_input_confs": 100,
                "minimization_failed": 0,
                "pb_input_confs": 95,
                "pb_pass_confs": 95,
                "pb_fail_confs": 0,
                "kept_confs": 95,
                "minimization_invalid": 5,
                "pb_check_fail_counts_json": "{}",
            },
        ]
    )
    summary = analyzer.build_generation_filter_summary(manifest)
    pb_failures = analyzer.build_generation_pb_failure_summary(manifest, summary)
    fixed = summary[summary["source"] == "rdkit_random_raw_fixed"].iloc[0]
    dynamic = summary[summary["source"] == "rdkit_random_raw_dynamic"].iloc[0]
    assert fixed["target_confs_mean"] == pytest.approx(100.0)
    assert fixed["pb_fail_total"] == pytest.approx(10.0)
    assert fixed["pb_fail_rate_mean"] == pytest.approx(0.1)
    assert fixed["finite_fail_total"] == pytest.approx(1.0)
    assert pd.isna(dynamic["finite_fail_total"])
    assert dynamic["pb_fail_rate_mean"] == pytest.approx(0.05)
    assert dynamic["kept_vs_target_rate_mean"] == pytest.approx(0.95)
    assert pb_failures[pb_failures["source"] == "rdkit_random_raw_fixed"].iloc[0]["pb_fail_count"] == 2
    assert pb_failures[pb_failures["source"] == "rdkit_random_raw_fixed"].iloc[0]["ligands_scope"] == "1/90"


def test_build_requested_summaries_include_references_and_generators():
    metrics = pd.DataFrame(
        [
            {
                "mol_id": "lig_a",
                "source": "casf_crystal",
                "family": "casf_crystal",
                "stage": "reference",
                "tier": "reference",
                "conformer_count": 1,
                "clash_input_confs": 1,
                "clash_pass_confs": 1,
                "clash_fail_confs": 0,
                "clash_pass_rate": 1.0,
                "pb_input_confs": 1,
                "pb_pass_confs": 1,
                "pb_fail_confs": 0,
                "pb_pass_rate": 1.0,
            },
            {
                "mol_id": "lig_a",
                "source": "chembl3d_sdf",
                "family": "chembl3d_sdf",
                "stage": "reference",
                "tier": "reference",
                "conformer_count": 1,
                "clash_input_confs": 1,
                "clash_pass_confs": 1,
                "clash_fail_confs": 0,
                "clash_pass_rate": 1.0,
                "pb_input_confs": 1,
                "pb_pass_confs": 1,
                "pb_fail_confs": 0,
                "pb_pass_rate": 1.0,
            },
            {
                "mol_id": "lig_a",
                "source": "casf_opt",
                "family": "casf_opt",
                "stage": "reference",
                "tier": "reference",
                "conformer_count": 1,
                "clash_input_confs": 1,
                "clash_pass_confs": 1,
                "clash_fail_confs": 0,
                "clash_pass_rate": 1.0,
                "pb_input_confs": 1,
                "pb_pass_confs": 1,
                "pb_fail_confs": 0,
                "pb_pass_rate": 1.0,
            },
            {
                "mol_id": "lig_a",
                "source": "chembl3d_gt",
                "family": "chembl3d_gt",
                "stage": "reference",
                "tier": "reference",
                "conformer_count": 2,
                "clash_input_confs": 2,
                "clash_pass_confs": 2,
                "clash_fail_confs": 0,
                "clash_pass_rate": 1.0,
                "pb_input_confs": 2,
                "pb_pass_confs": 1,
                "pb_fail_confs": 1,
                "pb_pass_rate": 0.5,
                "greedy_clusters_1p0": 2,
                "clusters_per_100_1p0": 100.0,
                "energy_median": 5.0,
                "casf_best_rmsd": 0.4,
                "casf_median_rmsd": 0.6,
                "casf_hit_0p25": 0.0,
                "casf_hit_0p5": 1.0,
                "casf_hit_0p75": 1.0,
                "casf_hit_2p0": 1.0,
                "casf_opt_best_rmsd": 0.5,
                "casf_opt_median_rmsd": 0.7,
                "casf_opt_hit_0p25": 0.0,
                "casf_opt_hit_0p5": 1.0,
                "casf_opt_hit_0p75": 1.0,
                "casf_opt_hit_2p0": 1.0,
                "pb_check_fail_counts_json": '{"Energy ratio":1}',
            },
            {
                "mol_id": "lig_a",
                "source": "chembl3d_gt_pb",
                "family": "chembl3d_gt_pb",
                "stage": "reference",
                "tier": "reference",
                "conformer_count": 1,
                "clash_input_confs": 1,
                "clash_pass_confs": 1,
                "clash_fail_confs": 0,
                "clash_pass_rate": 1.0,
                "pb_input_confs": 1,
                "pb_pass_confs": 1,
                "pb_fail_confs": 0,
                "pb_pass_rate": 1.0,
                "greedy_clusters_1p0": 1,
                "clusters_per_100_1p0": 100.0,
                "energy_median": 4.0,
                "casf_best_rmsd": 0.3,
                "casf_median_rmsd": 0.3,
                "casf_hit_0p25": 0.0,
                "casf_hit_0p5": 1.0,
                "casf_hit_0p75": 1.0,
                "casf_hit_2p0": 1.0,
                "casf_opt_best_rmsd": 0.4,
                "casf_opt_median_rmsd": 0.4,
                "casf_opt_hit_0p25": 0.0,
                "casf_opt_hit_0p5": 1.0,
                "casf_opt_hit_0p75": 1.0,
                "casf_opt_hit_2p0": 1.0,
                "pb_check_fail_counts_json": "{}",
            },
            {
                "mol_id": "lig_a",
                "source": "rdkit_random_raw_fixed",
                "family": "rdkit_random",
                "stage": "raw",
                "tier": "fixed",
                "conformer_count": 10,
                "greedy_clusters_1p0": 3,
                "clusters_per_100_1p0": 30.0,
                "energy_median": 2.0,
                "casf_best_rmsd": 0.7,
                "casf_median_rmsd": 0.9,
                "casf_hit_0p25": 0.0,
                "casf_hit_0p5": 0.0,
                "casf_hit_0p75": 1.0,
                "casf_hit_2p0": 1.0,
                "casf_opt_best_rmsd": 0.8,
                "casf_opt_median_rmsd": 1.0,
                "casf_opt_hit_0p25": 0.0,
                "casf_opt_hit_0p5": 0.0,
                "casf_opt_hit_0p75": 0.0,
                "casf_opt_hit_2p0": 1.0,
            },
        ]
    )
    refs = analyzer.build_reference_filter_summary(metrics)
    clusters = analyzer.build_cluster_summary(metrics)
    energies = analyzer.build_energy_summary(
        pd.DataFrame(
            [
                {
                    "mol_id": "lig_a",
                    "source": "rdkit_random_raw_fixed",
                    "family": "rdkit_random",
                    "stage": "raw",
                    "tier": "fixed",
                    "energy": 2.0,
                },
                {
                    "mol_id": "lig_a",
                    "source": "chembl3d_gt",
                    "family": "chembl3d_gt",
                    "stage": "reference",
                    "tier": "reference",
                    "energy": 5.0,
                },
                {
                    "mol_id": "lig_a",
                    "source": "chembl3d_gt_pb",
                    "family": "chembl3d_gt_pb",
                    "stage": "reference",
                    "tier": "reference",
                    "energy": 4.0,
                },
            ]
        )
    )
    hits = analyzer.build_casf_hit_summary(metrics)
    opt_hits = analyzer.build_casf_opt_hit_summary(metrics)
    assert set(refs["source"]) == {"casf_crystal", "casf_opt", "chembl3d_sdf", "chembl3d_gt", "chembl3d_gt_pb"}
    assert refs[refs["source"] == "casf_crystal"].iloc[0]["clash_fail_total"] == pytest.approx(0.0)
    assert refs[refs["source"] == "chembl3d_gt"].iloc[0]["pb_fail_total"] == pytest.approx(1.0)
    assert refs[refs["source"] == "chembl3d_gt_pb"].iloc[0]["pb_fail_total"] == pytest.approx(0.0)
    assert "ligands_scope" in refs.columns
    assert {"chembl3d_gt", "chembl3d_gt_pb"}.issubset(set(clusters["source"]))
    assert clusters[clusters["source"] == "chembl3d_gt"].iloc[0]["mean_confs_per_ligand"] == pytest.approx(2.0)
    assert clusters[clusters["source"] == "chembl3d_gt_pb"].iloc[0]["mean_confs_per_ligand"] == pytest.approx(1.0)
    assert "ligands_scope" in clusters.columns
    assert "rdkit_random_raw_fixed" in set(energies["source"])
    assert energies[energies["source"] == "chembl3d_gt"].iloc[0]["energy_median"] == pytest.approx(5.0)
    assert energies[energies["source"] == "chembl3d_gt_pb"].iloc[0]["energy_median"] == pytest.approx(4.0)
    assert {"casf_hit_0p25", "casf_hit_0p5", "casf_hit_0p75"}.issubset(hits.columns)
    assert {"chembl3d_gt", "chembl3d_gt_pb"}.issubset(set(hits["source"]))
    assert hits[hits["source"] == "rdkit_random_raw_fixed"].iloc[0]["casf_hit_0p75"] == pytest.approx(1.0)
    assert hits[hits["source"] == "chembl3d_gt"].iloc[0]["casf_hit_0p5"] == pytest.approx(1.0)
    assert hits[hits["source"] == "chembl3d_gt_pb"].iloc[0]["casf_hit_0p5"] == pytest.approx(1.0)
    assert {"chembl3d_gt", "chembl3d_gt_pb"}.issubset(set(opt_hits["source"]))
    assert opt_hits[opt_hits["source"] == "rdkit_random_raw_fixed"].iloc[0]["casf_opt_hit_2p0"] == pytest.approx(1.0)


def test_resolve_worker_count():
    assert analyzer._resolve_worker_count(None, 0) == 1
    assert analyzer._resolve_worker_count(8, 5) == 5
    assert analyzer._resolve_worker_count(1, 10) == 1
    assert analyzer._resolve_worker_count(0, 10) >= 1


def test_resolve_posebusters_parallelism(monkeypatch):
    monkeypatch.setattr(analyzer, "_available_cpu_count", lambda: 32)

    assert analyzer._resolve_posebusters_parallelism(None, None, 32) == (1, 1)
    assert analyzer._resolve_posebusters_parallelism(None, None, 8) == (4, 1)
    assert analyzer._resolve_posebusters_parallelism(None, None, 1) == (32, 1)
    assert analyzer._resolve_posebusters_parallelism(16, None, 8) == (4, 1)
    assert analyzer._resolve_posebusters_parallelism(None, 16, 8) == (4, 1)
    assert analyzer._resolve_posebusters_parallelism(2, 16, 1) == (2, 1)
    assert analyzer._resolve_posebusters_parallelism(1, None, 1) == (1, 32)


def test_select_analysis_mol_ids_filters_zero_rot_and_missing_sdfs(tmp_path):
    generation_dir = tmp_path / "generation"
    for method in analyzer.METHODS:
        (generation_dir / method).mkdir(parents=True)
        (generation_dir / method / "lig_ok.sdf").write_text("placeholder")
    manifest = pd.DataFrame(
        [
            {"mol_id": "lig_ok", "rotatable_bonds": 2, "generation_method": analyzer.METHODS[0]},
            {"mol_id": "lig_zero_rot", "rotatable_bonds": 0, "generation_method": analyzer.METHODS[0]},
            {"mol_id": "lig_missing_sdf", "rotatable_bonds": 3, "generation_method": analyzer.METHODS[0]},
        ]
    )
    chembl_map = pd.DataFrame(
        {"ligand_id": ["lig_ok", "lig_zero_rot", "lig_missing_sdf", "lig_not_in_manifest"]}
    )
    selected, excluded = analyzer.select_analysis_mol_ids(chembl_map, generation_dir, manifest)
    assert selected == ["lig_ok"]
    assert excluded["zero_rotatable_bonds"] == ["lig_zero_rot"]
    assert excluded["missing_generation_sdf"] == ["lig_missing_sdf"]
    assert excluded["not_in_manifest"] == ["lig_not_in_manifest"]


def test_load_casf_ligand_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="CASF ligand missing not found"):
        analyzer.load_casf_ligand("missing", tmp_path)


def test_try_load_casf_ligand_returns_none_when_missing(tmp_path):
    assert analyzer.try_load_casf_ligand("missing", tmp_path) is None


def test_clash_passes_marks_fail_instead_of_raising(monkeypatch):
    class BadMol:
        pass

    mols = [BadMol(), BadMol()]
    monkeypatch.setattr(analyzer, "get_dg_bounds", lambda _mol: (_ for _ in ()).throw(RuntimeError("bad bounds")))

    assert analyzer.clash_passes(mols) == [False, False]


def test_validate_metrics_completeness_allows_missing_casf_opt():
    rows = [
        {
            "mol_id": "lig_a",
            "source": source,
            "family": analyzer.source_metadata(source)[0],
            "stage": analyzer.source_metadata(source)[1],
            "tier": analyzer.source_metadata(source)[2],
            "conformer_count": 1,
            "chembl_load_status": "ok" if source in {"chembl3d_gt", "chembl3d_gt_pb"} else "",
            "pb_check_fail_counts_json": "{}",
            "pb_check_fail_rates_json": "{}",
            "casf_best_rmsd": 0.5,
            "casf_median_rmsd": 0.5,
            "casf_hit_0p25": 0.0,
            "casf_hit_0p5": 1.0,
            "casf_hit_0p75": 1.0,
            "casf_hit_2p0": 1.0,
            "casf_opt_best_rmsd": math.nan,
            "casf_opt_median_rmsd": math.nan,
            "casf_opt_hit_0p25": math.nan,
            "casf_opt_hit_0p5": math.nan,
            "casf_opt_hit_0p75": math.nan,
            "casf_opt_hit_2p0": math.nan,
        }
        for source in tuple(analyzer.METHODS) + ("casf_crystal", "chembl3d_sdf", "chembl3d_gt", "chembl3d_gt_pb")
    ]
    metrics = pd.DataFrame(rows)
    chembl_map = pd.DataFrame([{"ligand_id": "lig_a"}])
    analyzer.validate_metrics_completeness(metrics, ["lig_a"], chembl_map)


def test_analyze_source_records_clash_fail_for_bad_bounds(monkeypatch):
    mol = object()
    monkeypatch.setattr(analyzer, "get_dg_bounds", lambda _mol: (_ for _ in ()).throw(RuntimeError("bad bounds")))
    monkeypatch.setattr(
        analyzer,
        "mean_torsion_std_deg",
        lambda _mols: math.nan,
    )
    monkeypatch.setattr(analyzer, "pairwise_rmsd_stats", lambda _mols: {})
    monkeypatch.setattr(analyzer, "greedy_cluster_metrics", lambda _mols: {})
    monkeypatch.setattr(analyzer, "energy_stats", lambda _mols: {})

    row = analyzer.analyze_source(
        "lig_a",
        "casf_opt",
        [mol],
        mol,
        None,
        rotatable_bonds=2,
        run_reference_posebusters=False,
    )
    assert row["clash_input_confs"] == 1
    assert row["clash_pass_confs"] == 0
    assert row["clash_fail_confs"] == 1
    assert row["clash_pass_rate"] == pytest.approx(0.0)


def test_chembl_loader_roundtrip_with_fake_zarr(tmp_path):
    pytest.importorskip("zarr")
    pytest.importorskip("numpy")
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    from molgen3D.pharmacophore.chembl3d_conformer_loader import load_chembl3d_conformers

    topology_dir = tmp_path / "topologies"
    zarr_root = tmp_path / "zarr_database"
    group = "007"
    mol_id = "TEST_0"
    topology_dir.mkdir(parents=True)
    zarr_root.mkdir(parents=True)

    mol = Chem.MolFromSmiles("CC")
    mol = Chem.AddHs(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(idx, Point3D(float(idx), 0.0, 0.0))
    mol.AddConformer(conf)
    mol.SetProp("mol_id", mol_id)

    writer = Chem.SDWriter(str(topology_dir / f"{group}.sdf"))
    writer.write(mol)
    writer.close()

    import numpy as np
    import zarr

    group_path = zarr_root / group
    mol_id_arr = zarr.open_array(
        str(group_path / "mol_id"),
        mode="w",
        shape=(3,),
        dtype="S10",
        chunks=(1,),
    )
    coord_arr = zarr.open_array(
        str(group_path / "coord"),
        mode="w",
        shape=(3, mol.GetNumAtoms(), 3),
        dtype="f4",
        chunks=(1, mol.GetNumAtoms(), 3),
    )
    numbers_arr = zarr.open_array(
        str(group_path / "numbers"),
        mode="w",
        shape=(3, mol.GetNumAtoms()),
        dtype="i4",
        chunks=(1, mol.GetNumAtoms()),
    )
    encoded = mol_id.encode("utf-8")
    for row, row_mol_id in enumerate([encoded, b"OTHER_0", encoded]):
        mol_id_arr[row] = row_mol_id
        numbers_arr[row] = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        coords = []
        for idx in range(mol.GetNumAtoms()):
            coords.append([float(idx + row), 0.0, 0.0])
        coord_arr[row] = np.asarray(coords, dtype="f4")

    loaded = load_chembl3d_conformers(group, mol_id, topology_dir, zarr_root)
    assert len(loaded) == 2
    assert loaded[0].GetNumConformers() == 1
    assert loaded[1].GetConformer(0).GetAtomPosition(0).x == pytest.approx(2.0)

    bad_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    bad_numbers[0] = 999
    numbers_arr[2] = bad_numbers
    with pytest.raises(ValueError, match="Atomic numbers row"):
        load_chembl3d_conformers(group, mol_id, topology_dir, zarr_root)
