from __future__ import annotations

from pathlib import Path

from molgen3D.pharmacophore.conformer_sets_layout import (
    DEFAULT_CONFORMER_SETS_ROOT,
    DEFAULT_GENERATION_DIR,
    resolve_generation_dir,
    resolve_geometric_paths,
    resolve_paths,
)


def test_resolve_paths_for_current_dataset_layout() -> None:
    paths = resolve_paths(DEFAULT_CONFORMER_SETS_ROOT)
    assert paths.generation_dir == DEFAULT_GENERATION_DIR
    assert paths.analysis_dir == DEFAULT_CONFORMER_SETS_ROOT / "analysis"
    assert paths.tables_dir == paths.analysis_dir / "tables"
    assert paths.cache_dir == paths.analysis_dir / "cache"
    assert paths.manifest_path.name == "manifest.tsv"
    assert paths.manifest_path.exists()


def test_resolve_generation_dir_accepts_root_or_generation(tmp_path: Path) -> None:
    root = tmp_path / "conformer_sets_all"
    generation = root / "generation"
    generation.mkdir(parents=True)
    assert resolve_generation_dir(root) == generation
    assert resolve_generation_dir(generation) == generation


def test_resolve_generation_dir_legacy_flat_layout(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    (root / "rdkit_random_raw_fixed").mkdir(parents=True)
    assert resolve_generation_dir(root) == root


def test_resolve_geometric_paths_v2_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    generation = root / "generation"
    generation.mkdir(parents=True)
    (generation / "rdkit_random_raw_fixed").mkdir()
    paths = resolve_geometric_paths(root)
    assert paths.generation_dir == generation
    assert paths.analysis_dir == root / "analysis"
    assert paths.report_md == root / "analysis" / "geometric_report.md"
    assert paths.ligand_metrics_parts_dir == root / "analysis" / "cache" / "geometric_requested_parts"
    assert paths.chembl_mol_cache_dir == root / "analysis" / "cache" / "chembl3d_mols"
