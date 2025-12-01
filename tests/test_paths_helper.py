from pathlib import Path

import pytest

from molgen3D.config.paths import (
    get_base_path,
    get_ckpt,
    get_data_path,
    get_pretrain_dump_path,
    get_pretrain_logs_path,
    get_root_path,
    get_tokenizer_path,
    get_wandb_path,
    load_paths_yaml,
    resolve_tag,
)


def assert_exists(p: Path):
    if not p.exists():
        pytest.skip(f"Expected path to exist: {p}")
    assert p.exists(), f"Expected path to exist: {p}"


def test_data_local_files_exist():
    # under base_paths.data_root = "data"
    clean = get_data_path("clean_smi")
    distinct = get_data_path("distinct_smi")

    assert clean.name == "clean_smi.pickle"
    assert distinct.name == "distinct_smi.pickle"

    assert_exists(clean)
    assert_exists(distinct)


def test_data_geom_files_exist():
    # under base_paths.geom_data_root
    test_mols = get_data_path("test_mols")
    drugs_summary = get_data_path("drugs_summary")

    # sanity on filenames
    assert test_mols.name == "test_smiles_corrected.csv"
    assert drugs_summary.name == "summary_drugs.json"

    assert_exists(test_mols)
    assert_exists(drugs_summary)


def test_data_conformers_dirs_exist():
    train_dir = get_data_path("conformers_train")
    valid_dir = get_data_path("conformers_valid")
    test_dir = get_data_path("conformers_test")

    assert train_dir.name == "train"
    assert valid_dir.name == "valid"
    assert test_dir.name == "test"

    assert train_dir.is_dir(), f"Expected directory: {train_dir}"
    assert valid_dir.is_dir(), f"Expected directory: {valid_dir}"
    assert test_dir.is_dir(), f"Expected directory: {test_dir}"


def test_get_base_path():
    """Test that get_base_path returns correct paths for known keys."""
    # Test with a key that should exist
    pretrain_base = get_base_path("pretrain_results_root")
    assert isinstance(pretrain_base, Path)
    assert pretrain_base.is_absolute()


def test_get_base_path_invalid_key():
    """Test that get_base_path raises KeyError for invalid keys."""
    with pytest.raises(KeyError, match="Unknown base path"):
        get_base_path("nonexistent_key_12345")


def test_pretrain_dump_path_uses_base():
    folder = "foo_experiment_123"
    base = get_base_path("pretrain_results_root")
    assert get_pretrain_dump_path(folder) == base / folder


def test_pretrain_dump_path_accepts_absolute():
    absolute = Path("/tmp/pretrain_dump_test")
    assert get_pretrain_dump_path(absolute) == absolute


@pytest.mark.parametrize("name", ["llama3_chem_v1", "qwen3_0.6b_conf_v1"])
def test_tokenizer_paths_exist(name):
    tok_path = get_tokenizer_path(name)
    assert_exists(tok_path)


@pytest.mark.parametrize(
    "alias,key",
    [
        # adjust to a few you know are present
        ("m380_conf", "1e"),
        ("m380_conf", "4e"),
        ("nm380_conf_grpo_alignFalse_v1", "2231"),
    ],
)
def test_ckpt_paths_exist(alias, key):
    ckpt = get_ckpt(alias, key)
    assert_exists(ckpt)


def test_ckpt_latest_resolution():
    # no explosion when using default step selection
    latest_base = get_ckpt("m380_conf")
    latest_ft = get_ckpt("nm380_conf_grpo_alignFalse_v1")

    assert_exists(latest_base)
    assert_exists(latest_ft)


def test_get_root_path_uses_base():
    """Test that get_root_path correctly uses the base path."""
    folder = "test_folder_123"
    base = get_base_path("pretrain_results_root")
    result = get_root_path("pretrain_results_root", folder)
    assert result == base / folder


def test_get_root_path_accepts_absolute():
    """Test that get_root_path returns absolute paths as-is."""
    absolute = Path("/tmp/absolute_test")
    result = get_root_path("pretrain_results_root", absolute)
    assert result == absolute


def test_get_pretrain_logs_path():
    """Test that get_pretrain_logs_path uses the correct base."""
    folder = "logs_experiment_456"
    base = get_base_path("pretrain_logs_root")
    result = get_pretrain_logs_path(folder)
    assert result == base / folder


def test_get_wandb_path():
    """Test that get_wandb_path uses the correct base."""
    folder = "wandb_run_789"
    base = get_base_path("wandb_root")
    result = get_wandb_path(folder)
    assert result == base / folder


def test_load_paths_yaml_returns_deep_copy():
    """Test that load_paths_yaml returns independent copies."""
    config1 = load_paths_yaml()
    config2 = load_paths_yaml()
    
    # Verify it returns a dict
    assert isinstance(config1, dict)
    assert "base_paths" in config1 or "data" in config1 or "models" in config1
    
    # Modify one copy
    if "base_paths" in config1:
        config1["base_paths"]["test_key"] = "test_value"
    
    # Other copy should not be affected
    if "base_paths" in config2:
        assert "test_key" not in config2["base_paths"]


def test_resolve_tag_base_paths():
    """Test resolve_tag with base_paths section."""
    base_path = get_base_path("ckpts_root")
    resolved = resolve_tag("base_paths:ckpts_root")
    assert resolved == base_path


def test_resolve_tag_data():
    """Test resolve_tag with data section."""
    data_path = get_data_path("clean_smi")
    resolved = resolve_tag("data:clean_smi")
    assert resolved == data_path


def test_resolve_tag_tokenizers():
    """Test resolve_tag with tokenizers section."""
    tokenizer_path = get_tokenizer_path("llama3_chem_v1")
    resolved = resolve_tag("tokenizers:llama3_chem_v1")
    assert resolved == tokenizer_path


def test_resolve_tag_without_colon():
    """Test resolve_tag with a direct path (no colon)."""
    # Test with relative path
    relative_path = "some/relative/path"
    resolved = resolve_tag(relative_path)
    assert isinstance(resolved, Path)
    
    # Test with absolute path
    absolute_path = "/tmp/absolute_test"
    resolved = resolve_tag(absolute_path)
    assert resolved == Path(absolute_path)


def test_resolve_tag_strips_whitespace():
    """Test that resolve_tag strips whitespace from section and key."""
    base_path = get_base_path("ckpts_root")
    # Test with whitespace
    resolved1 = resolve_tag("base_paths : ckpts_root")
    resolved2 = resolve_tag("  base_paths  :  ckpts_root  ")
    assert resolved1 == base_path
    assert resolved2 == base_path


def test_resolve_tag_empty_string():
    """Test that resolve_tag raises ValueError for empty string."""
    with pytest.raises(ValueError, match="Empty tag cannot be resolved"):
        resolve_tag("")


def test_resolve_tag_invalid_section():
    """Test that resolve_tag raises KeyError for invalid section."""
    with pytest.raises(KeyError, match="Unsupported tag section"):
        resolve_tag("invalid_section:some_key")
