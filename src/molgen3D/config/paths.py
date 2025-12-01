from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import importlib.resources as pkg_resources
import copy
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]

# Keys that should use geom_data_root instead of data_root
GEOM_DATA_KEYS = {
    "rdkit_folder",
    "test_mols",
    "drugs_summary",
    "conformers_train",
    "conformers_valid",
    "conformers_test",
}


def _to_absolute_path(p: str | Path) -> Path:
    """Convert a path to absolute, resolving relative paths against REPO_ROOT."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


@lru_cache(maxsize=1)
def _cfg() -> dict:
    """Load and cache the paths.yaml configuration file."""
    paths_file = pkg_resources.files("molgen3D.config").joinpath("paths.yaml")
    with paths_file.open("r") as f:
        return yaml.safe_load(f) or {}


def _get_config_section(section: str) -> dict:
    """Get a section from the config, returning an empty dict if missing."""
    return _cfg().get(section, {})


def _get_ckpt_base_path(root_rel: str, base_paths: dict) -> str:
    """Determine the base path for a checkpoint based on root_rel pattern."""
    if "code_snapshot" in root_rel or "grpo_outputs" in root_rel:
        return base_paths.get("grpo_outputs_root", ".")
    if root_rel.startswith("2025-"):
        return base_paths.get("grpo_root", base_paths.get("ckpts_root", "."))
    return base_paths.get("hf_yerevann_root", ".")


def load_paths_yaml() -> dict:
    """
    Return a deep copy of the parsed paths.yaml so callers can inspect sections
    without risking shared-state mutations.
    """
    return copy.deepcopy(_cfg())


def get_ckpt(alias: str, key: str | None = None) -> Path:
    """
    Get the path to a checkpoint for a given model alias and step key.
    
    Args:
        alias: Model alias from the config
        key: Step key (e.g., "1e", "final"). If None, uses "final" if available,
             otherwise the last step alphabetically.
    
    Returns:
        Absolute path to the checkpoint directory
    """
    models = _get_config_section("models")
    entry = models.get(alias)
    if entry is None:
        raise KeyError(f"Unknown model alias '{alias}'.")

    steps = entry.get("steps") or {}
    if not steps:
        raise KeyError(f"Model '{alias}' has no steps defined.")

    if key is None:
        key = "final" if "final" in steps else sorted(steps.keys())[-1]
    if key not in steps:
        raise KeyError(
            f"Step '{key}' not found for '{alias}', "
            f"available: {sorted(steps.keys())}"
        )

    root_rel = entry["root"]
    step_rel = steps[key]
    base_paths = _get_config_section("base_paths")
    base = _get_ckpt_base_path(root_rel, base_paths)

    return _to_absolute_path(base) / root_rel / step_rel


def get_tokenizer_path(name: str) -> Path:
    cfg = _cfg()
    toks = cfg.get("tokenizers", {})
    if name not in toks:
        raise KeyError(f"Unknown tokenizer '{name}', available: {sorted(toks.keys())}")
    return _to_absolute_path(toks[name])


def get_base_path(key: str) -> Path:
    cfg = _cfg()
    base_paths = cfg.get("base_paths", {})
    if key not in base_paths:
        raise KeyError(f"Unknown base path '{key}', available: {sorted(base_paths.keys())}")
    return _to_absolute_path(base_paths[key])


def get_data_path(key: str) -> Path:
    cfg = _cfg()
    data_cfg = cfg.get("data", {})
    if key not in data_cfg:
        raise KeyError(f"Unknown data path '{key}', available: {sorted(data_cfg.keys())}")
    rel = Path(data_cfg[key])
    if rel.is_absolute():
        return rel

    base_paths = cfg.get("base_paths", {})
    geom_keys = {
        "rdkit_folder",
        "test_mols",
        "drugs_summary",
        "conformers_train",
        "conformers_valid",
        "conformers_test",
    }

    if key in geom_keys or str(rel).startswith(("geom_processed", "rdkit_folder")):
        base = base_paths.get("geom_data_root", base_paths.get("data_root", "."))
    else:
        base = base_paths.get("data_root", ".")

    return _to_absolute_path(base) / rel


def get_root_path(base_key: str, folder: str | Path) -> Path:
    """Return the path under the provided base key for the given folder."""
    folder_path = Path(folder)
    if folder_path.is_absolute():
        return folder_path

    base = get_base_path(base_key)
    return base / folder_path


def get_pretrain_dump_path(folder: str | Path, *, base_key: str = "pretrain_results_root") -> Path:
    """Return the path under `base_key` for the provided dump folder."""
    return get_root_path(base_key, folder)


def get_pretrain_logs_path(folder: str | Path) -> Path:
    return get_root_path("pretrain_logs_root", folder)


def get_wandb_path(folder: str | Path) -> Path:
    return get_root_path("wandb_root", folder)