from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import importlib.resources as pkg_resources
import copy
import yaml


_ENV_REPO_ROOT = os.environ.get("MOLGEN3D_REPO_ROOT")
_CANDIDATE_ROOT = (
    Path(_ENV_REPO_ROOT).expanduser().resolve()
    if _ENV_REPO_ROOT
    else Path(__file__).resolve().parents[3]
)
if not (_CANDIDATE_ROOT / "src" / "molgen3D").exists():
    cwd = Path.cwd().resolve()
    if (cwd / "src" / "molgen3D").exists():
        _CANDIDATE_ROOT = cwd
REPO_ROOT = _CANDIDATE_ROOT

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
    if root_rel.startswith("qwen3_06b"):
        return base_paths.get("qwen_yerevann_root", base_paths.get("hf_yerevann_root", "."))
    if "qwen3" in root_rel:
        return base_paths.get("qwen3_grpo_root", base_paths.get("grpo_root", "."))
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
    """
    Get the path to a tokenizer by name.
    
    Args:
        name: Tokenizer name from the config
    
    Returns:
        Absolute path to the tokenizer directory
    """
    tokenizers = _get_config_section("tokenizers")
    if name not in tokenizers:
        raise KeyError(f"Unknown tokenizer '{name}', available: {sorted(tokenizers.keys())}")
    return _to_absolute_path(tokenizers[name])


def get_base_path(key: str) -> Path:
    """
    Get a base path by key.
    
    Args:
        key: Base path key from the config
    
    Returns:
        Absolute path
    """
    base_paths = _get_config_section("base_paths")
    if key not in base_paths:
        raise KeyError(f"Unknown base path '{key}', available: {sorted(base_paths.keys())}")
    return _to_absolute_path(base_paths[key])


def get_data_path(key: str) -> Path:
    """
    Get a data path by key.
    
    Args:
        key: Data path key from the config
    
    Returns:
        Absolute path to the data file or directory
    """
    data_cfg = _get_config_section("data")
    if key not in data_cfg:
        raise KeyError(f"Unknown data path '{key}', available: {sorted(data_cfg.keys())}")
    
    rel = Path(data_cfg[key])
    if rel.is_absolute():
        return rel

    base_paths = _get_config_section("base_paths")
    if key in GEOM_DATA_KEYS or str(rel).startswith(("geom_processed", "rdkit_folder")):
        base = base_paths.get("geom_data_root", base_paths.get("data_root", "."))
    else:
        base = base_paths.get("data_root", ".")

    return _to_absolute_path(base) / rel


def get_root_path(base_key: str, folder: str | Path) -> Path:
    """
    Return the path under the provided base key for the given folder.
    
    Args:
        base_key: Base path key from the config
        folder: Folder name or path (if absolute, returned as-is)
    
    Returns:
        Absolute path
    """
    folder_path = Path(folder)
    if folder_path.is_absolute():
        return folder_path

    base = get_base_path(base_key)
    return base / folder_path


def get_pretrain_dump_path(folder: str | Path, *, base_key: str = "pretrain_results_root") -> Path:
    """
    Return the path under `base_key` for the provided dump folder.
    
    Args:
        folder: Folder name or path
        base_key: Base path key (default: "pretrain_results_root")
    
    Returns:
        Absolute path
    """
    return get_root_path(base_key, folder)


def get_pretrain_logs_path(folder: str | Path) -> Path:
    """
    Get the path for pretraining logs.
    
    Args:
        folder: Folder name or path
    
    Returns:
        Absolute path
    """
    return get_root_path("pretrain_logs_root", folder)


def get_wandb_path(folder: str | Path) -> Path:
    """
    Get the path for wandb logs.
    
    Args:
        folder: Folder name or path
    
    Returns:
        Absolute path
    """
    return get_root_path("wandb_root", folder)


def resolve_tag(tag: str) -> Path:
    """
    Resolve a structured tag like "base_paths:ckpts_root" into an absolute path.
    
    Supported sections: base_paths, data, tokenizers.
    If no colon is present, treats the tag as a direct path.
    
    Args:
        tag: Tag string in format "section:key" or a direct path
    
    Returns:
        Absolute path
    """
    if not tag:
        raise ValueError("Empty tag cannot be resolved")

    if ":" not in tag:
        candidate = Path(tag)
        return candidate if candidate.is_absolute() else _to_absolute_path(candidate)

    section, key = tag.split(":", 1)
    section = section.strip()
    key = key.strip()

    section_handlers = {
        "base_paths": get_base_path,
        "data": get_data_path,
        "tokenizers": get_tokenizer_path,
    }

    handler = section_handlers.get(section)
    if handler is None:
        raise KeyError(
            f"Unsupported tag section '{section}' in '{tag}'. "
            f"Expected one of: {', '.join(section_handlers.keys())}."
        )

    return handler(key)
