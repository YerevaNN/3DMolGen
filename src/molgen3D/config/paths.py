from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
import importlib.resources as pkg_resources
import copy
import yaml


_ENV_REPO_ROOT = os.environ.get("MOLGEN3D_REPO_ROOT")
_ENV_PROJECT_ROOT = os.environ.get("MOLGEN3D_PROJECT_ROOT")
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
    "pretokenized_prompts",
    "validation_pickle",
    "binned_conformers_train",
    "binned_conformers_valid",
    "binned_conformers_test",
    "filtered_conformers_train",
    "filtered_conformers_valid",
    "filtered_conformers_test",
    "binned_stripped_conformers_train",
    "binned_stripped_conformers_valid",
    "binned_stripped_conformers_test",
}


def _path_candidate_roots() -> list[Path]:
    """Return ordered roots used when resolving relative paths."""
    roots: list[Path] = []

    def _add_root(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in roots:
            roots.append(resolved)

    if _ENV_PROJECT_ROOT:
        _add_root(Path(_ENV_PROJECT_ROOT).expanduser())

    _add_root(REPO_ROOT)
    for ancestor in REPO_ROOT.parents[:2]:
        _add_root(ancestor)

    return roots


def _absolute_path_candidates(value: str | Path) -> list[Path]:
    """Return the absolute paths to try for a single candidate."""
    candidate = Path(value)
    if candidate.is_absolute():
        return [candidate]

    resolved: list[Path] = []
    seen: set[Path] = set()
    for root in _path_candidate_roots():
        path = (root / candidate).resolve()
        if path in seen:
            continue
        seen.add(path)
        resolved.append(path)

    if not resolved:
        return [candidate]
    return resolved


def _as_path_candidates(value: str | Path | Sequence[str | Path]) -> list[str | Path]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Path)):
        flattened: list[str | Path] = []
        for item in value:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, Path)):
                flattened.extend(list(item))
            else:
                flattened.append(item)
        return flattened
    return [value]


def _resolve_path_value(value: str | Path | Sequence[str | Path]) -> Path:
    """
    Resolve a config path value that may include fallback candidates.
    The first existing candidate is returned; otherwise, the first candidate.
    """
    candidates = _as_path_candidates(value)
    if not candidates:
        raise ValueError("Cannot resolve an empty set of path candidates")

    resolved: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        for candidate_path in _absolute_path_candidates(candidate):
            if candidate_path in seen:
                continue
            seen.add(candidate_path)
            resolved.append(candidate_path)
            if candidate_path.exists():
                return candidate_path

    return resolved[0]


@lru_cache(maxsize=1)
def _cfg() -> dict:
    """Load and cache the paths.yaml configuration file."""
    repo_path = REPO_ROOT / "src" / "molgen3D" / "config" / "paths.yaml"
    if repo_path.exists():
        with repo_path.open("r") as f:
            return yaml.safe_load(f) or {}

    paths_file = pkg_resources.files("molgen3D.config").joinpath("paths.yaml")
    with paths_file.open("r") as f:
        return yaml.safe_load(f) or {}


def _get_config_section(section: str) -> dict:
    """Get a section from the config, returning an empty dict if missing."""
    return _cfg().get(section, {})


def _get_ckpt_base_path(root_rel: str, base_paths: dict) -> Path:
    """Determine the base path for a checkpoint based on root_rel pattern."""

    def _resolve_from_keys(*keys: str, default: str = ".") -> Path:
        for key in keys:
            if key in base_paths:
                return _resolve_path_value(base_paths[key])
        return _resolve_path_value(default)

    if root_rel.startswith("qwen3_06b"):
        return _resolve_from_keys("qwen_yerevann_root", "hf_yerevann_root")
    if "qwen3" in root_rel:
        return _resolve_from_keys("qwen3_grpo_root", "grpo_root")
    if "code_snapshot" in root_rel or "grpo_outputs" in root_rel:
        return _resolve_from_keys("grpo_outputs_root")
    if root_rel.startswith("2025-"):
        return _resolve_from_keys("grpo_root", "ckpts_root")
    return _resolve_from_keys("hf_yerevann_root", default=".")


def _resolve_direct_path(value: str | Path) -> Path:
    """Resolve a single, possibly relative path without a section tag."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return _absolute_path_candidates(candidate)[0]


def load_paths_yaml() -> dict:
    """
    Return a deep copy of the parsed paths.yaml so callers can inspect sections
    without risking shared-state mutations.
    """
    return copy.deepcopy(_cfg())


_DATA_ROOT_KEYS = (
    "ckpts_root",
    "grpo_root",
    "qwen3_grpo_root",
    "hf_yerevann_root",
    "qwen_yerevann_root",
    "geom_data_root",
    "data_root",
    "project_root",
)


def _normalize_data_root(candidate: str | Path) -> Path:
    """Return an absolute resolved data root."""
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=False)


def _collect_data_roots() -> list[Path]:
    """Collect ordered roots to try when resolving data paths."""
    base_paths = load_paths_yaml().get("base_paths", {})
    roots: list[Path] = []
    seen: set[Path] = set()

    for key in _DATA_ROOT_KEYS:
        candidates = _as_path_candidates(base_paths.get(key))
        for candidate in candidates:
            path = Path(candidate)
            normalized = _normalize_data_root(path)
            if normalized not in seen:
                seen.add(normalized)
                roots.append(normalized)

    if not roots:
        roots.append(REPO_ROOT.resolve(strict=False))

    return roots


def _resolve_relative_data_path(relative: Path, roots: list[Path]) -> Path | None:
    for root in roots:
        for variant in _relative_variants(relative):
            candidate = (root / variant).resolve(strict=False)
            if candidate.exists():
                return candidate
    return None


def _resolve_absolute_data_path(absolute: Path, roots: list[Path]) -> Path | None:
    for root in roots:
        try:
            rel = absolute.relative_to(root)
        except ValueError:
            continue
        for fallback in roots:
            for variant in _relative_variants(rel):
                candidate = (fallback / variant).resolve(strict=False)
                if candidate.exists():
                    return candidate
    return None


def _relative_variants(relative: Path) -> list[Path]:
    variants = [relative]
    if relative.parts and relative.parts[0] == "geom_processed":
        remainder = Path(*relative.parts[1:])
        if remainder not in variants:
            variants.append(remainder)
    if relative.parts and relative.parts[0] == "3DMolGen_data":
        remainder = Path("data", *relative.parts[1:])
        if remainder not in variants:
            variants.append(remainder)
    if relative.parts and relative.parts[0] == "data":
        remainder = Path(*relative.parts[1:])
        if remainder not in variants:
            variants.append(remainder)
    if not (relative.parts and relative.parts[0] == "geom_processed"):
        prefixed = Path("geom_processed") / relative
        if prefixed not in variants:
            variants.append(prefixed)
    if not (relative.parts and relative.parts[0] == "3DMolGen_data"):
        prefixed = Path("3DMolGen_data") / relative
        if prefixed not in variants:
            variants.append(prefixed)
    if not (relative.parts and relative.parts[0] == "data"):
        prefixed = Path("data") / relative
        if prefixed not in variants:
            variants.append(prefixed)
    return variants


def resolve_data_path(value: str | Path) -> Path:
    """
    Resolve a data file path against configured base roots.
    """
    candidate = Path(value).expanduser()
    if candidate.exists():
        return candidate.resolve(strict=False)

    roots = _collect_data_roots()
    if candidate.is_absolute():
        resolved = _resolve_absolute_data_path(candidate.resolve(strict=False), roots)
        if resolved:
            return resolved
    else:
        resolved = _resolve_relative_data_path(candidate, roots)
        if resolved:
            return resolved

    return candidate.resolve(strict=False)


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

    return base / root_rel / step_rel


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
    return _resolve_path_value(tokenizers[name])


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
    return _resolve_path_value(base_paths[key])


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
    
    rel_candidates = _as_path_candidates(data_cfg[key])
    if not rel_candidates:
        raise ValueError(f"No data path candidates defined for '{key}'")

    base_paths = _get_config_section("base_paths")

    def _base_candidate_values(base_key: str) -> list[str | Path]:
        value = base_paths.get(base_key)
        if value is not None:
            return _as_path_candidates(value)
        if base_key == "geom_data_root":
            return _base_candidate_values("data_root")
        return ["."]

    default_path: Path | None = None
    for rel_candidate in rel_candidates:
        rel_path = Path(rel_candidate)
        if rel_path.is_absolute():
            if default_path is None:
                default_path = rel_path
            if rel_path.exists():
                return rel_path
            continue

        rel_str = str(rel_candidate)
        base_key = (
            "geom_data_root"
            if key in GEOM_DATA_KEYS or rel_str.startswith(("geom_processed", "rdkit_folder"))
            else "data_root"
        )

        for base_value in _base_candidate_values(base_key):
            for base_path in _absolute_path_candidates(base_value):
                candidate_path = base_path / rel_path
                if default_path is None:
                    default_path = candidate_path
                if candidate_path.exists():
                    return candidate_path

    if default_path is not None:
        return default_path

    return Path(rel_candidates[0])


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


def get_ckpt_tag_path(key: str) -> Path:
    """
    Resolve a checkpoint alias defined under the `ckpts` section of paths.yaml.

    The key format is `<alias>` or `<alias>/<subpath>`, where `alias` maps to an
    absolute (or repo-relative) directory. Any trailing subpath is appended to
    that base.
    """
    ckpt_cfg = _get_config_section("ckpts")
    if not ckpt_cfg:
        raise KeyError("No 'ckpts' section defined in paths.yaml.")

    normalized = key.strip()
    if not normalized:
        raise KeyError("Empty ckpts key cannot be resolved.")

    alias, sep, remainder = normalized.partition("/")
    alias = alias.strip()
    base = ckpt_cfg.get(alias)
    if base is None:
        raise KeyError(
            f"Unknown ckpts alias '{alias}', available: {sorted(ckpt_cfg.keys())}"
        )
    base_path = _resolve_path_value(base)
    return base_path / remainder if sep else base_path


def resolve_tag(tag: str) -> Path:
    """
    Resolve a structured tag like "base_paths:ckpts_root" into an absolute path.
    
    Supported sections: base_paths, data, tokenizers, ckpts.
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
        return candidate if candidate.is_absolute() else _resolve_direct_path(candidate)

    section, key = tag.split(":", 1)
    section = section.strip()
    key = key.strip()

    section_handlers = {
        "base_paths": get_base_path,
        "data": get_data_path,
        "tokenizers": get_tokenizer_path,
        "ckpts": get_ckpt_tag_path,
    }

    handler = section_handlers.get(section)
    if handler is None:
        raise KeyError(
            f"Unsupported tag section '{section}' in '{tag}'. "
            f"Expected one of: {', '.join(section_handlers.keys())}."
        )

    return handler(key)
