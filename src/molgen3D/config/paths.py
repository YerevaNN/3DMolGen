from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import importlib.resources as pkg_resources
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


@lru_cache(maxsize=1)
def _cfg() -> dict:
    paths_file = pkg_resources.files("molgen3D.config").joinpath("paths.yaml")
    with paths_file.open("r") as f:
        return yaml.safe_load(f) or {}


def get_ckpt(alias: str, key: str | None = None) -> Path:
    cfg = _cfg()
    base_paths = cfg.get("base_paths", {})
    models = cfg.get("models", {})

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

    if "code_snapshot" in root_rel or "grpo_outputs" in root_rel:
        base = base_paths.get("grpo_outputs_root", ".")
    elif root_rel.startswith("2025-"):
        base = base_paths.get("grpo_root", base_paths.get("ckpts_root", "."))
    else:
        base = base_paths.get("ckpts_root", ".")

    return _abs(base) / root_rel / step_rel


def get_tokenizer_path(name: str) -> Path:
    cfg = _cfg()
    toks = cfg.get("tokenizers", {})
    if name not in toks:
        raise KeyError(f"Unknown tokenizer '{name}', available: {sorted(toks.keys())}")
    return _abs(toks[name])


def get_data_path(key: str) -> Path:
    """
    data[key] is interpreted as relative to base_paths.data_root,
    unless it's absolute.
    """
    cfg = _cfg()
    data_cfg = cfg.get("data", {})
    if key not in data_cfg:
        raise KeyError(f"Unknown data key '{key}', available: {sorted(data_cfg.keys())}")

    path = Path(data_cfg[key])
    if path.is_absolute():
        return path

    data_root = cfg.get("base_paths", {}).get("data_root", ".")
    return _abs(data_root) / path

def get_base_path(key: str) -> Path:
    cfg = _cfg()
    base_paths = cfg.get("base_paths", {})
    return _abs(base_paths[key])