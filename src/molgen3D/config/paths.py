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


def get_base_path(key: str) -> Path:
    cfg = _cfg()
    base_paths = cfg.get("base_paths", {})
    if key not in base_paths:
        raise KeyError(f"Unknown base path '{key}', available: {sorted(base_paths.keys())}")
    return _abs(base_paths[key])


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

    return _abs(base) / rel


def get_pretrain_dump_path(folder: str | Path) -> Path:
    """Return the path under `pretrain_results_root` for the provided dump folder."""
    folder_path = Path(folder)
    if folder_path.is_absolute():
        return folder_path

    base = get_base_path("pretrain_results_root")
    return base / folder_path