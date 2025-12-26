from numpy.random import f
import logging
from contextlib import contextmanager
from molgen3D.data_processing.smiles_encoder_decoder import decode_cartesian_v2
from molgen3D.utils.utils import get_best_rmsd, load_json, load_pkl
from pathlib import Path
from rdkit import Chem, RDLogger
import os
from loguru import logger
import numpy as np
import torch
import sys
import types

# Global variables
_smiles_mapping = None
_geom_data_path = None

@contextmanager
def _suppress_rdkit_pickle_warnings():
    """Temporarily raise RDKit logger level to suppress pickle version warnings."""
    rd_logger = RDLogger.logger()
    previous_level = rd_logger.level
    rd_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        rd_logger.setLevel(previous_level)

def load_smiles_mapping(mapping_path: str) -> None:
    """Load the SMILES mapping from a JSON file.
    
    Args:
        mapping_path: Path to the JSON file containing the SMILES mapping
    """
    global _smiles_mapping
    if _smiles_mapping is None:
        _smiles_mapping = load_json(mapping_path)
        logger.info(f"Loaded SMILES mapping with {len(_smiles_mapping)} entries")

def set_geom_data_path(path: str) -> None:
    """Set the path to the GEOM data folder.
    
    Args:
        path: Path to the GEOM data folder
    """
    global _geom_data_path
    _geom_data_path = path
    logger.info(f"Set GEOM data path to: {path}")

def remove_chiral_info(mol):
    """
    Removes chiral center information from an RDKit molecule while preserving
    cis/trans (bond) stereochemistry.
    """
    for atom in mol.GetAtoms():
        if atom.HasProp("_ChiralityPossible") or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            # You might also want to clear the _CIPCode property if it exists,
            # though setting CHI_UNSPECIFIED is usually sufficient.
            if atom.HasProp("_CIPCode"):
                atom.ClearProp("_CIPCode")
    return mol

def load_ground_truths(key_mol_smiles, num_gt: int = 16):
    """Load ground truth conformers for a given canonical SMILES.

    Args:
        key_mol_smiles: Canonical SMILES string
        num_gt: Number of ground truth conformers to load

    Returns:
        List of RDKit molecules representing ground truth conformers
    """

    # Get the original GEOM SMILES from the mapping
    filepath = _smiles_mapping.get(key_mol_smiles)
    if filepath is None:
        logger.error("Missing SMILES mapping for pad key %s", key_mol_smiles)
        return None

    conformers = None
    try:
        with _suppress_rdkit_pickle_warnings():
            conformers = load_pkl(Path(filepath))
        return conformers
    except FileNotFoundError as e:
        logger.error(
            "Ground-truth pickle missing for %s at %s: %s",
            key_mol_smiles,
            filepath,
            e,
        )
    except Exception as e:
        logger.error(
            "Error loading ground truth for %s at %s: %s\n%r",
            key_mol_smiles,
            filepath,
            e,
            conformers,
        )
    return None

def get_rmsd(ground_truths, generated_conformer, align: bool = False) -> float:
    try:
        generated_mol = decode_cartesian_v2(generated_conformer)
        rmsds = []
        for ground_truth in ground_truths:
            try:
                rmsd = get_best_rmsd(generated_mol, ground_truth, use_alignmol=align)
                rmsds.append(rmsd)
            except Exception as e:
                logger.error(f"Error getting RMSD: {Chem.MolToSmiles(ground_truth, canonical=True)}\n{generated_conformer}\n{e}")
                rmsds.append(float('nan'))
        if not rmsds:
            return float('nan'), -1
        if not rmsds or np.all(np.isnan(rmsds)):
            return float('nan'), -1
        min_index = int(np.nanargmin(rmsds))
        return float(rmsds[min_index]), min_index
    except Exception as e:
        logger.error(f"Error in get_rmsd: {e}\n{generated_conformer}")
        return float('nan'), -1

def setup_logging(output_dir: str, log_level: str = "INFO"):
    """Setup logging for the run."""
    # Remove all existing handlers
    logger.remove()
    
    # Add file handler
    log_file = os.path.join(output_dir, "run.log")
    logger.add(log_file, rotation="100 MB", enqueue=True, format="{time:HH:mm} | {level} | {message}")
    
    # Add console handler
    logger.add(lambda msg: print(msg), level=log_level, enqueue=True)  

def create_code_snapshot(project_root: str, snapshot_dir: str):
    """Copy the GRPO package and its lightweight dependencies into the snapshot directory."""
    import shutil
    
    logger.info(f"Creating GRPO code snapshot in {snapshot_dir}")
    
    project_root_path = Path(project_root)
    source_root = project_root_path / "src" / "molgen3D"
    if not source_root.exists():
        raise FileNotFoundError(f"Expected source directory missing: {source_root}")

    destination_root = Path(snapshot_dir) / "molgen3D"
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    # Ensure package initializers exist for base modules we depend on.
    for relative_path in [
        "__init__.py",
        "training/__init__.py",
        "data_processing/__init__.py",
        "utils/__init__.py",
        "evaluation/__init__.py",
    ]:
        source_file = source_root / relative_path
        destination_file = destination_root / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        if source_file.exists():
            shutil.copy2(source_file, destination_file)
        else:
            destination_file.touch()

    # Copy the GRPO package plus the specific helper modules it depends on.
    modules_to_copy = [
        ("training/grpo", destination_root / "training" / "grpo"),
        ("data_processing", destination_root / "data_processing"),
        ("utils", destination_root / "utils"),
        ("evaluation", destination_root / "evaluation"),
    ]

    ignore_patterns = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    for relative_path, destination_dir in modules_to_copy:
        source_dir = source_root / relative_path
        if not source_dir.exists():
            logger.warning(f"Skipping missing module during snapshot: {source_dir}")
            continue
        shutil.copytree(
            source_dir,
            destination_dir,
            dirs_exist_ok=True,
            ignore=ignore_patterns,
        )

def dataclass_to_dict(obj):
    """Convert dataclass objects to dictionary recursively."""
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for field in obj.__dataclass_fields__:
            value = getattr(obj, field)
            result[field] = dataclass_to_dict(value)
        return result
    elif isinstance(obj, list):
        return [dataclass_to_dict(i) for i in obj]
    else:
        return obj

def get_torch_dtype(dtype_str: str):
    """Convert string dtype to torch dtype."""
    torch_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return torch_dtype_map.get(dtype_str, torch.bfloat16)


def ensure_torch_serialization_module():
    """Provide a shim for ``torch.utils.serialization`` when it is absent.

    Some PyTorch builds (notably those bundled with certain Liger kernels)
    remove the deprecated ``torch.utils.serialization`` module even though
    parts of ``torch.save`` still attempt to import it. When that happens,
    saving checkpoints fails with ``ModuleNotFoundError``. We expose a thin
    proxy module that forwards attribute lookups to ``torch.serialization``
    so that torch's legacy import continues to work.
    """

    module_name = "torch.utils.serialization"
    if module_name in sys.modules:
        return

    proxy_module = types.ModuleType(module_name)

    def _getattr(name):
        return getattr(torch.serialization, name)

    proxy_module.__getattr__ = _getattr  # type: ignore[attr-defined]
    proxy_module.__all__ = getattr(torch.serialization, "__all__", [])

    sys.modules[module_name] = proxy_module
    setattr(torch.utils, "serialization", proxy_module)

def save_config(config, output_dir: str):
    """Save configuration to output directory."""
    import yaml
    
    config_copy_path = os.path.join(output_dir, "config.yaml")
    with open(config_copy_path, "w") as f:
        yaml.safe_dump(dataclass_to_dict(config), f)
    logger.info(f"Saved updated config file to {config_copy_path}")
