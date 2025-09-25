from molgen3D.utils.utils import load_pkl
from molgen3D.data_processing.utils import decode_cartesian_raw
from molgen3D.utils.utils import load_pkl, load_json, get_best_rmsd
from rdkit import Chem
import os
from loguru import logger
import numpy as np
import torch

# Global variables
_smiles_mapping = None
_geom_data_path = None

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
    geom_smiles = _smiles_mapping.get(key_mol_smiles)
    if geom_smiles is None:
        logger.warning(f"No mapping found for SMILES: {key_mol_smiles}")
        return None
    
    filename = geom_smiles.replace("/", "_").replace("|", "_")

    try:
        mol_pickle = load_pkl(os.path.join("/nfs/ap/mnt/sxtn2/chem/GEOM_data/rdkit_folder", "drugs", filename + ".pickle"))
        conformers: list = mol_pickle["conformers"]
        if len(conformers) > 1:
            rng = np.random.default_rng()
            rng.shuffle(conformers)
        mol_confs = conformers[:num_gt]
        del mol_pickle
        mols = [conf['rd_mol'] for conf in mol_confs]
        return mols
    except Exception as e:
        logger.error(f"Error loading ground truth for {key_mol_smiles} {geom_smiles}: {e}")
        return None

def get_rmsd(ground_truths, generated_conformer, align: bool = False) -> float:
    try:
        generated_mol = decode_cartesian_raw(generated_conformer)
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
    """Create a minimal code snapshot containing only necessary files."""
    import subprocess
    
    logger.info(f"Creating code snapshot in {snapshot_dir}")
    
    # Copy only the molgen3D package structure needed for imports
    subprocess.run([
        "rsync", "-av", "--exclude=__pycache__", "--exclude=*.pyc",
        "--include=molgen3D/", "--include=molgen3D/__init__.py",
        "--include=molgen3D/grpo/", "--include=molgen3D/grpo/**",
        "--include=molgen3D/utils/", "--include=molgen3D/utils/**",
        "--exclude=*",
        f"{project_root}/", f"{snapshot_dir}/"
    ], check=True)

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

def save_config(config, output_dir: str):
    """Save configuration to output directory."""
    import yaml
    
    config_copy_path = os.path.join(output_dir, "config.yaml")
    with open(config_copy_path, "w") as f:
        yaml.safe_dump(dataclass_to_dict(config), f)
    logger.info(f"Saved updated config file to {config_copy_path}")
