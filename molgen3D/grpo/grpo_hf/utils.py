from molgen3D.utils.utils import load_pkl
from molgen3D.data_processing.utils import decode_cartesian_raw
from molgen3D.utils.utils import load_pkl, load_json, get_best_rmsd
from rdkit import Chem
import os
from loguru import logger

# Global variables
_smiles_mapping = None

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

def load_ground_truths(key_mol_smiles, num_gt=1):
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
    
    # Load the molecule from the pickle file
    geom_smiles = geom_smiles.replace("/", "_")
    try:
        mol_pickle = load_pkl(os.path.join("/nfs/ap/mnt/sxtn2/chem/GEOM_data/rdkit_folder", "drugs", geom_smiles + ".pickle"))
        mol_confs = mol_pickle["conformers"][:num_gt]
        mols = [conf['rd_mol'] for conf in mol_confs]
        return mols
    except Exception as e:
        logger.error(f"Error loading ground truth for {key_mol_smiles}: {e}")
        return None

def get_rmsd(ground_truth, generated_conformer, align=False):
    generated_mol = decode_cartesian_raw(generated_conformer)
    rmsd = get_best_rmsd(ground_truth, generated_mol, use_alignmol=False)
    return rmsd

def setup_logging(output_dir: str):
    """Setup logging for the run."""
    # Remove all existing handlers
    logger.remove()
    
    # Add file handler
    log_file = os.path.join(output_dir, "run.log")
    logger.add(log_file, rotation="100 MB", enqueue=True)
    
    # Add console handler
    logger.add(lambda msg: print(msg), level="INFO", enqueue=True)
