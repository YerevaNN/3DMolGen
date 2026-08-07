import ast
import os
import os.path as osp
import re
import json
import pickle
import random
import numpy as np
import cloudpickle
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
from loguru import logger as log
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
from rdkit import Chem
from rdkit.Geometry import Point3D

from molgen3D.data_processing.smiles_encoder_decoder import (
    BinConfig,
    encode_cartesian_binned,
    encode_cartesian_binned_v2,
    encode_cartesian_v2,
    encode_cartesian_with_config,
)

dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
qm9_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
drugs_types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
               'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
               'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
               'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}

EMBEDDING_REGISTRY = {
    "cartesian_v2": encode_cartesian_v2,
    "cartesian": encode_cartesian_v2,
    "cartesian_binned": encode_cartesian_binned,
    "cartesian_binned_v2": encode_cartesian_binned_v2,
    "uniform_binned": encode_cartesian_with_config,
    "quantile_binned": encode_cartesian_with_config,
}

REVISITED_SPLIT_FILE_MAP = {"train": "train", "valid": "val", "test": "test"}

def encode_cartesian_raw(mol, precision=4):
    """Legacy compatibility wrapper for the enriched representation."""

    return encode_cartesian_v2(mol, precision=precision)

def decode_cartesian_raw(embedded_smiles):
    """
    Reconstruct an RDKit Mol with 3D coordinates from an 'embedded' SMILES string,
    where each atom token is immediately followed by '<x,y,z>'.

    Parameters
    ----------
    embedded_smiles : str
        SMILES string with per-atom coordinates embedded, e.g.
        'Br<1.234,-0.567,2.890>C<1.890,0.123,0.456>(=C<...>/c<...>1c<...>...)'

    Returns
    -------
    mol : rdkit.Chem.Mol
        RDKit molecule with one conformer whose atom positions match the embedded coordinates.
    """
    # Regex: (1) atom-with-coords OR (2) other SMILES tokens
    atom_coord_pattern = r'(\[[^\[\]]+\]|Br|Cl|Si|[BCNOFPSIcbnosprse])<([^>]+)>'
    other_pattern = r'([0-9]+|=|#|-|\+|/|\\|\(|\))'
    regex = re.compile(f'{atom_coord_pattern}|{other_pattern}')
    
    tokens = []
    coords = []
    
    for match in regex.finditer(embedded_smiles):
        atom = match.group(1)
        coord_str = match.group(2)
        other = match.group(3)
        
        if atom:
            # We matched an atom with coords
            tokens.append(atom)
            x_str, y_str, z_str = coord_str.split(',')
            coords.append((float(x_str), float(y_str), float(z_str)))
        elif other:
            # We matched a SMILES syntax token
            tokens.append(other)
    
    # Rebuild clean SMILES by joining tokens
    clean_smiles = ''.join(tokens)
    
    # Create RDKit Mol (implicit Hs, but consistent with encoding where RemoveHs was used)
    mol = Chem.MolFromSmiles(clean_smiles)
    if mol is None:
        raise ValueError("Failed to parse SMILES: " + clean_smiles)
    
    # Build a new conformer and set atom positions
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(idx, Point3D(x, y, z))
    mol.AddConformer(conf, assignId=True)
    
    return mol

def save_processed_pickle(
    split_dir: str,
    geom_smiles: str,
    mols,
) -> str:

    filename = f"{geom_smiles.replace('/', '_')}"[:250] + ".pickle"
    output_path = osp.join(split_dir, filename)

    with open(output_path, "wb") as fh:
        cloudpickle.dump(mols, fh)

    return output_path


def save_grouped_pickle(output_path: str, iso_to_confs: Dict[str, List[Dict[str, Any]]]) -> None:
    parent = osp.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump(iso_to_confs, fh)


def get_embedding_func_and_config(
    embedding_type: str,
    bin_config_path: Optional[str] = None,
):
    bin_config = None
    if embedding_type in ("uniform_binned", "quantile_binned"):
        if bin_config_path is None:
            raise ValueError(
                f"--bin_config_path is required for embedding_type={embedding_type!r}."
            )
        bin_config = BinConfig.load(bin_config_path)
        log.info(
            "Loaded BinConfig from {} | mode={} L={:.4f} H={:.4f} n_bins={}",
            bin_config_path,
            bin_config.mode,
            bin_config.L,
            bin_config.H,
            bin_config.n_bins,
        )
    if embedding_type not in EMBEDDING_REGISTRY:
        raise ValueError(
            f"Unsupported embedding_type '{embedding_type}'. Options: {sorted(EMBEDDING_REGISTRY)}"
        )
    return EMBEDDING_REGISTRY[embedding_type], bin_config


def parse_coordinate_ranges(ranges: str) -> List[Tuple[float, float]]:
    try:
        parsed_ranges = ast.literal_eval(f"[{ranges}]")
        return [tuple(r) for r in parsed_ranges]
    except Exception as exc:
        log.error("Failed to parse ranges: {} | failure={}", ranges, exc)
        return [(-13.0, 13.0), (-13.0, 13.0), (-13.0, 13.0)]


def get_revisited_split_path(
    geom_raw_path: str,
    split_name: str,
    use_centered: bool = False,
) -> str:
    if split_name not in REVISITED_SPLIT_FILE_MAP:
        raise ValueError(f"Unsupported revisited split: {split_name!r}")
    suffix = "_data_centered.pickle" if use_centered else "_data.pickle"
    file_stem = REVISITED_SPLIT_FILE_MAP[split_name]
    return osp.join(geom_raw_path, f"{file_stem}{suffix}")


def copy_single_conformer_mol(mol: Chem.Mol) -> Chem.Mol:
    copied = Chem.Mol(mol)
    if copied.GetNumConformers() > 1:
        conf = Chem.Conformer(copied.GetConformer(0))
        copied.RemoveAllConformers()
        copied.AddConformer(conf, assignId=True)
    return copied


def _get_prop_float(props: Dict[str, Any], key: str) -> Optional[float]:
    if key in props:
        try:
            return float(props[key])
        except Exception:
            return None
    return None


def extract_conf_meta(
    conf_meta: Optional[Dict[str, Any]],
    mol: Chem.Mol,
) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    energy = weight = None
    conf_id = None

    if conf_meta:
        if "totalenergy" in conf_meta:
            energy = conf_meta.get("totalenergy")
        elif "relativeenergy" in conf_meta:
            energy = conf_meta.get("relativeenergy")

        if "boltzmannweight" in conf_meta:
            weight = conf_meta.get("boltzmannweight")
        elif "weight" in conf_meta:
            weight = conf_meta.get("weight")

        if "geom_id" in conf_meta:
            conf_id = conf_meta.get("geom_id")

    if mol is not None:
        mol_props = mol.GetPropsAsDict()
        if energy is None:
            energy = _get_prop_float(mol_props, "totalenergy")
        if weight is None:
            weight = _get_prop_float(mol_props, "boltzmannweight")
        if conf_id is None and "geom_id" in mol_props:
            try:
                conf_id = int(mol_props["geom_id"])
            except Exception:
                conf_id = None

        if mol.GetNumConformers() > 0:
            conf_props = mol.GetConformer().GetPropsAsDict()
            if energy is None:
                energy = _get_prop_float(conf_props, "totalenergy")
            if weight is None:
                weight = _get_prop_float(conf_props, "boltzmannweight")
            if conf_id is None and "geom_id" in conf_props:
                try:
                    conf_id = int(conf_props["geom_id"])
                except Exception:
                    conf_id = None

    try:
        energy = float(energy) if energy is not None else None
    except Exception:
        energy = None

    try:
        weight = float(weight) if weight is not None else None
    except Exception:
        weight = None

    return energy, weight, conf_id

class JsonlSplitWriter:
    def __init__(self, base_dir: str, split_name: str, chunk_size: int = 1_000_000):
        self.split_name = split_name
        self.base_dir = base_dir
        self.chunk_size = chunk_size
        self.buffer: List[str] = []
        self.chunk_idx = 0
        self.total_samples = 0
        os.makedirs(base_dir, exist_ok=True)

    def write(self, samples: Iterable[str]) -> None:
        for sample in samples:
            self.buffer.append(sample)
            self.total_samples += 1
            if len(self.buffer) >= self.chunk_size:
                self._flush()

    def close(self) -> None:
        if self.buffer:
            self._flush()

    def _flush(self) -> None:
        random.shuffle(self.buffer)
        file_path = osp.join(self.base_dir, f"{self.split_name}_data_{self.chunk_idx}.jsonl")
        with open(file_path, "a") as fh:
            fh.writelines(self.buffer)
        self.buffer.clear()
        self.chunk_idx += 1


def filter_mols(
    mol_dict: Dict[str, object],
    failures: Dict[str, int] = defaultdict(int),
    max_confs: int = 30,
) -> List[Chem.Mol]:
    confs = mol_dict["conformers"]
    smiles = mol_dict["smiles"]

    if smiles and "." in smiles:
        failures["dot_in_smiles"] += 1
        return []

    num_confs = len(confs)
    if max_confs is not None:
        num_confs = max_confs

    mol_from_smiles = Chem.MolFromSmiles(smiles)
    if mol_from_smiles is None:
        failures["mol_from_smiles_failed"] += 1
        return []

    selected: List[Chem.Mol] = []
    k = 0
    for conf in confs:
        mol = conf["rd_mol"]
        num_neighbors = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
        if np.max(num_neighbors) > 4:
            failures["large_degree"] += 1
            continue

        selected.append(mol)
        k += 1
        if k == num_confs:
            break

    return selected


def filter_revisited_mols(
    smiles: Optional[str],
    mols: List[Chem.Mol],
    failures: Dict[str, int] = defaultdict(int),
    max_confs: int = 30,
) -> List[Chem.Mol]:
    if smiles and "." in smiles:
        failures["dot_in_smiles"] += 1
        return []

    mol_from_smiles = Chem.MolFromSmiles(smiles) if smiles else None
    if mol_from_smiles is None:
        failures["mol_from_smiles_failed"] += 1
        return []

    num_confs = len(mols)
    if max_confs is not None:
        num_confs = max_confs

    selected: List[Chem.Mol] = []
    k = 0
    for mol in mols:
        if mol is None:
            failures["missing_mol"] += 1
            continue

        num_neighbors = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
        if not num_neighbors or np.max(num_neighbors) > 4:
            failures["large_degree"] += 1
            continue

        selected.append(mol)
        k += 1
        if k == num_confs:
            break

    return selected


def filter_revisited_conformers_keep_dotted(
    smiles: Optional[str],
    mols: List[Chem.Mol],
    failures: Dict[str, int],
) -> List[Tuple[Chem.Mol, int]]:
    if smiles and "." in smiles:
        failures["dot_in_smiles"] += 1

    mol_from_smiles = Chem.MolFromSmiles(smiles) if smiles else None
    if mol_from_smiles is None:
        failures["mol_from_smiles_failed"] += 1
        return []

    selected: List[Tuple[Chem.Mol, int]] = []
    for idx, mol in enumerate(mols):
        if mol is None:
            failures["missing_mol"] += 1
            continue

        num_neighbors = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
        if not num_neighbors or np.max(num_neighbors) > 4:
            failures["large_degree"] += 1
            continue

        selected.append((mol, idx))

    return selected

def load_pkl(file_path: Path) -> object:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return cloudpickle.load(f)

def load_file(file_path: str, mode: str = "r"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    # Add encoding for text modes, but not for binary modes
    if 'b' not in mode:
        with open(file_path, mode, encoding='utf-8', errors='replace') as f:
            return f.read()
    else:
        with open(file_path, mode) as f:
            return f.read()

def load_json(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "r", encoding='utf-8', errors='replace') as f:
        return json.load(f)
