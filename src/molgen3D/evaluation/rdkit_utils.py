"""RDKit utility functions - imported locally to avoid pickling issues."""

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Geometry import Point3D


def clean_confs(smi, confs):
    """Clean conformers by checking SMILES consistency."""
    good_ids = []
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(RemoveHs(c), isomericSmiles=False)
        if conf_smi == smi:
            good_ids.append(i)
    return [confs[i] for i in good_ids]


def correct_smiles(true_confs):
    """Find the most common SMILES from conformers."""
    from statistics import mode, StatisticsError

    conf_smis = []
    for c in true_confs:
        conf_smi = Chem.MolToSmiles(RemoveHs(c))
        conf_smis.append(conf_smi)

    try:
        common_smi = mode(conf_smis)
    except StatisticsError:
        return None  # these should be cleaned by hand

    if sum(common_smi == smi for smi in conf_smis) == len(conf_smis):
        return mode(conf_smis)
    else:
        print('consensus', common_smi)  # these should probably also be investigated manually
        return common_smi


def get_unique_smiles(confs):
    """Get unique SMILES and their counts."""
    from collections import Counter
    smiles = [Chem.MolToSmiles(RemoveHs(c), isomericSmiles=False) for c in confs]
    smiles_count = Counter(smiles)
    return smiles_count


def process_molecules_remove_hs(model_preds):
    """Process molecules by removing hydrogens from conformers."""
    return {smi: [RemoveHs(m) for m in confs] for smi, confs in model_preds.items()}


def _normalize_coords(mol):
    """Center molecule coordinates by subtracting the mean of atom positions.
    
    Equivalent to: feats[mask_bool, V:] -= feats[mask_bool, V:].mean(axis=0)
    """
    mol = Chem.Mol(mol)
    if mol.GetNumConformers() == 0:
        return mol
    conformer = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return mol
    
    coords = np.array([conformer.GetAtomPosition(i) for i in range(n_atoms)])
    mean_coords = coords.mean(axis=0)
    coords -= mean_coords
    
    for i in range(n_atoms):
        conformer.SetAtomPosition(i, Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
    return mol


def _best_rmsd(probe, ref, use_alignmol: bool):
    """Calculate RMSD between two molecules."""
    from rdkit.Chem import rdMolAlign as MA

    try:
        if use_alignmol:
            return float(MA.AlignMol(probe, ref))
        return float(MA.GetBestRMS(probe, ref))
    except Exception:
        return np.nan

