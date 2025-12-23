"""RDKit utility functions - imported locally to avoid pickling issues."""

from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import RemoveHs


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


def _best_rmsd(probe, ref, use_alignmol: bool):
    """Calculate RMSD between two molecules."""
    from rdkit.Chem import rdMolAlign as MA

    try:
        if use_alignmol:
            return float(MA.AlignMol(probe, ref))
        return float(MA.GetBestRMS(probe, ref))
    except Exception:
        import numpy as np
        return np.nan


def compute_key_matrix(
    key: str, true_confs: List, gen_mols: List, use_alignmol: bool
) -> Tuple[str, Dict[str, object], bool]:
    """Compute RMSD matrix for a single molecule key (sequential version).

    This function is in rdkit_utils (not run_eval) to enable pickling
    when used with ProcessPoolExecutor inside submitit jobs.

    Args:
        key: SMILES key identifying the molecule
        true_confs: List of ground truth conformers
        gen_mols: List of generated conformers
        use_alignmol: Whether to use AlignMol instead of GetBestRMS

    Returns:
        Tuple of (key, results_dict, all_nan_flag)
    """
    n_true = len(true_confs)
    n_gen = len(gen_mols)
    mat = np.full((n_true, n_gen), np.nan, dtype=float)
    for i_true, ref_mol in enumerate(true_confs):
        row = np.array(
            [_best_rmsd(gen_mol, ref_mol, use_alignmol) for gen_mol in gen_mols],
            dtype=float,
        )
        if row.shape == (n_gen,):
            mat[i_true] = row
    all_nan = bool(np.isnan(mat).all())
    return key, {"n_true": n_true, "n_model": n_gen, "rmsd": mat}, all_nan

