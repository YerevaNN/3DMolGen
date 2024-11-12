from rdkit import Chem
from posebusters import PoseBusters

from posebusters.modules.rmsd import check_rmsd


def get_conformer_statistics(mol: Chem.Mol, gt_mol: Chem.Mol | None = None):
    """
    Evaluates a molecule along its conformer, accoring to the following metrics:
    - Posebusters validity check

    Args:
    - mol (Chem.Mol): RDKit molecule object, with exactly one conformer at id 0
    - gt_mol (optional, Chem.Mol): RDKit molecule object, with at least one conformer, representing some "ground-truth" conformations
    """
    p = PoseBusters(config="mol")
    results = p.bust([mol], full_report=True)

    return {  
        "energy_ratio": results["energy_ratio"].iloc[0],
        "ensemble_avg_energy": results["ensemble_avg_energy"].iloc[0],
        "mol_pred_energy": results["mol_pred_energy"].iloc[0],
        "resuts": results
    }


def calculate_rmsd(mol: Chem.Mol, gt_mol: Chem.Mol):
    """
    Calculates the RMSD between the conformer of a molecule and a ground-truth conformer

    Args:
    - mol (Chem.Mol): RDKit molecule object, with exactly one conformer at id 0
    - gt_mol (Chem.Mol): RDKit molecule object, with at least one conformer, representing some "ground-truth" conformations
    """
    rmsd = check_rmsd(mol, gt_mol)

    return rmsd["results"]["rmsd"]







