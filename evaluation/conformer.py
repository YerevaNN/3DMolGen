from rdkit import Chem
from posebusters import PoseBusters
from yaml import safe_load

from posebusters.modules.rmsd import check_rmsd


def get_conformer_statistics(
    mol: Chem.Mol | list[Chem.Mol], config_path: str | None = None
) -> dict:
    """
    Evaluates a molecule along its conformer, accoring to the following metrics:
    - Posebusters validity check

    Args:
    - mol (Chem.Mol): RDKit molecule object, with exactly one conformer at id 0
    """
    if config_path:
        path = config_path
        with open(path, "r") as f:
            config = safe_load(f)
    else:
        config = "mol"

    p = PoseBusters(config=config)
    results = p.bust(mol, full_report=True)

    return {
        "results": results,
    }


def calculate_rmsd(mol: Chem.Mol, gt_mol: Chem.Mol) -> list[float]:
    """
    Calculates the RMSD between the conformer of a molecule and a ground-truth conformer.

    Args:
    - mol (Chem.Mol): RDKit molecule object, with exactly one conformer at id 0
    - gt_mol (Chem.Mol): RDKit molecule object, with at least one conformer, representing some "ground-truth" conformations
    """
    # Ensure that the GT molecule has exactly one conformer
    rmsd = check_rmsd(mol, gt_mol)

    return rmsd["results"]["rmsd"]
