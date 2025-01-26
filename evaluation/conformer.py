import re
import ast
import numpy as np
from rdkit import Chem
from posebusters import PoseBusters
from yaml import safe_load
from copy import deepcopy
from collections import OrderedDict
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

    return rmsd["results"]["kabsch_rmsd"]


def calculate_rmsd_matrix(mol: Chem.Mol, gt_mol: Chem.Mol) -> list[list[float]]:
    """
    Calculates the RMSD between the conformer of a molecule and a ground-truth conformer.

    Args:
    - mol (Chem.Mol): RDKit molecule object, with potentially multiple conformers
    - gt_mol (Chem.Mol): RDKit molecule object, with potentially multiple conformers, representing some "ground-truth" conformations

    returns a matrix of RMSD values, where each row represents a conformer of the GT molecule and each column a conformer of the molecule
    """
   
    gt_mol_placeholder = deepcopy(gt_mol)
    gt_mol_placeholder.RemoveAllConformers()

    mol_placeholder = deepcopy(mol)
    mol_placeholder.RemoveAllConformers()

    rmsd_matrix = []

    for c in gt_mol.GetConformers():
        gt_mol_placeholder.RemoveAllConformers()
        c = Chem.Conformer(c)
        gt_mol_placeholder.AddConformer(c, assignId=True)
        row = []

        for m in mol.GetConformers():
            mol_placeholder.RemoveAllConformers()
            m = Chem.Conformer(m)
            mol_placeholder.AddConformer(m, assignId=True)

            rmsd = check_rmsd(mol_placeholder, gt_mol_placeholder)
            row.append(rmsd["results"]["kabsch_rmsd"])
        rmsd_matrix.append(row)

    return np.array(rmsd_matrix)

def parse_embedded_smiles(smi: str) -> Chem.Mol:
    """
    Parses a SMILES string with embedded conformer coordinates and returns an RDKit Mol object with 3D coordinates.
    The SMILES string includes atomic coordinates in the format:
    AtomSymbol<X, Y, Z>
    Hydrogens may be present without explicit coordinates.
    Args:
        smi (str): The SMILES string with embedded conformer data.
    Returns:
        Chem.Mol: RDKit molecule object with conformer coordinates assigned.
    """
    # Regular expression to match atoms with embedded coordinates
    atom_regex = re.compile(r'([A-Za-z][A-Za-z]?)(?:<([\d\.\-eE,+ ]+)>)?')
    
    tokens = []
    coordinates = []
    index = 0
    length = len(smi)
    
    while index < length:
        match = atom_regex.match(smi, index)
        if match:
            atom, coord_str = match.groups()
            tokens.append(atom)
            if coord_str:
                # Extract X, Y, Z coordinates
                coords = [float(x.strip()) for x in coord_str.split(',')]
                if len(coords) != 3:
                    raise ValueError(f"Invalid coordinate format: {coord_str}")
                coordinates.append(coords)
            else:
                coordinates.append(None)  # Placeholder for atoms without coordinates
            index = match.end()
        else:
            # Handle non-atom characters (e.g., bonds, branches, ring closures)
            tokens.append(smi[index])
            coordinates.append(None)
            index += 1

    # Reconstruct the SMILES without coordinate annotations
    cleaned_smiles = ''
    atom_mapping = []  # Keep track of atom indices in the SMILES
    for idx, token in enumerate(tokens):
        if token.isalpha() or (len(token) > 1 and token[0].isalpha()):
            cleaned_smiles += token
            atom_mapping.append(idx)
        else:
            cleaned_smiles += token

    # Create RDKit Mol from cleaned SMILES
    mol = Chem.MolFromSmiles(cleaned_smiles)
    if mol is None:
        raise ValueError("Failed to parse SMILES string.")

    # Create a new conformer
    conformer = Chem.Conformer(mol.GetNumAtoms())
    
    # Map atom indices from SMILES to RDKit molecule
    for atom_idx in range(mol.GetNumAtoms()):
        smiles_idx = atom_mapping[atom_idx]
        coord = coordinates[smiles_idx]
        if coord is not None:
            conformer.SetAtomPosition(atom_idx, Chem.rdGeometry.Point3D(*coord))

    # Add the conformer to the molecule
    mol.AddConformer(conformer, assignId=True)
    return mol


def get_embedded_smiles(mol):
    original_smiles = Chem.MolToSmiles(mol)
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    atom_order = ast.literal_eval(mol.GetProp('_smilesAtomOutputOrder'))
    conf = mol.GetConformer()
    coordinates = conf.GetPositions()
    embedded = ''
    atom_coords = OrderedDict()
    for char in canonical_smiles:
        if char in ['B','b','C','c','N','n','O','o','P','p','S','s','F','f','Cl','cl','Br','br','I','I']:
            i = atom_order.pop(0)
            atom_coords[str(i) + "_" + char] = coordinates[i].tolist()
            embedded += f'{char}<{coordinates[i][0]},{coordinates[i][1]},{coordinates[i][2]}>'
        else:
            embedded += char

    return {"canonical_smiles": canonical_smiles, "original_smiles": original_smiles, "conformers": {"embedded_smiles": embedded, "atom_coords": atom_coords}}
    


if __name__ == "__main__":
    mol = Chem.SDMolSupplier("/auto/home/davit/3DMolGen/data/pcqm4m-v2-train.sdf")[100]
    print(get_embedded_smiles(mol))
    emb_smi = get_embedded_smiles(mol)["conformers"]["embedded_smiles"]
    mol2 = parse_embedded_smiles(emb_smi)
    
    print(Chem.MolToSmiles(mol2, canonical=True, isomericSmiles=False) == Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
