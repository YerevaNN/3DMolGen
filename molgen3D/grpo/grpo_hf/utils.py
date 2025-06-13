from molgen3D.utils.utils import load_pkl
from molgen3D.data_processing.utils import decode_cartesian_raw
from molgen3D.utils.utils import load_pkl, load_json, get_best_rmsd
from rdkit import Chem


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
    path = "/nfs/ap/mnt/sxtn2/chem/GEOM_data/rdkit_folder/drugs/"
    mol = Chem.MolFromSmiles(key_mol_smiles, sanitize=True)
    mol_removed_chiral = remove_chiral_info(mol)
    key_mol_smiles = Chem.MolToSmiles(mol_removed_chiral,
                                                          canonical=True, 
                                                          isomericSmiles=True, 
                                                          allHsExplicit=False)
    key_mol_smiles = key_mol_smiles.replace("/", "_")
    mol_pickle = load_pkl(path + key_mol_smiles + ".pickle")
    mol_confs = mol_pickle["conformers"][:num_gt]
    mols = [conf['rd_mol'] for conf in mol_confs]
    return mols

def get_rmsd(ground_truth, generated_conformer, align=False):
    generated_mol = decode_cartesian_raw(generated_conformer)
    rmsd = get_best_rmsd(ground_truth, generated_mol, use_alignmol=False)
    return rmsd
