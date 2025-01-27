from rdkit import Chem
from rdkit.Chem import AllChem

writer1 = Chem.SDWriter("mols1.sdf")
writer2 = Chem.SDWriter("mols2.sdf")
writer_pre = Chem.SDWriter("mols2_pre.sdf")

def create_atom_mapping_by_properties(mol1, mol2, useChirality=False):
    """
    Create an atom mapping between mol1 and mol2 by checking for a
    full substructure match (i.e., exact isomorphism) of mol2 in mol1.
    Returns a list of (idx_in_mol1, idx_in_mol2) tuples sorted by idx_in_mol1.
    
    :param mol1: RDKit Mol object
    :param mol2: RDKit Mol object
    :param useChirality: Whether to consider chirality in matching
    :return: List of tuples (atom_idx_in_mol1, atom_idx_in_mol2) 
             or None if no match is found
    """
    match_tuple = mol1.GetSubstructMatch(mol2, useChirality=useChirality)
    
    if not match_tuple:
        return None
    
    # match_tuple[i] gives the index in mol1 that matches the i-th atom of mol2.
    # We want a list of (mol1_index, mol2_index) pairs.
    mapping = [(match_tuple[m2_idx], m2_idx) for m2_idx in range(len(match_tuple))]
    mapping.sort(key=lambda x: x[0])
    
    return mapping

# def create_atom_mapping_by_properties(mol1, mol2):
#     atoms1 = [(atom.GetAtomicNum(), atom.GetDegree(), atom.GetIdx()) for atom in mol1.GetAtoms()]
#     atoms2 = [(atom.GetAtomicNum(), atom.GetDegree(), atom.GetIdx()) for atom in mol2.GetAtoms()]

#     # Sort atoms based on (atomic number, degree)
#     sorted_atoms1 = sorted(atoms1, key=lambda x: (x[0], x[1]))
#     sorted_atoms2 = sorted(atoms2, key=lambda x: (x[0], x[1]))

#     atom_map = []
#     for i in range(mol1.GetNumAtoms()):
#         atom_map.append((sorted_atoms1[i][2], sorted_atoms2[i][2])) # map original indices

#     return atom_map

def reorder_mol_atoms_by_map(mol, atom_map):
    num_atoms = mol.GetNumAtoms()
    if not atom_map or len(atom_map) != num_atoms:
        print("Error: Invalid atom_map for reordering.")
        return None

    new_mol = Chem.RWMol()  # Editable molecule
    old_to_new_indices = {} # Map from old index to new index in the new Mol

    # Add atoms in the new order based on atom_map (second index in tuple is new index)
    for index_mol1, index_mol2 in atom_map:
        atom = mol.GetAtomWithIdx(index_mol2) # Get atom from mol2 using index from atom_map (mol2 index)
        new_atom_index = new_mol.AddAtom(atom) # Atoms are added in order 0, 1, 2,... so new_atom_index == loop index
        old_to_new_indices[index_mol2] = new_atom_index # Keep track of mapping from mol2 original index -> new index

    new_conf = Chem.Conformer(num_atoms)
    old_conf = mol.GetConformer()

    # Set atom positions in the new conformer based on reordered atoms
    for index_mol1, index_mol2 in atom_map:
        new_conf.SetAtomPosition(index_mol1, old_conf.GetAtomPosition(index_mol2)) # Copy positions based on map (mol2 index)

    new_mol.AddConformer(new_conf, assignId=True) # Assign the new conformer

    # Add bonds - re-index bond atoms based on the mapping
    for bond in mol.GetBonds():
        atom_idx1_old = bond.GetBeginAtomIdx()
        atom_idx2_old = bond.GetEndAtomIdx()
        atom_idx1_new = old_to_new_indices[atom_idx1_old]
        atom_idx2_new = old_to_new_indices[atom_idx2_old]
        new_mol.AddBond(atom_idx1_new, atom_idx2_new, bond.GetBondType())

    return new_mol.GetMol() # Get immutable Mol object


def compare_mols_sdf(i, suppl1, suppl2):
    try:
        mol1 = next(suppl1)
        mol2 = next(suppl2)

        if not mol1 or not mol2:
            print(f"Error: Could not read molecule from one or both SDF files.")
            return None

        mol1 = Chem.RemoveHs(mol1)
        Chem.SanitizeMol(mol1)  # Ensure validity after removing Hs
        Chem.SanitizeMol(mol2)

        if mol1.GetNumAtoms() != mol2.GetNumAtoms():
            print(f"NumAtoms not equal: MOL {i}")
            return False

        smiles1 = Chem.MolToSmiles(mol1, canonical=True, isomericSmiles=False) # Canonical SMILES
        smiles2 = Chem.MolToSmiles(mol2, canonical=True, isomericSmiles=False) # Canonical SMILES

        if smiles1 != smiles2:
            print(smiles1, smiles2)
            print(f"Smiles not equal: MOL {i}")
            return False

        if mol1.GetNumConformers() == 0 or mol2.GetNumConformers() == 0:
            print(f"Error: One or both molecules lack 3D coordinates in SDF files.")
            return None
        writer_pre.write(mol2)
        # Align mol2 to mol1 based on coordinates
        try:
            mp = create_atom_mapping_by_properties(mol1, mol2)
            mol2 = reorder_mol_atoms_by_map(mol2, mp)
            AllChem.AlignMol(mol2, mol1, atomMap=[(i, i) for i in range(mol2.GetNumAtoms())])
            rmsd = AllChem.GetBestRMS(mol1, mol2)
            print(f"RMSD after alignment: {rmsd:.3f}")
        except Exception as align_err:
            print(f"Error during molecule alignment: {align_err}")
            return None

        writer2.write(mol2)
        writer1.write(mol1)
        return rmsd <= 0.1
    except StopIteration:
        print(f"Error: SDF file is empty: MOL {i}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


sdf1_path = "/auto/home/menuab/pcqm4m-v2-train.sdf" # Replace with path to SDF file WITH hydrogens
sdf2_path = "/auto/home/filya/3DMolGen/reconstructed_molecules.sdf" # Replace with path to SDF file WITHOUT hydrogens
try:
    suppl1 = Chem.SDMolSupplier(sdf1_path)
    suppl2 = Chem.SDMolSupplier(sdf2_path)
    if not suppl1 or not suppl2:
        print(f"Error: Could not open one or both SDF files: {sdf1_path}, {sdf2_path}")
except FileNotFoundError:
    print(f"Error: SDF file not found")

for i in range(10):
    are_same = compare_mols_sdf(i, suppl1, suppl2)
    if are_same is None:
        print("Comparison failed. Check error messages above.")
    elif are_same:
        print("Molecules are the same (ignoring hydrogens and 3D pose).")
    else:
        print("Molecules are different.")

writer1.close()
writer2.close()
writer_pre.close()
