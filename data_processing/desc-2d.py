import numpy as np
from rdkit import Chem
import ast
from scipy.spatial import distance
import sys

def is_collinear(p1, p2, p3, tolerance=1e-6):
    """Checks if three points are collinear."""
    v1 = p2 - p1
    v2 = p3 - p1
    cross_product = np.cross(v1, v2)
    return np.allclose(cross_product, [0, 0, 0], atol=tolerance)

def get_smiles_list(mol):
    smiles_list = []
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    atom_order = ast.literal_eval(mol.GetProp('_smilesAtomOutputOrder'))
    
    i = 0
    while i < len(canonical_smiles):
        if canonical_smiles[i].isupper():
            if i + 1 < len(canonical_smiles) and canonical_smiles[i+1].islower():
                atom_symbol = canonical_smiles[i:i+2]
                i += 1
            else:
                atom_symbol = canonical_smiles[i]
            original_index = atom_order.pop(0)
            smiles_list.append((atom_symbol, original_index))
        elif canonical_smiles[i] in ['b', 'c', 'n', 'o', 'p', 's', 'f', 'i']:
            atom_symbol = canonical_smiles[i]
            original_index = atom_order.pop(0)
            smiles_list.append((atom_symbol, original_index))
        i += 1

    return smiles_list

def calculate_descriptors(mol, atom_index, coords, smiles, sdf_to_smiles):
    """Calculates generation descriptors for a specific atom in a molecule."""
    def find_next_atom(atom_idx, exclude_atoms):
        print(atom_idx)
        this_atom = mol.GetAtomWithIdx(atom_idx)
        cur_id = sdf_to_smiles[atom_idx]
        neighbors = [neighbor.GetIdx() for neighbor in this_atom.GetNeighbors()]
        # Left
        for i in range(cur_id - 1, -1, -1):
            if smiles[i][1] in neighbors and smiles[i][1] not in exclude_atoms:
                return smiles[i][1]
        # Right
        for i in range(cur_id, len(smiles)):
            if smiles[i][1] in neighbors and smiles[i][1] not in exclude_atoms:
                return smiles[i][1]
        return -1
    
    def find_ref_points():
        exclude_focal = []
        focal_atom_index = -1
        c1_atom_index = -1
        c2_atom_index = -1
        cnt = 0
        while cnt < 20:
            focal_atom_index = find_next_atom(atom_index, exclude_focal)
            if focal_atom_index == -1:
                print(f"good f not found for atom {atom_index}")
                return np.array([])
            
            c1_atom_index = find_next_atom(focal_atom_index, [atom_index])
            if c1_atom_index == -1:
                exclude_focal.append(focal_atom_index)
                continue
                
            
            c2_atom_index = find_next_atom(c1_atom_index, [atom_index, focal_atom_index])
            if c2_atom_index == -1:
                # Try changing c1
                c1_atom_index = find_next_atom(focal_atom_index, [atom_index, c1_atom_index])
                if c1_atom_index == -1:
                    # Try changing f
                    exclude_focal.append(focal_atom_index)
                    continue
                c2_atom_index = find_next_atom(c1_atom_index, [atom_index, focal_atom_index])
                if c2_atom_index == -1: # Otherwise, we return the coords
                    # Try changing f
                    exclude_focal.append(focal_atom_index)
                    continue
            
            return coords[focal_atom_index], coords[c1_atom_index], coords[c2_atom_index]
        
        if c1_atom_index == -1:
            print(f"c1 not found for atom {atom_index}")
        elif c2_atom_index == -1:
            print(f"c2 not found for atom {atom_index}")
        return np.array([])

    refs = find_ref_points()
    print(refs)
    if len(refs) == 3:
        focal_atom_coord, c1_atom_coord, c2_atom_coord = refs
    else:
        return np.array([])

    if is_collinear(focal_atom_coord, c1_atom_coord, c2_atom_coord):
        print("collinear")
        return np.array([])

    # Calculate spherical coordinates
    current_atom_coord = coords[atom_index]

    r = distance.euclidean(current_atom_coord, focal_atom_coord)

    v_cf = c1_atom_coord - focal_atom_coord
    v_c2f = c2_atom_coord - focal_atom_coord
    v_if = current_atom_coord - focal_atom_coord

    # Calculate normal vector to the plane defined by focal and reference atoms
    normal_vector = np.cross(v_cf, v_c2f)

    # Calculate theta (polar angle)
    norm_v_if = np.linalg.norm(v_if)
    norm_normal_vector = np.linalg.norm(normal_vector)
    if norm_v_if > 1e-6 and norm_normal_vector > 1e-6:
        cos_theta = np.dot(v_if, normal_vector) / (norm_v_if * norm_normal_vector)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
    else:
        return np.array([])

    # Calculate phi (azimuthal angle)
    normal_vector_unit = normal_vector / norm_normal_vector
    proj_if = v_if - np.dot(v_if, normal_vector_unit) * normal_vector_unit

    norm_proj_if = np.linalg.norm(proj_if)
    norm_v_cf = np.linalg.norm(v_cf)
    if norm_proj_if > 1e-6 and norm_v_cf > 1e-6:
        cos_phi = np.dot(proj_if, v_cf) / (norm_proj_if * norm_v_cf)
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        phi = np.arccos(cos_phi)
        # Determine the sign of phi using the cross product
        cross_proj_cf = np.cross(v_cf, proj_if)
        if np.dot(normal_vector, cross_proj_cf) < 0:
            phi = -phi
    else:
        return np.array([])

    # Calculate understanding descriptors (bond lengths and angles)
    neighbor_indices = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(atom_index).GetNeighbors()]
    num_neighbors = len(neighbor_indices)
    bond_lengths = []
    bond_angles = []
    for neighbor_index in neighbor_indices:
        bond_lengths.append((distance.euclidean(current_atom_coord, coords[neighbor_index]), neighbor_index))

    bond_lengths = sorted(bond_lengths)[:4]
    closest_neighbor_indices = [index for length, index in bond_lengths]

    bond_angles = []
    for k_idx, neighbor_k_index in enumerate(closest_neighbor_indices):
        neighbor_k_coord = coords[neighbor_k_index]
        v_ik = neighbor_k_coord - current_atom_coord
        for l_idx in range(k_idx + 1, len(closest_neighbor_indices)):
            neighbor_l_index = closest_neighbor_indices[l_idx]
            neighbor_l_coord = coords[neighbor_l_index]
            v_il = neighbor_l_coord - current_atom_coord

            norm_v_ik = np.linalg.norm(v_ik)
            norm_v_il = np.linalg.norm(v_il)
            if norm_v_ik > 1e-6 and norm_v_il > 1e-6:
                cos_angle = np.dot(v_ik, v_il) / (norm_v_ik * norm_v_il)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                bond_angles.append(angle)

    bond_lengths = [b_length for b_length, b_idx in bond_lengths]

    # Pad bond lengths if less than 4 neighbors
    bond_lengths.extend([0.0] * (4 - num_neighbors))

    # Pad bond angles if less than 6 were found
    bond_angles.extend([0.0] * (6 - len(bond_angles)))

    # Combine descriptors
    generation_descriptor = np.array([r, theta, abs(phi), np.sign(phi)])
    understanding_descriptor = np.array(bond_lengths + bond_angles)
    descriptors = np.concatenate([generation_descriptor, understanding_descriptor])

    return descriptors

def get_mol_descriptors(mol):
    conformer = mol.GetConformer()
    coords = conformer.GetPositions()

    smiles = get_smiles_list(mol)
    sdf_to_smiles = {}
    for i, (ch, id) in enumerate(smiles):
        sdf_to_smiles[id] = i
    # print(smiles)
    # print(sdf_to_smiles)

    all_descriptors = []
    for i in range(mol.GetNumAtoms()):
        descriptors = calculate_descriptors(mol, i, coords, smiles, sdf_to_smiles)
        if descriptors.size:
            desc_str = ", ".join(f"{val:.3f}" for val in descriptors)
            atom_symbol = mol.GetAtomWithIdx(i).GetSymbol()
            all_descriptors.append(f"{atom_symbol}<{desc_str}>")
        else:
            all_descriptors.append(None)

    return all_descriptors


def process_and_find_descriptors(sdf):
    supplier = Chem.SDMolSupplier(sdf, sanitize=False, removeHs=False)
    for i in range(10):
        mol = supplier[i]
        if not mol:
            print(f"Failed to process molecule {i+1}")
            return
        else:
            print(f"Calculating descriptors for mol {i + 1}...")
            descriptors = get_mol_descriptors(mol)
            if any(descriptors): 
                for j, desc in enumerate(descriptors):
                    if desc:
                        print(f"Atom {j}: {desc}")
                    else:
                        print(f"Atom {j}: Descriptors could not be calculated (e.g., collinearity or no neighbors).")
            else:
                print(f"No descriptors could be calculated {i}-th molecule.")


if __name__ == '__main__':
    # sys.stdout = open("output.txt", "w") 
    sdf_file = '/auto/home/menuab/pcqm4m-v2-train.sdf' #'/auto/home/filya/3DMolGen/data_processing/molecule.sdf'
    try:
        process_and_find_descriptors(sdf_file)
    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print(f"Error: SDF file '{sdf_file}' not found.")
        