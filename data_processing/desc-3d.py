import numpy as np
from rdkit import Chem
from scipy.spatial import distance

def is_collinear(p1, p2, p3, tolerance=1e-6):
    """Checks if three points are collinear."""
    v1 = p2 - p1
    v2 = p3 - p1
    cross_product = np.cross(v1, v2)
    return np.allclose(cross_product, [0, 0, 0], atol=tolerance)

def calculate_generation_descriptors(mol, atom_index, coords):
    """Calculates generation descriptors for a specific atom in a molecule."""

    def find_next_atom(atom_idx, exclude_atoms):
        """For finding the reference points."""
        this_atom = mol.GetAtomWithIdx(atom_idx)
        neighbors = [neighbor.GetIdx() for neighbor in this_atom.GetNeighbors() if neighbor.GetIdx() not in exclude_atoms]
        print(neighbors)
        if not neighbors:
            return None
        dists = np.array([distance.euclidean(coords[n], coords[atom_idx]) for n in neighbors])
        return neighbors[np.argmin(dists)]
    
    current_atom_coord = coords[atom_index]

    # Find reference atoms
    focal_atom_index = find_next_atom(atom_index, [])
    print(focal_atom_index)
    if focal_atom_index is None:
        print(f"f not found for atom {atom_index}")
        return np.array([])
    
    c1_atom_index = find_next_atom(focal_atom_index, [atom_index])
    if c1_atom_index is None:
        print(f"c1 not found for atom {atom_index}")
        return np.array([])
    
    c2_atom_index = find_next_atom(c1_atom_index, [atom_index, focal_atom_index])
    if c2_atom_index is None:
        print(f"c2 not found for atom {atom_index}")
        return np.array([])
    
    focal_atom_coord = coords[focal_atom_index]
    c1_atom_coord = coords[c1_atom_index]
    c2_atom_coord = coords[c2_atom_index]

    if is_collinear(focal_atom_coord, c1_atom_coord, c2_atom_coord):
        print("collinear")
        return np.array([])

    # Calculate spherical coordinates
    r = distance.euclidean(current_atom_coord, focal_atom_coord)

    # Vectors for angles calculation
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

    # Project v_if onto the plane
    normal_vector_unit = normal_vector / norm_normal_vector
    proj_if = v_if - np.dot(v_if, normal_vector_unit) * normal_vector_unit

    # Calculate phi (azimuthal angle)
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
    print(atom_index, neighbor_indices)
    bond_lengths = []
    bond_angles = []

    for neighbor_index in neighbor_indices:
        bond_lengths.append(distance.euclidean(current_atom_coord, coords[neighbor_index]))

    # Pad bond lengths if less than 4 neighbors
    bond_lengths.extend([0.0] * (4 - num_neighbors))
    bond_lengths = sorted(bond_lengths)[:4] # Take the closest 4

    for k_idx, neighbor_k_index in enumerate(neighbor_indices[:4]):
        neighbor_k_coord = coords[neighbor_k_index]
        v_ik = neighbor_k_coord - current_atom_coord
        for l_idx, neighbor_l_index in enumerate(neighbor_indices[k_idx + 1:4]):
            neighbor_l_index_actual = neighbor_indices[k_idx + 1 + l_idx]
            neighbor_l_coord = coords[neighbor_l_index_actual]
            v_il = neighbor_l_coord - current_atom_coord

            norm_v_ik = np.linalg.norm(v_ik)
            norm_v_il = np.linalg.norm(v_il)
            if norm_v_ik > 1e-6 and norm_v_il > 1e-6:
                cos_angle = np.dot(v_ik, v_il) / (norm_v_ik * norm_v_il)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                bond_angles.append(angle)

    # Pad bond angles if less than 6 were found
    bond_angles.extend([0.0] * (6 - len(bond_angles)))
    bond_angles = sorted(bond_angles)[:6]

    # Combine descriptors
    generation_descriptor = np.array([r, theta, abs(phi), np.sign(phi)])
    understanding_descriptor = np.array(bond_lengths + bond_angles)
    descriptors = np.concatenate([generation_descriptor, understanding_descriptor])

    return descriptors

def get_mol_descriptors(sdf_file):
    """Calculates generation descriptors for all atoms in a molecule from an SDF file."""
    supplier = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)

    mol = supplier[0]
    if mol is None:
        raise ValueError("Could not read molecule from SDF file")

    conformer = mol.GetConformer()
    coords = conformer.GetPositions()

    all_descriptors = []
    for i in range(mol.GetNumAtoms()):
        descriptors = calculate_generation_descriptors(mol, i, coords)
        if descriptors.size:
            desc_str = ", ".join(f"{val:.3f}" for val in descriptors)
            atom_symbol = mol.GetAtomWithIdx(i).GetSymbol()
            all_descriptors.append(f"{atom_symbol}<{desc_str}>")
        else:
            all_descriptors.append(None)  # Use None to mark skipped atoms

    return all_descriptors

if __name__ == '__main__':
    sdf_file = '/auto/home/filya/3DMolGen/data_processing/molecule.sdf'
    try:
        descriptors = get_mol_descriptors(sdf_file)
        if any(descriptors):  # Check if any descriptors were calculated
            for i, desc in enumerate(descriptors):
                if desc:
                    print(f"Atom {i}: {desc}")
                else:
                    print(f"Atom {i}: Descriptors could not be calculated (e.g., collinearity or no neighbors).")
        else:
            print("No descriptors could be calculated for this molecule.")
    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print(f"Error: SDF file '{sdf_file}' not found.")