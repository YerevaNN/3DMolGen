import numpy as np
from rdkit import Chem
import ast
from scipy.spatial import distance
import sys

def is_collinear(p1, p2, p3, tolerance=1e-6):
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

def calculate_descriptors(mol, atom_index, coords, smiles_to_sdf, focal_atom_coord, c1_atom_coord, c2_atom_coord ):
    """Calculates generation descriptors for a specific atom in a molecule."""
    if atom_index == 0: 
        theta = np.pi / 2
        generation_descriptor = np.array([0, np.pi / 2, 0, np.sign(0)])
    elif atom_index == 1:
        atom_index = smiles_to_sdf[atom_index]
        generation_descriptor = np.array([distance.euclidean(coords[atom_index], focal_atom_coord), np.pi / 2, 0, np.sign(0)])
    elif atom_index == 2:
        atom_index = smiles_to_sdf[atom_index]
        proj_if = coords[atom_index] - focal_atom_coord #v_if=proj_if
        v_cf = c1_atom_coord - focal_atom_coord
        norm_proj_if = np.linalg.norm(proj_if)
        norm_v_cf = np.linalg.norm(v_cf)
        if norm_proj_if > 1e-6 and norm_v_cf > 1e-6:
            cos_phi = np.dot(proj_if, v_cf) / (norm_proj_if * norm_v_cf)
            cos_phi = np.clip(cos_phi, -1.0, 1.0)
            phi = np.arccos(cos_phi)
        else:
            return np.array([])
        generation_descriptor = np.array([distance.euclidean(coords[atom_index], focal_atom_coord), np.pi / 2, phi, np.sign(phi)])
    else:
        atom_index = smiles_to_sdf[atom_index]

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
            print(atom_index)
            return np.array([])
        
        generation_descriptor = np.array([r, theta, abs(phi), np.sign(phi)])
        
    return np.array(generation_descriptor)

def get_mol_descriptors(mol):
    conformer = mol.GetConformer()
    coords = conformer.GetPositions()

    smiles = get_smiles_list(mol)
    smiles_to_sdf = {}
    for i, (ch, id) in enumerate(smiles):
        smiles_to_sdf[i] = id
    # print(smiles)
    # print(smiles_to_sdf)

    f = coords[smiles_to_sdf[0]]
    c1 = coords[smiles_to_sdf[1]]
    c2 = coords[smiles_to_sdf[2]]

    all_descriptors = []
    for i in range(len(smiles)):
        descriptors = calculate_descriptors(mol, i, coords, smiles_to_sdf, f, c1, c2)
        if descriptors.size:
            desc_str = ", ".join(f"{val:.3f}" for val in descriptors)
            all_descriptors.append(f"{smiles[i][0]}<{desc_str}>")
        else:
            all_descriptors.append(None)

    return all_descriptors


def process_and_find_descriptors(sdf):
    supplier = Chem.SDMolSupplier(sdf, sanitize=False, removeHs=False)
    for i in range(497,498):
        if i % 1000 == 0:
            print(i)
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
                        print(f"Molecule{i} Atom {j}: Descriptors could not be calculated (e.g., collinearity or no neighbors).")
                        return
            else:
                print(f"No descriptors could be calculated {i}-th molecule.")


if __name__ == '__main__':
    sys.stdout = open("output.txt", "w") 
    sdf_file = '/auto/home/menuab/pcqm4m-v2-train.sdf' #'/auto/home/filya/3DMolGen/data_processing/molecule.sdf'
    try:
        process_and_find_descriptors(sdf_file)
    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print(f"Error: SDF file '{sdf_file}' not found.")
        