import numpy as np
from rdkit import Chem
import ast
from scipy.spatial import distance
import sys
import json

def is_collinear(p1, p2, p3, tolerance=1e-6):
    v1 = p2 - p1
    v2 = p3 - p1
    cross_product = np.cross(v1, v2)
    return np.allclose(cross_product, [0, 0, 0], atol=tolerance)

def get_new_atom(mol, atom_order):
    if not atom_order:
        return -1, "??"
    original_index = atom_order.pop(0)
    atom = mol.GetAtomWithIdx(original_index)
    if not atom:
        raise ValueError("Coundn't find the atom by the given index")
    while atom.GetSymbol() == "H" and atom_order:
        original_index = atom_order.pop(0)
        atom = mol.GetAtomWithIdx(original_index)
    if atom.GetSymbol() == "H":
        return -1, "??"
    return original_index, atom.GetSymbol()

def get_smiles(mol):
    smiles_list = []

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    # print("smiles", canonical_smiles)
    atom_order = ast.literal_eval(mol.GetProp('_smilesAtomOutputOrder'))
    original_index, atom_name = get_new_atom(mol, atom_order)
    
    i = 0
    while i < len(canonical_smiles):
        if canonical_smiles[i:i+len(atom_name)].upper() == atom_name.upper() and canonical_smiles[i] != 'H':
            smiles_list.append((atom_name, original_index))
            i += len(atom_name)
            original_index, atom_name = get_new_atom(mol, atom_order)
        else:
            i += 1

    smiles_to_sdf = {}
    sdf_to_smiles = {}
    for i, (ch, id) in enumerate(smiles_list):
        smiles_to_sdf[i] = id
        sdf_to_smiles[id] = i

    return smiles_list, smiles_to_sdf, sdf_to_smiles

def get_ans(mol, descriptors):
    ans = ""

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    atom_order = ast.literal_eval(mol.GetProp('_smilesAtomOutputOrder'))
    original_index, atom_name = get_new_atom(mol, atom_order)

    smiles_id = 0
    i = 0
    while i < len(canonical_smiles):
        if (canonical_smiles[i:i+len(atom_name)].upper() == atom_name.upper()) and canonical_smiles[i] != 'H':
            ans += canonical_smiles[i:i+len(atom_name)]
            ans += descriptors[smiles_id]
            smiles_id += 1
            i += len(atom_name)
            original_index, atom_name = get_new_atom(mol, atom_order)
        else:
            ans += canonical_smiles[i]
            i += 1
    return ans


def calculate_descriptors(mol, smiles_index, sdf_to_smiles, smiles_to_sdf, coords):
    """Calculates generation descriptors for a specific atom in a molecule."""
    atom_index = smiles_to_sdf[smiles_index]
    # Calculate the ref points 

    def find_next_atom(sdf_id):
        smiles_id = sdf_to_smiles[sdf_id]
        this_atom = mol.GetAtomWithIdx(sdf_id)
        neighbors = [neighbor.GetIdx() for neighbor in this_atom.GetNeighbors()]
        for i in range(smiles_id - 1, -1, -1):
            if smiles_to_sdf[i] in neighbors:
                return smiles_to_sdf[i]
        return -1

    f = -1
    c1 = -1
    c2 = -1
    focal_atom_coord = -1
    c1_atom_coord = -1
    c2_atom_coord = -1
    
    f = find_next_atom(atom_index)
    if f != -1:
        focal_atom_coord = coords[f]
        c1 = find_next_atom(f)
        if c1 != -1:
            c1_atom_coord = coords[c1]
            c2 = find_next_atom(c1)
            if c2 != -1:
                c2_atom_coord = coords[c2]

    # print("atom idx:", atom_index, "smiles idx:", smiles_index, "f:", f, "c1:", c1, "c2:", c2)

    # Calculate spherical coordinates
    if f == -1: 
        if smiles_index != 0:
            print(f"f was not found for atom {atom_index}, {sdf_to_smiles[atom_index]}")
            return np.array([-1])
        generation_descriptor = np.array([0, np.pi / 2, 0, np.sign(0)])
    elif c1 == -1:
        if smiles_index != 1:
            print(f"c1 was not found for atom {atom_index}, {sdf_to_smiles[atom_index]}")
        generation_descriptor = np.array([distance.euclidean(coords[atom_index], focal_atom_coord), np.pi / 2, 0, np.sign(0)])
    elif c2 == -1:
        # print("f-smiles:", sdf_to_smiles[f])
        # print("c1-smiles:", sdf_to_smiles[c1])
        if smiles_index != 2:
            print(f"c2 was not found for atom {atom_index}, {sdf_to_smiles[atom_index]}")
        proj_if = coords[atom_index] - focal_atom_coord #v_if=proj_if
        v_cf = c1_atom_coord - focal_atom_coord
        norm_proj_if = np.linalg.norm(proj_if)
        norm_v_cf = np.linalg.norm(v_cf)
        if norm_proj_if > 1e-6 and norm_v_cf > 1e-6:
            cos_phi = np.dot(proj_if, v_cf) / (norm_proj_if * norm_v_cf)
            cos_phi = np.clip(cos_phi, -1.0, 1.0)
            phi = np.arccos(cos_phi)
            normal_vector = [0, 0, 1]
            cross_proj_cf = np.cross(v_cf, proj_if)
            if np.dot(normal_vector, cross_proj_cf) < 0:
                phi = -phi
        else:
            print("can't divide by 0")
            return np.array([-1])
        generation_descriptor = np.array([distance.euclidean(coords[atom_index], focal_atom_coord), np.pi / 2, abs(phi), np.sign(phi)])
    else:
        if is_collinear(focal_atom_coord, c1_atom_coord, c2_atom_coord):
            print("collinear", atom_index)
            return np.array([-1])

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
            print("can't divide by 0")
            return np.array([-1])

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
            print("problem with", atom_index)
            return np.array([-1])
        
        generation_descriptor = np.array([r, theta, abs(phi), np.sign(phi)])
        
    return generation_descriptor


def get_mol_descriptors(mol):
    conformer = mol.GetConformer()
    coords = conformer.GetPositions()

    smiles, smiles_to_sdf, sdf_to_smiles = get_smiles(mol)

    all_descriptors = []
    for i in range(len(smiles)):
        descriptors = calculate_descriptors(mol, i, sdf_to_smiles, smiles_to_sdf, coords)
        if descriptors.size and descriptors[0] != -1:
            desc_str = ",".join(f"{val:.4f}" for val in descriptors)
            desc_str = desc_str[:-5] #give the sign as integer
            all_descriptors.append(f"<{desc_str}>")
        else:
            return "no descriptors"

    return get_ans(mol, all_descriptors)

def process_and_find_descriptors(sdf):
    supplier = Chem.SDMolSupplier(sdf) #, sanitize=False, removeHs=False)
    data = []
    for i, mol in enumerate(supplier): #3378606
        # if i == 2669976:
        #     continue
        if i % 1000 == 0:
            print(f"Mol {i}...")
        if not mol:
            print(f"Failed to process molecule {i}")
            return
        else:
            descriptors = get_mol_descriptors(mol)
            if descriptors != "no descriptors":
                json_string = json.dumps(descriptors)  
                data.append(json_string)
            else:
                print("Excluding this mol")

    with open(f"spherical.jsonl", "w") as file:
        for d in data:
            file.write(d)  
            file.write("\n")
        file.close()

if __name__ == '__main__':
    sys.stdout = open("output.txt", "w") 
    sdf_file = '/auto/home/menuab/pcqm4m-v2-train.sdf' #'/auto/home/filya/3DMolGen/data_processing/molecule.sdf'
    try:
        process_and_find_descriptors(sdf_file)
    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print(f"Error: SDF file '{sdf_file}' not found.")
        