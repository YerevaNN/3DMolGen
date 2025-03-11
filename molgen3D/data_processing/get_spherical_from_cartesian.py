import numpy as np
from rdkit import Chem
import ast
from scipy.spatial import distance
import sys
import json
import os
from ogb.lsc import PCQM4Mv2Dataset
from loguru import logger as log

exclude_h = False

def is_collinear(p1, p2, p3, tolerance=1e-3):
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
    # if exclude_h:
    #     while atom.GetSymbol() == "H" and atom_order:
    #         original_index = atom_order.pop(0)
    #         atom = mol.GetAtomWithIdx(original_index)
    #     if atom.GetSymbol() == "H":
    #         return -1, "??"
    return original_index, atom.GetSymbol()

def get_smiles(mol):
    smiles_list = []
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    # print("smiles", canonical_smiles)

    atom_order = ast.literal_eval(mol.GetProp('_smilesAtomOutputOrder'))

    original_index, atom_name = get_new_atom(mol, atom_order)
    
    i = 0
    while i < len(canonical_smiles):
        if canonical_smiles[i:i+len(atom_name)].upper() == atom_name.upper():
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
    # print("sdf_to_smiles ", sdf_to_smiles)
    # print("smiles_to_sdf ", smiles_to_sdf)
    # sys.stdout.flush()

    return smiles_list, smiles_to_sdf, sdf_to_smiles

def get_ans(mol, descriptors):
    ans = ""

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    atom_order = ast.literal_eval(mol.GetProp('_smilesAtomOutputOrder'))
    # print(atom_order)
    original_index, atom_name = get_new_atom(mol, atom_order)

    smiles_id = 0
    i = 0
    while i < len(canonical_smiles):
        if canonical_smiles[i:i+len(atom_name)].upper() == atom_name.upper():
            ans += canonical_smiles[i:i+len(atom_name)]
            ans += descriptors[smiles_id]
            smiles_id += 1
            i += len(atom_name)
            original_index, atom_name = get_new_atom(mol, atom_order)
        else:
            ans += canonical_smiles[i]
            i += 1
    return ans

def calculate_spherical_from_cartesian(current_atom_coord, focal_atom_coord, c1_atom_coord, c2_atom_coord):
    r = distance.euclidean(current_atom_coord, focal_atom_coord)

    v_cf = c1_atom_coord - focal_atom_coord
    v_c2f = c2_atom_coord - focal_atom_coord
    v_if = current_atom_coord - focal_atom_coord

    # Calculate normal vector to the plane defined by focal and reference atoms
    normal_vector = np.cross(v_cf, v_c2f)
    normal_vector_unit = normal_vector / np.linalg.norm(normal_vector) # norm_normal_vector is not 0, bcs f,c1,c2 are not collinear

    # Calculate theta (polar angle)
    cos_theta = np.dot(v_if, normal_vector_unit) / r # r is norm_v_if
    if(cos_theta > 1.0 or cos_theta < -1.0):
        log.error(f"cos_theta is not correct: {cos_theta}")
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Calculate phi (azimuthal angle)
    phi = 0
    proj_if = v_if - r * cos_theta * normal_vector_unit
    norm_proj_if = np.linalg.norm(proj_if)
    if norm_proj_if < 1e-7: #theta is 0, phi ?
        # log.error("here")
        phi = 0
    else:
        proj_if /= norm_proj_if
        v_cf /= np.linalg.norm(v_cf) # norm_v_cf is not 0, bcs cur c1 is not f
        
        cos_phi = np.dot(proj_if, v_cf) 
        if(cos_phi > 1.0 or cos_phi < -1.0):
            log.error(f"cos_theta is not correct: {cos_theta}")
            cos_phi = np.clip(cos_phi, -1.0, 1.0)
        
        phi = np.arccos(cos_phi)

        cross_proj_cf = np.cross(v_cf, proj_if)
        if np.dot(normal_vector, cross_proj_cf) < 0: # cospi = -1, cos0 = 1
            phi = -phi

    return r, theta, abs(phi), np.sign(phi)

def calculate_descriptors(mol, mol_id, smiles_index, sdf_to_smiles, smiles_to_sdf, coords):
    """Calculates generation descriptors for a specific atom in a molecule."""
    # def find_next_atom(begin_sdf_id, sdf_id1, atom_positions, exclude_atoms = [], sdf_id2 = -1):
    #     begin_smiles_id = sdf_to_smiles[begin_sdf_id]
    #     smiles_id1 = sdf_to_smiles[sdf_id1]
    #     this_atom1 = mol.GetAtomWithIdx(sdf_id1)
    #     neighbors1 = [neighbor.GetIdx() for neighbor in this_atom1.GetNeighbors() if neighbor not in exclude_atoms]
    #     neighbors2 = []
    #     smiles_id2 = -1
    #     if sdf_id2 != -1:
    #         smiles_id2 = sdf_to_smiles[sdf_id2]
    #         this_atom2 = mol.GetAtomWithIdx(sdf_id2)
    #         neighbors2 = [neighbor.GetIdx() for neighbor in this_atom2.GetNeighbors() if neighbor not in exclude_atoms]
    #     for i in range(begin_smiles_id - 1, -1, -1):
    #         if i == smiles_id1 or i == smiles_id2:
    #             continue
    #         if smiles_to_sdf[i] in neighbors1 or smiles_to_sdf[i] in neighbors2:
    #             if smiles_id2 != -1 and is_collinear(atom_positions[sdf_id1], atom_positions[sdf_id2], atom_positions[smiles_to_sdf[i]]):
    #                 continue
    #             return smiles_to_sdf[i]
    #     return -1

    def find_next_atom(begin_sdf_id, sdf_id1, atom_positions, sdf_id2 = -1):
        begin_smiles_id = sdf_to_smiles[begin_sdf_id]
        smiles_id1 = sdf_to_smiles[sdf_id1]
        this_atom1 = mol.GetAtomWithIdx(sdf_id1)
        neighbors1 = [neighbor.GetIdx() for neighbor in this_atom1.GetNeighbors()]
        neighbors2 = []
        smiles_id2 = -1
        if sdf_id2 != -1:
            smiles_id2 = sdf_to_smiles[sdf_id2]
            this_atom2 = mol.GetAtomWithIdx(sdf_id2)
            neighbors2 = [neighbor.GetIdx() for neighbor in this_atom2.GetNeighbors()]
        for i in range(begin_smiles_id - 1, -1, -1):
            if i == smiles_id1 or i == smiles_id2:
                continue
            if smiles_to_sdf[i] in neighbors1 or smiles_to_sdf[i] in neighbors2:
                if smiles_id2 != -1 and is_collinear(atom_positions[sdf_id1], atom_positions[sdf_id2], atom_positions[smiles_to_sdf[i]]):
                    continue
                return smiles_to_sdf[i]
        return -1
    
    
    def find_ref_points(atom_index):
        # def without_coordinates():
        #     best = []
        #     exclude_focal = []
        #     exclude_c1 = []
        #     while True:
        #         f = find_next_atom(atom_index, atom_index, coords, exclude_focal)
        #         if f == -1:
        #             while len(best) < 3:
        #                 best.append(-1)
        #             return best
        #         c1 = find_next_atom(atom_index, f, coords, exclude_c1)
        #         if c1 == -1:
        #             exclude_focal.append(f)
        #             exclude_c1 = []
        #             if len(best) < 1:
        #                 best = [f]
        #             continue
        #         c2 = find_next_atom(atom_index, c1, coords, [], f)
        #         if c2 != -1:
        #             # print(f"atom {atom_index} -> f:{focal_atom_index}, c1:{c1_atom_index}, c2:{c2_atom_index}")
        #             return f, c1, c2
        #         best = [f, c1]
        #         exclude_c1.append(c1)
        #         if len(exclude_focal) > len(coords) or len(exclude_c1) > len(coords):
        #             break

        #     # If the loop exits without finding c2, return the best found so far
        #     while len(best) < 3:
        #         best.append(-1)
        #     return best

        # f, c1, c2 = without_coordinates()
        # focal_atom_coord = -1
        # if f != -1:
        #     focal_atom_coord = coords[f]
        # c1_atom_coord = -1
        # if c1 != -1:
        #     c1_atom_coord = coords[c1]
        # c2_atom_coord = -1
        # if c2 != -1:
        #     c2_atom_coord = coords[c2]
        # return f, c1, c2, focal_atom_coord, c1_atom_coord, c2_atom_coord
        f = -1
        c1 = -1
        c2 = -1
        focal_atom_coord = -1
        c1_atom_coord = -1
        c2_atom_coord = -1

        f = find_next_atom(atom_index, atom_index, coords)
        if f != -1:
            focal_atom_coord = coords[f]
            c1 = find_next_atom(atom_index, f, coords)
            if c1 != -1:
                c1_atom_coord = coords[c1]
                c2 = find_next_atom(atom_index, c1, coords, f)
                if c2 != -1:
                    c2_atom_coord = coords[c2]
        return f, c1, c2, focal_atom_coord, c1_atom_coord, c2_atom_coord


    atom_index = smiles_to_sdf[smiles_index]
    current_atom_coord = coords[atom_index]

    f, c1, c2, focal_atom_coord, c1_atom_coord, c2_atom_coord = find_ref_points(atom_index)
    # log.info("smiles idx: {}, f: {}, c1: {}, c2: {}".format(
    #     sdf_to_smiles.get(atom_index, 'N/A'),
    #     sdf_to_smiles.get(f, 'N/A'),
    #     sdf_to_smiles.get(c1, 'N/A'),
    #     sdf_to_smiles.get(c2, 'N/A')
    # ))
    
    if f == -1: 
        if smiles_index != 0:
            raise ValueError(f"f was not found for mol {mol_id}, atom {atom_index}, {sdf_to_smiles[atom_index]}")
        return np.array([0, np.pi / 2, 0, np.sign(0)])
    if c1 == -1:
        if smiles_index != 1:
            raise ValueError(f"c1 was not found for atom {atom_index}, {sdf_to_smiles[atom_index]}")
        #assume that the point is on the OX ray
        return np.array([distance.euclidean(current_atom_coord, focal_atom_coord), np.pi / 2, 0, np.sign(0)])
    if c2 == -1:
        # if smiles_index != 2:
        #     log.error(f"c2 was not found for atom {atom_index}, {sdf_to_smiles[atom_index]}")
        
        # if smiles_index != 2 and not is_collinear(focal_atom_coord, c1_atom_coord, current_atom_coord):
        #     raise ValueError(f"buuu c2 was not found for atom {atom_index}, {sdf_to_smiles[atom_index]}")

        #assume that the point is on XY plane
        #?
        if is_collinear(focal_atom_coord, c1_atom_coord, current_atom_coord):
            # log.error("f,c1,i are collinear in atom", atom_index)
            
            # check if i is on the ray from f to c1
            f_c1 = c1_atom_coord - focal_atom_coord
            f_i = current_atom_coord - focal_atom_coord
            if np.dot(f_i, f_c1) < 0: # 180 degrees
                return np.array([distance.euclidean(current_atom_coord, focal_atom_coord), np.pi / 2, np.pi, 1])
            return np.array([distance.euclidean(current_atom_coord, focal_atom_coord), np.pi / 2, 0, 0])
        
        r, theta, phi, sign_phi = calculate_spherical_from_cartesian(current_atom_coord, focal_atom_coord, c1_atom_coord, current_atom_coord)
        return np.array([r, theta, phi, sign_phi])
        
    # Here we have f,c1,c2
    if is_collinear(focal_atom_coord, c1_atom_coord, c2_atom_coord): #shouldnt enter this
        raise ValueError("f,c1,c2 are colinear", atom_index)

    # Calculate spherical coordinates

    r, theta, phi, sign_phi = calculate_spherical_from_cartesian(current_atom_coord, focal_atom_coord, c1_atom_coord, c2_atom_coord)

    # e1 = v_cf / np.linalg.norm(v_cf)
    # e2 = np.cross(normal_vector_unit, e1)
    # proj_if = v_if - (np.dot(v_if, normal_vector_unit)) * normal_vector_unit
    # phi = np.arctan2(np.dot(proj_if, e2), np.dot(proj_if, e1))
            
    return np.array([r, theta, phi, sign_phi])

    
        
def get_json(mol, embedded, sample):
    return {"canonical_smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False), 
            "pcqm4v2_smiles": sample[0], 
            "pcqm4v2_label": sample[1], 
            "conformers": {"embedded_smiles": embedded}}
    

def get_mol_descriptors(mol, mol_id, precision=4):
    conformer = mol.GetConformer()
    coords = conformer.GetPositions()
    smiles, smiles_to_sdf, sdf_to_smiles = get_smiles(mol) 
    all_descriptors = []
    for i in range(len(smiles)):
        descriptors = calculate_descriptors(mol, mol_id, i, sdf_to_smiles, smiles_to_sdf, coords)
        if descriptors.size and descriptors[0] != -1:
            desc_str = ",".join(f"{val:.{precision}f}" for val in descriptors)
            desc_str = desc_str[:-5] # give the sign as an integer
            all_descriptors.append(f"<{desc_str}>")
        else:
            return "no descriptors"
    
    return all_descriptors

def embed_coordinates_spherical(mol, smiles, order, precision=4):
    all_descriptors = get_mol_descriptors(mol, 0, precision)
    return get_ans(mol, all_descriptors)

def process_and_find_descriptors(sdf, val_indices):
    # supplier = Chem.SDMolSupplier(sdf, sanitize=exclude_h, removeHs=exclude_h)
    supplier = Chem.SDMolSupplier("mol_9496-rec.sdf", sanitize=exclude_h, removeHs=exclude_h)
    dataset = PCQM4Mv2Dataset(root = '/auto/home/menuab', only_smiles = True)

    train_data = []
    val_data = []
    for i, mol in enumerate(supplier): #3378606
        # if i < 9496:
        #     continue
        # if i > 9496:
        #     break
        # writer = Chem.SDWriter("mol_9496-gen.sdf")
        # writer.write(mol)
        # writer.close()
        # if i == 200000:
        #     break
        if i % 1000 == 0:
            print(f"Mol {i}...")
            sys.stdout.flush()
        if not mol:
            raise ValueError(f"Failed to process mol {i}")
            # return
        else:
            all_descriptors = get_mol_descriptors(mol, i)
            descriptors = get_ans(mol, all_descriptors)
            if descriptors != "no descriptors":
                json_string = get_json(mol, descriptors, dataset[i])
                
                if i in val_indices:
                    val_data.append(json_string)
                else:
                    train_data.append(json_string)
            else:
                raise ValueError(f"Excluding mol {i}")

    current_file = None
    data_item = train_data[0]
    current_file = open(f"mol_9496_.jsonl", 'w')
    json.dump(data_item, current_file)
    current_file.write('\n')
    current_file.close()


    # train_folder = "train_embedded_spherical_with_H"
    # if exclude_h:
    #     train_folder = "train_embedded_spherical"
    # os.makedirs(train_folder, exist_ok=True)

    # file_counter = 0
    # current_file = None
    # new_file_freq = 1000000

    # for i, data_item in enumerate(train_data):
    #     if i % new_file_freq == 0:
    #         if current_file:
    #             current_file.close()
    #         current_file = open(os.path.join(train_folder, f"train_data_{file_counter}.jsonl"), 'w')
    #         file_counter += 1

    #     if current_file:
    #         json.dump(data_item, current_file)
    #         current_file.write('\n')

    # if current_file:
    #     current_file.close()

    # valid_folder = "valid_embedded_spherical_with_H"
    # if exclude_h:
    #     valid_folder = "valid_embedded_spherical"
    # os.makedirs(valid_folder, exist_ok=True)
    # with open(os.path.join(valid_folder, "valid_data.jsonl"), "w") as file:
    #     for d in val_data:
    #         json.dump(d, file)
    #         file.write("\n")
    #     file.close()

if __name__ == '__main__':
    out_file = "outputs/output_gen_with_H.txt"
    if exclude_h:
        out_file = "outputs/output_gen.txt"
    sys.stdout = open(out_file, "w") 
    sdf_file = '/auto/home/menuab/pcqm4m-v2-train.sdf'
    val_indices_file = "/auto/home/menuab/code/3DMolGen/data/pcqm/pcqm4v2_valid_indice.txt"
    val_indices = []
    with open(val_indices_file, 'r') as f:
        for line in f:
            try:
                index = int(line.strip())
                val_indices.append(index)
            except ValueError:
                raise ValueError(f"Skipping invalid line: {line.strip()}")
    process_and_find_descriptors(sdf_file, val_indices)