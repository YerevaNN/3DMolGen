import numpy as np
from rdkit import Chem
import re
import sys
import json
import numpy as np
from loguru import logger as log 
from get_spherical_from_cartesian import find_ref_points
exclude_h = False

def parse_embedded_smiles(embedded_smiles, embedding_type="spherical"):
    pattern = r'([A-Za-z][a-z]?)(<([^>]+)>)?'
    tokens = re.findall(pattern, embedded_smiles)
    
    descriptors = []
    
    for token in tokens:
        atom = token[0]
        descriptor_str = token[2]
        if descriptor_str:
            descriptor = [float(x) for x in descriptor_str.split(',')]
            if embedding_type == "spherical" and len(descriptor) != 4:
                raise ValueError(f"Invalid descriptor format for atom {atom}: '{descriptor_str}'")
            elif embedding_type == "cartesian" and len(descriptor) != 3:
                raise ValueError(f"Invalid descriptor format for atom {atom}: '{descriptor_str}'")
            descriptors.append(descriptor)
    
    smiles = re.sub(r'<[^>]+>', '', embedded_smiles) # exclude <...>s
    return smiles, descriptors

def relative(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def is_collinear(p1, p2, p3, tolerance=1e-3):
    v1 = p2 - p1
    v2 = p3 - p1
    cross_product = np.cross(v1, v2)
    return np.allclose(cross_product, [0, 0, 0], atol=tolerance)

def calc_spherical_to_cartesian(r, theta, phi, sign_phi, focal_atom_coord, c1_atom_coord, c2_atom_coord):
    v1 = c1_atom_coord - focal_atom_coord
    v2 = c2_atom_coord - focal_atom_coord
    
    # Orthonormal basis
    e1 = v1 / np.linalg.norm(v1)
    e3 = np.cross(v1, v2)
    e3 /= np.linalg.norm(e3) # norm_normal is not 0, bcs we excluded that(when the ref points are collinear) case from data
    e2 = np.cross(e3, e1)
    e2 /= np.linalg.norm(e2)

    # Create rotation matrix from local to global coordinates
    rotation_matrix = np.vstack([e1, e2, e3]).T  # 3x3 matrix
    
    # Convert spherical to Cartesian in local coordinates
    actual_phi = sign_phi * phi
    local_relative_pos = relative(r, theta, actual_phi)

    # Rotate to global coordinates
    global_relative_pos = rotation_matrix @ local_relative_pos
    
    # Assign global position
    pos = focal_atom_coord + global_relative_pos
    return pos

def assign_cartesian_coordinates(mol, descriptors, verbose=False, idx=None):
    """Returns the molecule with assigned 3D coordinates."""  
    conf = Chem.Conformer(mol.GetNumAtoms())
    mol.AddConformer(conf)
    conf = mol.GetConformer()
    atom_positions = {}
    
    for id, descriptor in enumerate(descriptors):
        # print("Atom", id)
        r, theta, phi, sign = descriptor

        sdf_to_smiles = {i: i for i in range(mol.GetNumAtoms())}
        smiles_to_sdf = {i: i for i in range(mol.GetNumAtoms())}
        f, c1, c2, focal_atom_coord, c1_atom_coord, c2_atom_coord = find_ref_points(mol, sdf_to_smiles, smiles_to_sdf, atom_positions, id)

        # print("smiles idx:", id, "f:", f, "c1:", c1, "c2:", c2)

        pos = []
        
        if f != -1 and c1 != -1 and c2 != -1:
            pos = calc_spherical_to_cartesian(r, theta, phi, sign, focal_atom_coord, c1_atom_coord, c2_atom_coord)

        elif f != -1 and c1 != -1:
            if id != 2 and verbose:
                print(f"c2 was not found for mol {idx}, atom {id}")
                sys.stdout.flush()

            v1 = c1_atom_coord - focal_atom_coord
            e1 = v1 / np.linalg.norm(v1)

            #?
            if abs(theta - np.pi/2) < 1e-3 and (abs(phi) < 1e-3 or abs(phi - np.pi) < 1e-3):
                if abs(phi) < 1e-3:
                    # the point is on the ray from f to c1 and has "r" distance from f 
                    # print("fc1i")
                    pos = focal_atom_coord + r * e1
                else:
                    # print("c1fi.")
                    # the point is on the ray from c1 to f and has "r" distance from f 
                    pos = focal_atom_coord - r * e1

                atom_positions[id] = pos
                conf.SetAtomPosition(id, tuple(pos))
                continue
            
            # rng = [0.5,0.5,1]
            rng = np.random.RandomState(42)
            while True:
                arbitrary_vector = rng.rand(3) - 0.5  # Generate a vector with components between -0.5 and 0.5
                if np.linalg.norm(np.cross(e1, arbitrary_vector)) > 1e-6: # Ensure not collinear
                    break # Exit loop if we found a non-collinear vector
            
            # Calculate e2 using cross product with the arbitrary vector, this will really be 
            e2_dir = np.cross(e1, arbitrary_vector)
            norm_e2_dir = np.linalg.norm(e2_dir)
            if norm_e2_dir < 1e-6 and verbose: # Vectors are collinear
                print(f"collinear during decoding for mol {idx}")
                sys.stdout.flush()
                if np.allclose(arbitrary_vector, [1, 0, 0]):
                    e2_dir = np.cross(e1, np.array([0, 1, 0]))
                elif np.allclose(arbitrary_vector, [0, 1, 0]):
                    e2_dir = np.cross(e1, np.array([0, 0, 1]))
                else:
                    e2_dir = np.cross(e1, np.array([1, 0, 0]))
                norm_e2_dir = np.linalg.norm(e2_dir)

            e2 = e2_dir / norm_e2_dir # Normalize e2

            e3 = np.cross(e1, e2)
            e3 /= np.linalg.norm(e3)
            
            # Create rotation matrix
            rotation_matrix = np.vstack([e1, e2, e3]).T  # 3x3 matrix
            
            # Convert spherical to Cartesian in local coordinates
            actual_phi = sign * phi
            local_relative_pos = relative(r, theta, actual_phi)
            
            # Rotate to global coordinates
            global_relative_pos = rotation_matrix @ local_relative_pos
            
            # Assign global position
            pos = focal_atom_coord + global_relative_pos
                
        elif f != -1:
            if id != 1 and verbose:
                print(f"c1 was not found for mol {idx}, atom {id}")
                sys.stdout.flush()
            pos = focal_atom_coord + np.array([r, 0.0, 0.0])

        else:
            if id != 0 and verbose:
                print(f"f was not found for mol {idx}, atom {id}")
                sys.stdout.flush()
            pos = np.array([0.0, 0.0, 0.0])

        atom_positions[id] = pos

        conf.SetAtomPosition(id, tuple(pos))
        # print(f"Coords: {pos}")

    return mol

def parse_molecule_with_spherical_coordinates(embedded_smiles, verbose=False, idx = None):
    smiles, descriptors = parse_embedded_smiles(embedded_smiles)
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if exclude_h:
        mol = Chem.RemoveHs(mol)
        mol = Chem.SanitizeMol(mol)
    if mol is None: 
        raise ValueError(f"Invalid SMILES '{smiles}'")
    if verbose:
        print(f"Parsing molecule with spherical coordinates: {embedded_smiles} mol: {idx}")
        sys.stdout.flush()
    return assign_cartesian_coordinates(mol, descriptors, verbose, idx)


def reconstruct_sdf_from_embedded_smiles(embedded_smiles_list, output_sdf):
    """Reconstructs molecules from embedded SMILES and writes to an SDF file."""
    writer = Chem.SDWriter(output_sdf)
    for idx, embedded_smiles in enumerate(embedded_smiles_list):
        if idx == 200000:
            break
        try:
            if idx % 5000 == 0:
                print(idx)
                sys.stdout.flush()

            smiles, descriptors = parse_embedded_smiles(embedded_smiles)
            # print(f"\nProcessing Molecule {idx}...")
            # print(f"SMILES: {smiles}")
            # print(f"Descriptors: {descriptors}")

            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if exclude_h:
                mol = Chem.RemoveHs(mol)
                mol = Chem.SanitizeMol(mol)
            if mol is None:
                print(f"Molecule {idx+1}: Invalid SMILES '{smiles}'.")
                continue

            mol_with_coords = assign_cartesian_coordinates(mol, descriptors)
            
            # Optional: Optimize geometry (e.g., using UFF)
            # AllChem.UFFOptimizeMolecule(mol_with_coords)
            
            writer.write(mol_with_coords)
            # print(f"Molecule {idx}: Successfully reconstructed and written to SDF.")
        
        except Exception as e:
            print(f"Molecule {idx+1}: Error - {str(e)}")
    
    writer.close()
    print(f"\nAll molecules have been written to '{output_sdf}'")
    sys.stdout.flush()

if __name__ == '__main__':
    out_file = "outputs/output_inv_with_H.txt"
    if exclude_h:
        out_file = "outputs/output_inv.txt"
    sys.stdout = open(out_file, "w") 

    data_file = "/auto/home/filya/3DMolGen/molgen3D/data_processing/mol_9496_.jsonl"
    # data_file = "/auto/home/filya/3DMolGen/train_embedded_spherical_with_H/train_data_0.jsonl"
    # if exclude_h:
    #     data_file = "/auto/home/filya/3DMolGen/train_embedded_spherical/train_data_0.jsonl"

    embedded_smiles_list = []
    with open(data_file, 'r') as f:
        for line_number, line in enumerate(f):
            try:
                json_object = json.loads(line)
                if "conformers" in json_object:
                    conformers_data = json_object["conformers"]
                    if isinstance(conformers_data, dict) and "embedded_smiles" in conformers_data:
                        embedded_smiles_list.append(conformers_data["embedded_smiles"])
                    else:
                        print(f"Warning: 'embedded_smiles' key not found within 'conformers' on line {line_number} or 'conformers' is not in expected format.")
                else:
                    print(f"Warning: 'conformers' key not found in JSON object on line {line_number}.")

            except json.JSONDecodeError as e:
                print(f"Error: JSONDecodeError on line {line_number}: {e}")
                print(f"Line causing error: {line.strip()}")

    output_sdf = "mol_9496-rec.sdf"
    # output_sdf = "reconstructed_mols/reconstructed_molecules_1_with_H.sdf"
    # if exclude_h:
    #     output_sdf = "reconstructed_mols/reconstructed_molecules_1.sdf"
    reconstruct_sdf_from_embedded_smiles(embedded_smiles_list, output_sdf)