import numpy as np
from rdkit import Chem
import re
import sys
import json
import numpy as np

# ignore = []  

def parse_embedded_smiles(embedded_smiles):
    pattern = r'([A-Za-z][a-z]?)(<([^>]+)>)?'
    tokens = re.findall(pattern, embedded_smiles)
    
    descriptors = []
    
    for token in tokens:
        atom = token[0]
        descriptor_str = token[2]
        if descriptor_str:
            try:
                descriptor = [float(x) for x in descriptor_str.split(',')]
                if len(descriptor) != 4:
                    raise ValueError
                descriptors.append(descriptor)
            except:
                raise ValueError(f"Invalid descriptor format for atom {atom}: '{descriptor_str}'")
        else: # H
            pass
    
    smiles = re.sub(r'<[^>]+>', '', embedded_smiles) # exclude <...>s
    return smiles, descriptors

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def is_collinear(p1, p2, p3, tolerance=1e-6):
    v1 = p2 - p1
    v2 = p3 - p1
    cross_product = np.cross(v1, v2)
    return np.allclose(cross_product, [0, 0, 0], atol=tolerance)

def assign_coordinates(mol_id, mol, descriptors):
    """Returns the molecule with assigned 3D coordinates."""    
    def find_next_atom(id):
        atom = mol.GetAtomWithIdx(id)
        neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        for i in range(id - 1, -1, -1):
            if i in neighbors:
                return i
        return -1

    conf = Chem.Conformer(mol.GetNumAtoms())
    mol.AddConformer(conf)
    conf = mol.GetConformer()
    
    atom_positions = {}
    
    for id, descriptor in enumerate(descriptors):
        # print("Atom", id)
        r, theta, phi, sign = descriptor
        f = -1
        c1 = -1
        c2 = -1
        f = find_next_atom(id)
        if f != -1:
            c1 = find_next_atom(f)
            if c1 != -1:
                c2 = find_next_atom(c1)
        # print("f", f, "c1", c1, "c2", c2)
        pos = []
        
        if f != -1 and c1 != -1 and c2 != -1:
            p_f = atom_positions.get(f)
            p_c1 = atom_positions.get(c1)
            p_c2 = atom_positions.get(c2)
            
            if p_f is None or p_c1 is None or p_c2 is None:
                print(f"Atom {id}: Reference atoms positions not fully defined.")
                pos = np.array([0.0, 0.0, 0.0])
            else:
                v1 = p_c1 - p_f
                v2 = p_c2 - p_f
                normal = np.cross(v1, v2)
                norm_normal = np.linalg.norm(normal)
                
                if norm_normal < 1e-6:
                    print(f"Atom {id}: Reference atoms are collinear. Assigning default position at origin.")
                    pos = np.array([0.0, 0.0, 0.0])
                else:
                    # Orthonormal basis
                    e1 = v1 / np.linalg.norm(v1)
                    e3 = normal / norm_normal
                    e2 = np.cross(e3, e1)
                    
                    # Create rotation matrix from local to global coordinates
                    rotation_matrix = np.vstack([e1, e2, e3]).T  # 3x3 matrix
                    
                    # Convert spherical to Cartesian in local coordinates
                    actual_phi = sign * phi
                    local_relative_pos = spherical_to_cartesian(r, theta, actual_phi)

                    # Rotate to global coordinates
                    global_relative_pos = rotation_matrix @ local_relative_pos
                    
                    # Assign global position
                    pos = p_f + global_relative_pos

        elif f != -1 and c1 != -1:
            if id != 2:
                print(f"c2 was not found for atom {id}")
                # ignore.append(mol_id)

            p_f = atom_positions.get(f)
            p_c1 = atom_positions.get(c1)
            if p_f is None or p_c1 is None:
                print(f"Atom {id}: Reference atoms positions not fully defined. Assigning default position at origin.")
                pos = np.array([0.0, 0.0, 0.0])
            else:
                v1 = p_c1 - p_f
                e1 = v1 / np.linalg.norm(v1)
                
                rng = np.random.RandomState(42)
                while True:
                    arbitrary_vector = rng.rand(3) - 0.5  # Generate a vector with components between -0.5 and 0.5
                    if np.linalg.norm(np.cross(e1, arbitrary_vector)) > 1e-6: # Ensure not collinear
                        break # Exit loop if we found a non-collinear vector
               
                # Calculate e2 using cross product with the arbitrary vector, this will really be 
                e2_dir = np.cross(e1, arbitrary_vector)
                norm_e2_dir = np.linalg.norm(e2_dir)
                if norm_e2_dir < 1e-6: # Vectors are collinear
                    print("collinear")
                    if np.allclose(arbitrary_vector, [1, 0, 0]):
                        e2_dir = np.cross(e1, np.array([0, 1, 0]))
                    elif np.allclose(arbitrary_vector, [0, 1, 0]):
                        e2_dir = np.cross(e1, np.array([0, 0, 1]))
                    else:
                        e2_dir = np.cross(e1, np.array([1, 0, 0]))
                    norm_e2_dir = np.linalg.norm(e2_dir)

                e2 = e2_dir / norm_e2_dir # Normalize e2

                e3 = np.cross(e1, e2)
                
                # Create rotation matrix
                rotation_matrix = np.vstack([e1, e2, e3]).T  # 3x3 matrix
                
                # Convert spherical to Cartesian in local coordinates
                actual_phi = sign * phi
                local_relative_pos = spherical_to_cartesian(r, theta, actual_phi)
                
                # Rotate to global coordinates
                global_relative_pos = rotation_matrix @ local_relative_pos
                
                # Assign global position
                pos = p_f + global_relative_pos
                
        elif f != -1:
            if id != 1:
                # ignore.append(mol_id)
                print(f"c1 was not found for atom {id}")
            p_f = atom_positions.get(f)
            if p_f is None:
                print(f"Atom {id}: Reference atom position not defined. Assigning default position at origin.")
                pos = np.array([0.0, 0.0, 0.0])
            else:
                pos = p_f + np.array([r, 0.0, 0.0])

        else:
            if id != 0:
                # ignore.append(mol_id)
                print(f"f was not found for atom {id}")
            pos = np.array([0.0, 0.0, 0.0])

        atom_positions[id] = pos

        conf.SetAtomPosition(id, tuple(pos))
        # print(f"Coords: {pos}")

    return mol

def reconstruct_sdf_from_embedded_smiles(embedded_smiles_list, output_sdf):
    """Reconstructs molecules from embedded SMILES and writes to an SDF file."""
    writer = Chem.SDWriter(output_sdf)
    if writer is None:
        raise ValueError(f"Could not create SDF writer for {output_sdf}")
    for idx, embedded_smiles in enumerate(embedded_smiles_list):
        try:
            if idx == 1000:
                break
            smiles, descriptors = parse_embedded_smiles(embedded_smiles)
            # print(f"\nProcessing Molecule {idx+1}...")
            # print(f"SMILES: {smiles}")
            # print(f"Descriptors: {descriptors}")

            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.RemoveHs(mol)
            Chem.SanitizeMol(mol)
            if mol is None:
                print(f"Molecule {idx+1}: Invalid SMILES '{smiles}'.")
                continue

            mol_with_coords = assign_coordinates(idx, mol, descriptors)
            
            # Optional: Optimize geometry (e.g., using UFF)
            # AllChem.UFFOptimizeMolecule(mol_with_coords)
            
            writer.write(mol_with_coords)
            # print(f"Molecule {idx+1}: Successfully reconstructed and written to SDF.")
        
        except Exception as e:
            print(f"Molecule {idx+1}: Error - {str(e)}")
    
    writer.close()
    print(f"\nAll molecules have been written to '{output_sdf}'")

sys.stdout = open("output-inv.txt", "w") 
data_file = "/auto/home/filya/3DMolGen/train1/train_data_0.jsonl"
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

output_sdf = "reconstructed_molecules.sdf"
reconstruct_sdf_from_embedded_smiles(embedded_smiles_list, output_sdf)

# ignore = np.array(ignore)
# np.save("/auto/home/filya/3DMolGen/data_processing/ignore.npy", ignore)