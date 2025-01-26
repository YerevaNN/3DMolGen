import numpy as np
from rdkit import Chem
import re

def parse_embedded_smiles(embedded_smiles):
    pattern = r'([A-Za-z][a-z]?)(<([^>]+)>)?'
    tokens = re.findall(pattern, embedded_smiles)
    
    atoms = []
    descriptors = []
    
    for token in tokens:
        atom = token[0]
        descriptor_str = token[2]
        if descriptor_str:
            try:
                atoms.append(atom)
                descriptor = [float(x) for x in descriptor_str.split(',')]
                if len(descriptor) != 4:
                    raise ValueError
                descriptors.append(descriptor)
            except:
                raise ValueError(f"Invalid descriptor format for atom {atom}: '{descriptor_str}'")
        else: # H
            pass
    
    smiles = re.sub(r'<[^>]+>', '', embedded_smiles)
    return smiles, atoms, descriptors

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

def assign_coordinates(mol, atoms, descriptors):
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
        print("Atom", id)
        r, theta, phi, sign = descriptor
        f = -1
        c1 = -1
        c2 = -1
        f = find_next_atom(id)
        if f != -1:
            c1 = find_next_atom(f)
            if c1 != -1:
                c2 = find_next_atom(c1)
        print("f", f, "c1", c1, "c2", c2)
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
                    local_relative_pos = spherical_to_cartesian(r, theta, phi) * sign
                    
                    # Rotate to global coordinates
                    global_relative_pos = rotation_matrix @ local_relative_pos
                    
                    # Assign global position
                    pos = p_f + global_relative_pos

        elif f != -1 and c1 != -1:
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
                local_relative_pos = spherical_to_cartesian(r, theta, phi) * sign
                
                # Rotate to global coordinates
                global_relative_pos = rotation_matrix @ local_relative_pos
                
                # Assign global position
                pos = p_f + global_relative_pos
                
        elif f != -1:
            p_f = atom_positions.get(f)
            if p_f is None:
                print(f"Atom {id}: Reference atom position not defined. Assigning default position at origin.")
                pos = np.array([0.0, 0.0, 0.0])
            else:
                pos = p_f + np.array([r, 0.0, 0.0])

        else:
            pos = np.array([0.0, 0.0, 0.0])
            print(f"Atom {id}: No reference atoms found. Assigning position at origin.")

        atom_positions[id] = pos

        conf.SetAtomPosition(id, tuple(pos))
        print(f"Coords: {pos}")

    return mol

def reconstruct_sdf_from_embedded_smiles(embedded_smiles_list, output_sdf):
    """Reconstructs molecules from embedded SMILES and writes to an SDF file."""
    writer = Chem.SDWriter(output_sdf)
    if writer is None:
        raise ValueError(f"Could not create SDF writer for {output_sdf}")
    
    for idx, embedded_smiles in enumerate(embedded_smiles_list):
        try:
            smiles, atoms, descriptors = parse_embedded_smiles(embedded_smiles)
            # print(f"\nProcessing Molecule {idx+1}:")
            # print(f"SMILES: {smiles}")
            # print(f"Atoms: {atoms}")
            # print(f"Descriptors: {descriptors}")

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Molecule {idx+1}: Invalid SMILES '{smiles}'.")
                continue
            # writer1 = Chem.SDWriter("mol1.sdf")
            # writer1.write(mol)
            # writer1.close()
            # mol = Chem.AddHs(mol)
            # writer2 = Chem.SDWriter("mol2.sdf")
            # writer2.write(mol)
            # writer2.close()

            mol_with_coords = assign_coordinates(mol, atoms, descriptors)
            
            # Optional: Optimize geometry (e.g., using UFF)
            # AllChem.UFFOptimizeMolecule(mol_with_coords)
            
            writer.write(mol_with_coords)
            print(f"Molecule {idx+1}: Successfully reconstructed and written to SDF.")
        
        except Exception as e:
            print(f"Molecule {idx+1}: Error - {str(e)}")
    
    writer.close()
    print(f"\nAll molecules have been written to '{output_sdf}'")

if __name__ == "__main__":
    embedded_smiles_list = [
        "C<0.0000,1.5708,0.0000,0>c<1.5107,1.5708,0.0000,0>1c<1.4015,1.5708,2.1141,1>c<1.3930,1.5936,2.1172,-1>c<1.4003,1.5746,2.1030,1>(C<1.5310,1.6010,2.0882,-1>2[C<1.4908,2.0614,2.0207,-1>H]c<1.3540,2.4228,2.4591,-1>3c<1.4545,1.5470,2.1450,-1>n<1.2937,1.6519,2.1781,-1>c<1.3970,1.5709,2.0394,1>c<1.3582,1.5811,2.1890,1>c<1.4451,1.5795,2.0978,1>3[N<1.3087,1.6038,2.0947,-1>]C<1.3973,1.5336,2.0709,-1>2=O<1.2168,1.7800,2.1312,-1>)c<1.3995,1.5778,2.0699,1>c<1.3945,1.5601,2.1043,1>1"
    ]
    output_sdf = "reconstructed_molecules.sdf"
    reconstruct_sdf_from_embedded_smiles(embedded_smiles_list, output_sdf)