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
        atoms.append(atom)
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
            descriptors.append([0.0, 0.0, 0.0, 0.0])
    
    smiles = re.sub(r'<[^>]+>', '', embedded_smiles)
    return smiles, atoms, descriptors

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def find_next_atom(mol, smiles_to_sdf, smiles_id):
    sdf_id = smiles_to_sdf.get(smiles_id, -1)
    if sdf_id == -1:
        print(f"Not found sdf from smiles {smiles_id}")
        return -1
    
    atom = mol.GetAtomWithIdx(sdf_id)
    neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
    
    for i in range(smiles_id - 1, -1, -1):
        if smiles_to_sdf[i] in neighbors:
            return smiles_to_sdf[i]
    return -1

def is_collinear(p1, p2, p3, tolerance=1e-6):
    v1 = p2 - p1
    v2 = p3 - p1
    cross_product = np.cross(v1, v2)
    return np.allclose(cross_product, [0, 0, 0], atol=tolerance)

def assign_coordinates(mol, atoms, descriptors):
    """Returns the molecule with assigned 3D coordinates."""
    conf = Chem.Conformer(mol.GetNumAtoms())
    mol.AddConformer(conf)

    writer = Chem.SDWriter("mol.sdf")
    writer.write(mol)
    writer.close()
    
    return mol

def reconstruct_sdf_from_embedded_smiles(embedded_smiles_list, output_sdf):
    """Reconstructs molecules from embedded SMILES and writes to an SDF file."""
    writer = Chem.SDWriter(output_sdf)
    if writer is None:
        raise ValueError(f"Could not create SDF writer for {output_sdf}")
    
    for idx, embedded_smiles in enumerate(embedded_smiles_list):
        try:
            smiles, atoms, descriptors = parse_embedded_smiles(embedded_smiles)
            print(smiles, atoms, descriptors)
            if not atoms:
                print(f"Molecule {idx}: No atoms found.")
                continue
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Molecule {idx}: Invalid SMILES '{smiles}'.")
                continue
            mol = Chem.AddHs(mol)
            
            mol_with_coords = assign_coordinates(mol, atoms, descriptors)
            
            # Optional: Optimize geometry (e.g., using UFF)
            # AllChem.UFFOptimizeMolecule(mol_with_coords)
            
            # writer.write(mol_with_coords)
            # print(f"Molecule {idx}: Successfully reconstructed.")
        
        except Exception as e:
            print(f"Molecule {idx}: Error - {str(e)}")
    
    # writer.close()
    # print(f"All molecules have been written to {output_sdf}")

if __name__ == "__main__":
    embedded_smiles_list = [
        "C<0.0000,1.5708,0.0000,0>c<1.5107,1.5708,0.0000,0>1c<1.4015,1.5708,2.1141,1>c<1.3930,1.5936,2.1172,-1>c<1.4003,1.5746,2.1030,1>(C<1.5310,1.6010,2.0882,-1>2[C<1.4908,2.0614,2.0207,-1>H]c<1.3540,2.4228,2.4591,-1>3c<1.4545,1.5470,2.1450,-1>n<1.2937,1.6519,2.1781,-1>c<1.3970,1.5709,2.0394,1>c<1.3582,1.5811,2.1890,1>c<1.4451,1.5795,2.0978,1>3[N<1.3087,1.6038,2.0947,-1>]C<1.3973,1.5336,2.0709,-1>2=O<1.2168,1.7800,2.1312,-1>)c<1.3995,1.5778,2.0699,1>c<1.3945,1.5601,2.1043,1>1"
    ]
    output_sdf = "reconstructed_molecules.sdf"
    reconstruct_sdf_from_embedded_smiles(embedded_smiles_list, output_sdf)