import re
import ast
from rdkit import Chem

def parse_molecule_with_coordinates(input_str):
    # Extract SMILES by removing coordinate annotations
    extracted_smiles = re.sub(r'<[^>]+>', '', input_str)
    
    # Parse the extracted SMILES
    mol = Chem.AddHs(Chem.MolFromSmiles(extracted_smiles))

    if mol is None:
        raise ValueError("Failed to parse the extracted SMILES.")
    canonical = Chem.MolToSmiles(mol, canonical=True, allHsExplicit=True)
    
    # Retrieve the atom output order from the molecule's properties
    if not mol.HasProp('_smilesAtomOutputOrder'):
        raise ValueError("SMILES atom output order not found.")
    atom_output_order = ast.literal_eval(mol.GetProp('_smilesAtomOutputOrder'))
    
    # Parse coordinates from the input string
    coords = []
    atom_pattern = re.compile(r'\[([^<]+)<([^>]+)>\]')
    for match in atom_pattern.finditer(input_str):
        coord_str = match.group(2)
        coord = list(map(float, coord_str.split(',')))
        coords.append(coord)
    
    # Verify coordinate count matches atom count
    if len(coords) != mol.GetNumAtoms():
        raise ValueError("Mismatch between number of coordinates and atoms.")
    
    # Create conformer and assign coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for input_idx, atom_idx in enumerate(atom_output_order):
        x, y, z = coords[input_idx]
        conf.SetAtomPosition(atom_idx, Chem.rdGeometry.Point3D(x, y, z))
    mol.AddConformer(conf)
    
    return mol