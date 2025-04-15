import re
import ast
from rdkit import Chem

def decode_cartesian_raw(input_str):
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

def encode_cartesian_raw(mol, smiles, order):
    # Get the conformer's positions
    conf = mol.GetConformer()
    
    # Split the SMILES into tokens
    tokens = []
    i = 0
    n = len(smiles)
    while i < n:
        if smiles[i] == '[':
            # Parse bracketed atom
            j = i + 1
            while j < n and smiles[j] != ']':
                j += 1
            if j >= n:
                j = n - 1
            tokens.append(('atom', smiles[i:j+1]))
            i = j + 1
        elif smiles[i] in {'-', '=', '#', ':', '/', '\\'}:
            # Bond symbols
            tokens.append(('bond', smiles[i]))
            i += 1
        elif smiles[i].isdigit() or smiles[i] == '%':
            # Handle ring numbers
            if smiles[i] == '%':
                if i + 2 < n and smiles[i+1].isdigit() and smiles[i+2].isdigit():
                    tokens.append(('ring', smiles[i:i+3]))
                    i += 3
                else:
                    tokens.append(('ring', smiles[i]))
                    i += 1
            else:
                j = i
                while j < n and smiles[j].isdigit():
                    j += 1
                tokens.append(('ring', smiles[i:j]))
                i = j
        elif smiles[i] in {'(', ')'}:
            # Branch
            tokens.append(('branch', smiles[i]))
            i += 1
        elif smiles[i].isupper() or smiles[i].islower():
            # Element symbol followed by optional digits
            start = i
            # Parse element
            if smiles[i].isupper() and i + 1 < n and smiles[i+1].islower():
                i += 2
            else:
                i += 1
            # Parse digits
            while i < n and smiles[i].isdigit():
                i += 1
            tokens.append(('atom', smiles[start:i]))
        else:
            # Unknown character, skip
            i += 1
    
    # Extract atom tokens and validate count
    atom_tokens = [token[1] for token in tokens if token[0] == 'atom']
    if len(atom_tokens) != len(order):
        raise ValueError("Mismatch between atom tokens count and order list length.")
    
    # Generate coordinate strings for each atom in order
    coord_strings = []
    for atom_idx in order:
        pos = conf.GetAtomPosition(atom_idx)
        coord_str = f"<{pos.x:.4f},{pos.y:.4f},{pos.z:.4f}>"
        coord_strings.append(coord_str)
    
    # Replace atom tokens with embedded coordinates
    current_atom = 0
    new_tokens = []
    for token in tokens:
        if token[0] == 'atom':
            new_token = f"{token[1][:-1]}{coord_strings[current_atom]}]"
            new_tokens.append(new_token)
            current_atom += 1
        else:
            new_tokens.append(token[1])
    
    # Join tokens to form the new SMILES
    embedded_smiles = ''.join(new_tokens)
    return embedded_smiles