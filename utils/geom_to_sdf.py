import os
import json
import pickle

from rdkit import Chem
from copy import deepcopy
from tqdm import tqdm

def convert_geom_drugs_to_sdf(json_path: str, output_sdf_path: str, include_2d_sdf: bool = False):
    """
    Convert GeomDrugs JSON to SDF file with molecule properties and conformers.
    
    Args:
        json_path: str, Path to the JSON file containing the GeomDrugs data.
        output_sdf_path: str, Path to save the output SDF file.
        include_2d_sdf: bool, Whether to also create a separate 2D SDF file, with only 2D representations.
    """
    with open(json_path, 'r') as f:
        geom_drugs_data = json.load(f)
    
    writer = Chem.SDWriter(output_sdf_path)
    if include_2d_sdf:
        writer_2d = Chem.SDWriter(output_sdf_path.replace(".sdf", "_2d.sdf"))

    errors = {
        "smi_parsing": 0,
        "pickle_loading": 0,
    }

    for smiles, mol_data in tqdm(geom_drugs_data.items(), desc="Processing Molecules"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            pickle_mol = Chem.MolFromSmiles(smiles)
            pickle_mol = Chem.AddHs(pickle_mol)

            if mol is None:
                print(f"Could not parse SMILES: {smiles}")
                errors["smi_parsing"] += 1
                continue
            
            try:
                # Ensure the pickle path is an absolute path or relative to the JSON file
                pickle_path = mol_data['pickle_path']
                if not os.path.isabs(pickle_path):
                    # If pickle path is relative, make it relative to the JSON file's directory
                    json_dir = os.path.dirname(os.path.abspath(json_path))
                    pickle_path = os.path.join(json_dir, pickle_path)
                
                with open(pickle_path, 'rb') as pickle_file:
                    pickle_data = pickle.load(pickle_file)
                
                for conf in pickle_data['conformers']:
                    rd_mol = conf['rd_mol']
                    conformer = Chem.Conformer(rd_mol.GetConformer(0))
                    pickle_mol.AddConformer(conformer, assignId=True)

            except Exception as e:
                print(f"Could not load pickle file {pickle_path}: {e}")
                errors["pickle_loading"] += 1
            
            # Set the molecule name as the pickle path
            mol.SetProp('_Name', mol_data['pickle_path'])
            pickle_mol.SetProp('_Name', mol_data['pickle_path'])
            
            # Add all JSON keys as properties
            for key, value in mol_data.items():
                mol.SetProp(key, str(value))
                pickle_mol.SetProp(key, str(value))
            
            writer.write(pickle_mol)
            if include_2d_sdf:
                writer_2d.write(mol)
        
        except Exception as e:
            print(f"Error processing molecule {smiles}: {e}")
    
    # Close the writer
    writer.close()
    
    print(f"Conversion complete. SDF file saved to {output_sdf_path}")

# Example usage
if __name__ == "__main__":
    input_json_path = "/auto/home/davit/3DMolGen/data/geomdrugs/censo/summary.json"
    output_sdf_path = "/auto/home/davit/3DMolGen/data/eval/output_censo.sdf"
    convert_geom_drugs_to_sdf(input_json_path, output_sdf_path, include_2d_sdf=True)