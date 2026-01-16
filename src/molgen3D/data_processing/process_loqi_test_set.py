import os
import torch
import cloudpickle
import csv
from rdkit import Chem
from tqdm import tqdm
from collections import defaultdict, Counter, OrderedDict
import argparse

def process_loqi(pt_path, output_dir):
    print(f"Loading LoQI test set from {pt_path}...")
    try:
        # Load the PyTorch Geometric InMemoryDataset
        data = torch.load(pt_path, map_location='cpu', weights_only=False)
        if isinstance(data, tuple) and len(data) == 2:
            collated_data, slices = data
            print(f"Detected InMemoryDataset with {len(slices['smiles']) - 1} samples.")
        else:
            print(f"Unexpected data format: {type(data)}. Trying to iterate directly.")
            collated_data = data
            slices = None
    except Exception as e:
        print(f"Error loading {pt_path}: {e}")
        return

    mol_dict = defaultdict(list)
    
    num_samples = len(collated_data.mol)
    print(f"Processing {num_samples} Mol objects...")
    
    for i in tqdm(range(num_samples)):
        mol = collated_data.mol[i]
        # Generate canonical SMILES without hydrogens to use as dictionary key
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)
        mol_dict[smiles].append(mol)

    os.makedirs(output_dir, exist_ok=True)
    
    # Building the schema as per process_test_dataset.py
    processed_loqi_test = {}
    folder_name = "loqi"
    
    print("Converting to MolGen3D schema...")
    for geom_smiles, true_confs in tqdm(mol_dict.items()):
        num_confs = len(true_confs)
        
        # Calculate sub_smiles_counts (Counter of canonical SMILES for conformers)
        gn_count = Counter([
            Chem.MolToSmiles(Chem.RemoveHs(c), canonical=True, isomericSmiles=True) 
            for c in true_confs
        ])
        
        sample_dict = {
            "geom_smiles": geom_smiles,
            "geom_smiles_c": geom_smiles,  # No separate correction map for now
            "confs": true_confs,
            "num_confs": num_confs,
            "pickle_path": f"{folder_name}/{geom_smiles.replace('/', '_')}.pickle",
            "sub_smiles_counts": gn_count,
            "corrected_smi": None,  # process_type "distinct" default
        }
        processed_loqi_test[geom_smiles] = sample_dict

    # Sort by SMILES length as per original script
    sorted_data = OrderedDict(
        sorted(processed_loqi_test.items(), key=lambda item: len(item[1]['geom_smiles']))
    )

    pkl_path = os.path.join(output_dir, "test_mols.pkl")
    # Also save the final processed file in data_root if possible, 
    # but for now we follow the user's requested output_dir.
    schema_pkl_path = os.path.join(output_dir, "loqi_smi.pickle")
    csv_path = os.path.join(output_dir, "test_smiles.csv")

    print(f"Total unique SMILES: {len(mol_dict)}")
    
    print(f"Saving dictionary to {pkl_path}...")
    with open(pkl_path, 'wb') as f:
        cloudpickle.dump(dict(mol_dict), f)

    print(f"Saving schema-compliant pickle to {schema_pkl_path}...")
    with open(schema_pkl_path, 'wb') as f:
        cloudpickle.dump(sorted_data, f, protocol=4)

    print(f"Saving summary CSV to {csv_path}...")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "n_conformers", "corrected_smiles"])
        for smiles, mols in mol_dict.items():
            writer.writerow([smiles, len(mols), smiles])
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="test_small_h.pt")
    parser.add_argument("--output_dir", type=str, default="data/LOQI")
    args = parser.parse_args()
    
    process_loqi(args.input, args.output_dir)
