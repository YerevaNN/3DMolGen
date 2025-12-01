from rdkit import Chem
import json
import datamol as dm
import os
import cloudpickle
import argparse
from collections import Counter
import random
import numpy as np
from collections import OrderedDict
from molgen3D.evaluation.rdkit_utils import correct_smiles, clean_confs, get_unique_smiles
from molgen3D.config.paths import get_ckpt, get_tokenizer_path, get_data_path, get_base_path


random.seed(43)
np.random.seed(43)

def load_pkl(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "rb") as f:
        return cloudpickle.load(f)

def process_dataset(dataset: str, process_type: str, base_path: str):
    # Determine paths based on dataset
    if dataset.upper() == "DRUGS":
        folder_name = "DRUGS"
        output_name = f"{process_type}_smi.pickle"
    elif dataset.upper() == "QM9":
        folder_name = "QM9"
        output_name = "qm9_smi.pickle"
    elif dataset.upper() == "XL":
        folder_name = "XL"
        output_name = "xl_smi.pickle"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    dataset_path = os.path.join(base_path, folder_name)
    test_mols_path = os.path.join(dataset_path, "test_smiles.csv")
    test_pkl_path = os.path.join(dataset_path, "test_mols.pkl")
    
    destination_path = get_base_path("data_root")
    
    print(f"Processing {dataset} dataset...")
    print(f"Test molecules path: {test_mols_path}")
    print(f"Pickle file path: {test_pkl_path}")
    print(f"Output file: {os.path.join(destination_path, output_name)}")

    # Load the dictionary of molecules (smiles -> [mol_objects])
    print("Loading test_mols.pkl...")
    mol_dic = load_pkl(test_pkl_path)
    print(f"Loaded {len(mol_dic)} molecules from pickle.")

    # Read the CSV
    with open(test_mols_path, 'r') as f:
        print(f"Reading CSV header: {f.readline().strip()}")
        test_mols = [(m.strip().split(',')) for m in f.readlines()]
    

    test_mols_parsed = []
    for m in test_mols:
        if len(m) >= 3:
            test_mols_parsed.append((m[0].strip(), int(m[1]), m[2].strip()))
        else:
            print(f"Skipping invalid line: {m}")

    processed_drugs_test = {}
    conf_count, mol_count = 0, 0
    
    for i in range(len(test_mols_parsed)):
        geom_smiles = test_mols_parsed[i][0]
        num_confs_csv = test_mols_parsed[i][1]
        geom_smiles_corrected = test_mols_parsed[i][2]

        try:
            # Retrieve conformers from the loaded pickle dictionary
            if geom_smiles not in mol_dic:
                print(f"Warning: SMILES {geom_smiles} not found in pickle file. Skipping.")
                continue
                
            true_confs = mol_dic[geom_smiles]
            
            num_confs = len(true_confs)

            if process_type == "clean":
                true_confs = clean_confs(geom_smiles, true_confs)
                num_confs = len(true_confs)
                if num_confs == 0:
                    continue
                corrected_smi = correct_smiles(true_confs)
            else:
                corrected_smi = None 

            gn_count = Counter([Chem.MolToSmiles(Chem.RemoveHs(c), canonical=True, isomericSmiles=True) for c in true_confs])

            sample_dict = {
                "geom_smiles": geom_smiles,
                "geom_smiles_c": geom_smiles_corrected,
                "confs": true_confs,
                "num_confs": num_confs,
                "pickle_path": None,
                "sub_smiles_counts": gn_count,
                "corrected_smi": corrected_smi,
            }
            processed_drugs_test[geom_smiles] = sample_dict
            mol_count += 1
            if i % 100 == 0:
                print(f"Processed {i}/{len(test_mols_parsed)}: num original confs {num_confs_csv} num correct confs {num_confs}, {geom_smiles[:20]}...")
            
        except Exception as e:
            print(f"{i} {geom_smiles} --- Error: {e}")
            import traceback
            traceback.print_exc()
        
        conf_count += num_confs

    print(f"number of processed molecules: {len(processed_drugs_test.keys())}")
    print(f"number of processed conformers: {conf_count}")
    print(f"number of processed molecules: {mol_count}")

    sorted_data = OrderedDict(
        sorted(processed_drugs_test.items(), key=lambda item: len(item[1]['geom_smiles']))
    )

    with open(os.path.join(destination_path, output_name),'wb') as f:
        cloudpickle.dump(sorted_data, f, protocol=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process test dataset for MolGen3D")
    parser.add_argument("--dataset", type=str, choices=["drugs", "qm9", "xl"], required=True, help="Dataset to process")
    parser.add_argument("--process_type", type=str, default="distinct", choices=["distinct", "clean"], help="Process type (default: distinct)")
    parser.add_argument("--base_path", type=str, default="/nfs/ap/mnt/sxtn2/chem/GEOM_data/torsional_diff_gdrive/extracted", help="Base path to GEOM data")
    
    args = parser.parse_args()
    
    process_dataset(args.dataset, args.process_type, args.base_path)