import pickle as pkl
import numpy as np
from tqdm import tqdm
import os
import copy

def main():
    data_path = "/auto/home/vover/3DMolGen/data/distinct_smi.pickle"
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        data = pkl.load(f)

    # Filtering Range
    lower_bound = -13.0
    upper_bound = 13.0

    original_total_mols = len(data)
    original_total_confs = 0
    
    filtered_data = {}
    
    kept_total_confs = 0
    removed_confs_count = 0
    removed_mols_count = 0

    print(f"Filtering conformers with coordinates outside [{lower_bound}, {upper_bound}]...")
    
    for geom_smiles, entry in tqdm(data.items()):
        confs = entry.get("confs", [])
        original_total_confs += len(confs)
        
        valid_confs = []
        for conf in confs:
            conformer = conf.GetConformer()
            is_valid = True
            
            # Check X, Y, and Z for every atom
            for i in range(conformer.GetNumAtoms()):
                pos = conformer.GetAtomPosition(i)
                if not (lower_bound <= pos.x <= upper_bound and 
                        lower_bound <= pos.y <= upper_bound and 
                        lower_bound <= pos.z <= upper_bound):
                    is_valid = False
                    break
            
            if is_valid:
                valid_confs.append(conf)
                kept_total_confs += 1
            else:
                removed_confs_count += 1
        
        if valid_confs:
            # Keep the molecule but only with valid conformers
            new_entry = copy.copy(entry)
            new_entry["confs"] = valid_confs
            new_entry["num_confs"] = len(valid_confs)
            filtered_data[geom_smiles] = new_entry
        else:
            removed_mols_count += 1

    print(f"\n=== Filtering Stats (Range: [{lower_bound}, {upper_bound}]) ===")
    print(f"Molecules:")
    print(f"  Original: {original_total_mols}")
    print(f"  Removed:  {removed_mols_count} ({100*removed_mols_count/original_total_mols:.2f}%)")
    print(f"  Kept:     {len(filtered_data)}")
    
    print(f"\nConformers:")
    print(f"  Original: {original_total_confs}")
    print(f"  Removed:  {removed_confs_count} ({100*removed_confs_count/original_total_confs:.4f}%)")
    print(f"  Kept:     {kept_total_confs}")

    # Optional: Save the filtered dataset
    # save_path = "/auto/home/vover/3DMolGen/data/distinct_smi_filtered_13.pickle"
    # with open(save_path, "wb") as f:
    #     pkl.dump(filtered_data, f)
    # print(f"\nFiltered dataset saved to {save_path}")

if __name__ == "__main__":
    main()
