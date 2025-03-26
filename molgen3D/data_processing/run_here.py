import sys
sys.path.append('/auto/home/filya/3DMolGen/molgen3D/')

import os
# Create the molecules directory at the start
os.makedirs("molecules1", exist_ok=True)
os.makedirs("molecules1/error", exist_ok=True)

import json
import os.path as osp
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import ast
from loguru import logger as log
from tqdm import tqdm  


import datamol as dm
from rdkit import Chem
from rdkit.Chem import AllChem

from data_processing.preprocessing_forked_ET_Flow import load_pkl, load_json, embed_coordinates
from evaluation.inference import parse_molecule_with_coordinates
from data_processing.get_cartesian_from_spherical import parse_molecule_with_spherical_coordinates
from data_processing.get_spherical_from_cartesian import embed_coordinates_spherical

# Mapping for embedding and decoding functions:
embedding_func_selector = {
    "cartesian": embed_coordinates,
    "spherical": embed_coordinates_spherical
}
decoding_func_selector = {
    "cartesian": parse_molecule_with_coordinates,
    "spherical": parse_molecule_with_spherical_coordinates
}

def save_to_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)

def stat_log(rmsds, cnt):
    rmsds = np.array(rmsds)
    max_rmsd = np.max(rmsds) if len(rmsds) > 0 else None
    mean_rmsd = np.mean(rmsds) if len(rmsds) > 0 else None
    std_rmsd = np.std(rmsds) if len(rmsds) > 0 else None 
    percentile95 = np.percentile(rmsds, 95) if len(rmsds) > 0 else None
    print(f"Maximum RMSD: {max_rmsd}")
    print(f"Mean: {mean_rmsd}")
    print(f"Std: {std_rmsd}")
    print(f"95th Percentile: {percentile95}")

    rmsds = np.array(rmsds)
    count_gt_0_2 = np.sum(rmsds > 0.2)
    count_gt_0_4 = np.sum(rmsds > 0.4)
    count_gt_0_6 = np.sum(rmsds > 0.6)
    count_gt_0_8 = np.sum(rmsds > 0.8)
    count_gt_1 = np.sum(rmsds > 1)
    
    print(f"Count of RMSDs > 0.2: {count_gt_0_2}")
    print(f"Count of RMSDs > 0.4: {count_gt_0_4}")
    print(f"Count of RMSDs > 0.6: {count_gt_0_6}")
    print(f"Count of RMSDs > 0.8: {count_gt_0_8}")
    print(f"Count of RMSDs > 1: {count_gt_1}")

def process_chunk(start_idx, end_idx, raw_path, embedding_type, partition, precision):
    """
    Process a chunk of molecules from the QM9 partition.
    Each process writes its own SDF files (with names indicating its index range) and returns computed lists.
    """
    print("Processing chunk", start_idx, end_idx)
    sys.stdout.flush()
    rmsds = []
    excluded = []
    error = []
    
    mols = load_json(osp.join(raw_path, f"summary_{partition}.json"))
    # Convert the dictionary to a list for easier slicing
    mol_items = list(mols.items())
    
    for idx in range(start_idx, min(end_idx, len(mol_items))):
        mol_id, mol_dict = mol_items[idx]
        try:
            mol_pickle = load_pkl(os.path.join(raw_path, mol_dict["pickle_path"]))
            confs = mol_pickle["conformers"]    
            for conf_id, conf in enumerate(confs):
                mol, geom_id = conf["rd_mol"], conf["geom_id"]
                canonical_smiles = dm.to_smiles(
                    mol,
                    canonical=True,
                    explicit_hs=True,
                    with_atom_indices=False,
                    isomeric=False,
                )
                if '.' in canonical_smiles:
                    continue
                atom_order = list(map(int, ast.literal_eval(mol.GetProp('_smilesAtomOutputOrder'))))
                embedded_smiles = embedding_func_selector[embedding_type](mol, canonical_smiles, atom_order, precision)
                mol1 = decoding_func_selector[embedding_type](embedded_smiles)
                rmsd = AllChem.GetBestRMS(mol, mol1)
                if rmsd > 0.4:
                    excluded.append((geom_id, conf_id))
                    embedded_smiles = embedding_func_selector[embedding_type](mol, canonical_smiles, atom_order, precision, verbose=True, idx=idx)
                    mol1 = decoding_func_selector[embedding_type](embedded_smiles, True, idx)
                    # Save the molecule with high RMSD in a separate file
                    with Chem.SDWriter(f"molecules1/high_rmsd_{idx}.sdf") as writer1:
                        writer1.write(mol)
                    with Chem.SDWriter(f"molecules1/high_rmsd_recovered_{idx}.sdf") as writer2:
                        writer2.write(mol1)
                    print(f"rmsd:{rmsd} for id:{idx}, geom_id:{geom_id}")
                    sys.stdout.flush()
                    
                rmsds.append(rmsd)

        except Exception as e:
            with Chem.SDWriter(f"molecules1/error/error_{idx}.sdf") as error_writer:
                error_writer.write(mol)
            error.append(idx)
            print(f"Error: {e} for molecule {idx}, canonical_smiles: {canonical_smiles}")
            sys.stdout.flush()
    
    # Return the collected results for this chunk
    return {"rmsds": rmsds, "excluded": excluded, "error": error}

def process_wrapper(args):
    return process_chunk(*args)

def main():
    # Create output directories if not present
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("indices", exist_ok=True)

    sys.stdout = open("outputs/run_here3.txt", "w")

    raw_path = "/mnt/sxtn2/chem/GEOM_data/rdkit_folder"

    embedding_type = "spherical"
    precision = 4

    partitions = ["qm9"]
    for partition in partitions:
        summary_path = osp.join(raw_path, f"summary_{partition}.json")
        mols = load_json(summary_path)
        total = len(mols)
        start_index = 0
        end_index = total
        
        num_workers = min(cpu_count(), 30)  # Use up to 30 workers or the number of available CPUs
        print(f"Using {num_workers} workers")
        sys.stdout.flush()
        chunk_size = (total - 1) // num_workers + 1

        # Prepare arguments for each worker
        args_list = [
            (start_index + i * chunk_size, min(start_index + (i + 1) * chunk_size, end_index), raw_path, embedding_type, partition, precision)
            for i in range(num_workers)
            if start_index + i * chunk_size < end_index
        ]

        print("List", args_list)
        sys.stdout.flush()

        # Use multiprocessing to process chunks in parallel
        with Pool(num_workers) as pool:
            results = pool.map(process_wrapper, args_list)

        # Aggregate results from all workers
        total_rmsds = []
        # total_excluded = []
        # total_error = []
        for result in results:
            total_rmsds.extend(result["rmsds"])
            # total_excluded.extend(result["excluded"])
            # total_error.extend(result["error"])
        
        # cnt = len(total_excluded)
        cnt = 0
        stat_log(total_rmsds, cnt)
        
        # Write the aggregated results to JSON files
        # os.mkdir(f"indices_{partition}", exist_ok=True)
        # save_to_json(total_excluded, f"indices_{partition}/excluded_molecules.json")
        # save_to_json(total_error, f"indices_{partition}/error_molecules.json")

if __name__ == "__main__":
    main()