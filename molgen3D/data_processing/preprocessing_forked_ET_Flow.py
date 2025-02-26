import argparse
import os
import ast
from rdkit import Chem
import os.path as osp
from typing import Dict
import pickle
import json
import datamol as dm
import numpy as np
from loguru import logger as log
from tqdm import tqdm
<<<<<<< HEAD:data_processing/preprocessing_forked_ET_Flow.py
from get_spherical_from_cartesian import get_smiles, get_mol_descriptors
import re
=======
import random
random.seed(42)
>>>>>>> 392db4e (restructure project (dependencies not fixed)):molgen3D/data_processing/preprocessing_forked_ET_Flow.py

def load_pkl(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "rb") as f:
        return pickle.load(f)
    
def load_json(path):
<<<<<<< HEAD:data_processing/preprocessing_forked_ET_Flow.py
    with open(path, "r") as fp:  # Unpickling
=======
    """Loads json file"""
    with open(path, "r") as fp:  
>>>>>>> 392db4e (restructure project (dependencies not fixed)):molgen3D/data_processing/preprocessing_forked_ET_Flow.py
        return json.load(fp)
    
def embed_coordinates(mol, smiles, order):
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


def get_spherical_embedded(mol, canonical_smiles, embedded_smiles, geom_id, precision=4):
    descriptors_smiles_indexation = get_mol_descriptors(mol, geom_id)
    smiles_list, smiles_to_sdf, sdf_to_smiles = get_smiles(mol)
    descriptors = []
    for i in range(len(descriptors_smiles_indexation)):
        descriptors.append(descriptors_smiles_indexation[sdf_to_smiles[i]]) #descriptors[i] -> of the i-th atom in sdf indexation, sdf_to_smiles[i]-th atom in smiles indexation
    for atm_ind in range(len(descriptors)):
        desc_string = descriptors[atm_ind]
        number_strings = re.findall(r"[-+]?\d*\.\d+|\d+", desc_string)
        descriptor = [float(num) for num in number_strings]
        r, theta, abs_phi, sign_phi = descriptor
        embedded_smiles = embedded_smiles.replace(f":{atm_ind}]", f"<{r:.{precision}f},{theta:.{precision}f},{abs_phi:.{precision}f},{int(sign_phi)}>]")
    return {"canonical_smiles": canonical_smiles,
            "geom_id": geom_id, 
            "embedded_smiles": embedded_smiles}

embedding_func_selector = {
<<<<<<< HEAD:data_processing/preprocessing_forked_ET_Flow.py
    "cartesian": get_cartesian_embedded,
    "spherical": get_spherical_embedded
=======
    "cartesian": embed_coordinates
>>>>>>> 392db4e (restructure project (dependencies not fixed)):molgen3D/data_processing/preprocessing_forked_ET_Flow.py
}

def read_mol(
    mol_id: str,
    mol_dict,
    base_path: str,
    embedding_func
) -> Dict[str, np.ndarray]:
    precision = 4
    data = []
    try:
        mol_pickle = load_pkl(osp.join(base_path, mol_dict["pickle_path"]))
        confs = mol_pickle["conformers"]        
        for conf in confs:
            mol, geom_id = conf["rd_mol"], conf["geom_id"]
            canonical_smiles = dm.to_smiles(
                mol,
                canonical=True,
                explicit_hs=True,
                with_atom_indices=False,
                isomeric=False,
            )
            atom_order = list(map(int, ast.literal_eval(mol.GetProp('_smilesAtomOutputOrder'))))
            embedded_smiles = embed_coordinates(mol, canonical_smiles, atom_order)
            sample = {"canonical_smiles": canonical_smiles,
                      "geom_id": geom_id, 
                      "embedded_smiles": embedded_smiles}
            # print(sample)
            json_string = json.dumps(sample)
            data.append(json_string)

        return data

    except Exception as e:
        print(f"Skipping: {mol_id} due to {e}")
        return None

def save_processed_data(data, dest_path):
    data_chunk = 1000000
    for partition in ["train", "valid"]:
        os.makedirs(osp.join(dest_path, partition), exist_ok=True)
        part = data[partition]
        random.shuffle(part)
        for i in range(0, len(part), data_chunk):
            with open(osp.join(*[dest_path, partition, f"{partition}_data_{i//data_chunk}.jsonl"]), "w") as file:
                file.writelines(part[i:i+data_chunk])

def preprocess(raw_path: str, dest_folder_path: str, indices_path, embedding_type) -> None:
    log.info(f"Reading files from {raw_path}")
    partitions = ["qm9", "drugs"]
    total_confs, total_mols = 0, 0
    embedding_func = embedding_func_selector[embedding_type]

    for partition in partitions:
        train_list, valid_list = [], []
        
        dest_path = osp.join(dest_folder_path, partition.upper())
        train_indices = set(sorted(np.load(osp.join(*[indices_path, partition.upper(),
                                                      "train_indices.npy"]), allow_pickle=True)))
        val_indices = set(sorted(np.load(osp.join(*[indices_path, partition.upper(),
                                                    "val_indices.npy"]), allow_pickle=True)))
        log.info(f"{partition} indices contain train:{len(train_indices)}, valid:{len(val_indices)},"\
                 f" total:{len(train_indices)+len(val_indices)} samples")
        
        mols = load_json(osp.join(raw_path, f"summary_{partition}.json"))

        for _, (mol_id, mol_dict) in tqdm(
            enumerate(mols.items()),
            total=len(mols),
            desc=f"Processing molecules of {partition}",
        ):
            res = read_mol(
                mol_id,
                mol_dict,
                raw_path,
                embedding_func
            )

            if res is None:
                continue

            for en, key in enumerate(res, start=total_confs):
                if en in train_indices:
                    train_list.append(f"{key}\n")
                elif en in val_indices:
                    valid_list.append(f"{key}\n")

            total_mols += 1
            total_confs += len(res)
        
        save_processed_data(
            {"train": train_list, "valid": valid_list},
            dest_path
        )

    log.info(
        f"Processed: {total_mols} molecules, {total_confs} conformers."
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geom_path",
        "-p",
        type=str,
        required=False,
        default="/mnt/sxtn2/chem/GEOM_data/rdkit_folder",
        help="Path to the geom dataset rdkit folder",
    )
    parser.add_argument(
        "--dest",
        "-d",
        type=str,
        required=False,
        default="/mnt/sxtn2/chem/GEOM_data/geom_processed",
        help="Path to the destinaiton folder",
    )
    parser.add_argument(
        "--embedding_type",
        "-et",
        type=str,
        required=False,
        default="cartesian",
        help="Type of the embeddings. (cartesian, spherical)",
    )
    # destination path to store
    args = parser.parse_args()

    # get path to raw file
    path = args.geom_path
    assert osp.exists(path), f"Path {path} not found"
    
    #ET-Flow indices path
    indices_path = "/mnt/sxtn2/chem/GEOM_data/et_flow_indice/"

    # get distanation path
    dest = osp.join(args.dest, args.embedding_type + "_nonisomeric")
    os.makedirs(dest, exist_ok=True)
    log.info(f"Processed files will be saved at destination path: {dest}")

    # preprocess
    preprocess(path, dest, indices_path, args.embedding_type)