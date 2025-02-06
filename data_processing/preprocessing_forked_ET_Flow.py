import argparse
import os
import os.path as osp
from typing import Dict
import pickle
import json
import datamol as dm
import numpy as np
from loguru import logger as log
from tqdm import tqdm
from get_spherical_from_cartesian import get_smiles, get_mol_descriptors
import re

def load_pkl(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "rb") as f:
        return pickle.load(f)
    
def load_json(path):
    with open(path, "r") as fp:  # Unpickling
        return json.load(fp)
    
def get_cartesian_embedded(mol, canonical_smiles, embedded_smiles, geom_id, precision=4):
    positions = mol.GetConformer().GetPositions()
    for atm_ind in range(0, len(positions)):
        x, y, z = positions[atm_ind]
        embedded_smiles = embedded_smiles.replace(f":{atm_ind}]", f"<{x:.{precision}f},{y:.{precision}f},{z:.{precision}f}>]")
    return {"canonical_smiles": canonical_smiles,
            "geom_id": geom_id, 
            "embedded_smiles": embedded_smiles}

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
    "cartesian": get_cartesian_embedded,
    "spherical": get_spherical_embedded
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
        sample_mol = confs[0]["rd_mol"]
        canonical_smiles = dm.to_smiles(
                sample_mol,
                canonical=True,
                explicit_hs=True,
                with_atom_indices=False,
                isomeric=True,
            )
        smiles_with_ind = dm.to_smiles(
                sample_mol,
                canonical=True,
                explicit_hs=True,
                with_atom_indices=True,
                isomeric=True,
                )
        for conf in confs:
            mol, geom_id = conf["rd_mol"], conf["geom_id"]
            sample = embedding_func(mol,
                                    canonical_smiles, 
                                    smiles_with_ind, 
                                    geom_id, 
                                    precision)
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
        required=True,
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
    dest = osp.join(args.dest, args.embedding_type)
    os.makedirs(dest, exist_ok=True)
    log.info(f"Processed files will be saved at destination path: {dest}")

    # preprocess
    preprocess(path, dest, indices_path, args.embedding_type)