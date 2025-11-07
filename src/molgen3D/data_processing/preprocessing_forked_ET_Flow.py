import argparse
import os
import ast
from rdkit import Chem
import os.path as osp
from typing import Dict
import json
import datamol as dm
import numpy as np
from loguru import logger as log
from tqdm import tqdm
import re
import random
random.seed(42)

from molgen3D.utils.utils import load_pkl, load_json
from molgen3D.utils.data_processing_utils import encode_cartesian_raw    

encoding_func_selector = {
    "cartesian": encode_cartesian_raw,
    # "spherical": get_spherical_embedded
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
    embedding_func = encoding_func_selector[embedding_type]

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