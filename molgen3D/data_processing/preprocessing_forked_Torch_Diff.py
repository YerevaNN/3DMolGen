import argparse
import os
import glob
from multiprocessing import Pool
from collections import defaultdict
from rdkit import Chem
import os.path as osp
from typing import Dict
import json
from functools import partial
import numpy as np
from loguru import logger as log
from tqdm import tqdm
import random
random.seed(42)

from molgen3D.utils.utils import load_pkl
from molgen3D.data_processing.utils import encode_cartesian_raw, dihedral_pattern

encoding_func_selector = {
    "cartesian": encode_cartesian_raw,
    # "spherical": get_spherical_embedded
}

def read_mol(
    mol_path: str,
    embedding_func: callable,
    failures: Dict[str, int],
    precision: int,
    ) -> Dict[str, np.ndarray]:
    
    mol_object = load_pkl(mol_path)
    geom_smiles = mol_object["smiles"]
    confs = mol_object["conformers"]  

    if '.' in geom_smiles:
        failures['dot_in_smile'] += 1
        return None

    # filter mols rdkit can't intrinsically handle
    geom_smiles_mol = Chem.MolFromSmiles(geom_smiles)
    if not geom_smiles_mol:
        failures['mol_from_smiles_failed'] += 1
        return None

    mol = mol_object['conformers'][0]['rd_mol']
    N = mol.GetNumAtoms()
    if not mol.HasSubstructMatch(dihedral_pattern):
        failures['no_substruct_match'] += 1
        return None

    if N < 4:
        failures['mol_too_small'] += 1
        return None

    canonical_smi = Chem.MolToSmiles(geom_smiles_mol, canonical=True, isomericSmiles=False)

    mols, weights = [], []
    for conf in confs:
        mol = conf['rd_mol']

        # filter for conformers that may have reacted
        try:
            conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), isomericSmiles=False)
        except Exception as e:
            print(e)
            continue
        if conf_canonical_smi != canonical_smi:
            failures['conf_canonical_smi_failed'] += 1
            continue

        mols.append(mol)
        weights.append(conf['boltzmannweight'])

    # return None if no non-reactive conformers were found
    if len(mols) == 0:
        return None

    normalized_weights = list(np.array(weights) / np.sum(weights))
    if np.isnan(normalized_weights).sum() != 0:
        normalized_weights = [1 / len(weights)] * len(weights)

    embedded_smiles = [embedding_func(mol, precision) for mol in mols]
    embedded_smiles = [es for _, es in sorted(zip(normalized_weights, embedded_smiles), reverse=True)][:30]
    samples = []
    for es in embedded_smiles:
        json_string = json.dumps({"canonical_smiles": es[1], "embedded_smiles": es[0]})
        samples.append(f"{json_string}\n")
    
    return samples

def save_processed_data(dataset, dest_path):
    data_chunk = 1000000
    for partition in ["train", "valid"]:
        save_path = osp.join(dest_path, partition)
        os.makedirs(save_path, exist_ok=True)
        log.info(f"Saving {partition} data to {save_path}")
        part = dataset[partition]
        random.shuffle(part)
        for i in range(0, len(part), data_chunk):
            with open(osp.join(*[dest_path, partition, f"{partition}_data_{i//data_chunk}.jsonl"]), "w") as file:
                file.writelines(part[i:i+data_chunk])

def preprocess(raw_path: str, 
               dest_folder_path: str, 
               indices_path, 
               embedding_type,
               num_workers=4, 
               precision=4,
               dataset_type="drugs",
               ) -> None: 
    total_confs, total_mols = 0, 0
    dataset = {"train": [], "valid": []}
    embedding_func = encoding_func_selector[embedding_type]
    failures = defaultdict(int)
    log.info(f"Reading files from {raw_path}")
    for split_idx, split_name in enumerate(["train", "valid"]):
        split = sorted(np.load(indices_path, allow_pickle=True)[split_idx])
        smiles = np.array(sorted(glob.glob(osp.join(raw_path, 'drugs/*.pickle'))))
        mol_paths = smiles[split]
        log.info(f"Processing split {split_name} with {len(mol_paths)} samples")

        # log.info(f"Preparing to process {len(mol_paths)} smiles")
        if num_workers > 1:
            p = Pool(num_workers)
            p.__enter__()
        read_mol_partial = partial(read_mol, 
                                   embedding_func=embedding_func, 
                                   failures=failures, 
                                   precision=precision)
        with tqdm(total=len(mol_paths)) as pbar:
            map_fn = p.imap if num_workers > 1 else map
            for data in map_fn(read_mol_partial, mol_paths):
                if data:
                    num_confs = len(data)
                    total_confs += num_confs
                    total_mols += 1
                    dataset[split_name].extend(data)
                pbar.update()

        log.info(f"Finished processing {dataset_type} {split_name} split.")
        log.info(f"Total conformers: {total_confs} total molecules: {total_mols}")
        log.info(f"Failures: {dict(failures)}")
        log.info(f"Processed {len(dataset[split_name])} samples for {split_name} split.")

    dest_path = osp.join(dest_folder_path, dataset_type.upper())
    os.makedirs(dest_path, exist_ok=True)   
    save_processed_data(dataset, dest_path)

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
    parser.add_argument(
        "--num_workers",
        "-nw",
        type=int,
        required=False,
        default=4,
        help="Number of workers to use for processing",
    )
    parser.add_argument(
        "--precision",
        type=int,
        required=False,
        default=4,
        help="Precision for the coordinates in the output smiles",
    )   
    parser.add_argument(
        "--dataset_type",
        "-dt",
        type=str,
        required=False,
        default="drugs",
        help="Type of the dataset (drugs, qm9)",
    )
    # destination path to store
    args = parser.parse_args()

    # get path to raw file
    path = args.geom_path
    assert osp.exists(path), f"Path {path} not found"
    
    #GeoMol indices path
    indices_path = "/mnt/sxtn2/chem/GEOM_data/splits/splits/split0.npy"

    # get distanation path
    dest = osp.join(args.dest, args.embedding_type + "isomeric")
    os.makedirs(dest, exist_ok=True)
    log.info(f"Processed files will be saved at destination path: {dest}")

    # preprocess
    preprocess(raw_path=path, 
               dest_folder_path=dest, 
               indices_path=indices_path,
               embedding_type=args.embedding_type,
               num_workers=args.num_workers, 
               precision=args.precision,
               dataset_type=args.dataset_type)
