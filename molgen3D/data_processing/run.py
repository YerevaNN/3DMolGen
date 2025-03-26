import sys
sys.path.append('/auto/home/filya/3DMolGen/molgen3D')

import os
import json
import os.path as osp

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
from get_cartesian_from_spherical import parse_molecule_with_spherical_coordinates
from get_spherical_from_cartesian import embed_coordinates_spherical

import importlib
import get_cartesian_from_spherical as to_reload1
import get_spherical_from_cartesian as to_reload2
importlib.reload(to_reload1)
importlib.reload(to_reload2)

global_rmsds = []
excluded = []
error = []
cnt = 0

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

def stat_log(rmsds):
    max_rmsd = np.max(rmsds)
    mean_rmsd = np.mean(rmsds)
    percentile95 = np.percentile(rmsds, 95)
    print(f"Maximum RMSD: {max_rmsd}")
    print(f"Mean RMSD: {mean_rmsd}")
    print(f"95th Percentile: {percentile95}")
    print(f"Cnt of excluded molecule conformers: {cnt}")
    # plt.figure(figsize=(8, 6))
    # plt.hist(rmsds, bins=50, alpha=0.7, color='blue', edgecolor='black')
    # plt.xlabel("RMSD")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of RMSDs")
    # plt.show()
    # print(np.sort(rmsds)[::-1])

def validate(raw_path, embedding_type, precision): #limit, dest_folder_path, indices_path,
    global cnt
    
    partitions = ["qm9"]

    if not os.path.exists("molecules"):
        os.makedirs("molecules")
    writer = Chem.SDWriter("molecules/special_mol.sdf")
    writer1 = Chem.SDWriter("molecules/special-rec.sdf")
    writer_error = Chem.SDWriter("molecules/error.sdf")

    embedding_function = embedding_func_selector[embedding_type]
    decoding_function = decoding_func_selector[embedding_type]

    for partition in partitions:
        mols = load_json(osp.join(raw_path, f"summary_{partition}.json"))
        for id, (mol_id, mol_dict) in tqdm(
            enumerate(mols.items()),
            total=len(mols),
            desc=f"Processing molecules of {partition}",
        ):
            if id < 21000:
                continue
            if id > 25500:
                break
            if id % 1000 == 999:
                stat_log(global_rmsds)
                sys.stdout.flush()
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
                    embedded_smiles = embedding_function(mol, canonical_smiles, atom_order, precision)
                    
                    # sample = {"canonical_smiles": canonical_smiles,
                    #             "geom_embed_coordinatesid": geom_id, 
                    #             "embedded_smiles": embedded_smiles}
                    mol1 = decoding_function(embedded_smiles)
                    rmsd = AllChem.GetBestRMS(mol, mol1)
                    if rmsd > 0.45:
                        cnt += 1
                        writer.write(mol)
                        writer1.write(mol1)
                        excluded.append((geom_id, conf_id))
                        raise ValueError(f"rmsd:{rmsd}")
                        # print(f"skipping -> mol_id:{geom_id}, con_id:{conf_id}, rmsd:{rmsd}")
                    else:
                        global_rmsds.append(rmsd)
                    

            except Exception as e:
                cnt += 1
                writer_error.write(mol)
                error.append(mol_id)
                print(f"Error: {e} for molecule {canonical_smiles}, id {id}")

    
    writer.close()
    writer1.close()
    writer_error.close()
    
    rmsds = np.array(global_rmsds)

    return rmsds


if not os.path.exists("outputs"):
    os.makedirs("outputs")
if not os.path.exists("indices"):
    os.makedirs("indices")
out_file = "outputs/run.txt"
sys.stdout = open(out_file, "w") 

raw_path = "/mnt/sxtn2/chem/GEOM_data/rdkit_folder"
embedding_type = "spherical"
rmsds = validate(raw_path, embedding_type, precision=4)
stat_log(rmsds)

save_to_json(excluded, "indices/excluded_molecules.json")
save_to_json(error, "indices/error_molecules.json")
