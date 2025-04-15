from rdkit import Chem
import json
import datamol as dm
import os
import cloudpickle
import random
import numpy as np

random.seed(43)
np.random.seed(43)

base_path = "/mnt/sxtn2/chem/GEOM_data"

test_mols_path = os.path.join(base_path, "geom_processed/test_smiles_corrected.csv")
drugs_file_path = os.path.join(base_path, "rdkit_folder/summary_drugs.json")
destination_path = "./drugs_test_inference_.pickle"

def load_pkl(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "rb") as f:
        return cloudpickle.load(f)

with open(drugs_file_path, "r") as f:
    drugs_summ = json.load(f)

with open(test_mols_path, 'r') as f:
    print(f.readline())
    test_mols = [(m.split(',')) for m in f.readlines()]
test_mols = [(m[0].strip(), int(m[1]), m[2].strip()) for m in test_mols]

def select_confs(mol_dic, geom_smiles_corrected):
    selected_mols = []
    confs = mol_dic['conformers']
    canonical_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(geom_smiles_corrected, sanitize=False), isomericSmiles=False)
    
    # random.shuffle(confs)  # shuffle confs
    
    # smiles_list = []
    for conf in confs:
        
        mol = conf['rd_mol']
        n_neighbors = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
        # if len(selected_mols) == 10:
        #     break
        try:
            conf_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), isomericSmiles=False)
            # if np.max(n_neighbors) > 4:
            #     # print("neighbors more than 4")
            #     continue
            if geom_smiles_corrected != conf_smiles:
                print(f"smiles mismatch {geom_smiles_corrected}\n{canonical_smiles=}\n{geom_smiles=}")
            #     continue
            # else:
            #     selected_mols.append(mol)
        except:
            print('failed parsing')
            continue
    # print(f"{mol_dic['uniqueconfs']=} {len(selected_mols)=}")
        
    # selected_smiles = max(set(smiles_list), key=smiles_list.count)

    # for conf in confs:
    #     if len(selected_mols) == 10:
    #         break
    #     mol = conf['rd_mol']
    #     try:
    #         conf_smi = dm.to_smiles(mol, canonical=True, isomeric=False, explicit_hs=False)
    #     except:
    #         print('failed parsing')
    #         continue
    #     if conf_smi == selected_smiles:
    #         selected_mols.append(mol)
    #         canonical_smiles = dm.to_smiles(mol, canonical=True, isomeric=False, explicit_hs=True)
    #     else:
    #         print(f"smiles mismatch \n{mol_dic['uniqueconfs']}---{geom_smiles}\n{selected_smiles=}\n{conf_smi=}")

    # print(f"{mol_dic['uniqueconfs']} {len(selected_mols)}")
    canonical_smiles_H = dm.to_smiles(selected_mols[0], canonical=True, isomeric=False, explicit_hs=True)
    return selected_mols, canonical_smiles_H

processed_drugs_test = {}
conf_count, mol_count = 0, 0
for i in range(len(test_mols)):
    try:
        geom_smiles = test_mols[i][0]
        geom_smiles_corrected = test_mols[i][2]
        mol_pickle = load_pkl(os.path.join(*[base_path, "rdkit_folder", drugs_summ[geom_smiles]['pickle_path']]))
        selected_mols, canonical_smiles = select_confs(mol_pickle, geom_smiles_corrected)
        num_confs = len(selected_mols)
        sample_dict = {
            "geom_smiles": geom_smiles,
            "geom_smiles_c": geom_smiles_corrected,
            "confs": selected_mols,
            "num_confs": num_confs,
            "pickle_path": drugs_summ[geom_smiles]['pickle_path'],
            "canonical_smiles": canonical_smiles
            }
        processed_drugs_test[geom_smiles] = sample_dict
        mol_count += 1
        print(geom_smiles, num_confs)
    except:
        print(i , '---')
    conf_count += num_confs

print(f"number of processed molecules: {len(processed_drugs_test.keys())}")
print(f"number of processed conformers: {conf_count}")
print(f"number of processed molecules: {mol_count}")

# with open(destination_path,'wb') as f:
#     cloudpickle.dump(processed_drugs_test, f, protocol=4)