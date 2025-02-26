from rdkit import Chem
import json
import datamol as dm
import os
import pickle

base_path = "/mnt/sxtn2/chem/GEOM_data"


test_mols_path = os.path.join(base_path, "geom_processed/test_smiles_corrected.csv")
drugs_file_path = os.path.join(base_path, "rdkit_folder/summary_drugs.json")
destination_path = "./true_confs.pickle"

def load_pkl(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "rb") as f:
        return pickle.load(f)
    
with open(drugs_file_path, "r") as f:
    drugs_summ = json.load(f)

with open(test_mols_path, 'r') as f:
    print(f.readline())
    test_mols = [(m.split(',')[:2]) for m in f.readlines()]
test_mols = [(m[0].strip(), int(m[1])) for m in test_mols]

true_confs = {}
for i in range(len(test_mols)):
    conf_list = []
    try:
        geom_smiles = test_mols[i][0]
        geom_smiles_corrected = test_mols[i][2]
        mol_pickle = load_pkl(os.path.join(*[base_path, "rdkit_folder", drugs_summ[test_mols[i][0]]['pickle_path']]))
        conf = mol_pickle['conformers'][0]['rd_mol']
        conf_smiles = dm.to_smiles(conf, canonical=True, isomeric=True, explicit_hs=True)
        for conf in mol_pickle["conformers"]:
            conf_list.append(conf["rd_mol"])
        if i == 0:
            print(conf_list)
    except:
        print(i , '---')
    true_confs[geom_smiles] = conf_list

with open(destination_path,'wb') as f:
    pickle.dump(true_confs, f, protocol=pickle.HIGHEST_PROTOCOL)
