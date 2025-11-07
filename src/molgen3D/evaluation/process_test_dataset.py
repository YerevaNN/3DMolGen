from rdkit import Chem
import json
import datamol as dm
import os
import cloudpickle
from collections import Counter
import random
import numpy as np
from collections import OrderedDict
from molgen3D.evaluation.utils import correct_smiles, clean_confs, get_unique_smiles

random.seed(43)
np.random.seed(43)

base_path = "/mnt/sxtn2/chem/GEOM_data"

test_mols_path = os.path.join(base_path, "geom_processed/test_smiles_corrected.csv")
drugs_file_path = os.path.join(base_path, "rdkit_folder/summary_drugs.json")
destination_path = "/auto/home/menuab/code/3DMolGen/data/inference_set_clean_smiles/drugs_test_dict.pickle"

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

processed_drugs_test = {}
conf_count, mol_count = 0, 0
# zero_confs_count, less_confs_count = 0, 0
for i in range(len(test_mols)):
    geom_smiles = test_mols[i][0]
    try:
        geom_smiles_corrected = test_mols[i][2]
        mol_dic = load_pkl(os.path.join(*[base_path, "rdkit_folder", drugs_summ[geom_smiles]['pickle_path']]))

        true_confs = [conf['rd_mol'] for conf in mol_dic['conformers']]
        true_confs = clean_confs(geom_smiles, true_confs)
        num_confs = len(true_confs)
        if num_confs == 0:
            continue
        # if num_confs == 0:
        #     print(i, num_confs, test_mols[i][1], geom_smiles, '---')
        #     zero_confs_count += 1
        #     continue
        # if num_confs != test_mols[i][1]:
        #     less_confs_count += 1

        corrected_smi = correct_smiles(true_confs)
        # if corrected_smi != geom_smiles_corrected:
        #     print(num_confs)
        #     print(f"corrected smile mismatch: \n{corrected_smi=}\n{geom_smiles=}\n{geom_smiles_corrected=}")
        #     print('***')

        # smiles_dict = get_unique_smiles(true_confs)
        gn_count = Counter([Chem.MolToSmiles(Chem.RemoveHs(c)) for c in true_confs])

        sample_dict = {
            "geom_smiles": geom_smiles,
            "geom_smiles_c": geom_smiles_corrected,
            "confs": true_confs,
            "num_confs": num_confs,
            "pickle_path": drugs_summ[geom_smiles]['pickle_path'],
            "sub_smiles_counts": gn_count,
            "corrected_smi": corrected_smi,
            }
        processed_drugs_test[geom_smiles] = sample_dict
        mol_count += 1
        print(f"num original confs {test_mols[i][1]} num correct confs {num_confs}, {geom_smiles}")
        
    except:
        print(i , geom_smiles, '---')
    conf_count += num_confs

print(f"number of processed molecules: {len(processed_drugs_test.keys())}")
print(f"number of processed conformers: {conf_count}")
print(f"number of processed molecules: {mol_count}")

sorted_data = OrderedDict(
    sorted(processed_drugs_test.items(), key=lambda item: len(item[1]['geom_smiles']))
)
with open(destination_path,'wb') as f:
    cloudpickle.dump(sorted_data, f, protocol=4)