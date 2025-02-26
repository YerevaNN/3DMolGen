from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import random
import json
import os

random.seed(0)
np.random.seed(0)

dataset = 'qm9'
exp_dir = f'../trained_models/{dataset}'
suffix = ''

with open("/auto/home/menuab/code/3DMolGen/drugs_test_mols_inference_2x.pickle", 'rb') as f:
    model_preds = pickle.load(f)

base_path = "/mnt/sxtn2/chem/GEOM_data"


test_mols_path = os.path.join(base_path, "geom_processed/test_smiles_corrected.csv")
drugs_file_path = os.path.join(base_path, "rdkit_folder/summary_drugs.json")
destination_path = "./drugs_test_inference.jsonl"

with open(drugs_file_path, "r") as f:
    drugs_summ = json.load(f)

with open(test_mols_path, 'r') as f:
    print(f.readline())
    test_mols = [(m.split(',')[:2]) for m in f.readlines()]
test_data = [(m[0].strip(), int(m[1])) for m in test_mols]

with open("/auto/home/menuab/code/3DMolGen/true_confs.pickle", 'rb') as f:
    true_mols = pickle.load(f)


def calc_performance_stats(true_confs, model_confs):
    
    threshold = np.arange(0, 2.5, .125)
    rmsd_list = []
    for tc in true_confs:
        for mc in model_confs:

            try:
                rmsd_val = AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
            except RuntimeError:
                return None
            rmsd_list.append(rmsd_val)

    rmsd_array = np.array(rmsd_list).reshape(len(true_confs), len(model_confs))

    coverage_recall = np.sum(rmsd_array.min(axis=1, keepdims=True) < threshold, axis=0) / len(true_confs)
    amr_recall = rmsd_array.min(axis=1).mean()

    coverage_precision = np.sum(rmsd_array.min(axis=0, keepdims=True) < np.expand_dims(threshold, 1), axis=1) / len(model_confs)
    amr_precision = rmsd_array.min(axis=0).mean()
    
    return coverage_recall, amr_recall, coverage_precision, amr_precision


def clean_confs(smi, confs):
    good_ids = []
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c), isomericSmiles=False)
        if conf_smi == smi:
            good_ids.append(i)
    return [confs[i] for i in good_ids]


coverage_recall, amr_recall, coverage_precision, amr_precision = [], [], [], []
test_smiles = []
threshold_ranges = np.arange(0, 2.5, .125)  # change for QM9

for smi, n_confs in tqdm(test_data):
    if not Chem.MolFromSmiles(smi):
        continue
    
    try:
        model_confs = model_preds[smi]
    except KeyError:
        print(f'no model prediction available: {smi}')
        coverage_recall.append(threshold_ranges*0)
        amr_recall.append(np.nan)
        coverage_precision.append(threshold_ranges*0)
        amr_precision.append(np.nan)
        test_smiles.append(smi)
        continue

    # failure if model can't generate confs                                                                                                                                                         
    if len(model_confs) == 0:
        print(f'model failed: {smi}')
        coverage_recall.append(threshold_ranges*0)
        amr_recall.append(np.nan)
        coverage_precision.append(threshold_ranges*0)
        amr_precision.append(np.nan)
        test_smiles.append(smi)
        continue
    
    try:
        true_confs = true_mols[smi]
    except KeyError:
        print(f'cannot find ground truth conformer file: {smi}')
        continue
    
    # remove reacted conformers
    true_confs = clean_confs(smi, true_confs)
    if len(true_confs) == 0:
        print(f'poor ground truth conformers: {smi}')
        continue
        
    stats = calc_performance_stats(true_confs, model_confs)
    if not stats:
        print(f'failure calculating stats: {smi, smi}')
        continue
        
    cr, mr, cp, mp = stats
    coverage_recall.append(cr)
    amr_recall.append(mr)
    coverage_precision.append(cp)
    amr_precision.append(mp)
    test_smiles.append(smi)

# np.save(f'{exp_dir}/stats{suffix}.npy', [coverage_recall, amr_recall, coverage_precision, amr_precision, test_smiles])

coverage_recall_vals = [stat[10] for stat in coverage_recall]
coverage_precision_vals = [stat[10] for stat in coverage_precision]

print(f'Recall Coverage: Mean = {np.mean(coverage_recall_vals)*100:.2f}, Median = {np.median(coverage_recall_vals)*100:.2f}')
print(f'Recall AMR: Mean = {np.nanmean(amr_recall):.4f}, Median = {np.nanmedian(amr_recall):.4f}')
print()
print(f'Precision Coverage: Mean = {np.mean(coverage_precision_vals)*100:.2f}, Median = {np.median(coverage_precision_vals)*100:.2f}')
print(f'Precision AMR: Mean = {np.nanmean(amr_precision):.4f}, Median = {np.nanmedian(amr_precision):.4f}')
