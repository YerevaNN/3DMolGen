from rdkit import Chem
import os.path as osp
from statistics import mode, StatisticsError
import numpy as np
import pandas as pd

def clean_confs(smi, confs):
    good_ids = []
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c), isomericSmiles=False)
        if conf_smi == smi:
            good_ids.append(i)
    return [confs[i] for i in good_ids]

def correct_smiles(true_confs):

    conf_smis = []
    for c in true_confs:
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c))
        conf_smis.append(conf_smi)

    try:
        common_smi = mode(conf_smis)
    except StatisticsError:
        return None  # these should be cleaned by hand

    if np.sum([common_smi == smi for smi in conf_smis]) == len(conf_smis):
        return mode(conf_smis)
    else:
        print('consensus', common_smi)  # these should probably also be investigated manually
        return common_smi
        # return None

def extract_between(text, start_marker, end_marker):
    start = text.find(start_marker)
    if start != -1:
        start += len(start_marker)  # Move to the end of the start marker
        end = text.find(end_marker, start)
        if end != -1:
            return text[start:end]
    return ""  # Return empty if markers are not found

def get_unique_smiles(confs):
    smiles_counts: dict[str, int] = {}
    for conformer in confs:
        smiles: str = Chem.MolToSmiles(Chem.RemoveHs(conformer), isomericSmiles=False)
        if smiles not in smiles_counts:
            smiles_counts[smiles] = 0
        smiles_counts[smiles] += 1
    # unique_smiles: set[str] = set(smiles_counts.keys())
    return smiles_counts