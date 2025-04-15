import os
import pickle
import json
import numpy as np
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdmolops import RemoveHs

def load_pkl(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "rb") as f:
        return pickle.load(f)
    
def load_json(path):
    with open(path, "r") as fp:  
        return json.load(fp)

def get_best_rmsd(probe, ref, use_alignmol=False):
    probe = RemoveHs(probe)
    # display(probe)
    ref = RemoveHs(ref)
    # display(ref)
    try:
        if use_alignmol:
            return MA.AlignMol(probe, ref)
        else:
            rmsd = MA.GetBestRMS(probe, ref)
    except:  # noqa
        rmsd = np.nan

    return rmsd
