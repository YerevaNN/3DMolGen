import os
import pickle
import multiprocessing
import numpy as np
from posebusters import PoseBusters
import pandas as pd
import random
from loky import ProcessPoolExecutor
import itertools
import submitit

def _bust_smi(smi, mols, config, full_report):
    try:
        b = PoseBusters(config=config)
        # run checks and collect per‐conformer results
        dfb = b.bust(mols, None, None, full_report=full_report)
        # average per-test metrics
        m = dfb.mean().to_dict()
        # percent of conformers that passed all tests
        m['pass_percentage'] = dfb.all(axis=1).mean() * 100
        m['smiles'] = smi
        m['error'] = ''
        return m
    except Exception as e:
        return {'smiles': smi, 'error': str(e)}

def run_all_posebusters(data, config="mol", full_report=False,
                        max_workers=16, fail_threshold=0.0):
    num_smiles = len(data)
    num_conformers = sum(len(mols) for mols in data.values())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            _bust_smi,
            data.keys(),
            data.values(),
            itertools.repeat(config),
            itertools.repeat(full_report),
        ))
    df = pd.DataFrame(results)
    error_smiles = df.loc[df['error'] != '', 'smiles'].tolist()
    if 'failure_rate' in df.columns:
        bad = df['failure_rate'] > fail_threshold
        fail_smiles = df.loc[bad, 'smiles'].tolist()
    else:
        fail_smiles = []
    summary = df[df['error']==''].mean(numeric_only=True).to_frame().T
    summary.insert(0, 'smiles', 'ALL')
    summary.insert(1, 'num_smiles', num_smiles)
    summary.insert(2, 'num_conformers', num_conformers)
    return df, summary, fail_smiles, error_smiles

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    print("This module is intended to be imported, not run directly.")