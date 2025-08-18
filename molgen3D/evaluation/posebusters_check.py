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
from tqdm.auto import tqdm  # Import tqdm for progress bars

_POSEBUSTERS_CACHE = None

def _init_posebusters(config):
    # Each process initializes its own PoseBusters instance once (expensive import/setup)
    global _POSEBUSTERS_CACHE
    if _POSEBUSTERS_CACHE is None:
        _POSEBUSTERS_CACHE = PoseBusters(config=config)
    return _POSEBUSTERS_CACHE

def _bust_smi(smi, mols, config, full_report):
    try:
        b = _init_posebusters(config)
        dfb = b.bust(mols, None, None, full_report=full_report)
        m = dfb.mean().to_dict()
        m['pass_percentage'] = dfb.all(axis=1).mean() * 100
        m['smiles'] = smi
        m['error'] = ''
        return m
    except Exception as e:
        return {'smiles': smi, 'error': str(e)}

def run_all_posebusters(data, config="mol", full_report=False,
                        max_workers=16, fail_threshold=0.0, chunk_size=200):
    """Run PoseBusters in parallel with process-level caching & chunking.
    chunk_size limits the number of molecules submitted at once to reduce scheduler overhead
    and memory spikes when there are thousands of molecules.
    """
    num_smiles = len(data)
    if num_smiles == 0:
        empty_df = pd.DataFrame(columns=['smiles','error'])
        summary = pd.DataFrame([{ 'smiles':'ALL','num_smiles':0,'num_conformers':0,'pass_percentage':0 }])
        return empty_df, summary, [], []

    num_conformers = sum(len(mols) for mols in data.values())
    print(f"Running Posebusters on {num_smiles} molecules with {num_conformers} total conformers...")

    smiles_items = list(data.items())

    # Adapt worker count: too many workers on small sets causes overhead
    if num_smiles < max_workers:
        max_workers = max(1, min(num_smiles, max_workers))

    results = []
    # Process in chunks to reduce peak memory
    for start in range(0, num_smiles, chunk_size):
        chunk = smiles_items[start:start+chunk_size]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_bust_smi, smi, mols, config, full_report) for smi, mols in chunk]
            for future in tqdm(submitit.helpers.as_completed(futures), total=len(futures), desc=f"Posebusters chunk {start//chunk_size+1}", unit="mol"):
                results.append(future.result())

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

    print(f"Posebusters completed: {len(results) - len(error_smiles)} molecules processed successfully")
    if error_smiles:
        print(f"Errors encountered for {len(error_smiles)} molecules")

    return df, summary, fail_smiles, error_smiles

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    print("This module is intended to be imported, not run directly.")