import os
import pickle
import multiprocessing
import numpy as np
from posebusters import PoseBusters
import pandas as pd
import random
from concurrent.futures import ProcessPoolExecutor
import itertools
import submitit
from tqdm import tqdm

_POSEBUSTERS_CACHE = None

def _init_posebusters(config):
    # Each process initializes its own PoseBusters instance once (expensive import/setup)
    global _POSEBUSTERS_CACHE
    if _POSEBUSTERS_CACHE is None:
        _POSEBUSTERS_CACHE = PoseBusters(config=config)
    return _POSEBUSTERS_CACHE

def _bust_smi(smi, mols, config, full_report):
    # Validate input
    if not mols:
        return {"smiles": smi, "error": "No molecules provided"}
    
    if any(mol is None for mol in mols):
        return {"smiles": smi, "error": "Some molecules are None"}
    
    # Check if molecules have 3D coordinates
    from rdkit import Chem
    mols_3d = []
    for mol in mols:
        if mol.GetNumConformers() == 0:
            # Try to add hydrogens and generate 3D coords if missing
            try:
                mol = Chem.AddHs(mol)
                from rdkit.Chem import AllChem
                AllChem.EmbedMolecule(mol, randomSeed=42)
            except Exception:
                pass  # Continue with original molecule
        mols_3d.append(mol)
    
    mols = mols_3d
    # Debug logging
    # print(f"Processing molecule with {len(mols)} conformers")
    
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
    # Debug logging
    print(f"PoseBusters received {len(data)} molecules to process")
    
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for data, got {type(data)}")
    
    num_smiles = len(data)
    if num_smiles == 0:
        print("PoseBusters: No molecules to process")
        empty_df = pd.DataFrame(columns=['smiles','error'])
        summary = pd.DataFrame([{ 'smiles':'ALL','num_smiles':0,'num_conformers':0,'pass_percentage':0 }])
        return empty_df, summary, [], []

    num_conformers = sum(len(mols) for mols in data.values())
    print(f"Running PoseBusters on {num_smiles} SMILES with {num_conformers} total conformers using {max_workers} workers (config: {config})")

    smiles_items = list(data.items())

    # Adapt worker count: too many workers on small sets causes overhead
    if num_smiles < max_workers:
        max_workers = max(1, min(num_smiles, max_workers))

    results = []
    # Process in chunks to reduce peak memory
    total_chunks = (num_smiles + chunk_size - 1) // chunk_size
    print(f"Processing in {total_chunks} chunks...")
    
    from tqdm import tqdm as overall_tqdm
    print("\n" + "="*80)
    print("STARTING POSEBUSTERS EVALUATION")
    print("="*80)
    overall_pbar = overall_tqdm(total=num_smiles, desc="POSEBUSTERS OVERALL", unit="mol", 
                                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}", ncols=80)
    
    for chunk_idx, start in enumerate(range(0, num_smiles, chunk_size)):
        chunk = smiles_items[start:start+chunk_size]
        print(f"Processing PoseBusters chunk {chunk_idx+1}/{total_chunks} ({len(chunk)} molecules) with {max_workers} workers")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_bust_smi, smi, mols, config, full_report) for smi, mols in chunk]
            # Use concurrent.futures.as_completed instead of submitit.helpers.as_completed
            from concurrent.futures import as_completed
            chunk_pbar = tqdm(total=len(futures), desc=f"🔬 Chunk {chunk_idx+1}/{total_chunks}", unit="mol", 
                           bar_format="  {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]", 
                           disable=None)  # Let tqdm decide if it can display
            processed_in_chunk = 0
            for future in as_completed(futures):
                results.append(future.result())
                chunk_pbar.update(1)
                overall_pbar.update(1)  # Update overall progress
                processed_in_chunk += 1
                if processed_in_chunk % 10 == 0:
                    print(f"  → Processed {processed_in_chunk}/{len(futures)} molecules in chunk {chunk_idx+1}")
            chunk_pbar.close()
            print(f"  ✓ Completed chunk {chunk_idx+1}/{total_chunks} ({processed_in_chunk} molecules)")
    overall_pbar.close()
    print("\n" + "="*80)
    print("✅ POSEBUSTERS EVALUATION COMPLETED")
    print("="*80)
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

    success_count = len(results) - len(error_smiles)
    print(f"PoseBusters completed: {success_count}/{num_smiles} molecules processed successfully")
    if success_count < num_smiles:
        print(f"  - {len(error_smiles)} failed with errors")
        print(f"  - {len(fail_smiles)} failed posebuster checks")
    if error_smiles:
        print(f"Errors encountered for {len(error_smiles)} molecules")

    return df, summary, fail_smiles, error_smiles

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    print("This module is intended to be imported, not run directly.")