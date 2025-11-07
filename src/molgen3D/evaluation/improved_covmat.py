from copy import deepcopy
from functools import partial
from multiprocessing import Pool, shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Dict, Tuple
import numpy as np
import pandas as pd
from loguru import logger as log
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdchem import Conformer
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Geometry import Point3D
from tqdm import tqdm
import os
import json
import pickle
import multiprocessing as mp
import time


def get_best_rmsd(probe, ref, use_alignmol=False):
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)

    try:
        if use_alignmol:
            return MA.AlignMol(probe, ref)
        else:
            rmsd = MA.GetBestRMS(probe, ref)
    except:
        rmsd = None
    return rmsd


def get_rmsd_batch(ref_mol, gen_mols: List, useFF=False, use_alignmol=False):
    """Optimized version that processes multiple generated molecules efficiently."""
    num_gen = len(gen_mols)
    rmsd_vals = []
    
    # Apply force field optimization in batch if needed
    if useFF:
        for gen_mol in gen_mols:
            MMFFOptimizeMolecule(gen_mol)
    
    # Calculate RMSD values
    for gen_mol in gen_mols:
        rmsd_vals.append(get_best_rmsd(gen_mol, ref_mol, use_alignmol=use_alignmol))

    return rmsd_vals


def worker_fn_improved(job_batch, useFF=False, use_alignmol=False):
    """Improved worker function that processes multiple jobs in a batch."""
    results = []
    
    for smi, i_true, ref_mol, gen_mols in job_batch:
        rmsd_vals = get_rmsd_batch(ref_mol, gen_mols, useFF=useFF, use_alignmol=use_alignmol)
        results.append((smi, i_true, rmsd_vals))
    
    return results


def calc_performance_stats(rmsd_array, threshold):
    coverage_recall = np.mean(
        np.nanmin(rmsd_array, axis=1, keepdims=True) < threshold, axis=0
    )
    amr_recall = np.mean(np.nanmin(rmsd_array, axis=1))
    coverage_precision = np.mean(
        np.nanmin(rmsd_array, axis=0, keepdims=True) < np.expand_dims(threshold, 1),
        axis=1,
    )
    amr_precision = np.mean(np.nanmin(rmsd_array, axis=0))

    return coverage_recall, amr_recall, coverage_precision, amr_precision


class ImprovedCovMatEvaluator:
    """Improved Coverage Recall Metrics Calculator with better parallelization."""

    def __init__(
        self,
        num_workers: int = 8,
        use_force_field: bool = False,
        use_alignmol: bool = False,
        thresholds: np.ndarray = np.arange(0.05, 3.05, 0.05),
        batch_size: int = 10,
        chunk_size: int = 100,
        print_fn: Callable = print,
    ):
        self.num_workers = num_workers
        self.use_force_field = use_force_field
        self.use_alignmol = use_alignmol
        self.thresholds = np.array(thresholds).flatten()
        self.batch_size = batch_size  # Number of jobs per worker batch
        self.chunk_size = chunk_size  # Number of molecules per chunk
        self.print_fn = print_fn

    def __call__(self, ref_data: Dict, gen_data: Dict, start_idx: int = 0) -> Tuple[Dict, Dict, List]:
        rmsd_results = {}
        jobs = []
        missing_mols = []
        
        log.info(f"Preparing evaluation jobs for {len(ref_data)} reference molecules...")
        
        # Prepare all jobs
        for smiles in ref_data.keys():
            canonical_smiles = ref_data[smiles]['canonical_smiles']
            gen_mols = gen_data.get(canonical_smiles, None)
            
            if not gen_mols:
                missing_mols.append(smiles)
                continue

            ref_confs = ref_data[smiles]['confs']
            num_true = len(ref_confs)
            num_gen = len(gen_mols)

            rmsd_results[smiles] = {
                "n_true": num_true,
                "n_model": num_gen,
                "rmsd": np.nan * np.ones((num_true, num_gen)),
            }
            
            for i in range(num_true):
                jobs.append((smiles, i, ref_confs[i], gen_mols))

        log.info(f"Created {len(jobs)} total jobs, missing {len(missing_mols)} molecules")

        def populate_results(results_batch):
            for smi, i_true, rmsd_vals in results_batch:
                rmsd_results[smi]["rmsd"][i_true] = rmsd_vals

        start_time = time.time()

        if self.num_workers > 1:
            # Use ProcessPoolExecutor for better control and error handling
            self._process_parallel(jobs, populate_results)
        else:
            # Sequential processing
            self._process_sequential(jobs, populate_results)

        processing_time = time.time() - start_time
        log.info(f"RMSD calculation completed in {processing_time:.2f} seconds")

        # Calculate statistics
        stats = []
        for res in rmsd_results.values():
            stats_ = calc_performance_stats(res["rmsd"], self.thresholds)
            stats.append(stats_)
        
        coverage_recall, amr_recall, coverage_precision, amr_precision = zip(*stats)

        results = {
            "CoverageR": np.array(coverage_recall),
            "MatchingR": np.array(amr_recall),
            "thresholds": self.thresholds,
            "CoverageP": np.array(coverage_precision),
            "MatchingP": np.array(amr_precision),
        }

        return results, rmsd_results, missing_mols

    def _process_parallel(self, jobs: List, populate_results: Callable):
        """Process jobs using ProcessPoolExecutor with batching."""
        # Create batches of jobs
        job_batches = [jobs[i:i + self.batch_size] 
                      for i in range(0, len(jobs), self.batch_size)]
        
        log.info(f"Processing {len(job_batches)} batches with {self.num_workers} workers")

        fn = partial(
            worker_fn_improved, 
            useFF=self.use_force_field, 
            use_alignmol=self.use_alignmol
        )

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all batches
            futures = [executor.submit(fn, batch) for batch in job_batches]
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                try:
                    results_batch = future.result()
                    populate_results(results_batch)
                except Exception as e:
                    log.error(f"Error processing batch: {e}")

    def _process_sequential(self, jobs: List, populate_results: Callable):
        """Process jobs sequentially."""
        log.info("Processing jobs sequentially...")
        
        fn = partial(
            worker_fn_improved,
            useFF=self.use_force_field, 
            use_alignmol=self.use_alignmol
        )
        
        # Process in batches even for sequential processing
        job_batches = [jobs[i:i + self.batch_size] 
                      for i in range(0, len(jobs), self.batch_size)]
        
        for batch in tqdm(job_batches, desc="Processing batches"):
            results_batch = fn(batch)
            populate_results(results_batch)


class MemoryEfficientCovMatEvaluator:
    """Memory-efficient evaluator for very large datasets."""
    
    def __init__(
        self,
        num_workers: int = 8,
        use_force_field: bool = False,
        use_alignmol: bool = False,
        thresholds: np.ndarray = np.arange(0.05, 3.05, 0.05),
        max_molecules_in_memory: int = 1000,
    ):
        self.num_workers = num_workers
        self.use_force_field = use_force_field
        self.use_alignmol = use_alignmol
        self.thresholds = np.array(thresholds).flatten()
        self.max_molecules_in_memory = max_molecules_in_memory

    def __call__(self, ref_data: Dict, gen_data: Dict) -> Tuple[Dict, Dict, List]:
        """Process data in chunks to manage memory usage."""
        
        molecules = list(ref_data.keys())
        num_molecules = len(molecules)
        chunk_size = min(self.max_molecules_in_memory, num_molecules)
        
        log.info(f"Processing {num_molecules} molecules in chunks of {chunk_size}")
        
        # Initialize results storage
        all_rmsd_results = {}
        all_missing_mols = []
        all_stats = []
        
        # Process in chunks
        for i in range(0, num_molecules, chunk_size):
            chunk_molecules = molecules[i:i + chunk_size]
            
            log.info(f"Processing chunk {i//chunk_size + 1}/{(num_molecules + chunk_size - 1)//chunk_size}")
            
            # Create subset of data for this chunk
            chunk_ref_data = {mol: ref_data[mol] for mol in chunk_molecules}
            
            # Use regular evaluator for this chunk
            evaluator = ImprovedCovMatEvaluator(
                num_workers=self.num_workers,
                use_force_field=self.use_force_field,
                use_alignmol=self.use_alignmol,
                thresholds=self.thresholds
            )
            
            results, rmsd_results, missing_mols = evaluator(chunk_ref_data, gen_data)
            
            # Accumulate results
            all_rmsd_results.update(rmsd_results)
            all_missing_mols.extend(missing_mols)
            
            # Store statistics for each molecule
            for res in rmsd_results.values():
                stats_ = calc_performance_stats(res["rmsd"], self.thresholds)
                all_stats.append(stats_)
        
        # Combine all statistics
        if all_stats:
            coverage_recall, amr_recall, coverage_precision, amr_precision = zip(*all_stats)
            
            final_results = {
                "CoverageR": np.array(coverage_recall),
                "MatchingR": np.array(amr_recall),
                "thresholds": self.thresholds,
                "CoverageP": np.array(coverage_precision),
                "MatchingP": np.array(amr_precision),
            }
        else:
            final_results = {
                "CoverageR": np.array([]),
                "MatchingR": np.array([]),
                "thresholds": self.thresholds,
                "CoverageP": np.array([]),
                "MatchingP": np.array([]),
            }
        
        return final_results, all_rmsd_results, all_missing_mols


def print_covmat_results(results, print_fn=print):
    """Print coverage matrix results in a formatted way."""
    df = pd.DataFrame.from_dict(
        {
            "Threshold": results["thresholds"],
            "COV-R_mean": np.mean(results["CoverageR"], 0),
            "COV-R_median": np.median(results["CoverageR"], 0),
            "COV-P_mean": np.mean(results["CoverageP"], 0),
            "COV-P_median": np.median(results["CoverageP"], 0),
        }
    )
    matching_metrics = {
        "MAT-R_mean": np.mean(results["MatchingR"]),
        "MAT-R_median": np.median(results["MatchingR"]),
        "MAT-P_mean": np.mean(results["MatchingP"]),
        "MAT-P_median": np.median(results["MatchingP"]),
    }
    return df, matching_metrics


if __name__ == "__main__":
    # Example usage
    log.info("Testing improved CovMatEvaluator...")
    
    # Load your data here
    # with open("path_to_model_preds.pickle", 'rb') as f:
    #     model_preds = pickle.load(f)
    # 
    # with open("path_to_true_mols.pickle", 'rb') as f:
    #     true_mols = pickle.load(f)
    
    # # Test different configurations
    # evaluator_configs = [
    #     {"num_workers": mp.cpu_count(), "batch_size": 10},
    #     {"num_workers": mp.cpu_count()//2, "batch_size": 20},
    #     {"num_workers": 1, "batch_size": 50},  # Sequential baseline
    # ]
    # 
    # for i, config in enumerate(evaluator_configs):
    #     log.info(f"Testing configuration {i+1}: {config}")
    #     
    #     evaluator = ImprovedCovMatEvaluator(**config)
    #     
    #     start_time = time.time()
    #     results, rmsd_results, missing_mols = evaluator(ref_data=true_mols, gen_data=model_preds)
    #     end_time = time.time()
    #     
    #     log.info(f"Configuration {i+1} completed in {end_time - start_time:.2f} seconds")
    #     
    #     cov_df, matching_metrics = print_covmat_results(results)
    #     log.info(f"Results: {matching_metrics}")
    
    print("Improved CovMatEvaluator is ready for use!") 