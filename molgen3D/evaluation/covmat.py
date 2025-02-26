from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from typing import Callable, List

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

def get_best_rmsd(probe, ref, use_alignmol=False):
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)

    try:
        if use_alignmol:
            return MA.AlignMol(probe, ref)
        else:
            rmsd = MA.GetBestRMS(probe, ref)
    except:  # noqa
        # rmsd = np.nan
        rmsd = None
    return rmsd

def get_rmsd(ref_mol, gen_mols: List, useFF=False, use_alignmol=False):
    num_gen = len(gen_mols)
    rmsd_vals = []
    for i in range(num_gen):
        gen_mol = gen_mols[i]
        if useFF:
            # print('Applying FF on generated molecules...')
            MMFFOptimizeMolecule(gen_mol)
        rmsd_vals.append(get_best_rmsd(gen_mol, ref_mol, use_alignmol=use_alignmol))

    return rmsd_vals

def calc_performance_stats(rmsd_array, threshold):
    # if rmsd_array == np.nan:
    #     print(rmsd_array)
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


def worker_fn(job, useFF=False, use_alignmol=False):
    smi, i_true, ref_mol, gen_mols = job
    rmsd_vals = get_rmsd(ref_mol, gen_mols, useFF=useFF, use_alignmol=use_alignmol)
    return smi, i_true, rmsd_vals


class CovMatEvaluator(object):
    """Coverage Recall Metrics Calculation for GEOM-Dataset"""

    def __init__(
        self,
        num_workers: int = 8,
        use_force_field: bool = False,
        use_alignmol: bool = False,
        thresholds: np.ndarray = np.arange(0.05, 3.05, 0.05),
        ratio: int = 2,
        filter_disconnected: bool = True,
        print_fn: Callable = print,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.use_force_field = use_force_field
        self.use_alignmol = use_alignmol
        self.thresholds = np.array(thresholds).flatten()

        self.ratio = ratio
        self.filter_disconnected = filter_disconnected

        self.print_fn = print_fn

    def __call__(self, ref_data, gen_data, start_idx=0):
        rmsd_results = {}
        jobs = []
        # true_mols = {}
        # gen_mols = {}
        missing_mols = 0
        for smiles in ref_data.keys():

            gen_mols = gen_data.get(smiles,None)
            if not gen_mols:
                # print(f"mol doesnt exist for {smiles}")
                missing_mols += 1
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

    
        def populate_results(res):
            smi, i_true, rmsd_vals = res
            rmsd_results[smi]["rmsd"][i_true] = rmsd_vals

        if self.num_workers > 1:
            p = Pool(self.num_workers)
            map_fn = p.imap_unordered
            p.__enter__()
        else:
            map_fn = map

        fn = partial(
            worker_fn, useFF=self.use_force_field, use_alignmol=self.use_alignmol
        )
        for res in tqdm(map_fn(fn, jobs), total=len(jobs)):
            populate_results(res)

        if self.num_workers > 1:
            p.__exit__(None, None, None)

        stats = []
        # print(rmsd_results)
        for res in rmsd_results.values():
            stats_ = calc_performance_stats(res["rmsd"], self.thresholds)
            stats.append(stats_)
        coverage_recall, amr_recall, coverage_precision, amr_precision = zip(*stats)

        results = {
            "CoverageR": np.array(coverage_recall),  # (num_mols, num_threshold)
            "MatchingR": np.array(amr_recall),  # (num_mols)
            "thresholds": self.thresholds,
            "CoverageP": np.array(coverage_precision),  # (num_mols, num_threshold)
            "MatchingP": np.array(amr_precision),  # (num_mols)
        }
        # print_conformation_eval_results(results)
        # print(f"no generations for {missing_mols} mols")

        
        return results, rmsd_results, missing_mols
        

def print_covmat_results(results, print_fn=print):
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
    
    with open("/auto/home/menuab/code/3DMolGen/gen_results/1e_1x_greedy/generation_resutls.pickle", 'rb') as f:
        model_preds = pickle.load(f)

    with open("/auto/home/menuab/code/3DMolGen/drugs_test_inference.pickle", 'rb') as f:
        true_mols = pickle.load(f)

    evaluator = CovMatEvaluator(num_workers=2)
    results, rmsd_results, missing_mols = evaluator(ref_data=true_mols, gen_data=model_preds)
    # print(results)
    log.info("Evaluation finished...")

    # get dataframe of results
    cov_df, matching_metrics = print_covmat_results(results)
    # print(cov_df)
    # print(matching_metrics)