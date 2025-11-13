import os
import pickle
import multiprocessing
import math
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from posebusters import PoseBusters
import pandas as pd
import random
from concurrent.futures import ProcessPoolExecutor
import itertools
from tqdm import tqdm

try:
    import submitit  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    submitit = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

_POSEBUSTERS_CACHE = None
_SHARED_CONFORMERS: List["Mol"] = []

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

AggregationType = Literal["avg", "bool"]


def _compute_pass_fraction(report: pd.DataFrame) -> np.ndarray:
    """Compute per-conformer pass fraction from a PoseBusters report."""
    bool_df = report.select_dtypes(include=["bool"])
    if bool_df.empty:
        raise ValueError("PoseBusters report does not contain boolean checks to aggregate.")
    return bool_df.astype(float).mean(axis=1).to_numpy(dtype=float)


def _build_chunks(items: Sequence[Tuple[int, "Mol"]], max_workers: int) -> List[List[Tuple[int, "Mol"]]]:
    """
    Each workers gets chunk_size conformers to process.
    
    Args:
        items: Sequence of tuples (index, molecule)
        max_workers: Maximum number of workers to use

    Returns:
        List of lists of tuples (index, molecule) where each inner list is a chunk of conformers to process by a worker.
        Looks like this: [ [(0, conf1), (1, conf2), ...], [(X, confX), (X+1, confX+1), ...], ...] 
        where X is the chunk_size and the indices are consecutive.
    """
    if not items:
        return []
    if max_workers <= 1:
        return [list(items)]
    chunk_size = (len(items) + max_workers - 1) // max_workers # divide total # conformers by max_workers and round up
    # optimization idea:
    # just return a list of indices ranges, so we know the range of indices to splice for each worker?
    # so when spawning workers, we can just pass the indices range to the worker and let it splice the molecules from the original list?
    # or deep-copy the splice from the original list and pass that to the worker?
    return [list(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]


def _build_chunks_optimized(items_len: int, max_workers: int) -> List[Tuple[int, int]]:
    """Split a flattened conformer list into contiguous index ranges.

    Args:
        items_len: Total number of conformers in the flattened list.
        max_workers: Maximum number of worker processes available.

    Returns:
        List of ``(start, end)`` tuples describing half-open index intervals
        assigned to each worker, where ``start`` is inclusive and ``end`` is exclusive.
    """
    if items_len <= 0:
        return []
    if max_workers <= 1:
        return [(0, items_len)]
    chunk_size = math.ceil(items_len / max_workers)
    ranges: List[Tuple[int, int]] = []
    for start in range(0, items_len, chunk_size):
        end = min(start + chunk_size, items_len)
        ranges.append((start, end))
    return ranges
    

def _posebusters_chunk_worker(chunk: Sequence[Tuple[int, "Mol"]], config: str, full_report: bool) -> pd.DataFrame:
    if not chunk:
        return pd.DataFrame()
    indices, molecules = zip(*chunk)
    runner = _init_posebusters(config)
    df = runner.bust(list(molecules), None, None, full_report=full_report)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("PoseBusters returned unexpected data type.")
    df = df.reset_index(drop=True)
    df.insert(0, "conformer_index", list(indices))
    return df


def _posebusters_chunk_worker_by_range(index_range: Tuple[int, int], config: str, full_report: bool) -> pd.DataFrame:
    """Worker that looks up conformers by index from shared memory."""
    start, end = index_range
    if end <= start:
        return pd.DataFrame()
    if not _SHARED_CONFORMERS:
        raise RuntimeError("Shared conformer buffer is not initialized in worker process.")
    subset = _SHARED_CONFORMERS[start:end] # subset is a list of conformers from the shared memory (RDKit Mols)
    indices = list(range(start, end))
    runner = _init_posebusters(config)
    df = runner.bust(list(subset), None, None, full_report=full_report)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("PoseBusters returned unexpected data type.")
    df = df.reset_index(drop=True)
    df.insert(0, "conformer_index", indices)
    return df


def bust_cpu(
    rd_confs: Sequence["Mol"],
    *,
    num_workers: int = 8,
    aggregation_type: AggregationType = "avg",
    full_report: bool = False,
    config: str = "mol",
) -> Tuple[List[float], float]:
    """Run PoseBusters on a list of conformers and aggregate pass rates.

    Args:
        rd_confs: Sequence of RDKit molecules with 3D conformers to evaluate.
        num_workers: Number of processes to use when evaluating the conformers.
        aggregation_type: Either ``\"avg\"`` (fraction of tests passed) or
            ``\"bool\"`` (1 if all tests pass, 0 otherwise).
        full_report: Whether to request the full PoseBusters report.
        config: PoseBusters configuration key (``\"mol\"`` or ``\"redock\"``) (Assume `mol` for now).

    Returns:
        A tuple ``(pass_rates, average_pass_rate)`` where ``pass_rates`` is ordered
        with the input conformers and ``average_pass_rate`` is their arithmetic mean.
    """
    if aggregation_type not in {"avg", "bool"}:
        raise ValueError(f"Unsupported aggregation_type: {aggregation_type}")

    conformers: List["Mol"] = list(rd_confs)
    if not conformers:
        return [], float("nan")

    if any(mol is None for mol in conformers):
        raise ValueError("Encountered None entry in conformer list.")

    indexed_confs: List[Tuple[int, "Mol"]] = list(enumerate(conformers))
    max_workers = max(1, min(num_workers, len(indexed_confs)))

    frames: List[pd.DataFrame] = [] # list of dataframes, one for each worker (reduce/aggregate them later )

    def _can_use_fork_sharing() -> bool:
        """
        Defensive code to check if we can use fork sharing.
        If we are not running on linux or the start method is not "fork", we cannot use fork sharing.
        """
        try:
            method = multiprocessing.get_start_method(allow_none=True)  # type: ignore[arg-type]
        except TypeError:
            method = multiprocessing.get_start_method()
        except RuntimeError:
            method = None
        if method is None:
            try:
                method = multiprocessing.get_context().get_start_method()
            except RuntimeError:
                method = None
        return method == "fork"

    use_fork_sharing = _can_use_fork_sharing()

    tasks: List[object]
    worker_fn: object
    global _SHARED_CONFORMERS

    if use_fork_sharing: # if we can use fork sharing, we can use the shared memory to pass the conformers to the workers
        _SHARED_CONFORMERS = conformers
        task_ranges = _build_chunks_optimized(len(conformers), max_workers)
        if not task_ranges:
            return [], float("nan")
        tasks = task_ranges
        worker_fn = _posebusters_chunk_worker_by_range
    else: # if we cannot use fork sharing, we need to build chunks of conformers to pass to the workers
        chunks = _build_chunks(indexed_confs, max_workers)
        if not chunks:
            return [], float("nan")
        tasks = chunks
        worker_fn = _posebusters_chunk_worker

    try:
        if max_workers == 1:
            frame = worker_fn(tasks[0], config, full_report)  # type: ignore[index]
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frames.append(frame)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # submit the tasks to the executor, and wait for the results
                futures = [
                    executor.submit(worker_fn, task, config, full_report)  # type: ignore[arg-type]
                    for task in tasks
                    if (isinstance(task, tuple) and task[1] > task[0]) or (not isinstance(task, tuple) and task)
                ]
                for future in futures:
                    result = future.result()
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        frames.append(result)
    finally:
        if use_fork_sharing:
            _SHARED_CONFORMERS = []

    if not frames:
        return [], float("nan")

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return [], float("nan")
    combined.sort_values("conformer_index", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    pass_fraction = _compute_pass_fraction(combined)
    if aggregation_type == "avg": # if we are aggregating by average, we can just convert the pass fraction to a list of floats
        pass_rates = [float(value) for value in pass_fraction.tolist()]
    else: # if we are aggregating by boolean, we need to convert the pass fraction to a list of 1s and 0s
        pass_rates = [1.0 if np.isclose(value, 1.0) else 0.0 for value in pass_fraction.tolist()]

    average_pass_rate = float(np.mean(pass_rates)) if pass_rates else float("nan")
    return pass_rates, average_pass_rate


def bust_full_gens(
    smiles_to_confs: Dict[str, Sequence["Mol"]],
    *,
    num_workers: int = 8,
    config: str = "mol",
    full_report: bool = False,
) -> Tuple[pd.DataFrame, float]:
    """Evaluate PoseBusters pass rates for an entire generation dictionary.

    Args:
        smiles_to_confs: Mapping from canonical SMILES strings to sequences of RDKit conformers.
            Each conformer should be an RDKit ``Mol`` containing a single 3D conformer.
        num_workers: Number of worker processes to use. For ideal performance, align with
            the number of available CPU cores minus other workloads.
        config: PoseBusters configuration key (``"mol"`` or ``"redock"``).
        full_report: Whether to request the full PoseBusters report (slower, more columns).

    Returns:
        A tuple ``(summary_df, overall_pass_rate)`` where ``summary_df`` contains one row per
        SMILES with aggregated pass statistics (boolean style, i.e. a conformer counts as 1
        only if all PoseBusters checks pass) and ``overall_pass_rate`` is the arithmetic mean
        over all conformers in the input.
    """
    if not isinstance(smiles_to_confs, dict):
        raise TypeError(f"Expected dict for smiles_to_confs, got {type(smiles_to_confs)}")

    flattened: List["Mol"] = []
    conformer_smiles: List[str] = []
    for smiles, confs in smiles_to_confs.items():
        if not confs:
            continue
        for mol in confs:
            if mol is None:
                raise ValueError(f"Encountered None conformer for SMILES {smiles}")
            flattened.append(mol)
            conformer_smiles.append(smiles)

    if not flattened:
        empty_summary = pd.DataFrame(columns=["smiles", "num_conformers", "num_passed", "pass_pct"])
        return empty_summary, float("nan")

    pass_rates, overall_pass_rate = bust_cpu(
        flattened,
        num_workers=num_workers,
        aggregation_type="bool",
        full_report=full_report,
        config=config,
    )

    per_smiles: Dict[str, Dict[str, float]] = {}
    for smiles, passed in zip(conformer_smiles, pass_rates):
        entry = per_smiles.setdefault(smiles, {"num_conformers": 0, "num_passed": 0.0})
        entry["num_conformers"] += 1
        entry["num_passed"] += float(passed)

    records = []
    for smiles, stats in per_smiles.items():
        num_conf = int(stats["num_conformers"])
        num_passed = float(stats["num_passed"])
        pass_pct = num_passed / num_conf if num_conf else float("nan")
        records.append(
            {
                "smiles": smiles,
                "num_conformers": num_conf,
                "num_passed": num_passed,
                "pass_pct": pass_pct,
            }
        )

    summary_df = pd.DataFrame(records).sort_values("smiles").reset_index(drop=True)
    return summary_df, overall_pass_rate

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