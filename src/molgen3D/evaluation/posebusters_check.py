import json
import logging
import math
import multiprocessing
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from posebusters import PoseBusters
except ImportError:  # pragma: no cover - optional dependency
    PoseBusters = None  # type: ignore[assignment]

try:
    import submitit  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    submitit = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

_POSEBUSTERS_CACHE = None
_SHARED_CONFORMERS: List["Mol"] = []
logger = logging.getLogger(__name__)

def _init_posebusters(config):
    # Each process initializes its own PoseBusters instance once (expensive import/setup)
    global _POSEBUSTERS_CACHE
    if PoseBusters is None:
        raise ImportError(
            "PoseBusters is not installed. Please install the 'posebusters' package to run these checks."
        )
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
    """Worker that processes a chunk of conformers."""
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
    indices = list(range(start, end)) # indices is a list of indices of the conformers to process
    # initialize the PoseBusters instance (each process initializes its own PoseBusters instance once (expensive import/setup)
    # (maybe it can be optimized to be a global instance that is shared?)) (i think this is the case, but i need to check)
    runner = _init_posebusters(config)
    df = runner.bust(list(subset), None, None, full_report=full_report)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("PoseBusters returned unexpected data type.")
    df = df.reset_index(drop=True)
    df.insert(0, "conformer_index", indices)
    return df


def _can_use_fork_sharing() -> bool:
    """Return True when we can rely on ``fork`` semantics for shared memory."""
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


def _collect_posebusters_report(
    rd_confs: Sequence["Mol"],
    num_workers: int,
    full_report: bool,
    config: str,
) -> pd.DataFrame:
    """Run PoseBusters and return the concatenated per-conformer report."""
    conformers: List["Mol"] = list(rd_confs)
    if not conformers:
        return pd.DataFrame()

    if any(mol is None for mol in conformers):
        raise ValueError("Encountered None entry in conformer list.")

    indexed_confs: List[Tuple[int, "Mol"]] = list(enumerate(conformers))
    max_workers = max(1, min(num_workers, len(indexed_confs)))

    frames: List[pd.DataFrame] = []
    # fork sharing is a way to share memory between processes, it is a way to avoid the overhead of creating a new process for each conformer
    use_fork_sharing = _can_use_fork_sharing()

    tasks: List[object]
    worker_fn: object
    # shared conformers is a list of conformers that are shared between processes, it is a way to avoid the overhead of creating a new process for each conformer
    global _SHARED_CONFORMERS 

    # here we are deciding which worker function to use based on the use_fork_sharing flag
    if use_fork_sharing:
        _SHARED_CONFORMERS = conformers
        # task ranges are a list of tuples, each tuple contains the start and end index of a chunk of conformers to process
        task_ranges = _build_chunks_optimized(len(conformers), max_workers)
        if not task_ranges:
            return pd.DataFrame()
        tasks = task_ranges
        worker_fn = _posebusters_chunk_worker_by_range
    else:
        chunks = _build_chunks(indexed_confs, max_workers)
        if not chunks:
            return pd.DataFrame()
        tasks = chunks
        worker_fn = _posebusters_chunk_worker

    try:
        if max_workers == 1:
            frame = worker_fn(tasks[0], config, full_report)  # type: ignore[index]
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frames.append(frame)
        else:
            # here we are using the ProcessPoolExecutor to run the worker function for each chunk of conformers
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # here we are submitting the worker function for each chunk of conformers to the ProcessPoolExecutor
                futures = [
                    executor.submit(worker_fn, task, config, full_report)  # type: ignore[arg-type]
                    for task in tasks
                    if (isinstance(task, tuple) and task[1] > task[0]) or (not isinstance(task, tuple) and task)
                ]
                for future in futures:
                    result = future.result()
                    # here we are appending the result of the worker function to the frames list
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        frames.append(result)
    finally:
        if use_fork_sharing:
            # here we are clearing the shared conformers list
            _SHARED_CONFORMERS = []

    if not frames:
        return pd.DataFrame()

    # here we are concatenating the frames list into a single dataframe
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    combined.sort_values("conformer_index", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


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

    report_df = _collect_posebusters_report(
        rd_confs,
        num_workers=num_workers,
        full_report=full_report,
        config=config,
    )
    if report_df.empty:
        return [], float("nan")

    pass_fraction = _compute_pass_fraction(report_df)
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
    fail_threshold: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Evaluate PoseBusters pass rates for an entire generation dictionary.

    Args:
        smiles_to_confs: Mapping from SMILES strings to lists of RDKit conformers.
        num_workers: Number of worker processes devoted to PoseBusters.
        config: PoseBusters configuration key (``"mol"`` or ``"redock"``).
        full_report: Whether to request the extended PoseBusters report.
        fail_threshold: Allowed failure rate per SMILES (computed using boolean aggregation).
            Note: This parameter is kept for API compatibility but is not used.
            Users can filter ``per_smiles_df`` by ``pass_percentage`` directly.

    Returns:
        Tuple ``(per_smiles_df, summary_df, overall_pass_rate)``.

        * ``per_smiles_df`` matches the schema in ``tests/evaluation/sample_by_smiles_df.csv``:
          PoseBusters boolean checks per SMILES, ``pass_percentage`` (0–100), ``smiles``,
          ``num_of_conformers``, and ``error``. Users can filter this DataFrame by
          ``pass_percentage`` to identify failed SMILES.
        * ``summary_df`` mirrors ``tests/evaluation/sample_summary_df.csv``: a single ``"ALL"``
          row containing dataset-wide counts, PoseBusters check means, and the overall pass rate.
        * ``overall_pass_rate`` is the conformer-level arithmetic mean of ``pass_bool``
          (guards against averaging per-SMILES percentages, e.g., 0.33 vs. 0.375).

    Note:
        PoseBusters does not return an ``error`` column - it raises exceptions on errors.
        The ``error`` column in ``per_smiles_df`` is part of the schema but will always be empty.
    """
    if not isinstance(smiles_to_confs, dict):
        raise TypeError(f"Expected dict for smiles_to_confs, got {type(smiles_to_confs)}")

    flattened: List["Mol"] = []
    # Book-keeping strategy for per-conformer → SMILES provenance:
    #   unique_smiles               : ["CCO", "c1ccccc1", "CN", ...]  (each string stored once)
    #   smiles_to_idx               : {"CCO": 0, "c1ccccc1": 1, ...}  (string → integer ID)
    #   flattened_idx_to_smiles_idx : [0, 0, 1, 2, 2, ...]            (parallel to `flattened`)
    #
    # Example with {"CCO": [mol0, mol1], "c1cc": [mol2], "CN": [mol3, mol4]}:
    #   flattened                  = [mol0, mol1, mol2, mol3, mol4]
    #   flattened_idx_to_smiles_idx= [0, 0, 1, 2, 2]
    #   unique_smiles[flattened_idx_to_smiles_idx[3]] == "CN"
    #
    # This mimics the conceptual list [(smiles_id, conformer), ...] while keeping memory
    # usage low by storing small ints instead of repeating long SMILES strings.
    flattened_idx_to_smiles_idx: List[int] = []
    unique_smiles: List[str] = []
    smiles_to_idx: Dict[str, int] = {}
    # Track any input sanitization we perform (missing conformers, None entries, etc.) so
    # reporting layers can surface those details alongside PoseBusters errors.
    input_validation_notes: Dict[str, List[str]] = {}

    # Step 1 — defensive preprocessing: filter invalid conformers but record what was removed.
    for smiles, confs in smiles_to_confs.items():
        if not confs: # Case 1: if there are no conformers for a given SMILES, add it to the input_validation_notes set
            input_validation_notes.setdefault(smiles, []).append("no_conformers")
            continue

        dropped_none = 0
        valid_confs: List["Mol"] = []

        for mol in confs:
            if mol is None:  # Case 2: if there is a None conformer, skip it
                dropped_none += 1
                continue
            else:
                valid_confs.append(mol)
        
        # if there are no valid conformers for a given SMILES, skip the rest of the loop and continue to the next SMILES
        if not valid_confs:
            continue

        if dropped_none == len(confs):  # Case 3: if there are no valid conformers for a given SMILES, skip it
            input_validation_notes.setdefault(smiles, []).append(f"{dropped_none}_none_entries")
            continue  # skip the rest of the loop and continue to the next SMILES

        elif dropped_none > 0: # Case 4: if there are some None conformers for a given SMILES, add it to the input_validation_notes set
            input_validation_notes.setdefault(smiles, []).append(f"{dropped_none}_none_entries")

        # Step 2 — assign a stable integer ID for this SMILES once, reusing it for all conformers.
        smiles_idx = smiles_to_idx.setdefault(smiles, len(unique_smiles))
        if smiles_idx == len(unique_smiles): 
            unique_smiles.append(smiles)

        # Step 3 — append valid conformers and remember which SMILES each belongs to.
        flattened.extend(valid_confs)
        flattened_idx_to_smiles_idx.extend([smiles_idx] * len(valid_confs))

    if input_validation_notes:
        logger.info(
            "PoseBusters input validation filtered %d SMILES: %s",
            len(input_validation_notes),
            json.dumps(input_validation_notes, sort_keys=True),
        )

    per_smiles_template = ["smiles", "num_of_conformers", "pass_percentage", "error"]
    summary_template = ["smiles", "num_smiles", "num_conformers", "pass_percentage"]
    if not flattened:
        empty_per_smiles = pd.DataFrame(columns=per_smiles_template)
        empty_summary = pd.DataFrame(columns=summary_template)
        empty_summary.attrs["per_smiles_df"] = empty_per_smiles
        empty_summary.attrs["input_validation_notes"] = input_validation_notes
        return empty_per_smiles, empty_summary, float("nan")

    # Step 4 — run PoseBusters on the flattened conformer list (parallelized upstream).
    per_conformer_df = _collect_posebusters_report(
        flattened,
        num_workers=num_workers,
        full_report=full_report,
        config=config,
    )
    if per_conformer_df.empty:
        empty_per_smiles = pd.DataFrame(columns=per_smiles_template)
        empty_summary = pd.DataFrame(columns=summary_template)
        empty_summary.attrs["per_smiles_df"] = empty_per_smiles
        empty_summary.attrs["input_validation_notes"] = input_validation_notes
        return empty_per_smiles, empty_summary, float("nan")

    # Step 5 — recover SMILES provenance and per-conformer pass metrics.
    pass_fraction = _compute_pass_fraction(per_conformer_df)
    per_conformer_df["pass_fraction"] = pass_fraction
    per_conformer_df["pass_bool"] = (np.isclose(pass_fraction, 1.0)).astype(float)
    per_conformer_df["smiles"] = per_conformer_df["conformer_index"].map(
        lambda idx: unique_smiles[flattened_idx_to_smiles_idx[int(idx)]]
    )

    # Step 6 — aggregate per-SMILES statistics (matches tests/evaluation/sample_by_smiles_df.csv).
    bool_columns = per_conformer_df.select_dtypes(include=["bool"]).columns.tolist()
    per_smiles_records: List[Dict[str, float]] = []
    grouped = per_conformer_df.groupby("smiles", sort=True)
    for smiles, frame in grouped:
        bool_vals = frame["pass_bool"]
        record: Dict[str, float] = {}
        for col in bool_columns:
            record[col] = float(frame[col].mean())
        record["pass_percentage"] = float(bool_vals.mean() * 100.0)
        record["smiles"] = smiles
        record["num_of_conformers"] = int(frame.shape[0])
        record["error"] = ""
        if "error" in frame.columns:
            error_msgs = sorted({msg for msg in frame["error"].astype(str).tolist() if msg})
            record["error"] = ";".join(error_msgs)
        per_smiles_records.append(record)

    per_smiles_columns = [*bool_columns, "pass_percentage", "smiles", "num_of_conformers", "error"]
    per_smiles_df = (
        pd.DataFrame(per_smiles_records)
        .reindex(columns=per_smiles_columns)
        .sort_values("smiles")
        .reset_index(drop=True)
    )
    overall_pass_rate = float(per_conformer_df["pass_bool"].mean())

    # Step 7 — create the single-row summary (matches tests/evaluation/sample_summary_df.csv).
    summary_record: Dict[str, float] = {
        "smiles": "ALL",
        "num_smiles": float(per_smiles_df.shape[0]),
        "num_conformers": float(per_conformer_df.shape[0]),
    }
    for col in bool_columns:
        summary_record[col] = float(per_conformer_df[col].mean())
    summary_record["pass_percentage"] = float(overall_pass_rate * 100.0)
    summary_columns = ["smiles", "num_smiles", "num_conformers", *bool_columns, "pass_percentage"]
    summary_df = pd.DataFrame([summary_record], columns=summary_columns)
    summary_df.attrs["per_smiles_df"] = per_smiles_df
    summary_df.attrs["input_validation_notes"] = input_validation_notes

    # Note: fail_smiles and error_smiles removed as redundant.
    # - fail_smiles: Users can filter per_smiles_df by pass_percentage directly
    # - error_smiles: PoseBusters doesn't return an error column, it raises exceptions.
    #   The error column in per_smiles_df is part of the schema but always empty.

    return per_smiles_df, summary_df, overall_pass_rate

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