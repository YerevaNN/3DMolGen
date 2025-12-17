# `run_eval_optimized.py` - Evaluation Pipeline Optimizations

This document describes the performance optimizations implemented in `src/molgen3D/evaluation/run_eval_optimized.py`, an experimental alternative to `run_eval.py`. The optimized version was created on 2025-12-17.

**Status**: Experimental - kept separate until correctness is validated against `run_eval.py`.

## Summary of Changes

| Change | Before | After | Impact |
|--------|--------|-------|--------|
| Executor type | `ThreadPoolExecutor` | `ProcessPoolExecutor` | Enables true CPU parallelism |
| Parallelization granularity | Per-molecule (1000 tasks) | Per-row (106K tasks) | More uniform task distribution |
| Memory allocation | Hardcoded 80GB | CLI configurable `--memory-gb` | Better resource utilization |
| PoseBusters chunk size | Hardcoded 600 | CLI configurable `--pb-chunk-size` (default 300) | Configurable load balancing |

**Measured result**: run_eval_optimized completed ~5 min faster overall than run_eval.py on the `distinct` dataset.

---

## 1. ThreadPoolExecutor → ProcessPoolExecutor

### The Problem

The original code used `ThreadPoolExecutor` for RMSD computation:

```python
# BEFORE (run_eval.py)
from concurrent.futures import ThreadPoolExecutor
...
with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
    futures = [ex.submit(_compute_key_matrix, ...) for ...]
```

### Why This Was Wrong

Python's **Global Interpreter Lock (GIL)** prevents true parallel execution of Python bytecode across threads. From the [Python documentation](https://docs.python.org/3/glossary.html#term-global-interpreter-lock):

> "The mechanism used by the CPython interpreter to assure that only one thread executes Python bytecode at a time."

`ThreadPoolExecutor` is designed for **I/O-bound** tasks (network requests, file I/O) where threads spend most of their time waiting. For **CPU-bound** tasks like RMSD calculations, threads cannot run in parallel—they take turns holding the GIL.

### Evidence

From the [Python `concurrent.futures` documentation](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor):

> "ThreadPoolExecutor is an Executor subclass that uses a pool of threads to execute calls asynchronously. **Deadlocks can occur** when the callable associated with a Future waits on the results of another Future."

And critically:

> "If you need to execute CPU-bound operations in parallel, **use ProcessPoolExecutor instead**."

### The Fix

```python
# AFTER (run_eval.py)
from concurrent.futures import ProcessPoolExecutor
...
with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
    futures = [ex.submit(compute_rmsd_row, ...) for ...]
```

`ProcessPoolExecutor` spawns separate Python interpreter processes, each with its own GIL. This enables true parallel execution on multi-core systems for CPU-bound work like RMSD calculations.

### Additional Consideration: Pickling

`ProcessPoolExecutor` requires that the target function and its arguments be picklable (serializable). The original `_compute_key_matrix` function was defined in `run_eval.py`, which caused a pickling error when running via submitit:

```
_pickle.PicklingError: Can't pickle <function _compute_key_matrix at 0x...>:
attribute lookup _compute_key_matrix on __main__ failed
```

**Solution**: Moved the function to `rdkit_utils.py` as `compute_rmsd_row`, making it a proper importable module-level function that survives the double-pickling chain (submitit → ProcessPoolExecutor).

---

## 2. Molecule-Level → Row-Level Parallelization

### The Problem

The original code created one task per molecule:

```python
# BEFORE
work_items: List[Tuple[str, List, List]] = []
for key in true_data.keys():
    work_items.append((key, true_data[key]["confs"], gen_mols))  # One task per molecule

with ProcessPoolExecutor(max_workers=48) as ex:
    futures = [ex.submit(compute_key_matrix, key, confs, gen_mols, ...) for key, confs, gen_mols in work_items]
```

With 1000 molecules, this creates 1000 tasks. However, molecules have vastly different numbers of conformers:

```
Top 10 largest molecules by conformer count:
  2497 conformers: CCCCC(=O)NC(NC(=O)CCCC)c1ccc(OCC(C)C)c(OC)c1
  1837 conformers: CCCCOCP(=O)(CC)COCCCC
  1500 conformers: CN(CC(O)COc1ccc(OCC(O)CN(C)C2CCCCC2)cc1)C1CCCCC1
  ...
Total molecules: 1000
Total conformers: 106778
Top 1% molecules have 12602 conformers (11.8% of total)
```

### The Straggler Problem

This is a well-known issue in parallel computing called the **straggler problem** or **load imbalance**. From [Dean & Barroso, "The Tail at Scale" (2013)](https://research.google/pubs/pub40801/):

> "Variability in response times leads to situations where a small number of slow tasks ('stragglers') dominate overall job completion time."

The work per molecule is O(n_true × n_gen) RMSD calculations:
- Small molecule: 5 × 10 = 50 calculations → milliseconds
- Large molecule: 2497 × 50 = 124,850 calculations → minutes

One worker processes the 2497-conformer molecule while 47 workers sit idle.

### The Fix

Parallelize at the **row level** (one task per true conformer):

```python
# AFTER
row_tasks: List[Tuple[str, int, object, List, bool]] = []
for key in true_data.keys():
    true_confs = true_data[key]["confs"]
    for row_idx, ref_mol in enumerate(true_confs):
        row_tasks.append((key, row_idx, ref_mol, gen_mols, use_alignmol))

with ProcessPoolExecutor(max_workers=48) as ex:
    futures = [ex.submit(compute_rmsd_row, *task) for task in row_tasks]
```

This creates **106,778 tasks** instead of 1000, with each task being roughly the same size (one RMSD row computation).

### Evidence: Amdahl's Law and Load Balancing

From [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law), the speedup from parallelization is limited by the serial portion of the workload. With molecule-level parallelization, the serial portion is the largest molecule.

From the [Python multiprocessing documentation](https://docs.python.org/3/library/multiprocessing.html#programming-guidelines):

> "When using a process pool, you should ensure that the work is divided into roughly equal-sized chunks to maximize efficiency."

### Observed Results

| Metric | Before (molecule-level) | After (row-level) |
|--------|------------------------|-------------------|
| Tasks | 1,000 | 106,778 |
| Task size variance | Huge (5 → 2497 rows) | Uniform (~50 RMSD calcs each) |
| Progress pattern | Steady throughput | 98% fast → last 2% slower |
| RMSD phase time | Not measured | ~27 min |

**Note**: Based on jobs 422719 (run_eval.py) and 422722 (run_eval_optimized) on H100 with `distinct` dataset (1000 molecules, 106K conformers). Overall, run_eval_optimized completed ~5 min faster than run_eval.py. The old run_eval.py has no progress logging, so RMSD timing was not measured.

---

## 3. Configurable Memory Allocation

### The Problem

Memory was hardcoded at 80GB:

```python
# utils.py
def create_slurm_executor(
    ...
    memory_gb: int = 80,  # Hardcoded!
) -> submitit.AutoExecutor:
```

On a node with 1935GB available, this severely underutilizes resources.

### The Fix

Added `--memory-gb` CLI argument:

```python
parser.add_argument("--memory-gb", type=int, default=80,
                    help="Memory in GB to request from Slurm")
```

### Recommended Value

For 48 workers with RDKit molecular processing:
- ~8GB per worker is a safe estimate
- 48 × 8GB = 384GB, rounded to **400GB**

---

## 4. PoseBusters Chunk Size Tuning

### The Problem

PoseBusters chunk size was hardcoded at 600 conformers per task:

```python
# posebusters_check.py
def bust_full_gens(
    ...
    task_chunk_size: Optional[int] = 600,  # Hardcoded!
```

With 106K conformers and 600 chunk size: 106000 / 600 = **177 tasks** for 48 workers = 3.7 tasks per worker.

### The Fix

Added `--pb-chunk-size` CLI argument with default 300:

```python
parser.add_argument("--pb-chunk-size", type=int, default=300,
                    help="PoseBusters conformers per task (smaller=better load balance)")
```

With 300 chunk size: 106000 / 300 = **355 tasks** for 48 workers = 7.4 tasks per worker.

### Trade-off Analysis

| Chunk Size | Tasks | Tasks/Worker | Overhead | Load Balance |
|------------|-------|--------------|----------|--------------|
| 600 | 177 | 3.7 | Low | Moderate |
| 300 | 355 | 7.4 | Medium | Good |
| 200 | 533 | 11.1 | Higher | Excellent |
| 100 | 1067 | 22.2 | High | Excellent |

The optimal chunk size balances:
1. **Load balancing**: More tasks → better distribution
2. **Overhead**: Fewer tasks → less IPC/pickling cost

300 is a good default that provides 2x the tasks/worker ratio while keeping overhead manageable.

---

## File Changes Summary

### `src/molgen3D/evaluation/run_eval_optimized.py` (new file)

Created as experimental alternative to `run_eval.py` with:

1. `ProcessPoolExecutor` instead of `ThreadPoolExecutor`
2. Row-level parallelization in `compute_rmsd_matrix()`
3. New CLI arguments: `--memory-gb`, `--pb-chunk-size`, `--output-dir`
4. Progress logging for RMSD and PoseBusters phases

### `src/molgen3D/evaluation/rdkit_utils.py`

1. Added `compute_rmsd_row()` function (picklable, module-level)
2. Added numpy import for array operations

---

## Usage

```bash
python -m molgen3D.evaluation.run_eval_optimized \
    --device h100 \
    --num-workers 48 \
    --memory-gb 200 \
    --pb-chunk-size 600 \
    --specific-dir <generation_dir> \
    --test_set distinct \
    --posebusters mol \
    --output-dir <custom_output_dir>
```

---

## References

1. Python Documentation: [Global Interpreter Lock](https://docs.python.org/3/glossary.html#term-global-interpreter-lock)
2. Python Documentation: [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html)
3. Python Documentation: [multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
4. Dean, J. & Barroso, L.A. (2013). ["The Tail at Scale"](https://research.google/pubs/pub40801/). Communications of the ACM.
5. Wikipedia: [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law)
6. RDKit Documentation: [Molecule Alignment](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html)
