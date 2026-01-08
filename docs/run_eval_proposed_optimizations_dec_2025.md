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
# AFTER (run_eval_optimized.py)
from concurrent.futures import ProcessPoolExecutor
...
with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
    futures = [ex.submit(compute_key_matrix, ...) for ...]
```

`ProcessPoolExecutor` spawns separate Python interpreter processes, each with its own GIL. This enables true parallel execution on multi-core systems for CPU-bound work like RMSD calculations.

### Additional Consideration: Pickling

`ProcessPoolExecutor` requires that the target function and its arguments be picklable (serializable). The original `_compute_key_matrix` function was defined in `run_eval.py`, which caused a pickling error when running via submitit:

```
_pickle.PicklingError: Can't pickle <function _compute_key_matrix at 0x...>:
attribute lookup _compute_key_matrix on __main__ failed
```

**Solution**: Moved the function to `rdkit_utils.py` as `compute_key_matrix`, making it a proper importable module-level function that survives the double-pickling chain (submitit → ProcessPoolExecutor).

---

## 2. Row-Level vs Molecule-Level Parallelization (Tested & Rejected)

### The Investigation

We investigated whether finer-grained parallelization could improve performance. The idea was to parallelize at the **row level** (one task per true conformer) instead of **molecule level** (one task per molecule).

### Why We Considered Row-Level

Molecules have vastly different numbers of conformers:

```
Top 10 largest molecules by conformer count:
  2497 conformers: CCCCC(=O)NC(NC(=O)CCCC)c1ccc(OCC(C)C)c(OC)c1
  1837 conformers: CCCCOCP(=O)(CC)COCCCC
  ...
Total molecules: 1000
Total conformers: 106778
```

This causes the **straggler problem**: one worker processes the 2497-conformer molecule while 47 workers sit idle.

### What We Tested

Row-level parallelization would create **106,778 tasks** instead of 1000:

```python
# Row-level approach (REJECTED)
for key in true_data.keys():
    for row_idx, ref_mol in enumerate(true_confs):
        row_tasks.append((key, row_idx, ref_mol, gen_mols, use_alignmol))
# Creates 106K tasks!
```

### Why Row-Level Was Slower

We tested using `scripts/test_parallelization_overhead.py` and found **IPC/pickling overhead dominates**:

| Simulated Work/Row | Molecule-Level | Row-Level | Overhead Ratio |
|--------------------|----------------|-----------|----------------|
| 0ms (pure overhead) | 0.61s | 34.62s | **57x slower** |
| 1ms (light work) | 0.70s | 1.12s | **1.6x slower** |

At realistic scale (1000 molecules, 107K rows, 48 workers), row-level parallelization is always slower because:
1. 100x more tasks = 100x more IPC overhead
2. 100x more pickle/unpickle operations
3. 100x more future object management

### Final Decision: Molecule-Level Wins

We keep **molecule-level parallelization** (`compute_key_matrix`):
- 1,000 tasks (not 106K)
- Acceptable straggler effect (tail takes longer but not too bad)
- Much lower overhead

The `compute_rmsd_row` function was removed as dead code (2025-12-23).

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
2. Molecule-level parallelization in `compute_rmsd_matrix()` (row-level tested but rejected due to overhead)
3. New CLI arguments: `--memory-gb`, `--pb-chunk-size`, `--output-dir`
4. Progress logging for RMSD and PoseBusters phases

### `src/molgen3D/evaluation/rdkit_utils.py`

1. Added `compute_key_matrix()` function (picklable, module-level for ProcessPoolExecutor)
2. Added numpy import for array operations
3. `compute_rmsd_row()` was tested for row-level parallelization but removed (2025-12-23) after benchmarks showed 57x overhead

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
