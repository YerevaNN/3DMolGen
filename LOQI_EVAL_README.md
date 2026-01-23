# LOQI Energy Evaluation

LOQI-style energy evaluation for generated molecular conformers using AIMNet2. This evaluation computes quantum mechanical energies, performs geometry optimization, and analyzes conformer quality.

## Quick Start

### Single Directory (Local)

```bash
python run_loqi_eval.py --gen-dir 20260122_093145_m600_qwen_pre_4seq_3e_loqi
```

### Batch Mode (Multiple Directories)

Evaluate all directories missing LOQI results:

```bash
# Run locally
python run_loqi_eval.py --device local --max-recent 5

# Submit to slurm A100 queue
python run_loqi_eval.py --device a100 --max-recent 5

# Submit to slurm H100 queue
python run_loqi_eval.py --device h100 --max-recent 5
```

### Module Usage

```bash
python -m molgen3D.evaluation.loqi_energy_eval \
    --gen-dir 20260122_093145_m600_qwen_pre_4seq_3e_loqi \
    --device local \
    --batch-size 64
```

## Options

- `--gen-dir` / `--specific-dir`: Specific directory to evaluate (triggers single mode)
- `--device`: Execution mode (default: `local`)
  - `local`: Run locally with auto device detection
  - `cuda` / `cpu`: Run locally on specific device
  - `a100`: Submit to slurm A100 queue (1 GPU per job)
  - `h100`: Submit to slurm H100 queue (1 GPU per job)
  - `all`: Submit to any available slurm queue
- `--batch-size`: Batch size for AIMNet2 (default: 64)
- `--no-opt`: Skip optimization for 5x speedup
- `--fmax`: Force convergence threshold (default: 2e-3 eV/Å)
- `--max-steps`: Maximum optimization steps (default: 5000)
- `--max-recent`: Number of recent directories to process in batch mode (default: 3)

## What It Does

The evaluation automatically:
1. Loads generated conformers from `outputs/gen_results/{gen-dir}/generation_results.pickle`
2. Loads ground truth from `data/loqi_smi.pickle` (configured in `paths.yaml`)
3. Uses AIMNet2 model from `/nfs/ap/mnt/sxtn2/chem/GEOM_data/wb97m_cpcms_v2_0.jpt`
4. Computes initial energies and forces
5. Optimizes geometries using FIRE algorithm (unless `--no-opt`)
6. Validates topology preservation
7. Computes geometry metrics (bond lengths, angles, dihedrals)
8. Saves results to `{gen-dir}/loqi_eval/`

## Output Files

Results are saved in `outputs/gen_results/{gen-dir}/loqi_eval/`:

### 1. loqi_summary.txt
High-level aggregate metrics:
- Number of molecules and conformers evaluated
- Mean/median relative energies (kcal/mol)
- Optimization convergence and topology preservation rates
- Geometry change statistics

### 2. loqi_per_molecule.csv
Detailed per-molecule statistics:
- Energy statistics for each SMILES
- Number of conformers generated
- Relative energies vs reference (initial and optimized)
- Topology preservation rate per molecule
- Quality metrics (within 0.1 kcal/mol, better than reference)

### 3. loqi_results.pickle
Complete evaluation data including:
- All LOQI metrics
- Per-molecule DataFrame
- Full AIMNet2 outputs
- Reference molecule metrics

## Key Metrics

### Energy Metrics
- **mean_relative_energy**: Average energy difference from reference (kcal/mol)
- **median_relative_energy**: Median energy difference from reference (kcal/mol)
- **min_relative_energy**: Best (lowest) energy difference across all conformers

### Optimization Metrics (when `--no-opt` is not used)
- **opt_converged**: Fraction of conformers that converged during optimization
- **preserved_topology**: Fraction maintaining correct bonding after optimization
- **opt_avg_energy_drop**: Average energy decrease during optimization (kcal/mol)
- **opt_median_relative_energy**: Median optimized energy vs reference
- **opt_min_conformers**: Fraction within 0.1 kcal/mol of reference
- **opt_better_min_conformers**: Fraction better than reference (< -0.1 kcal/mol)

### Geometry Metrics
- **opt_bond_lengths_diff**: Average bond length change after optimization (Å)
- **opt_bond_angles_diff**: Average bond angle change after optimization (degrees)
- **opt_dihedrals_diff**: Average dihedral angle change after optimization (degrees)

## Implementation

The evaluation follows the LOQI methodology:

1. **Preprocessing**: Remove hydrogens, validate molecules
2. **Molecule Pairing**: Match each generated conformer with its reference
3. **Energy Calculation**: Compute QM energies using AIMNet2
4. **Optimization**: Use FIRE algorithm to find local energy minima
5. **Topology Check**: Validate bonding patterns after optimization
6. **Geometry Analysis**: Compute structural differences using RDKit

## Code Structure

- [loqi_energy_eval.py](src/molgen3D/evaluation/loqi_energy_eval.py): Main evaluation logic
- [aimnet2_metrics.py](src/molgen3D/evaluation/aimnet2_metrics.py): AIMNet2 calculator wrapper
- [aimnet2_utils.py](src/molgen3D/evaluation/aimnet2_utils.py): FIRE optimizer and geometry utilities
- [run_loqi_eval.py](run_loqi_eval.py): Convenient runner script

## Performance Optimization

See [LOQI_PERFORMANCE_GUIDE.md](LOQI_PERFORMANCE_GUIDE.md) for detailed performance tips.

**Quick tips**:
- Increase batch size for faster GPU utilization: `--batch-size 128`
- Skip optimization for 5x speedup (energy-only): `--no-opt`
- Use larger convergence threshold for faster optimization: `--fmax 5e-3`

**Achieved speedups**:
- 2x faster with default optimizations
- 5.6x faster with `--no-opt`

## Requirements

- PyTorch with CUDA support
- RDKit
- AIMNet2 model file (.jpt)
- tqdm, pandas, numpy

## Slurm Integration

The evaluation supports submitting jobs to slurm clusters:

```bash
# Submit 3 most recent unevaluated directories to A100 queue
python run_loqi_eval.py --device a100 --max-recent 3

# Submit specific directory to H100 queue with custom batch size
python run_loqi_eval.py --device h100 \
    --specific-dir 20260122_093145_m600_qwen_pre_4seq_3e_loqi \
    --batch-size 128

# Evaluate all missing directories locally
python run_loqi_eval.py --device local --max-recent 10
```

Each slurm job:
- Uses 1 GPU (required for AIMNet2)
- Uses 4 CPUs
- Job name: `loqi_eval`
- Results saved to `{gen_dir}/loqi_eval/`

## Examples

```bash
# Single directory, local GPU
python run_loqi_eval.py --gen-dir YOUR_DIR

# Single directory, fast mode (no optimization)
python run_loqi_eval.py --gen-dir YOUR_DIR --no-opt

# Single directory, CPU only
python run_loqi_eval.py --gen-dir YOUR_DIR --device cpu

# Batch mode, submit to A100 queue
python run_loqi_eval.py --device a100 --max-recent 5

# Custom optimization parameters
python run_loqi_eval.py --gen-dir YOUR_DIR \
    --fmax 1e-3 --max-steps 10000 --batch-size 128
```
