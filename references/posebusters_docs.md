# PoseBusters Python API

## Overview

PoseBusters is a validation framework for molecular docking and conformer generation. The main `PoseBusters` class orchestrates test execution across molecules, running modular validation checks and reporting results as pandas DataFrames.

---

## PoseBusters Class

### Constructor

```python
class PoseBusters(config: str | dict[str, Any] = 'redock', top_n: int | None = None)
```

**Parameters:**

- `config` (str | dict): Configuration preset or custom configuration dictionary
  - Defaults to `'redock'`
  - Can be a configuration dict for custom test sets
- `top_n` (int | None): Limit results to top N entries
  - Defaults to `None` (no limit)

---

## Core Methods

### `bust()`

Run all tests on a single molecule or set of molecules.

```python
def bust(
    mol_pred: Iterable[Mol | Path],
    mol_true: Mol | Path | None = None,
    mol_cond: Mol | Path | None = None,
    full_report: bool = False
) -> pd.DataFrame
```

**Parameters:**

- `mol_pred` (Iterable[Mol | Path]): Generated molecule(s) to test
  - Examples: docked ligand, generated conformer
  - Can be RDKit Mol objects or file paths
  - Can be iterable of multiple molecules
- `mol_true` (Mol | Path | None): Reference/ground truth molecule
  - Examples: crystal ligand, reference conformer
  - Can be RDKit Mol object or file path
  - Defaults to `None`
- `mol_cond` (Mol | Path | None): Conditioning molecule
  - Examples: protein, binding pocket
  - Can be RDKit Mol object or file path
  - Defaults to `None`
- `full_report` (bool): Include all columns in output
  - If `False`: only boolean test results
  - If `True`: includes detailed metrics and intermediate values
  - Defaults to `False`

**Returns:**

- `pd.DataFrame`: Test results with one row per molecule and columns for each test

**Notes:**

- Molecules can be provided as RDKit molecule objects or file paths
- At least `mol_pred` is required; `mol_true` and `mol_cond` are optional depending on test requirements
- Each molecule should have exactly one conformer (except where noted)

---

### `bust_table()`

Run all tests on multiple molecules provided in a pandas DataFrame.

```python
def bust_table(
    mol_table: DataFrame,
    full_report: bool = False
) -> DataFrame
```

**Parameters:**

- `mol_table` (DataFrame): Input table with molecule data
  - Required columns: `"mol_pred"`, `"mol_true"`, `"mol_cond"`
  - Values can be RDKit Mol objects or file paths
- `full_report` (bool): Include all columns in output
  - Defaults to `False`

**Returns:**

- `DataFrame`: Test results merged with input table

**Example:**

```python
# Create input DataFrame
molecules = pd.DataFrame({
    'mol_pred': ['pose_1.pdb', 'pose_2.pdb'],
    'mol_true': ['crystal_1.pdb', 'crystal_2.pdb'],
    'mol_cond': ['protein.pdb', 'protein.pdb']
})

# Run tests
pb = PoseBusters(config='redock')
results = pb.bust_table(molecules, full_report=True)
```

---

## Validation Modules

PoseBusters validates molecules using modular checks. Each module is a function that accepts one or more molecule inputs and returns test results as a dictionary.

### Module Structure

**Inputs:**
- Accept one or more of: `mol_pred`, `mol_true`, `mol_cond` as RDKit molecules
- Additional parameters must have default values

**Outputs:**
- Dictionary with required `"results"` key containing test outcomes
- Optional keys for additional metrics (e.g., bond lengths, angles)

---

### Distance Geometry

Validates molecular geometry using RDKit distance geometry bounds.

```python
def check_geometry(
    mol_pred: Mol,
    threshold_bad_bond_length: float = 0.2,
    threshold_clash: float = 0.2,
    threshold_bad_angle: float = 0.2,
    bound_matrix_params: dict[str, Any] = {
        'doTriangleSmoothing': True,
        'scaleVDW': True,
        'set15bounds': True,
        'useMacrocycle14config': False
    },
    ignore_hydrogens: bool = True,
    sanitize: bool = True
) -> dict[str, Any]
```

**Parameters:**

- `mol_pred` (Mol): Predicted molecule (docked ligand)
  - Must have exactly one conformer
- `threshold_bad_bond_length` (float): Bond length deviation tolerance
  - Value as relative percentage (0.2 = ±20%)
  - Defaults to `0.2`
- `threshold_clash` (float): Atomic clash tolerance
  - 0.2 means atoms can be up to 80% of lower bound apart
  - Defaults to `0.2`
- `threshold_bad_angle` (float): Bond angle deviation tolerance
  - Value as relative percentage
  - Defaults to `0.2`
- `bound_matrix_params` (dict): Parameters for RDKit's `GetMoleculeBoundsMatrix()`
  - Keys: `doTriangleSmoothing`, `scaleVDW`, `set15bounds`, `useMacrocycle14config`
  - Defaults shown above
- `ignore_hydrogens` (bool): Exclude hydrogens from checks
  - Defaults to `True`
- `sanitize` (bool): Sanitize molecule before validation
  - Recommended to keep `True`
  - Defaults to `True`

**Returns:**

- `dict`: PoseBusters results dictionary with geometry validation tests

**Tests Check:**
- Bond lengths within expected ranges
- Bond angles within expected ranges
- Atomic clashes/overlaps

---

### Energy Ratio

Validates docked ligand energy compared to conformer ensemble.

```python
def check_energy_ratio(
    mol_pred: Mol,
    threshold_energy_ratio: float = 7.0,
    ensemble_number_conformations: int = 100
) -> dict[str, Any]
```

**Parameters:**

- `mol_pred` (Mol): Predicted molecule (docked ligand)
  - Must have exactly one conformer
- `threshold_energy_ratio` (float): Maximum acceptable energy ratio
  - Defaults to `7.0`
- `ensemble_number_conformations` (int): Number of conformations for ensemble
  - Generated to establish baseline distribution
  - Defaults to `100`

**Returns:**

- `dict`: PoseBusters results dictionary with energy ratio validation

**Tests Check:**
- Whether docked ligand energy is within threshold of ensemble average

---

### Flatness

Validates planarity of aromatic and other flat substructures.

```python
def check_flatness(
    mol_pred: Mol,
    threshold_flatness: float = 0.1,
    flat_systems: dict[str, str] = {
        'aromatic_5_membered_rings_sp2': '[ar5^2]1[ar5^2][ar5^2][ar5^2][ar5^2]1',
        'aromatic_6_membered_rings_sp2': '[ar6^2]1[ar6^2][ar6^2][ar6^2][ar6^2][ar6^2]1',
        'trigonal_planar_double_bonds': '[C;X3;^2](*)(*)=[C;X3;^2](*)(*)' 
    }
) -> dict[str, Any]
```

**Parameters:**

- `mol_pred` (Mol): Molecule to validate
  - Must have exactly one conformer
- `threshold_flatness` (float): Maximum allowed distance from plane (Ångströms)
  - Defaults to `0.1`
- `flat_systems` (dict): SMARTS patterns defining flat substructures
  - Keys: pattern name (descriptive)
  - Values: SMARTS pattern strings
  - Default patterns detect aromatic rings and double bonds

**Returns:**

- `dict`: PoseBusters results dictionary with flatness validation

**Tests Check:**
- Aromatic rings are planar (5 and 6 membered)
- Trigonal planar double bonds are flat

---

### Identity

Validates that two molecules are chemically identical.

```python
def check_identity(
    mol_pred: Mol,
    mol_true: Mol,
    inchi_options: str = ''
) -> dict[str, Any]
```

**Parameters:**

- `mol_pred` (Mol): Predicted molecule
- `mol_true` (Mol): Ground truth molecule
  - Must have at least one conformer
- `inchi_options` (str): Options passed to RDKit InChI module
  - Defaults to `''` (no options)
  - See RDKit documentation for available options

**Returns:**

- `dict`: PoseBusters results dictionary with identity validation

**Tests Check:**
- Molecular structure equivalence (ignoring 3D coordinates)

---

### Intermolecular Distance

Validates spatial relationship between two molecules.

```python
def check_intermolecular_distance(
    mol_pred: Mol,
    mol_cond: Mol,
    radius_type: str = 'vdw',
    radius_scale: float = 1.0,
    clash_cutoff: float = 0.75,
    ignore_types: set[str] = {'hydrogens'},
    max_distance: float = 5.0,
    search_distance: float = 6.0
) -> dict[str, Any]
```

**Parameters:**

- `mol_pred` (Mol): Predicted molecule (docked ligand)
  - Must have one conformer
- `mol_cond` (Mol): Conditioning molecule (protein)
  - Must have one conformer
- `radius_type` (str): Atomic radius type
  - Options: `'vdw'` (van der Waals), `'covalent'`
  - Defaults to `'vdw'`
- `radius_scale` (float): Scaling factor for atomic radii
  - Defaults to `1.0`
- `clash_cutoff` (float): Overlap threshold for clash detection
  - Defaults to `0.75` Ångströms
- `ignore_types` (set[str]): Atom types to exclude from checks
  - Options: `'hydrogens'`, `'protein'`, `'organic_cofactors'`, `'inorganic_cofactors'`, `'waters'`
  - Defaults to `{'hydrogens'}`
- `max_distance` (float): Maximum allowed distance between molecules (Ångströms)
  - Defaults to `5.0`
- `search_distance` (float): Radius for neighbor search (Ångströms)
  - Defaults to `6.0`

**Returns:**

- `dict`: PoseBusters results dictionary with intermolecular distance validation

**Tests Check:**
- No atomic clashes between molecules
- Molecules within maximum distance threshold
- Molecules not too far apart (likely invalid pose)

---

### Loading

Validates that molecule files were loaded successfully.

```python
def check_loading(
    mol_pred: Any = None,
    mol_true: Any = None,
    mol_cond: Any = None
) -> dict[str, dict[str, bool]]
```

**Parameters:**

- `mol_pred` (Any): Predicted molecule or path
  - Defaults to `None`
- `mol_true` (Any): Ground truth molecule or path
  - Defaults to `None`
- `mol_cond` (Any): Conditioning molecule or path
  - Defaults to `None`

**Returns:**

- `dict`: PoseBusters results dictionary with loading status

**Tests Check:**
- Successful file loading and parsing
- Valid RDKit molecule objects

---

### RMSD

Calculate Root Mean Square Deviation and related metrics.

```python
def check_rmsd(
    mol_pred: Mol,
    mol_true: Mol,
    rmsd_threshold: float = 2.0
) -> dict[str, dict[str, bool | float]]
```

**Parameters:**

- `mol_pred` (Mol): Predicted molecule (docked ligand)
  - Must have exactly one conformer
- `mol_true` (Mol): Ground truth molecule (crystal ligand)
  - Must have at least one conformer
  - If multiple: lowest RMSD is reported
- `rmsd_threshold` (float): RMSD threshold (Ångströms)
  - Used to determine if prediction is within acceptable range
  - Defaults to `2.0`

**Returns:**

- `dict`: Results dictionary containing:
  - Boolean: whether RMSD is within threshold
  - Float: actual RMSD value

**Notes:**

- If `mol_true` has multiple conformers, the function automatically finds the closest match
- RMSD is a metric of 3D structure similarity (lower is better)

---

### Volume Overlap

Validates volume overlap between ligand and protein.

```python
def check_volume_overlap(
    mol_pred: Mol,
    mol_cond: Mol,
    clash_cutoff: float = 0.05,
    vdw_scale: float = 0.8,
    ignore_types: set[str] = {'hydrogens'},
    search_distance: float = 6.0
) -> dict[str, dict]
```

**Parameters:**

- `mol_pred` (Mol): Predicted molecule (docked ligand)
  - Must have one conformer
- `mol_cond` (Mol): Conditioning molecule (protein)
  - Must have one conformer
- `clash_cutoff` (float): Maximum allowed volume overlap
  - Value is maximum share of `mol_pred` volume allowed to overlap with `mol_cond`
  - Defaults to `0.05` (5%)
- `vdw_scale` (float): Scaling factor for van der Waals radii
  - Defines volume around each atom
  - Defaults to `0.8`
- `ignore_types` (set[str]): Atom types in `mol_cond` to exclude
  - Options: `'hydrogens'`, `'protein'`, `'organic_cofactors'`, `'inorganic_cofactors'`, `'waters'`
  - Defaults to `{'hydrogens'}`
- `search_distance` (float): Radius for spatial search (Ångströms)
  - Defaults to `6.0`

**Returns:**

- `dict`: PoseBusters results dictionary with volume overlap validation

**Tests Check:**
- Volume overlap within acceptable threshold
- No significant steric clashes

---

## Usage Examples

### Basic Single Molecule Validation

```python
from posebusters import PoseBusters
from rdkit import Chem

# Load molecules
docked = Chem.MolFromPDBFile('docked_ligand.pdb')
crystal = Chem.MolFromPDBFile('crystal_ligand.pdb')
protein = Chem.MolFromPDBFile('protein.pdb')

# Initialize validator
pb = PoseBusters(config='redock')

# Run all tests
results = pb.bust(
    mol_pred=docked,
    mol_true=crystal,
    mol_cond=protein,
    full_report=True
)

print(results)
```

### Batch Validation with DataFrame

```python
import pandas as pd
from posebusters import PoseBusters

# Create input table
molecules = pd.DataFrame({
    'mol_pred': ['pose_1.pdb', 'pose_2.pdb', 'pose_3.pdb'],
    'mol_true': ['crystal.pdb', 'crystal.pdb', 'crystal.pdb'],
    'mol_cond': ['protein.pdb', 'protein.pdb', 'protein.pdb']
})

# Validate all molecules
pb = PoseBusters()
results = pb.bust_table(molecules, full_report=True)

# Filter for valid poses
valid = results[results['identity'] & results['rmsd']]
print(f"Valid poses: {len(valid)} / {len(results)}")
```

### Custom Configuration

```python
from posebusters import PoseBusters

# Custom config: only specific tests
config = {
    'tests': [
        'identity',
        'rmsd',
        'distance_geometry'
    ]
}

pb = PoseBusters(config=config)
results = pb.bust(mol_pred=docked, mol_true=crystal)
```

### Individual Module Usage

```python
from posebusters.modules import distance_geometry, rmsd
from rdkit import Chem

mol = Chem.MolFromPDBFile('ligand.pdb')
ref = Chem.MolFromPDBFile('reference.pdb')

# Check geometry
geom_results = distance_geometry.check_geometry(mol)
print(geom_results['results'])

# Check RMSD
rmsd_results = rmsd.check_rmsd(mol, ref, rmsd_threshold=2.0)
print(f"RMSD: {rmsd_results['results']['rmsd']}")
```

---

## Key Concepts

### Conformers

Most molecules in PoseBusters should have exactly **one conformer** (3D structure). A conformer is a specific 3D arrangement of atoms.

- For ligand conformer generation: `mol_pred` has one generated conformer
- For references: `mol_true` can have multiple conformers (best is automatically selected)

### Input Flexibility

Molecules can be provided as:
- **RDKit Mol objects**: Already loaded in memory
- **File paths**: Paths to PDB, MOL, SDF, or other supported formats
  - Can be strings or `Path` objects

### Result Format

Results are returned as `pd.DataFrame`:
- Each row = one molecule tested
- Each column = one test
- Values are typically boolean (passed/failed) or numeric (metrics)
- Use `full_report=True` for detailed intermediate values

### Thresholds and Tolerances

Most checks have configurable thresholds:
- Stricter thresholds (lower values): more molecules fail validation
- Looser thresholds (higher values): more molecules pass
- Defaults represent reasonable starting points for most use cases

---

## Configuration Presets

### `'redock'` (Default)

Full validation suite for docking redocking tasks.

- Includes: geometry, energy, flatness, identity, intermolecular distance, RMSD, volume overlap
- Suitable for: crystal ligand redocking to same protein

### Custom Configurations

Pass a dictionary to customize tests and parameters:

```python
config = {
    'tests': ['geometry', 'rmsd'],
    'geometry': {'threshold_bad_bond_length': 0.15},
    'rmsd': {'rmsd_threshold': 1.5}
}

pb = PoseBusters(config=config)
```

---

## Performance Considerations

**Conformer Generation:**
- Energy ratio module generates 100 conformers by default
- Can be slow for large molecules
- Consider reducing `ensemble_number_conformations` for faster validation

**Batch Processing:**
- Use `bust_table()` for multiple molecules
- More efficient than calling `bust()` in a loop
- Enables parallelization across molecules

**Memory:**
- Load molecules lazily using file paths rather than pre-loading all as Mol objects
- PoseBusters will handle loading at test time

---

## Related Documentation

- [RDKit Molecule Documentation](https://www.rdkit.org/)
- [Distance Geometry](https://www.rdkit.org/docs/)
- [SMARTS Patterns](https://www.rdkit.org/docs/RDKit_Book.html#smarts-queries)