# CASF/ChEMBL3D conformer-set pipeline

This report describes the exact-match input construction and the current
`generate_casf_smiles_conformer_sets.py` generation workflow.

## CASF16 to ChEMBL3D Matching

`scripts/match_casf16_chembl3d_exact.py` builds the intersection CSV consumed by
generation.

1. CASF ligand `.mol2` files are loaded with RDKit using `sanitize=True,
   removeHs=False`; if that fails, the script retries `sanitize=False` and then
   calls `Chem.SanitizeMol`.
2. CASF molecules are normalized as `RemoveHs -> AddHs(addCoords=True)`. Hydrogen
   coordinates are set to `(0, 0, 0)` so hydrogen placement does not carry pose
   information.
3. Four CASF SMILES are recorded:
   - `casf_explicit_isomeric_smiles`: canonical isomeric SMILES on the H-added molecule
   - `casf_explicit_nonisomeric_smiles`: canonical non-isomeric SMILES on the H-added molecule
   - `casf_heavy_isomeric_smiles`: canonical isomeric SMILES after `RemoveHs`
   - `casf_heavy_nonisomeric_smiles`: canonical non-isomeric SMILES after `RemoveHs`
4. ChEMBL3D candidates come from `chembl3d_topology_smiles_index.csv`. The script
   parses both supported index schemas, keeps rows with a 3-digit group, and
   matches only on `casf_heavy_isomeric_smiles == chembl3d_isomeric_smiles`.
5. Duplicate ChEMBL3D rows are deduplicated by `(isomeric_smiles, group, mol_id)`,
   keeping the row with the largest `conformer_count`.
6. A hit is eligible only if the ChEMBL3D topology SDF entry exists in
   `topologies/{group}.sdf` and the prepared ChEMBL3D reference has at least one
   rotatable torsion.
7. Output rows contain CASF SMILES variants, selected `chembl3d_group`,
   `chembl3d_mol_id`, `chembl3d_isomeric_smiles`, and ChEMBL3D `conformer_count`.

## Generation Inputs

Each CSV row maps one CASF ligand to one ChEMBL3D topology molecule. Generation
loads that ChEMBL3D topology molecule as the reference/torsion seed. RDKit random
pipelines use the same topology with all conformers removed as `base_mol`.

| Setting | Value |
|---|---:|
| Fixed pool size | `1000` |
| Generation batch size | `1000` |
| Dynamic target | `max(1, -20 + 22 * rotatable_bonds)` |
| ChEMBL-count target | `max(1, conformer_count)` from the intersection CSV |
| Torsion perturbation | all rotatable torsions, uniform `+/-120 deg` |
| Initial torsion-min pre-pool | `1500` DG-clash passers |
| Torsion-min refill pre-pool | `500` DG-clash passers |
| Minimizer | `MMFF94s`, `500` iterations, accepted statuses `0` and `1` |

Each ligand produces four generation families and three sampling tiers:

- Families: `rdkit_random_raw`, `rdkit_random_minimized`, `torsion_raw`,
  `torsion_minimized`
- Tiers: `fixed`, `dynamic`, `chembl_count`
- Total outputs per ligand: `4 x 3 = 12` SDFs plus `.indices.tsv` sidecars

## Generation Families

### `rdkit_random_raw`

1. Embed RDKit conformers with `ETKDGv3`, `useRandomCoords=True`,
   `enforceChirality=True`, `pruneRmsThresh=-1.0`.
2. Reject only conformers with non-finite coordinates.
3. Accumulate until the pre-PoseBusters fixed pool has `1000` conformers.

### `rdkit_random_minimized`

1. Generate RDKit random conformers as above.
2. Minimize each conformer with `MMFF94s`.
3. Reject minimization exceptions and statuses outside `{0, 1}`.
4. No DG clash filter is applied before or after minimization.
5. Accumulate until the pre-PoseBusters fixed pool has `1000` minimized conformers.

### `torsion_raw`

1. Start from the ChEMBL3D reference conformer.
2. Rotate all detected rotatable torsions by random deltas in `[-120 deg, 120 deg]`.
3. Apply the DG clash filter at cutoff `0.7`.
4. Accumulate until the pre-PoseBusters fixed pool has `1000` DG-clash passers.

### `torsion_minimized`

1. Generate torsion-perturbed conformers and pre-filter with DG clash cutoff `0.7`.
2. First minimize tranche uses `1500` pre-filter passers; later refills use `500`.
3. Minimize with `MMFF94s`.
4. Apply the same DG clash filter after minimization.
5. Accumulate post-minimize DG-clash passers until the fixed pool has `1000`.

DG clash filter details: RDKit
`GetMoleculeBoundsMatrix(set15bounds=True, scaleVDW=True,
doTriangleSmoothing=True, useMacrocycle14config=False)`; hydrogens ignored;
direct bonds and valence-angle `(1,3)` pairs excluded; reject when
`distance < 0.7 * DG_lower_bound`.

## Sampling and PoseBusters

Sampling happens from the full `1000` pre-PoseBusters fixed pool:

- `fixed`: all `1000`
- `dynamic`: random subset of size `max(1, -20 + 22 * RB)`, capped by pool size
- `chembl_count`: random subset of size `max(1, conformer_count)`, capped by pool size

PoseBusters is run once on the fixed pool, then per-conformer pass/fail labels are
mapped onto the `dynamic` and `chembl_count` subsets. This keeps failed conformers
in the sampling frame while avoiding repeated PB work.

The PoseBusters configuration is an inline Python dict; no named PB config or
external config file is read. The true/reference molecule is supplied so identity
checks are active. Boolean checks recorded:

- File loads: predicted and reference molecule load
- Sanitisation: RDKit sanity checks
- InChI convertible
- All atoms connected
- No radicals
- Molecular formula, bonds, tetrahedral chirality, double-bond stereochemistry
  via PoseBusters `identity` with `inchi_options="w"`
- Bond lengths, bond angles, internal steric clash via distance geometry with:
  - `threshold_bad_bond_length=0.25`
  - `threshold_bad_angle=0.25`
  - `threshold_clash=0.3`
  - `ignore_hydrogens=True`
  - `sanitize=True`
- Planar aromatic rings: 5/6-membered aromatic SMARTS, `threshold_flatness=0.25 A`
- Planar double bonds: trigonal carbon-carbon double-bond SMARTS,
  `threshold_flatness=0.25 A`
- Energy ratio: threshold `100.0`, ensemble size `50`, `inchi_strict=False`

## Manifest and Outputs

Per-ligand array tasks write `generation/manifest_parts/{mol_id}.tsv`; the merge
job writes `generation/manifest.tsv`.

Manifest rows record:

- target, selected, generated, kept, walltime
- finite/clash/minimization/post-min-clash rejection counts
- minimization and post-min-clash rejection rates
- waste ratio: `generated_candidates / selected_confs`
- overall PoseBusters pass/fail counts and pass rate
- per-PoseBusters-check failure counts/rates as JSON
- output SDF path and status

SDFs are written to `generation/{method}/{mol_id}.sdf`; sidecars
`generation/{method}/{mol_id}.indices.tsv` record selected fixed-pool indices.

## Slurm Submission

`scripts/submit_casf.sh generate [core|ref]` submits one ligand per array
task and a dependent merge job. The wrapper defaults to the ChEMBL3D conda Python,
exports `PYTHON`, `CHEMBL_MAP_CSV`, ligand dir, topology root, output dir,
`NUM_THREADS`, and `MINIMIZE_WORKERS`, and uses `gpu:0`, `16G`, and 8 CPUs by
default.
