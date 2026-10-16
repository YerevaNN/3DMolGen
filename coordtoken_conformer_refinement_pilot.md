# CoordToken Conformer Refinement Pilot

Continue the existing CoordToken refinement experiments inside the current CoordToken repository.

Use the folder containing the previous PoseBench refinement experiment as the working directory. Do not create a new top-level project and do not modify or overwrite the previous PoseBench results.

Create a new subfolder inside the current refinement directory:

```text
platinum_diverse_refinement/
```

## Goal

Evaluate frozen CoordToken as a zero-shot post-hoc refiner of the published
BCL::Conf MES50 conformer ensembles for Platinum Diverse 2017_01.

The first pilot should answer:

> Can CoordToken improve physical validity while preserving or improving agreement with experimentally observed bound conformations?

## Data

Use the checksum-verified ACS Figshare supplements accompanying
`10.1021/acs.jcim.0c01140`:

- The 2,859 experimental protein-bound ligand conformations.
- The published resonance-aware maximum-ensemble-size-50 conformers
  (`clustered.50.withresonance.8000.sdf.gz`).
- Supplement `.s003` for the original native/input structures and protocol.

The archive contains 132,165 conformers across all 2,859 molecules (3–50 per
molecule). Its molecule IDs are benchmark indices; this release does not
define train/validation/test evaluation splits.

Before running the full experiment, report:

- Number of molecules and conformers found.
- Heavy-atom count statistics.
- CoordToken sequence-length statistics.
- Unsupported atom types or SMILES tokens.
- Molecules exceeding CoordToken limits.
- Parsing and atom-mapping failures.

Do not silently exclude failures.

## Refinement

For every published generated conformer:

1. Keep the molecular graph, atom identities, bonds, formal charges, and stereochemistry fixed.
2. Run one frozen CoordToken encode-decode pass.
3. Use the same CoordToken checkpoint and preprocessing code used in the previous refinement experiment.
4. Reuse the existing atom-mapping and reconstruction code where possible.
5. Symmetry-aware align the reconstructed conformer back to the input conformer using Kabsch alignment.
6. Preserve the original centroid and rigid-body placement.
7. Save both the original and CoordToken-refined conformers.

CoordToken must only refine internal geometry. It must not perform docking or change molecular topology.

## Compared Conditions

Evaluate exactly two conditions in this pilot:

1. Original published conformer ensemble.
2. The same ensemble after CoordToken refinement.

Use exactly the same successfully paired molecules and conformers for both conditions.

## Evaluation

### Conformer-generation metrics

Platinum Diverse contains one experimental bound conformer per molecule.
Compute the official symmetry-aware heavy-atom nearest-reference RMSD and
recovery at both reported benchmark thresholds:

- Mean and median minimum RMSD.
- Recovery at 0.6 Å.
- Recovery at 1.0 Å.

Do not label these results COV-P or MAT-P: with only one reference per molecule,
those are not independent precision metrics. Also report ensemble diversity
retention (unique count, duplicate rate, and pairwise RMSD distribution) so
apparent recovery gains cannot hide conformer collapse.

### Physical-validity metrics

Run PoseBusters with the molecule-only configuration on every original and
refined conformer. Protein/redocking checks are inapplicable because generated
free conformers are not docked in the crystallographic protein frame.

Report:

- Overall PoseBusters validity.
- Every individual PoseBusters check.
- Number and percentage passing each check.
- Number changed from fail to pass.
- Number changed from pass to fail.

### Paired analysis

For each molecule, report whether refinement:

- Improved nearest-reference RMSD.
- Left it effectively unchanged.
- Worsened nearest-reference RMSD.
- Changed PoseBusters from invalid to valid.
- Changed PoseBusters from valid to invalid.

Use paired bootstrap 95% confidence intervals over molecules for the main metric differences.

## Leakage Check

Build or load the canonical-SMILES sets used in CoordToken training and validation.
Treat this as a separately reported leakage audit: the checkpoint training
corpus is a packed 190-million-record dataset and does not expose benchmark
split labels directly.

Report benchmark overlap with:

- CoordToken training data.
- CoordToken validation data.

Save all overlapping molecule IDs.

Produce both:

1. Results on the full supported benchmark.
2. Results after removing train/validation SMILES overlaps.

Do not remove overlaps without reporting them.

## Required Outputs

Save inside:

```text
platinum_diverse_refinement/
```

Required files:

```text
README.md
config.yaml
data_summary.json
unsupported_molecules.csv
overlap_molecules.csv
per_conformer_results.parquet
per_molecule_results.csv
summary_metrics.csv
posebusters_check_summary.csv
original_conformers.sdf
coordtoken_refined_conformers.sdf
```

Also create a concise final report containing:

- Dataset and supported-sample counts.
- COV-R, COV-P, MAT-R, and MAT-P before and after refinement.
- Overall PoseBusters validity before and after refinement.
- Individual PoseBusters check changes.
- Improve / unchanged / worsen counts.
- Paired confidence intervals.
- Full-data and leakage-filtered results.
- Four qualitative examples:
  - strongest improvement;
  - typical improvement;
  - almost unchanged;
  - clearly harmed.

## Reproducibility

The `README.md` must include:

- Exact environment.
- CoordToken checkpoint path.
- Dataset source and preparation steps.
- Exact commands for refinement.
- Exact commands for evaluation.
- Random seeds.
- Definitions of paired and excluded samples.

Reuse the previous PoseBench refinement implementation rather than duplicating working utilities, but keep all new experiment-specific code and outputs inside `platinum_diverse_refinement/`.
