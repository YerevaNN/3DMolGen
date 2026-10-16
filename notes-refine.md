# CoordToken refinement of PoseBench predictions

> [!WARNING]
> The PoseBench RMSD and joint RMSD-and-validity values below are superseded by
> [posebench_rmsd_rescoring.md](posebench_rmsd_rescoring.md). PoseBench v0.6.0
> mutated and aligned the crystal ligand during fragment selection before
> applying `CalcRMS`. The ligand-only PB-valid columns remain correct.

## Paper table: ligand-valid poses within 2 \AA

We evaluate one frozen CoordToken encode--decode pass on the predicted ligand
poses from nine PoseBench methods. The benchmark contains 130 requested
complexes in each of three repeats, so every percentage below uses the common
denominator

\[
3 \times 130 = 390.
\]

A requested complex without an evaluated prediction is counted as a failure.
This makes the reported value an end-to-end success rate, rather than a
conditional rate among the poses that happened to be available. `Paired n`
is retained to show prediction coverage. RFAA has only one available repeat
(94 evaluated poses), so its /390 rate is necessarily low; it should not be
read as a conditional assessment of the 94 poses it produced.

We report two endpoints. The first is the fraction of all 390 requested
complexes for which the predicted ligand simultaneously:

1. has symmetry-aware heavy-atom RMSD at most 2 \AA\ from the crystal ligand;
2. passes every applicable ligand-only PoseBusters check.

The second is ligand-only PoseBusters validity by itself. The ligand-only
composite contains the six chemical-validity checks and six
intramolecular-geometry checks. It excludes all protein-, pocket-, cofactor-,
and intermolecular-overlap checks. Separate cRMSD and RMSD-only results are
intentionally omitted.

Each endpoint cell gives `successes/390 (percentage)` and reads **original →
refine-all → selective**. The two transition columns follow the same order;
each pair is `fixed/broken` relative to the original.

| Method | RMSD ≤ 2 \AA\ & ligand-only valid | Ligand-only PB-valid | Paired n | Poses invalid→valid / valid→invalid | PB tests fixed/broken |
|---|---|---|---:|---|---|
| NeuralPLexer | 202/390 (51.8%) → 222/390 (56.9%) → **230/390 (59.0%)** | 230/390 (59.0%) → 268/390 (68.7%) → **276/390 (70.8%)** | 371 | – → 46/8 → 46/0 | – → 123/24 → 123/16 |
| RFAA | 68/390 (17.4%) → 67/390 (17.2%) → **68/390 (17.4%)** | 77/390 (19.7%) → 78/390 (20.0%) → **79/390 (20.3%)** | 94 | – → 2/1 → 2/0 | – → 5/1 → 5/0 |
| DynamicBind | 251/390 (64.4%) → 253/390 (64.9%) → **253/390 (64.9%)** | 328/390 (84.1%) → 330/390 (84.6%) → **330/390 (84.6%)** | 348 | – → 2/0 → 2/0 | – → 7/0 → 7/0 |
| DiffDock | 287/390 (73.6%) → 290/390 (74.4%) → **293/390 (75.1%)** | 322/390 (82.6%) → 329/390 (84.4%) → **329/390 (84.4%)** | 352 | – → 7/0 → 7/0 | – → 17/4 → 17/4 |
| AF3-Single-Seq | 207/390 (53.1%) → 213/390 (54.6%) → **213/390 (54.6%)** | 256/390 (65.6%) → 263/390 (67.4%) → **263/390 (67.4%)** | 354 | – → 7/0 → 7/0 | – → 9/0 → 9/0 |
| AutoDock Vina | 104/390 (26.7%) → 103/390 (26.4%) → **105/390 (26.9%)** | 177/390 (45.4%) → 179/390 (45.9%) → **180/390 (46.2%)** | 188 | – → 3/1 → 3/0 | – → 4/3 → 4/2 |
| Chai-1 | 255/390 (65.4%) → 260/390 (66.7%) → **263/390 (67.4%)** | 275/390 (70.5%) → 280/390 (71.8%) → **283/390 (72.6%)** | 359 | – → 8/3 → 8/0 | – → 11/6 → 11/3 |
| AF3 | 285/390 (73.1%) → 282/390 (72.3%) → **285/390 (73.1%)** | 291/390 (74.6%) → 288/390 (73.8%) → **291/390 (74.6%)** | 354 | – → 0/3 → 0/0 | – → 0/3 → 0/0 |
| Chai-1-Single-Seq | 272/390 (69.7%) → 273/390 (70.0%) → **276/390 (70.8%)** | 294/390 (75.4%) → 293/390 (75.1%) → **298/390 (76.4%)** | 353 | – → 4/5 → 4/0 | – → 5/6 → 5/0 |

### Instruction for Prism: handling RFAA in the figure

Keep RFAA in the numerical table for completeness and transparency, but **do
not include RFAA in the histogram or in visual comparisons across methods**.
RFAA is not comparable to the other eight methods because only one of the
three requested repeats is available: it has 94 evaluated poses from one
130-complex repeat, whereas the other methods were evaluated against the full
three-repeat target of 390 cases. Treating the two unavailable RFAA repeats as
model failures would make its histogram bar artificially low and visually
misleading.

The histogram caption should state:

> RFAA is omitted from the histogram because only one of three benchmark
> repeats is available (94 evaluated poses from one 130-complex repeat), so its
> result is not directly comparable to the three-repeat results. It is retained
> in the table for completeness.

Do not use RFAA when summarizing the number of methods improved, calculating an
across-method average, ranking methods, or describing the range of histogram
changes. Such statements should refer explicitly to the **eight comparable
three-repeat methods**.

## How the experiment and aggregation were performed

- **Original** uses each method's unmodified predicted ligand pose.
- **Refine all** applies one CoordToken encode--decode pass to every available
  predicted ligand while leaving the molecular graph fixed.
- **Selective refinement** applies that pass only when the original ligand
  fails the ligand-only PoseBusters composite; originally valid ligands retain
  their original coordinates.
- Explicit hydrogens are removed for evaluation. RMSD is the symmetry-aware
  heavy-atom RMSD after optimal rigid alignment to the experimental ligand.
- A success requires both RMSD \(\le 2\) \AA\ and ligand-only validity on the
  same pose. Missing predictions and unavailable RMSDs contribute zero to the
  numerator but remain in the common denominator of 390.
- Ligand-only PB-valid is also normalized by 390, with missing predictions
  counted as failures.
- The pose-transition audit counts ligand-only-invalid poses repaired to valid
  and ligand-only-valid poses broken to invalid. The PB-test audit counts
  individual ligand-only checks changing from fail to pass and pass to fail.
  These are raw paired-pose transition counts, so they are not divided by 390.
- To remain comparable with the PoseBench release, the same published NaN mask
  is applied across original, refine-all, and selective conditions. PoseBusters
  can sometimes return an arbitrary RMSD for a formula-mismatched ligand, so
  these masked cases must not be reintroduced as successes.
- CoordToken could not tokenize two RFAA poses containing unsupported
  hypervalent-phosphorus tokens (`[P@TB20]` and `[P@TB2H]`). Their original
  coordinates were retained, so refinement failures were not silently dropped.
- The source is PoseBench v0.6.0 (2025-02-09), using raw per-complex predictions
  rather than values copied from its summary table. Where both pipelines define
  RMSD, the audited values agree exactly for eight of nine methods.
- NeuralPLexer is the exception: 173 of 371 RMSDs differ from the published
  release because the available coordinates appear to correspond to a
  different (unrelaxed versus relaxed) pose variant. Its row therefore reports
  the coordinates actually refined and evaluated here, not a reproduction of
  PoseBench's published NeuralPLexer baseline.

## Reproducibility and provenance

DiffDock and Vina per-complex inputs are under:

`/mnt/weka/fgeikyan/CoordToken/posebench_{diffdock,vina}_no_postprocessing_reproduction/outputs/`

The other seven methods are under:

`/mnt/weka/fgeikyan/CoordToken/posebench_remaining_methods_no_postprocessing_reproduction/outputs/<method>/`

The original and refine-all results come from the per-complex CSV files in the
`baseline_summary`, `coordinate_summary`, and `coordtoken_summary` directories.
Selective refinement and fallback rescoring were computed post hoc from those
files. The selective-refinement scripts used in the original session were not
committed to this repository, which remains a reproducibility limitation that
must be disclosed if this table is used in the paper.

### Suggested paper caption

> **CoordToken refinement of PoseBench ligand poses.** Success requires both a
> symmetry-aware heavy-atom RMSD of at most 2 \AA\ to the crystal ligand and
> passage of all ligand-only chemical and intramolecular PoseBusters checks.
> Ligand-only PoseBusters validity is also reported separately.
> Percentages use all 390 requested complex--repeat cases as the denominator;
> missing or unevaluable predictions count as failures. Refine-all applies one
> frozen CoordToken encode--decode pass to every available pose, whereas
> selective refinement applies it only to poses that initially fail the
> ligand-only validity gate. RFAA has only one available repeat (94 poses).

## Platinum Diverse: refinement of already-clean conformer ensembles

Platinum Diverse is a separate stress test and must not be pooled with the
PoseBench docking table. We apply the same frozen one-pass CoordToken
encode--decode operation to the published resonance-aware BCL::Conf MES50
ensembles for Platinum Diverse 2017_01: **2,859 molecules and 132,165
conformers**. Molecular graphs remain fixed and only coordinates change. Each
molecule has one experimental bound conformation.

PoseBusters runs in molecule-only mode, so PB-valid here is already the
ligand-only chemical-plus-intramolecular composite. Selective refinement acts
only on the 2,380 originally PB-invalid conformers (1.8% of the pool) and keeps
the other conformers unchanged.

| Metric | Original | Refine all | Selective refinement | Selective change |
|---|---:|---:|---:|---:|
| PB-valid conformers, macro over molecules (%) | 98.10 | 84.28 | **98.67** | +0.56 pp |
| Originally invalid conformers repaired (%) | -- | 656/2,380 (27.6%) | **656/2,380 (27.6%)** | -- |
| Recovery at 0.6 \AA\ (%) | 63.80 | 61.53 | **63.80** | +0.00 pp |
| Recovery at 1.0 \AA\ (%) | 86.60 | 84.44 | **86.60** | +0.00 pp |
| Mean nearest symmetry-aware RMSD (\AA) | 0.5433 | 0.5738 | **0.5436** | +0.0003 \AA |

The macro PB percentages above first compute the valid-conformer fraction for
each molecule and then average over 2,859 molecules. They therefore must not be
reconstructed from the pooled conformer counts below. The pooled micro PB rate
over all 132,165 conformers is 98.20% before, 83.27% after refine-all, and
98.70% after selective refinement. Both summaries are correct but answer
different weighting questions; the paper table uses the macro values.

| Audit | Original → refine all → selective |
|---|---|
| Conformers invalid→valid / valid→invalid | – → 656/20,390 → 656/0 |
| Individual PB tests fixed/broken | – → 1,084/45,344 → 1,084/1,053 |

Blanket refinement is strongly harmful on this clean input: it repairs 656
invalid conformers but breaks 20,390 valid ones, reducing macro PB validity by
13.8 percentage points and worsening both recovery and nearest RMSD. Most
damage appears in bond lengths and bond angles, exposing CoordToken's
reconstruction/quantization floor when the input geometry is already cleaner
than the reconstruction.

The validity gate changes the result. Selective refinement repairs **656 of the
2,380 originally PB-invalid conformers (27.6%)** without breaking any
originally valid conformer, raises macro PB validity from 98.10% to 98.67%,
restores both recovery thresholds exactly, and leaves nearest RMSD essentially
unchanged. This is a counterexample to any
claim that CoordToken is a universally improving geometry optimizer; the
supported conclusion is that refinement can repair invalid geometry when
coupled to an input-validity gate.

### Instruction for Prism: Platinum presentation

Do **not** add Platinum as another bar in the PoseBench eight-method histogram.
It is a conformer-ensemble experiment with 132,165 conformers and per-molecule
macro PB weighting, whereas PoseBench evaluates requested complex--repeat
cases over a `/390` denominator. Present Platinum as its own table, panel, or
failure-case paragraph. Clearly label PB validity as macro over molecules and
keep the pooled transition counts as a separate audit.

Suggested caption:

> **CoordToken refinement on Platinum Diverse.** The input contains 2,859
> molecules and 132,165 already-clean BCL::Conf conformers. PB validity is the
> mean per-molecule fraction of conformers passing all molecule-only checks.
> Blanket refinement damages valid conformers, whereas selective refinement
> applies one frozen CoordToken pass only to the 2,380 originally invalid
> conformers, repairing 656 of them (27.6%) without breaking an originally
> valid conformer. Recovery and nearest RMSD use symmetry-aware heavy-atom comparison
> with the experimental bound conformation.

Source files:

`/mnt/weka/fgeikyan/CoordToken/posebusters_refinement/platinum_diverse_refinement/`

The main artifacts are `summary_metrics.csv`, `posebusters_check_summary.csv`,
`per_conformer_results.csv`, and `per_molecule_results.csv`. Bootstrap
confidence intervals use 5,000 molecule-clustered replicates with seed
20260730.
