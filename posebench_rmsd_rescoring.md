# Corrective PoseBench pose-RMSD rescore

Status: complete, 2026-09-03.

## Decision

The previous PoseBench `RMSD <= 2 A` and joint RMSD-and-validity results are
superseded. They were derived through the PoseBench v0.6.0 primary-ligand
selection path, which called RDKit `GetBestRMS(reference, prediction)` and then
wrote the mutated reference molecule for downstream scoring. Because
`GetBestRMS` rigidly aligns its probe argument in place, the nominally
unaligned PoseBusters `CalcRMS` calculation received an already aligned crystal
ligand. The old values therefore behaved like conformational RMSDs and do not
measure pose accuracy.

The ligand-only PoseBusters validity results are unaffected. They depend on the
predicted ligand's chemistry and internal geometry, not on the crystal
ligand's global coordinates.

## Corrected procedure

No tokenizer inference or refinement was rerun. The correction reads the saved
original and CoordToken-refined SDF files and:

1. reloads every crystal ligand from its untouched benchmark SDF;
2. performs primary-fragment selection using copies whenever `GetBestRMS` is
   needed as a tie-breaker;
3. rigidly translates each saved decoded ligand so that its heavy-atom centroid
   exactly equals the corresponding original predicted-pose centroid;
4. removes explicit hydrogens and computes symmetry-aware pose RMSD using
   RDKit `CalcRMS`, without ligand superposition;
5. computes `GetBestRMS` on separate copies only as a diagnostic;
6. applies the original ligand-only PoseBusters flags and the same gating rule;
7. divides every success count by the common 390 requested complex--repeat
   cases. Missing and unevaluable cases remain denominator failures.

The explicit centroid translation in step 3 is important. The saved decoder
outputs had already had the input centroid added back, but the reconstructed
centered coordinates retained a small residual mean. The additional rigid
translation was 0.00769 A at the median and 0.11267 A at the maximum. It makes
the intended policy exact without changing internal geometry or rerunning the
model.

For two RFAA ligands that CoordToken could not tokenize, the original
coordinates are retained as the refined fallback, as in the previous table.
RFAA still has only one available repeat and must remain excluded from the
cross-method histogram and eight-method summary.

## Corrected results

Each metric reads **original -> refine all -> PB-gated**. `Evaluated` is the
number of original predictions that reached the saved evaluation; `RMSD n` is
the number for which the corrected atom correspondence produced an RMSD. Every
percentage nevertheless uses 390.

| Method | Evaluated | RMSD n | Pose RMSD <= 2 A | Pose RMSD <= 2 A and ligand-only valid | Ligand-only PB-valid |
|---|---:|---:|---|---|---|
| NeuralPLexer | 371 | 371 | 49/390 (12.6%) -> 49/390 (12.6%) -> 49/390 (12.6%) | 35/390 (9.0%) -> 42/390 (10.8%) -> **43/390 (11.0%)** | 230/390 (59.0%) -> 268/390 (68.7%) -> **276/390 (70.8%)** |
| RFAA | 100 | 94 | 24/390 (6.2%) -> 24/390 (6.2%) -> 24/390 (6.2%) | 20/390 (5.1%) -> 20/390 (5.1%) -> **20/390 (5.1%)** | 77/390 (19.7%) -> 78/390 (20.0%) -> **79/390 (20.3%)** |
| DynamicBind | 348 | 348 | 57/390 (14.6%) -> 57/390 (14.6%) -> 57/390 (14.6%) | 52/390 (13.3%) -> 54/390 (13.8%) -> **54/390 (13.8%)** | 328/390 (84.1%) -> 330/390 (84.6%) -> **330/390 (84.6%)** |
| DiffDock | 352 | 352 | 110/390 (28.2%) -> 110/390 (28.2%) -> 110/390 (28.2%) | 105/390 (26.9%) -> 106/390 (27.2%) -> **106/390 (27.2%)** | 322/390 (82.6%) -> 329/390 (84.4%) -> **329/390 (84.4%)** |
| AF3-Single-Seq | 375 | 354 | 3/390 (0.8%) -> 3/390 (0.8%) -> 3/390 (0.8%) | 3/390 (0.8%) -> 3/390 (0.8%) -> **3/390 (0.8%)** | 256/390 (65.6%) -> 263/390 (67.4%) -> **263/390 (67.4%)** |
| AutoDock Vina | 188 | 188 | 14/390 (3.6%) -> 14/390 (3.6%) -> 14/390 (3.6%) | 14/390 (3.6%) -> 14/390 (3.6%) -> **14/390 (3.6%)** | 177/390 (45.4%) -> 179/390 (45.9%) -> **180/390 (46.2%)** |
| Chai-1 | 362 | 347 | 139/390 (35.6%) -> 139/390 (35.6%) -> 139/390 (35.6%) | 118/390 (30.3%) -> 122/390 (31.3%) -> **122/390 (31.3%)** | 275/390 (70.5%) -> 280/390 (71.8%) -> **283/390 (72.6%)** |
| AF3 | 375 | 354 | 201/390 (51.5%) -> 201/390 (51.5%) -> 201/390 (51.5%) | 171/390 (43.8%) -> 168/390 (43.1%) -> **171/390 (43.8%)** | 291/390 (74.6%) -> 288/390 (73.8%) -> **291/390 (74.6%)** |
| Chai-1-Single-Seq | 364 | 350 | 128/390 (32.8%) -> 128/390 (32.8%) -> 128/390 (32.8%) | 119/390 (30.5%) -> 121/390 (31.0%) -> **122/390 (31.3%)** | 294/390 (75.4%) -> 293/390 (75.1%) -> **298/390 (76.4%)** |

Among the eight comparable three-repeat methods, gating improves joint pose
success for five methods (NeuralPLexer, DynamicBind, DiffDock, Chai-1, and
Chai-1-Single-Seq) and preserves it for three (AF3-Single-Seq, AutoDock Vina,
and AF3). The largest change is NeuralPLexer, 35/390 to 43/390, or 9.0% to
11.0%. The earlier statements that seven of eight methods improve and that
NeuralPLexer changes from 51.8% to 59.0% must be removed.

The pose-RMSD numerator is unchanged by refinement for every method. With the
centroid held exactly fixed, one CoordToken pass changes internal coordinates
but does not move any case across the 2 A pose threshold. Changes in the joint
endpoint therefore come from ligand-validity repairs or damage.

There are 77 original evaluated poses without a corrected RMSD: AF3 21,
AF3-Single-Seq 21, Chai-1 15, Chai-1-Single-Seq 14, and RFAA 6. Every one of
these 77 poses fails the ligand-only validity composite, so none can contribute
to the joint endpoint and their unavailable RMSDs do not change its numerator.

## Regression checks

All checks pass with RDKit 2025.09.6:

- 5,670 before/after assertions confirmed that reference coordinates remained
  unchanged during fragment selection and scoring.
- Original and refined heavy-atom centroids agree after restoration; the
  maximum residual difference is `8.76e-14 A` under a `1e-9 A` assertion.
- Across 2,758 finite original pairs, `CalcRMS` and `GetBestRMS` are equal in
  zero cases at `1e-6` tolerance. `CalcRMS >= GetBestRMS` in every case.
- Recomputed diagnostic `GetBestRMS` values reproduce the saved values within
  `2.83e-5 A`, consistent with SDF coordinate precision.
- DiffDock repeat 1, `7WUY_76N`, again has a 32.177489 A centroid separation.
  Its corrected `CalcRMS` is 32.611221 A, whereas `GetBestRMS` is 2.016061 A.
  The obsolete v0.6.0 path moved the crystal ligand by 32.18 A onto the
  prediction before applying `CalcRMS`.

## Reproduction and artifacts

Run from the `3dmolgen_bigdata` root:

```bash
python scripts/posebench_rescore_unaligned.py --workers 12
pytest -q tests/test_posebench_rescore_unaligned.py
```

Implementation and regression tests:

- `scripts/posebench_rescore_unaligned.py`
- `tests/test_posebench_rescore_unaligned.py`

Generated artifacts:

- `outputs/posebench_rmsd_rescoring/corrected_per_pose.csv`
- `outputs/posebench_rmsd_rescoring/corrected_summary.csv`
- `outputs/posebench_rmsd_rescoring/regression_checks.json`
- `outputs/posebench_rmsd_rescoring/rmsd_unavailable.csv`
- `outputs/posebench_rmsd_rescoring/scoring_failures.csv`

The old v0.6.0-derived CSVs were deliberately left untouched as provenance.

## Paper-ready wording

> We assess pose accuracy using PoseBusters' symmetry-aware heavy-atom RMSD,
> implemented with RDKit `CalcRMS` without ligand superposition. Primary-ligand
> fragment selection uses defensive molecule copies so that any diagnostic
> `GetBestRMS` call cannot modify the crystal coordinates. CoordToken operates
> on ligand coordinates centered at their heavy-atom centroid; after decoding,
> each output is translated back to exactly the centroid of its input predicted
> pose. Percentages use all 390 requested complex--repeat cases, with missing or
> unevaluable predictions counted as failures.

Suggested result sentence:

> Among eight comparable PoseBench methods, ligand-validity-gated CoordToken
> refinement improves joint pose-RMSD-and-validity success for five and
> preserves it for three. The largest increase is NeuralPLexer, from 35/390
> (9.0%) to 43/390 (11.0%).
