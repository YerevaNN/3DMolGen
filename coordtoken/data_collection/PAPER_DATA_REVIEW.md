# Data review of neurips_2026_ai4dd.pdf

Reviewed all 19 PDF pages on 2026-09-25. References below use the PDF page and
printed line numbers. This is a review of the supplied draft and local evidence,
not confirmation that every local artifact produced a particular reported table.

## Findings that affect collection and split claims

1. **The global separation claim is contradicted by the audits.** Section 2,
   p.2 lines 69–75, says canonical-SMILES overlap is prevented across sources,
   with an exception inside nabla. A.3, p.11 lines 415–420, describes filtering
   nabla training against other datasets' holdouts, but does not establish the
   converse exclusion. We measured 281,848 other-source training rows matching
   nabla scaffold/structure tests by full Standard InChIKey and 6,390 matching
   nabla validation. These are remaining overlaps in the local training CSV.

2. **The nabla exception conflates three evaluation regimes.** A.3 lines 415–417
   motivates preserving all nabla internal overlap because otherwise most training
   disappears. The intended known-molecule overlap belongs to conformation tests.
   It does not justify scaffold/structure test molecule overlap. The full-key
   audit finds 482 nabla training rows matching scaffold/structure tests and 21
   matching validation. A selective exclusion removes 503 nabla rows, not the
   millions removed by treating the conformation test as a molecule holdout.
   This count applies to the merged corpus's nabla portion; the separately packed
   nabla-only training pool has different membership and needs its own key audit.

3. **Table 1 row 3 is explicitly a conformation result.** Its caption on p.3
   states that nabla rows use the test-conformations subset. Shared molecular
   identity alone therefore does not invalidate its reconstruction number. It
   must be labeled as known-molecule/unseen-conformation evaluation, and exact or
   near-identical test geometry exposure still needs testing. The all-test row
   mixes evaluation regimes. Report conformation, novel-molecule, and audited
   scaffold results separately rather than relying on the aggregate alone.

4. **Table 6 obscures the evaluation population.** On p.11, the nabla test count
   combines 3,907,073 records across all three subsets, while the same 0.074 RMSD
   appears in Table 1's conformation row. We need the actual evaluation manifest
   and per-subset errors to determine whether this is a reused subset score or a
   coincidentally equal pooled value. Recompute the global micro average from
   per-sample results, with explicit included files and sample IDs. Add a macro
   average so a large source does not obscure small-source performance.

5. **Split-size reconciliation is needed, but not every table count is wrong.**
   A fresh CSV recount gives 1,817,389 validation rows (the PDF total is 1,817,390).
   Test files contain 5,568,079 rows including 23,404 reserved GEOM-revisited
   records. Excluding that reference-only file yields 5,544,675, matching Table 6.
   The current GEOM test file has 711,228 rows, matching that table. This checks
   the local CSV populations, not which snapshots were evaluated or trained on.

6. **Downstream GEOM requires its own exclusion.** A.11 lines 525–528 says SFT
   and evaluation use original GEOM splits before tokenizer-corpus filtering.
   A.12 lines 617–621 explicitly discloses 22 test identities present in training.
   The recovered downstream manifest additionally reports 586 train/validation
   key collisions involving 13,536 training conformers, 22 train/test collisions
   involving 392 training conformers, and 5 validation/test collisions involving
   150 validation conformers. Its policy is `audit_only_no_exclusions`. These
   stored counts need to be reproduced under the new pinned identity policy;
   do not silently inherit them as newly verified molecular identities.

7. **Identity provenance has a stereo mismatch.** Of the 1,000 downstream GEOM
   benchmark records, three stored metadata keys differ from keys regenerated
   from the saved standardized SMILES with the current declared text policy.
   One stored InChI contains stereo layers absent from the saved SMILES. The
   registry preserves each disagreement rather than pretending a text-derived
   key is the original stereo-resolved identifier. Coordinate-derived identity
   must be audited separately from the conditioning graph used by the model.

8. **External refinement holdouts also matter.** Table 5 uses GEOM test identities;
   Table 13 uses PoseBench experimental complexes; Table 14 uses Platinum Diverse
   experimental ligands; GEOM-XL is called out of distribution in A.5. Freeze
   these reference identities and quantify exposure to every training stage,
   including the pretrained tokenizer. A compound can enter through a different
   source even when a benchmark's own split is sound. Molecule novelty, scaffold
   novelty, protein novelty, and geometry novelty are different properties.

9. **Historic sampling is not exactly reproducible from the recovered scripts.**
   PubChem uses periodic CID-run selection; ZINC uses seeded block selection.
   Historical unordered sets, file order and parallel quotas can alter membership.
   For the revision, pin input hashes, actual sample IDs, split ownership, software
   and command lines. Describe measured eligible populations and failed records,
   not an unsupported IID random sample or exact historical replay.

10. **Names alone are not sample identifiers.** Some merged rows have repeated
    names such as `BindingMoad_BindingMoad`. New release manifests must use an
    immutable source snapshot plus source row/byte offset (and, where available,
    original molecule/conformer IDs); do not join evaluations on the name column
    without testing uniqueness. Preserve upstream provenance and record which
    records were excluded instead of silently editing old datasets.

## Identity and split policy for the first revision

- Generate full Standard InChIKeys from consistently decoded enriched text.
  Preserve encoded stereo, isotopes and charges; apply no custom salt stripping
  or stereochemistry assignment from coordinates. Standard InChI itself can
  normalize recognized tautomers. Do not call this exact stereochemical equivalence
  for all conformers, or replace it silently with the first 14 characters.
- Protect nabla validation, structure-test and scaffold-test identities against
  every training source, including nabla. Keep the conformation test separate;
  molecular identity overlap is intentional there, exact geometry leakage is not.
- Quarantine the 13 unconvertible training records in a separate CSV with reasons.
  They are not certified clean simply because no key was produced.
- This removes 288,741 overlap rows plus 13 quarantined records and leaves
  189,925,494 rows from the 190,214,248-row input. This is a nabla-focused revision,
  not the final globally decontaminated publication dataset.
- Audit all other retained validation/test files, reserved GEOM-revisited data,
  and the actual downstream GEOM references before producing the final union
  exclusion. Preserve reference files; record validation/test conflicts explicitly.
- Do not claim scaffold-disjoint training from molecule-key exclusion. Measure
  Bemis–Murcko scaffold exposure and its data-retention cost separately before
  deciding whether to enforce it or label evaluation as molecule-disjoint only.

## Release gates before training publication models

| Gate | Required evidence | Current status |
| --- | --- | --- |
| Input snapshot | source paths, sizes, checksums, software and code hashes | available for audit references; revision has chunk hashes |
| Molecular holdouts | zero successful-key train/heldout intersections for declared references | nabla revision written; broader audit running |
| Validation/test ownership | overlaps and a frozen test-priority policy | broader reference audit required |
| Conformation holdouts | no exact or near-duplicate test geometries under declared RMSD/atom mapping | not measured |
| Scaffold regime | scaffold exposure counts, policy and retained data size | not measured |
| Structural QC | parseability, finite coordinates, token/atom/coordinate alignment, supported chemistry and sequence limits | identity failures quarantined; remaining QC pending |
| Multiplicity and provenance | conformer counts per identity/source, exact duplicates, weighting policy | retain all nonexcluded rows for first revision; deeper audit pending |
| Model linkage | exact dataset/packed-data/checkpoint hashes and evaluation manifests | downstream checkpoint manifest found; Table 1/6 linkage incomplete |
| Independent verification | counted output, audit reconciliation, file hash and reproducible failure behavior | implemented for first revision |

Passing molecule-level gates does not by itself prove that every scientific
notion of leakage has been eliminated. State the guarantee precisely.

## Consequences for the stronger-model phase

Retrain the tokenizer on the frozen corrected corpus before claiming corrected
results. Rebuild packed data and derived FSQ sequences; do not reuse encodings or
checkpoints without identifying the tokenizer snapshot that created them. Use the
same corrected molecule populations for both downstream representations, including
SFT and validation. Freeze hyperparameters using clean validation; hold test data
for final evaluation. Repeat seed-controlled comparisons and report paired
uncertainty and failure accounting after the data gates, not before.

Additional manuscript issues to revisit after collection is fixed: PoseBench
ligand-optimal alignment (A.9) evaluates ligand shape and does not alone establish
receptor-frame docking-pose accuracy; the two ETFlow outliers drive the MAT-P
improvement (A.8); and A.12 acknowledges different failure rates and possible
Cartesian under-training. These are interpretation/experimental-design issues,
not reasons to silently discard difficult evaluation cases.

## Located artifacts

- Supplied paper: `/mnt/weka/fgeikyan/CoordToken/neurips_2026_ai4dd.pdf`
- A manuscript source also exists at
  `/mnt/weka/fgeikyan/3dmolgen_bigdata/coordtoken_paper_revised.tex`; equivalence to
  the supplied PDF has not been established and that file was not edited.
- Historical collection package: GitHub `YerevaNN/CoordToken`, branch
  `data-collection-190m`, commit `48e1f7f897a1bba04ac4be34b3e485134c09451d`.
- Downstream manifests:
  `/mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_redesigned_geom_fsq/manifests/`.
- Pinned downstream test pickle:
  `/mnt/weka/fgeikyan/tokenizer/distinct_smi_filtered_13.pickle`.
- Pinned tokenizer checkpoint hash from those manifests:
  `270d4cab25f723662005163357f413338e2836c5ef08ae0a9aa3cca1d0c538ff`.
- Local split recount: `../logs/paper_review/current_split_counts.json`.
- Frozen reference registry: `runs/holdout_registry_v3/registry.json`.
