# Data release acceptance criteria

A revision is ready for a publication training run only when its declared scope
passes these checks. This is an engineering/scientific acceptance list, not a
claim that the current intermediate output already passes every item.

1. **Frozen identities.** Every holdout is registered with its purpose, original
   source, split, version and content hash. The molecular identity policy and
   handling of salts, protonation, stereo, isotopes and tautomers are explicit.
   Metadata/text/3D disagreement reports accompany any identity transformation.
2. **Explicit split regimes.** Known-molecule conformations, molecule holdouts and
   scaffold holdouts are distinct. Test priority applies when repairing molecular
   validation/test conflicts. Do not silently move or shrink a public benchmark
   to improve scores. Record every exclusion and the changed evaluation population.
3. **Training exclusion.** All successful-key intersections with every declared
   molecular holdout are zero after filtering across every source and training
   stage. Keep aggregate row and unique-identity counts, not only examples.
4. **No unresolved records in the clean release.** Parse/key failures and malformed
   or nonfinite coordinates are quarantined with reasons. Check atom-token and
   coordinate counts, atom order, supported vocabulary, chemical valence and
   sequence limits. Physically strained conformers require source-aware quality
   reporting; do not remove them simply because a model performs poorly on them.
5. **Conformer exposure.** Audit exact enriched samples and coordinate duplicates
   independently from molecular identities. For intentional same-molecule tests,
   compare geometries with defined hydrogen, symmetry, atom-mapping, alignment and
   RMSD-tolerance policies. Record near duplicates and their effect on evaluation.
6. **Distribution and weighting.** Count conformers and unique identities per
   source, molecule size/charge/stereo, and conformers per molecule. Prevent
   accidental source duplication from masquerading as new independent data.
   Record any caps or sampling decisions deterministically; do not silently
   change a source's scientifically meaningful conformer distribution.
7. **Provenance and reproducibility.** Immutable source snapshots, stable row IDs,
   code/environment versions, random seeds and saved split assignments accompany
   the release. Historical sources that cannot be recovered are explicitly marked.
   Document source licenses before redistribution. New filtering is reproducible
   from its specified input; it is not claimed to recreate unknown historical runs.
8. **Independent output checks.** Conservation of input/retained/excluded/quarantine
   counts, per-source/per-identity removal reconciliation, content hashes, and a
   fresh read-back pass. An independent molecular overlap check is required for
   the final full release, not only success messages from the writer.
9. **Derived artifact linkage.** Rebuild packed indices and frozen-tokenizer codes
   from the final corpus. Store data/packed/encoder/checkpoint hashes in each run.
   Exclude downstream holdouts from the tokenizer as well as the language model;
   use the same corrected examples for both representation baselines.
10. **Evidence-backed reporting.** Generate Table 6 counts from the release manifest.
    Publish per-subset reconstruction and micro/macro aggregates with denominators,
    bootstrap units and failure accounting. Existing results remain labeled with
    their old data snapshot until retraining/re-evaluation supports new claims.

Model tuning starts after these data decisions are frozen. Improving the tokenizer
or language model cannot retroactively repair test exposure in an existing run.
