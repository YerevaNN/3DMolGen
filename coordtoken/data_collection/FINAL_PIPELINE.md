# October 6 final data pipeline

## Released files

Shared-storage root: `/mnt/weka/fgeikyan/3dmolgen_bigdata/data_handoff/merged_deduplicated_20261006`. The `merged_final` link in its parent points here. Read [release_status.json](evidence/release_20261006/release_status.json) together with the manifest: the manifest was written before the post-deduplication audit, while release status records its successful completion.

| File | Conformer rows | SHA-256 |
|---|---:|---|
| `merged_train.csv` | 189,792,028 | `24d13d817438ce5123cd3753e33cbf39d2555768456ab3de5a7516f6b3eeec96` |
| `merged_val.csv` | 1,802,906 | `535a599591f2d1b9737dd77807fdbb93e7e231503575c05c35b0b9f6c242c9f7` |
| `merged_test.csv` | 5,553,421 | `089c2afd4af973b77e7a894d2eda04544afff5c9e5530e47d6d00aaa999e0211` |

These are the CoordToken corpus files, distinct from the later CASF/protomer-filtered corpus. `merged_20261006` is the intermediate merge before exact deduplication; do not substitute it for this release. The merged CSV is not a new physical global shuffle. Tokenizer packing and seeded training permutations are in the `atom-register-tokenizer` branch, also bundled under `coordtoken/research/` in the 3DMolGen `coordtoken-integration-handoff` branch.

## Inputs and execution order

Start from the corrected grouped source snapshot recorded in [prepared_input_manifest.json](prepared_snapshot_20261005/prepared_input_manifest.json), not an already filtered merged release. ChEMBL uses the source-checked replacement populations documented in [its installation receipt](prepared_snapshot_20261005/chembl_origin/owned_chembl_install.json); these differ in membership from the defective backup files. Some surviving sources had earlier filtering; this is not an untouched raw-source reconstruction. PubChem/ZINC are sampled collections, not the full databases. The historical SDFs are not all retained.

1. Inspect the [prepared snapshot and repair evidence](prepared_snapshot_20261005/README.md). Restore or provide the recorded source/reference files and verify their hashes. Nine unresolved source rows remain quarantined: six ChEMBL, one PubChem, two ZINC. Ten other PubChem rows have source-backed repairs and restoration evidence.
2. Run [Group A prepare and processing](group_a_split/README.md). Existing supplied partitions are preserved. Fixed test identities take priority over validation, and validation over training. The prepared snapshot and pinned parser/RDKit are required.
3. Run [Group B indexing, splitting, and independent holdout verification](group_b_split/README.md), in order KIBA-3D, OMol25-small-mols, PubChem, ZINC. Whole full-InChIKey groups target 0.5% of source rows per holdout, respecting previous ownership. SHA-256 ranking with seed 42 removes set-order dependence. Quota excess/shortfall is explicitly reported.
4. Run [Group C](group_c_split/README.md) using the completed A/B evidence. Keep nabla conformations, structures, and scaffolds test files separate. Protect structure/scaffold-test molecular identities against training; conformation-test molecular overlap is intentional. New nabla validation identities are selected only where they do not occur in retained A/B training. Verify combined ownership and hashes.
5. Run [merge](merge_release/README.md), then [exact audit](exact_duplicate_audit/README.md), [exact deduplication](exact_deduplication/README.md), and the post-deduplication exact audit. Preserve source IDs and duplicate aliases. Promote only after all checks pass, using the documented finalization step.

Each linked README contains executable CLI examples. Use new output directories, configure actual paths, and keep source files immutable. Snapshot repair scripts retain their original absolute paths and are provenance records, not a portable one-command raw-data downloader. The stages pin their dependencies/evidence; do not remove these checks to force a run through a different input snapshot.

## Identity and verification scope

Full Standard InChIKeys come from sanitized graph SMILES decoded from the stored representation. They preserve encoded stereo/isotope information; neither first-block-only matching nor custom salt stripping is used. Protecting molecules in a scaffold test does not guarantee corpus-wide scaffold disjointness. Historical decorative brackets can have lost original chemical distinctions; a corrected parser cannot recover them without source graphs.

InChIKey matching assigns/excludes molecular identities; it does **not** collapse all conformations of a molecule. Exact enriched-text deduplication separately removes identical samples: 1,734,217 train, 7,102 validation, and 7,557 test rows. The completed post-audit finds zero within-split exact repeats and zero exact cross-split matches. It does not certify geometric near-duplicate absence. The separate GEOM-Revisited benchmark retains 2,037 exact matches to the GEOM test set and must not be pooled as independent observations.

The independent molecular audit freshly reparses materialized holdouts; retained training checks use the indexed keys/assignments and verified file hashes. This is not a fresh second RDKit pass over every final training row. The recorded production parser environment is RDKit 2025.09.6; the October 17 synthetic regression rerun used 2025.09.3 and does not replace that production provenance.

## Source provenance and checks

[Source manifest](evidence/release_20261006/sources.json), [category manifest](evidence/release_20261006/category_manifest.json), and [duplicate source statistics](evidence/release_20261006/duplicate_source_statistics.json) provide source/category accounting. Stage evidence and prepared-snapshot hashes preserve input lineage. No new full-scale processing was run merely to publish this package.

```bash
for stage in group_a_split group_b_split group_c_split merge_release exact_duplicate_audit exact_deduplication; do
  python -m unittest discover -s "data_collection/$stage" -p 'test_*.py' -v || exit 1
done
```
