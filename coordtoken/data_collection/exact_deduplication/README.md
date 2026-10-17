# Exact deduplication of merged splits

This step consumes the verified merged release and its completed exact-sample
audit. Within each merged train/validation/test file, it retains the first
occurrence of each exact `enriched_text` string in existing file order. Distinct
conformations remain distinct. It writes a new release and leaves inputs intact.
GEOM-Revisited remains an independent, unchanged evaluation benchmark; shared
samples between it and the GEOM test set remain in both benchmarks.

Every deletion is rechecked against its retained owner using the complete text
and full InChIKey. The output row bytes and stable source IDs remain unchanged.
`duplicate_aliases.csv` preserves every removed row's original metadata and its
retained record ID, including cross-dataset duplicates. `sources.json` preserves
the prepared-input paths and hashes. Dataset ownership follows original merged
file order, rather than a claim that one source is scientifically preferable.

The output `manifest.json` records row counts and checksums; `category_manifest.json`
records per-dataset/split counts. The latter describes categories within merged
files, not separate physical per-dataset CSVs. Parallel filtering and assembly
are followed by parallel region readback and full-file SHA-256 calculation.
A fresh exact-duplicate audit must pass before this release is promoted.

```bash
python -m unittest discover -s data_collection/exact_deduplication -p 'test_*.py' -v
sbatch --output=/path/to/dedup-%j.log data_collection/exact_deduplication/run.sbatch \
  /absolute/path/to/data_collection/exact_deduplication \
  /absolute/path/to/merged_release /absolute/path/to/completed_exact_audit \
  /absolute/path/to/new_deduplicated_release
```

Tests cover cross-source ownership and aliases, preserving distinct conformers,
refusing incorrect deletion matches, source integrity, and unchanged chunks.

## Completed release: 2026-10-06

Status: ready_deduplicated_release

| Split | Retained rows | Removed exact repeats |
| --- | ---: | ---: |
| train | 189,792,028 | 1,734,217 |
| val | 1,802,906 | 7,102 |
| test | 5,553,421 | 7,557 |

All three merged splits have zero exact duplicate strings internally and zero exact cross-split overlap. GEOM-Revisited remains a separate benchmark; 2,037 of its samples also occur in the GEOM test set.

| Split | Removed row matched same source file | Different file, same dataset | Different dataset |
| --- | ---: | ---: | ---: |
| train | 1,286,145 | 0 | 448,072 |
| val | 7,098 | 0 | 4 |
| test | 7,554 | 0 | 3 |

Each removed row is classified relative to the retained first occurrence. These counts include every removed row, not just distinct shared molecular strings.

| Split | Removed dataset | Retained dataset | Removed rows |
| --- | --- | --- | ---: |
| train | OMol25_small_mols | OMol25_bio_mols | 447,940 |
| train | OMol25_bio_mols | HiQBind | 62 |
| train | OMol25_small_mols | HiQBind | 57 |
| train | BindingNet-Low | BindingNet-High | 8 |
| val | OMol25_small_mols | OMol25_bio_mols | 4 |
| train | Plinder | BindingMoad | 3 |
| train | BindingNet-Mid | BindingNet-Low | 2 |
| test | OMol25_small_mols | OMol25_bio_mols | 2 |
| test | Plinder | BindingMoad | 1 |

`duplicate_aliases.csv` maps every removed record to its retained record while preserving its original source metadata. `sources.json` maps source IDs to prepared input files. These indices are not raw upstream SDF/pickle indices.

The materialization manifest records the initial audit-pending state.
`release_status.json` is the final readiness record after the independent audit.

Release: `/mnt/weka/fgeikyan/3dmolgen_bigdata/data_handoff/merged_deduplicated_20261006`

[Completion evidence](evidence/deduplicated_20261006.json) and
[source statistics](evidence/source_statistics_20261006.json).
