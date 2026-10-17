# Merged CSVs with source row provenance

The merger preserves the verified Group A/B/C split assignments. It writes
`merged_train.csv`, `merged_val.csv`, and `merged_test.csv`, with columns:

| Column | Meaning |
| --- | --- |
| `name` | Unique stable record ID: `<source_file_id>:<source_row_index>` |
| `enriched_text` | Unchanged decoded CSV field from the input |
| `dataset` | Dataset name; all nabla subsets use `nablaDFT` |
| `source_name` | Exact original input `name` field |
| `source_file_id` | Source identifier recorded in `sources.json` |
| `source_row_index` | Zero-based CSV data-row index before this split/filter run, excluding header |
| `inchikey` | Full Standard InChIKey used by the split pipeline |
| `test_subset` | Nabla `conformations`, `scaffolds`, or `structures`; empty otherwise |

`sources.json` records each prepared input path, checksum, row count, original
split/pool, dataset, and source ID. IDs are derived from the source path and
checksum. The row indices refer to prepared input CSVs, not original raw
SDF/pickle/database record indices. Output order is source-manifest order then
source row order; no random shuffle or exact-sample deduplication occurs.

Workers align the original records with every retained and excluded split
chunk, verifying complete source/chunk hashes, split-part hashes, counts, and
row-aligned key hashes. This preserves provenance even when names or complete
records repeat. The conformation test is preserved; its previously unneeded
keys are computed using the same frozen parser and RDKit version.

Workers write separate temporary chunks. Assembly assigns disjoint byte ranges
in the three final files and copies chunks in parallel with `pwrite`. A complete
readback checks each region's checksum and row count, and computes whole-file
SHA-256 checksums. `manifest.json` is written only after all checks pass.

```bash
python -m unittest discover -s data_collection/merge_release -p 'test_*.py' -v
sbatch --output=/path/to/merge-%j.log data_collection/merge_release/run.sbatch \
  /absolute/path/to/data_collection/merge_release \
  /absolute/path/to/data_handoff /absolute/path/to/new_merged_release
```

The launcher requests 64 CPUs and 256 GB of memory. The number of assembly
workers is capped at 32 to limit concurrent filesystem writes. The completed
molecular overlap checks are inherited through the pinned input manifest;
merging does not introduce new split assignments.

## Completed release: 2026-10-06

Job 314163 completed the merge and full readback in 720.6 seconds using 64
workers. All source alignment, key sidecar, split-part, row-count, and merged
readback checks passed. Five provenance/integrity regression tests passed before
submission. The first attempt (314162) stopped during preflight on a CRLF header;
the corrected header check preserves byte-level source verification.

Release directory:
`/mnt/weka/fgeikyan/3dmolgen_bigdata/data_handoff/merged_20261006/`

| File | Rows | Bytes |
| --- | ---: | ---: |
| `merged_train.csv` | 191,526,245 | 144,470,480,309 |
| `merged_val.csv` | 1,810,008 | 1,419,365,658 |
| `merged_test.csv` | 5,560,978 | 4,014,520,155 |

The directory also contains `sources.json`, `manifest.json`, and `SHA256SUMS`.
See [completion evidence](evidence/merged_20261006.json) for hashes and checks,
and [dataset counts and methods](../paper_dataset_split_revision.md).
GEOM-Revisited remains a separate 23,404-row evaluation test file referenced by
the prior combined split manifest; evaluation totals include it, whereas the
merged test CSV does not. No additional exact-sample deduplication was performed.
