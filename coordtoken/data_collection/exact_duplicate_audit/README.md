# Exact-sample duplicate audit

This read-only audit compares the exact decoded `enriched_text` CSV field across
the verified merged train, validation and test files, plus the separately held
GEOM-Revisited test file. Names, molecular equivalence, coordinate rounding,
rotation and atom ordering do not define equality here. Distinct encoded
conformations remain distinct samples.

64 workers scan byte regions from the verified merge receipts, checking region
hashes and row counts. They record SHA-256 text digests, dataset/split category,
and source byte offset into 256 buckets. Up to 32 reducers sort these records
and reread every candidate occurrence to verify full-string equality. Hash
collisions cannot create a false duplicate report. Inputs are never modified.

`report.json` summarizes duplicate text groups, rows participating in them,
redundant occurrences globally and within each split/dataset, cross-split and
cross-dataset intersections, and examples. `buckets/*_groups.jsonl` records every
confirmed group with the original names and byte offsets of its occurrences.
Counts of redundant rows are statistical; the audit does not choose deletions.
Cross-split pair counts measure distinct shared text strings, not row pairs.

```bash
python -m unittest discover -s data_collection/exact_duplicate_audit -p 'test_*.py' -v
sbatch --output=/path/to/audit-%j.log data_collection/exact_duplicate_audit/run.sbatch \
  /absolute/path/to/data_collection/exact_duplicate_audit \
  /absolute/path/to/merged_release /absolute/path/to/new_audit_output
```

Tests cover repeated names with distinct texts, exact repetitions across
multiple datasets and splits, rejection of altered source regions, and forced
hash collisions requiring full-string confirmation.

## Completed audit: 2026-10-06

Job 314196 scanned all 198,920,635 rows and confirmed all candidate matches in
93.9 seconds. No source files were modified. No exact `enriched_text` string is
shared between train, validation, and test, including the conformation-test
exception and the separate GEOM-Revisited benchmark.

| Scope | Redundant occurrences beyond the first |
| --- | ---: |
| Merged training | 1,734,217 |
| Merged validation | 7,102 |
| Merged test | 7,557 |
| All evaluation tests, including GEOM-Revisited | 9,594 |

GEOM-Revisited contains no internal exact repeats, but 2,037 of its texts also
appear in GEOM's test split. This is overlap between two test benchmarks.
Across all inputs, 1,689,701 repeated text groups contain 3,440,614 rows;
1,750,913 occurrences are redundant globally. No hash collisions were found.

Training duplicates are about 0.905% of training rows. Removing training repeats
would leave 189,792,028 rows, but no deduplication has been applied. Most repeats
come from within ZINC training (1,133,909 redundant rows), and 427,062 distinct
texts are shared by the OMol25 bio/small-molecule training sets.

See [completion evidence](evidence/exact_duplicates_20261006.json). Detailed
examples and all occurrence ledgers are stored at
`/mnt/weka/fgeikyan/3dmolgen_bigdata/data_handoff/exact_duplicates_20261006/`.
