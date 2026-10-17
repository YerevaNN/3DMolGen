# Final CoordToken data pipeline

The completed October 6 release contains **189,792,028 training**, **1,802,906 validation**, and **5,553,421 test** conformations. Start with [FINAL_PIPELINE.md](FINAL_PIPELINE.md), which identifies the prepared inputs, stage order, commands, and verification evidence. The final merged training file is not globally shuffled; training uses separately seeded row permutations.

| Stage | Code and instructions |
|---|---|
| Prepared inputs, repairs and quarantine | [Prepared snapshot](prepared_snapshot_20261005/README.md) |
| Preserve supplied Group A partitions; protect fixed molecular holdouts | [Group A](group_a_split/README.md) |
| Deterministically split Group B by full Standard InChIKey | [Group B](group_b_split/README.md) |
| Separate nabla regimes and construct validation | [Group C](group_c_split/README.md) |
| Merge and preserve source identities | [Merge](merge_release/README.md) |
| Audit exact enriched-text duplicates | [Exact audit](exact_duplicate_audit/README.md) |
| Remove repeated samples and verify the resulting release | [Deduplication](exact_deduplication/README.md) |

[Final release evidence](evidence/release_20261006/release_status.json) records the final file counts and hashes, completed post-deduplication audit, and the separate GEOM-Revisited overlap. Stage reports describe intermediate outputs; their counts are not the final deduplicated counts. [October 17 regression checks](evidence/integration_tests_20261017.json) passed on synthetic fixtures; they are not a second full-scale replay.

The [historical collection documentation](HISTORICAL_190M.md), `historical/`, and `run_pipeline.py` describe the earlier 190,214,248-row collection. That wrapper is not the entry point for the October InChIKey pipeline. Data/weights are external; this branch publishes code, small manifests, and evidence.
