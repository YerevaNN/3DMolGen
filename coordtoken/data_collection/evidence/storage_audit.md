# Storage audit — September 16, 2026

Read-only discovery was performed from host `np`.

| Location | Observation |
| --- | --- |
| `/auto/home/filya/fsq` | Recovered collection/filtering source, launchers, summaries, and March job logs |
| `/auto/home/filya/fsq_remote/CoordToken` | Local checkout of `YerevaNN/CoordToken`; unrelated staged checkpoint preserved |
| `/nfs/h100/raid/chem/3D_big_data*` | Not mounted/available here, including outside the sandbox |
| `h100.yc2.io` | Resolves to `172.26.32.55`; SSH reached the host but authentication for `filya` failed |
| `/nfs/dgx/raid/chem/TokenizerData` | 82 source/backup files found; metadata recorded in `available_inputs.json` |
| `TokenizerData/backup_excluded_splits_v2` | Saved train/val/test files for ChEMBL, PubChem, OMol-small, and KIBA |
| `TokenizerData/train/OMol25_bio_mols.csv` | Only 20 bytes (header-sized), insufficient to replace the missing grouped snapshot |
| `TokenizerData/train/pubchem3d.csv` | Missing in the current train directory; PubChem CSVs still exist at the root and in backups |
| `/auto/home/vover/revisited.pickle` | Available; hashed and used to rebuild the GEOM-revisited reference |
| `/nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_processed/geom_cartesian_v3/processed_strings` | Original GEOM processed-string directory exists |
| `/auto/home/filya/FlowrData/BindingMoad` | BindingMOAD source archives exist |
| `/mnt/big/3dtokenizer` | nablaDFT model checkpoints found, not the 190m split snapshots |
| `/mnt/weka/fgeikyan/fsq/shuffle_index_merged_train` | Referenced by training scripts, but no such local mount was found |
| Hugging Face `FilyaGeikyan/CoordToken-data` | Published final filtered archive found; immutable revision and compressed checksum recorded in README |

The mounted DGX chemistry directory and accessible AP chemistry roots did not
expose the named `3D_big_data*` snapshots in the directories searched. This is
not an exhaustive search of every host or inaccessible directory. The original
H100 disk could not be inspected because SSH authentication was unavailable.

To finish tracing the historical pipeline, access H100 (or a disk snapshot) and
locate `3D_big_data/grp_a`, `grp_b`, `grp_c`, `grp_d`, `3D_big_data_new`,
`3D_big_data_new_dedup_state`, `3D_big_data_new_dedup`, and
`3D_big_data_new_dedup_no_geom_revisited`. Preserve the exact per-file CSV bytes,
drop lists, SQLite duplicate state, reference CSV, and split membership before
rerunning any historical scripts. None of these source datasets were modified.
