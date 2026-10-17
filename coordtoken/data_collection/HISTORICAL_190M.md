# Historical 190M collection and earlier cleaning notes

This folder collects the source-conversion, split/filter, deduplication, and
aggregation code recovered from `/auto/home/filya/fsq`. The March 23 merge log
records **190,214,248 training rows**. The original output was
`/nfs/h100/raid/chem/3D_big_data_new_dedup_no_geom_revisited/merged_train.csv`.

The published `merged_train.csv.zst` is **already filtered**. It is the final
training artifact, not an input to the filtering pipeline.

## New InChIKey cleaning revision

For the October 2026 pass starting from corrected **grouped inputs**, see
[Group A processing](group_a_split/README.md). It preserves Group A partitions,
protects full InChIKeys from fixed holdouts, and keeps nabla's conformation-test
molecular overlap exception. Group A remains provisional until newly selected
Group B/C holdouts are applied globally.
The next stage is [iterative Group B splitting](group_b_split/README.md), with
whole-InChIKey assignment and automatic index-to-split job dependencies.
The [final nabla step](group_c_split/README.md) fills validation only with keys
absent from retained Group A/B training and verifies the combined molecular split.

For the full Standard InChIKey audits, non-destructive filter, tests, completed
nabla correction evidence, and planned global release, see
[INCHIKEY_CLEANING.md](INCHIKEY_CLEANING.md).
[Manuscript changes and methods wording](FILTERING_METHODS.md) distinguish
completed filtering from pending release checks. These tools revise the existing
merged corpus; they do not reconstruct unavailable historical source snapshots.

## Main scripts for review

| Task | Recovered implementation |
| --- | --- |
| Initial train/test filtering | [remove_test_val_smiles.py](historical/remove_test_val_smiles.py) |
| Filtering with ChEMBL included | [add_chembl_to_group_a.py](historical/add_chembl_to_group_a.py) |
| KIBA, OMol-small, PubChem, ZINC splitting | [split_group_b_merged.py](historical/split_group_b_merged.py) |
| nablaDFT splitting | [split_group_c.py](historical/split_group_c.py) |
| Exact-sample deduplication | [dedupe_samples_and_report_intersections.py](historical/dedupe_samples_and_report_intersections.py) |
| GEOM-revisited test exclusion | [remove_geom_revisited_test_smiles.py](historical/remove_geom_revisited_test_smiles.py) |
| Final training aggregation | [merge_train_only_streaming.py](historical/merge_train_only_streaming.py) |

## What can be reproduced

- `historical/` contains the recovered scripts and their local helper modules.
  The algorithms and original paths are retained. These are the files present
  on disk when collected, not an independently recovered version of every file
  at the time of its historical job.
- `run_pipeline.py` runs the existing final-stage scripts from explicit input
  snapshots into a new directory. It records commands, code fingerprints, input
  metadata, logs, environment information, and the final training count. Optional
  SHA-256 hashing fingerprints the input files and merged output.
- The original H100 split snapshots are not mounted on the current machine.
  The H100 hostname resolves and SSH responds, but the current login cannot
  authenticate. **This does not mean the data was deleted.**
- A full reconstruction of the historical train/val/test membership is not
  established. Group B/C shuffle Python sets; their `--seed 42` does not freeze
  the iteration order. The historical hash seed and complete input fingerprints
  were not recorded. Raw PubChem/ZINC conversion also uses shared worker quotas,
  and concatenation starts from filesystem glob order.
- The GEOM-revisited reference was rebuilt from the available pickle and matched
  the historical **23,404 rows / 1,041 base SMILES**. Its original byte identity
  cannot be checked without the original reference CSV.

These limitations are part of the provenance, not a claim of 100% reproducibility.
The precise data inventory and fingerprints are in
[`evidence/available_inputs.json`](evidence/available_inputs.json).

## Layout

| Location | Purpose |
| --- | --- |
| `historical/data-collection/` | Source CSV conversion, shard preparation, concatenation, and earlier re-splitting |
| `historical/*.py` | Filtering, grouping, final splitting, duplicate removal, merging, and checks |
| `historical/train/utils.py` | Encoder used by the GEOM-revisited reference builder |
| `historical/data-collection/utils.py` | Encoder used by the original source converters |
| `historical/old_data_utils/` | Older embedded-SMILES aggregation; not the final 190m merge |
| `historical/train_on_nabladft/tokenize_utils.py` | Dependency of that older aggregator |
| `historical/*.sh`, `historical/data-collection/*.sbatch` | Recorded cluster launch settings, including original absolute paths |
| `evidence/` | Original job outputs, summary tables, source fingerprints, and discovery records |

The two encoder helpers differ and must not be replaced with the repository-root
`utils.py`. Their directory layout is deliberately preserved. Historical launchers
still point at the original `/auto/home/filya/fsq` checkout; treat them as execution
records. Use the runner below for final-stage replay from this checkout.

## Replay saved CSV snapshots

Use Python 3.10.14 and the recorded dependencies as a starting environment.
Dependency versions were observed on September 16, 2026; they are not a recovered
lockfile from the March jobs. The CSV replay needs only `tqdm` beyond the standard
library; RDKit/PyTorch are needed for upstream source conversion.

```bash
python -m pip install -r data_collection/requirements.txt
cp data_collection/config.example.json data_collection/config.local.json
```

Edit `config.local.json` to point at the actual input snapshot, reference CSV,
and a **new** output directory. Relative paths resolve against the config file,
not the shell's working directory. Original source paths are shown in the example;
the output defaults to a separate `data_collection/runs/190m-replay` directory.

| `start_at` | `source_root` must contain |
| --- | --- |
| `deduplicate` | Original `3D_big_data_new/{train,val,test}/*.csv`, after Group A/B/C splitting |
| `filter` | Original `3D_big_data_new_dedup/{train,val,test}/*.csv` |
| `merge` | Already filtered `3D_big_data_new_dedup_no_geom_revisited/train/*.csv` |

For `deduplicate`/`filter`, `reference_csv` must be the GEOM-revisited held-out
reference. `merge` does not use it. Never use the published merged CSV as a
source split directory. The runner does not append Group D to the training set.

```bash
python data_collection/run_pipeline.py --config data_collection/config.local.json --plan
python data_collection/run_pipeline.py --config data_collection/config.local.json
python data_collection/run_pipeline.py --config data_collection/config.local.json --execute --hash-inputs
```

`--plan` prints the commands even without mounted data. Without `--execute`, the
runner checks the input paths and CSV headers without writing files. Execution
refuses an existing output directory or overlapping source/output roots. Failed
runs retain their logs and a failed manifest; use a new output directory for a
retry. Inputs must remain immutable for the entire run. Metadata checks are not
a substitute for immutable snapshots; `--hash-inputs` additionally reads every
input and output byte and can add significant I/O at this scale.

The expected merge count is 190,214,248. A count match alone does not establish
identical molecules or byte-for-byte identity. For another dataset, set
`expected_train_rows` explicitly, or use `null` to disable that comparison.
The original jobs requested up to 512 GB RAM for exact deduplication; streaming
the final merge does not make the entire pipeline memory bounded. No full-scale
replay was performed when assembling this folder.

## Recorded pipeline

1. Convert the dataset sources into `name,enriched_text` CSVs using
   `historical/data-collection/csv_*.py`. Flowr conversions use the saved split
   indices; GEOM reads processed JSONL strings; PubChem/ZINC use staged pickle
   shards; ChEMBL uses its processed data; nablaDFT uses summary/XYZ inputs.
   The corresponding `.sbatch` files preserve the observed arguments and paths.
2. `remove_test_val_smiles.py` builds the initial filtered data from TokenizerData.
   It excludes KIBA-3D, OMol-small, ChEMBL, ZINC, and PubChem at that stage.
   Base SMILES are obtained by stripping `<...>` coordinate tags, not by another
   RDKit canonicalization. Training loses validation/test SMILES; validation
   loses test SMILES; test rows are retained.
3. `add_chembl_to_group_a.py` combines the existing filtered/grouped base with
   freshly converted ChEMBL and applies the global test/validation exclusions.
   `merge_nabladft_3d.py` prepares Group C from the saved train/validation backup.
   Earlier regrouping, isotope-removal, and reset utilities are retained as
   history; there is no complete logged raw-to-group orchestration to replay.
4. `split_group_b_merged.py` processes KIBA-3D, OMol25_small_mols, PubChem, then
   ZINC, respecting earlier split membership and aiming for 0.5% each for
   validation/test by row count while keeping a SMILES together. It drops
   contaminated SMILES. `split_group_c.py` targets 63,381 nablaDFT validation
   rows; the logged job produced 63,384 because whole SMILES groups are selected.
5. `dedupe_samples_and_report_intersections.py` compares exact `enriched_text`,
   not base SMILES. Its stats phase builds cross-file drop lists; its apply
   phase removes within-file and cross-file duplicates. Ownership is ordered
   by `test > val > train`, then dataset name; the first row in the owning file
   survives. The March jobs ran stats and apply separately; the replay runner
   invokes the same two phases consecutively via `--phase all`.
6. `build_group_d_geom_revisited_test.py` produces the held-out reference from
   `/auto/home/vover/revisited.pickle`, using precision 4. Then
   `remove_geom_revisited_test_smiles.py` removes its base SMILES from the other
   train/validation/test CSVs. The logged filtering run processed 56 files.
7. `merge_train_only_streaming.py` merges train files in sorted filename order,
   preserves row order within each file, prefixes names with the dataset, and
   skips empty `enriched_text`. The March 23 job wrote 190,214,248 rows.

`redistribute_data.py` and `add_conformers_back.py` represent an earlier path;
they are not additional steps to apply to the final filtered artifact.
`merge_global_splits.py` is an earlier merge utility; the final 190m job used
the train-only streaming script.

## Evidence caveats

- `evidence/logs/merge_train_new_469030.out` is the source for the final training
  count and its per-dataset components.
- The saved `3d_big_data_new_dedup_no_geom_revisited_split_counts.csv` agrees
  with the merge log for training, but its test counts retain pre-GEOM-filter
  values. For example, it says 711,228 GEOM test rows, while the filtering log
  reports removing 2,193 from that count. It must not be used as a verified
  final validation/test manifest.
- The overlap report that ignores nablaDFT records **624 ignored train/validation
  SMILES overlaps**. A reported zero after that exclusion is not proof of strict
  global train/validation separation. Run `check_global_overlap.py` without
  `--ignore-val-dataset-vs-other-train` to report all overlaps.
- Source modification times can postdate jobs (for example Group C). A recovered
  script is not proof that every byte matches its historical executed version.
- `evidence/source_manifest.json` records both original and packaged SHA-256
  values. Line endings, trailing whitespace, and final newlines were normalized.
  Every Python copy was checked against its source for identical syntax trees;
  no algorithm changes were made in the historical copies.

## Released training artifact

The existing repository README links the already-filtered dataset
[`FilyaGeikyan/CoordToken-data`](https://huggingface.co/datasets/FilyaGeikyan/CoordToken-data).
Metadata inspected on September 16, 2026 identifies:

- Revision: `0aa03ea21894d2c7bef1b84d474d051b5ac593f4`
- File: `merged_train.csv.zst` (46,956,911,350 bytes)
- Compressed-file SHA-256: `30ad18ddd9e58f4ac18fd3706f1d35fe600a806aabc5830a1d7c611491415bb7`

```bash
hf download FilyaGeikyan/CoordToken-data merged_train.csv.zst --repo-type dataset --revision 0aa03ea21894d2c7bef1b84d474d051b5ac593f4 --local-dir /path/to/release
sha256sum /path/to/release/merged_train.csv.zst
```

The checksum comes from Hugging Face's file metadata; the 47 GB archive was not
downloaded or decompressed during this audit. Access to the released artifact
does not recover its missing source splits or establish a raw-data rebuild.
