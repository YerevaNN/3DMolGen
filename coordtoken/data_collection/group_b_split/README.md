# Iterative Group B InChIKey splits

This continues the verified [Group A pass](../group_a_split/README.md) using the
prepared grouped CSVs. Order: KIBA-3D, OMol25_small_mols, pubchem3d, zinc.
It uses full Standard InChIKeys from the same pinned parser and RDKit version
as Group A. The three unresolved PubChem/ZINC rows remain excluded from the
prepared inputs. Newly encountered format/key failures are preserved in
quarantine evidence and never treated as novel molecules.

For each dataset, test and validation target approximately 0.5% of **input
sample rows**, selecting whole molecular-key groups. Existing global test keys
take priority over validation keys. Matching existing holdouts are selected
first. If their candidate rows exceed the quota, whole groups are subsampled;
the unselected groups are excluded, never reassigned to training. Deficits are
filled only with keys absent from existing train/validation/test ownership.

The initial registry protects fixed Group A tests, nabla structure/scaffold
tests, and GEOM-revisited, plus Group A validation and the reserved quarantined
ChEMBL validation key. Existing Group A and nabla training keys block new holdout
filling. Earlier Group B assignments join the registry after each dataset.
Later Group B pools are still unassigned; they inherit these decisions when
processed. This preserves the iterative method rather than pretending later
unassigned pools are already training sets.

Selection uses SHA-256 ranking of `(seed, dataset, split, full key)`, with seed
42, for reproducible pseudorandom order independent of set iteration or worker
completion. Whole groups can overshoot the target. If insufficient eligible
groups exist, the shortfall is reported rather than violating ownership.
Nabla's conformation test is not a molecular blacklist. There is no scaffold-wide
or stereo-free matching, salt stripping, or conformer collapsing.

## Execution

Completed on 2026-10-06: index job `313668` (1h06m20s), split job `313670`
(2h08m20s), both exit code zero. All 183,260,646 input/reference rows produced
keys without new failures. Final Group B training/validation assignments had
zero prohibited overlaps with the final ownership registry.

| Dataset | Training | Validation | Test | Excluded |
| --- | ---: | ---: | ---: | ---: |
| KIBA-3D | 260,129 | 1,533 | 1,576 | 17,770 |
| OMol25_small_mols | 4,033,245 | 21,597 | 37,384 | 226,716 |
| pubchem3d | 93,802,743 | 473,756 | 473,758 | 0 |
| zinc | 74,401,034 | 375,763 | 375,763 | 0 |

The 244,486 exclusions are existing-holdout candidates outside the sampled
quotas; none were returned to training. OMol's test target was 21,595 rows, but
the largest selected whole molecular group has 24,255 rows, producing a final
37,384-row test set. This overshoot preserves all conformers of a key together.
[Completion evidence](evidence/group_b_20261006.json) includes per-dataset
selection counts, output hashes, and remaining Group C/global verification work.

The first stage indexes the four Group B pools and the existing nabla training
pool in parallel. The second stage uses SQLite to aggregate key counts and
ownership, processes datasets sequentially, and writes CSV chunks in parallel.
The prepared October 2026 input contains 174,502,767 Group B rows and 8,757,879
nabla training-reference rows.

```bash
python data_collection/group_b_split/prepare_index.py \
  --preflight /path/to/resplit_20261005 \
  --group-a /path/to/completed-group-a-run \
  --output /path/to/new-group-b-experiment/manifest.json

python data_collection/group_b_split/index_inputs.py \
  --manifest /path/to/new-group-b-experiment/manifest.json \
  --output /path/to/new-group-b-experiment/index_run --workers 64

split_code_sha=$(sha256sum data_collection/group_b_split/split_pools.py | cut -d' ' -f1)
python data_collection/group_b_split/split_pools.py \
  --index-run /path/to/new-group-b-experiment/index_run \
  --output /path/to/new-group-b-experiment/split_run \
  --runner-sha256 "$split_code_sha" --workers 32
```

Only the standard library and RDKit are required. The preflight must contain
the same artifacts documented for Group A. Its recorded RDKit version must
match the environment. Both stage output directories must be new. Failed
outputs remain as evidence; use a new directory for retries.

`index.sbatch CODE_DIRECTORY MANIFEST NEW_INDEX_DIRECTORY` requests 64 CPUs and
256 GB. `split.sbatch CODE_DIRECTORY INDEX_DIRECTORY NEW_SPLIT_DIRECTORY SHA256`
requests 32 CPUs and 128 GB. Adjust site/account settings as needed. Submit the
second job using `--dependency=afterok:INDEX_JOB_ID`, so it cannot consume an
incomplete index. Both scripts accept `--output` through `sbatch` for persistent
logs. Writable controller databases use up to 8 GiB of page cache each; worker
read caches are limited to 64 MiB each.

## Validation and artifacts

The index verifies every original file hash, code hashes, RDKit, row counts,
and stable input metadata. Chunk receipts pin reports and key-count files.
Writers verify receipt hashes, row-aligned key hashes, and source-record hashes,
then apply one immutable assignment per key. Exact original quarantine
membership uses source path, name, and row hash.

`split_run/candidate/{train,val,test}/*.csv` holds the 12 outputs. Dataset
allocation reports record targets, selected matching/new holdout rows, and
shortfalls. Dataset completion reports record exclusions, quarantine counts,
and output hashes. Final assembly checks fresh output hashes and row counts.
The final report checks every Group B training/validation assignment against
the registry after **all four** datasets, guarding against later holdout leaks.
`parts/` preserves excluded rows and reasons. `index_run/index/` preserves raw
failure evidence and reusable row-key sidecars. `progress.json` in each stage
reports its state. Per-dataset SQLite files preserve complete key assignments;
`ownership.sqlite` carries global ownership forward to Group C.

Completion is not the final corpus release. Group C validation must still be
constructed and its identities applied globally. Final corpus checks and
exact-sample deduplication remain separate from molecule-level splitting.

```bash
python -m unittest discover -s data_collection/group_b_split -p 'test_*.py' -v
```

Tests exercise complete indexing/allocation/writing across two datasets,
test-over-validation precedence, existing-training protection (including nabla),
whole-conformer assignment, quota exclusions, deterministic selection under
different insertion orders, target shortfalls, invalid-row quarantine, byte
chunk boundaries, output accounting, and source immutability.

## Independent check of materialized holdouts

`verify_holdout_precedence.py` reparses every actual Group A/B validation/test
CSV plus the fixed nabla structure/scaffold and GEOM-revisited tests through
RDKit's direct `MolToInchiKey` API. It builds fresh test/validation sets without
using `ownership.sqlite`, checks every validation set against the test union,
and checks retained training assignments against that union. It also rereads
the full hashes of all 54 written A/B CSVs and the three fixed reference files.
Training uses the saved row-key/assignment evidence rather than a fresh
conversion of all 183 million indexed rows; the report states this boundary.

```bash
python data_collection/group_b_split/verify_holdout_precedence.py \
  --group-a /path/to/completed-group-a-run \
  --group-b /path/to/completed-group-b-split-run \
  --output /path/to/new-independent-verification --workers 48
```

The verifier's regression includes a deliberately contaminated materialized
validation CSV, even while the saved assignment database remains clean, to
confirm that the fresh holdout scan detects the conflict.

The completed independent check (job `314132`, 2026-10-06, 4m39s) reparsed all
5,707,138 holdout rows in 39 files. The written test and validation unions share
zero keys; indexed retained Group A/B training has zero matches to their union.
All 57 complete CSV hashes matched. See the
[verification report](evidence/independent_holdout_verification_20261006.json).

PubChem's zero excluded rows does not mean zero overlap with existing holdouts.
Its 223,138 existing-test matches and 147,575 validation-only matches fit below
the quotas and were assigned to their respective splits. Among the 2,612 keys
shared by the original test/validation reference lists, PubChem contains 270
keys representing 2,461 rows. Every one was assigned to test; none entered
validation or training. Test priority is applied before validation matching.
