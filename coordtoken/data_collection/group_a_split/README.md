# Group A processing from prepared grouped inputs

This pass preserves Group A's existing train/validation/test assignments and
removes molecular overlap using full Standard InChIKeys. It operates on the
corrected surviving grouped CSVs, not the previously merged 190M release or the
experimental source-graph rebuild. Some inputs already contain historical
filtering; this does not reconstruct untouched original datasets.

Test identities take precedence over validation and training; validation
identities take precedence over training. The test registry includes all fixed
Group A tests, nabla structure and scaffold tests, and GEOM-revisited. Nabla's
conformation test is deliberately not a molecular blacklist. Scaffold-wide
matching, stereo-free matching, salt removal, and conformer collapsing are not
applied. Distinct samples with the same key remain together unless excluded by
a higher-priority partition. Fixed tests remain separate; shared test identities
are reported.

Six unresolved ChEMBL records remain quarantined in the prepared input snapshot
(five training, one validation). Quarantine membership uses the source path,
record name, and original row SHA-256. ChEMBL names are local to each source:
`chembl3d_110806` in training is not the quarantined validation record with the
same name. The quarantined validation key remains reserved against training.
The three unresolved PubChem/ZINC records belong to Group B and are outside this
pass.

## Completed run: 2026-10-06

Slurm job `313611` completed successfully in 8 minutes 52 seconds using code
commit `416a3f9`. All 42 CSV outputs passed the runner's checks and matched an
independent recomputation from the first attempt's complete identity index.
No new Group A parse/key failures were found. The six preexisting ChEMBL
quarantine records remain excluded.

| Partition | Input rows | Retained rows | Removed rows |
| --- | ---: | ---: | ---: |
| Training | 11,094,343 | 10,339,711 | 754,632 |
| Validation | 901,495 | 874,019 | 27,476 |
| Test | 765,424 | 765,424 | 0 |

[Completion evidence](evidence/group_a_20261006.json) records per-file input and
output hashes, counts, checks, code revision, and remaining global-split work.
The previous job `313567` failed because its quarantine guard assumed ChEMBL
names were unique across splits; the source-path/name/row-hash guard fixes that
false positive. Failed partial outputs were not promoted or overwritten.

## Run

Use the same RDKit version as the verified preflight (the October 2026 snapshot
uses RDKit 2025.09.6). Only the Python standard library and RDKit are required.

`--preflight` must point to the prepared snapshot's evidence directory containing
`prepared_input_manifest.json`, `split_policy.json`,
`quarantine_resolution/current_quarantine.json`,
`reserved_quarantine_reference_keys.json`, and the verified
`references/{manifest,final_report}.json`. Input CSV paths come from those files;
they are never inferred from a folder glob. Large source and generated files
are not committed to Git.

```bash
python data_collection/group_a_split/prepare.py \
  --preflight /path/to/resplit_20261005 \
  --output /path/to/new-experiment/manifest.json

python data_collection/group_a_split/process_group_a.py \
  --manifest /path/to/new-experiment/manifest.json \
  --output /path/to/new-experiment/run --workers 32
```

For Slurm, adjust the account/partition/resources as appropriate:

```bash
sbatch --output=/path/to/new-experiment/slurm-%j.log \
  data_collection/group_a_split/run.sbatch \
  "$PWD/data_collection/group_a_split" \
  /path/to/new-experiment/manifest.json /path/to/new-experiment/run
```

The output directory must be new. A failed run is retained for diagnosis;
create a new manifest/output directory when rerunning changed code.

## Verification and outputs

The runner pins code hashes and RDKit, verifies every input's complete SHA-256,
then extracts identities in parallel byte chunks. The included `audit_parser.py`
is the parser snapshot used by the completed prepared-input audit. It validates
the enriched syntax and coordinates, normalizes legacy organic-atom brackets,
and obtains keys through sanitized SMILES parsing. It does not recover chemical
information already lost in the original representation.

Reference parsing/key failures stop processing. Training failures are retained
in quarantine evidence and excluded. Each writer checks source-byte and key
sidecar alignment, original quarantine membership, and retained-key exclusion.
Final assembly rereads output hashes and verifies complete row accounting.

- `candidate/{train,val,test}/*.csv`: 42 processed Group A CSVs.
- `index/`: row-aligned keys, unique-key counts, and parse/key-failure evidence.
- `parts/`: kept/excluded row bytes and exclusion reasons with source offsets.
- `test_keys.json`, `validation_keys.json`: identity registries for later stages.
- `reference_summary.json`: unique-key counts and fixed-test intersections.
- `progress.json`, `report.json`: live status and completion checks/counts.

**Completion is provisional.** Newly selected Group B/C validation/test keys
must be applied to Group A before the global dataset is final. This pass does
not perform exact-sample deduplication or certify conformation-level exposure.

Run the end-to-end regression:

```bash
python -m unittest discover -s data_collection/group_a_split -p 'test_*.py' -v
```

It covers molecular ownership, the nabla conformation exception, stereo/isotope
distinctions, multiple conformers, byte-boundary alignment, identity failures,
input immutability, same-name records in different splits, and rejection of the
exact original quarantined record.
