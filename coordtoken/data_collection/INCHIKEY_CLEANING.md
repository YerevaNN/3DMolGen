# InChIKey auditing and data correction

These utilities add molecular-identity auditing and non-destructive filtering to
the historical collection package on branch `data-collection-190m`. Historical
scripts and their provenance remain unchanged. See [the collection README](README.md)
for the recovered pipeline and its reproducibility limits.

Start with [the paper/data review](PAPER_DATA_REVIEW.md). The supplied paper makes
broader data-separation claims than the existing corpus supports. The first
revision addresses nabla molecular holdouts; a final global data release requires
the remaining checks listed in that review.

## Matching policy

Full Standard InChIKey from decoded molecular text, retaining encoded stereo,
isotopes and charges. No coordinate-derived stereo assignment, custom tautomer
normalization or salt stripping. Standard InChI's own normalization applies.
CoordToken's decorative atom brackets are decoded consistently before RDKit use.
InChI conversion warnings and failures are distinct; failures are never silently
counted as valid molecular nonmatches.

Molecule, scaffold and conformer separation are different guarantees. Nabla test
conformations intentionally share molecules with training, so they are not a
molecule-exclusion reference. This does not authorize repeated held-out geometries.

## Tested filtering from a completed audit

`inchikey_filter.py` applies `scripts/audit_inchikey_overlap.py` results to a new
CSV. Reusing the completed audit avoids recomputing keys for all 190M rows while
still regenerating every selected removal identity before writing the output.

```bash
python data_collection/inchikey_filter.py \
  --audit logs/inchikey_overlap/nabla_all_val_test_190m \
  --reference-label val/nablaDFT \
  --reference-label test/nablaDFT_scaffolds \
  --reference-label test/nablaDFT_structures \
  --failure-policy quarantine \
  --output data_collection/runs/nabla_holdouts_v1 \
  --workers 16 --resume
```

The input is the already-filtered historical corpus; this creates a new revision,
not a reconstruction of unavailable unfiltered source data.

This specific command excludes matching rows from **every source, including
nabla training**. The generic optional `--preserve-source` flag is not used.

- Requires completed audit state, consistent database/summary counts, unchanged
  input metadata and the audited RDKit version.
- Reference labels are explicit; subset membership comes from audited bit masks.
- Recomputes keys for selected removal texts and checks reference membership.
- Processes resumable byte-range chunks, preserving retained rows byte for byte
  and in input order. One CSV record per physical line is required; malformed or
  multiline input is rejected rather than silently skipped.
- Reconciles every `(source, stripped text)` removal count against the audit and
  confirms the full input row count before publication.
- Verifies chunk checksums during ordered assembly and records the final SHA-256.
- Publishes without overwriting an existing dataset. Quarantined rows and reasons
  are saved separately. Resume verifies the same policy/code/input snapshot.
- Deletes only temporary data chunks from this run after successful publication;
  provenance, counts, chunk hashes, quarantine and removal ledgers remain.

Dependencies: Linux (fork workers and file locks), Python 3.10+, and RDKit with InChI support.
Install the pinned dependency with `python -m pip install -r data_collection/requirements-inchi.txt`. Actual validated runtime:
Python 3.13.12, RDKit 2025.09.6. Resume requires the audited environment; do not
regenerate partial results with a different identity implementation/version.

Tests:

```bash
python -m unittest discover -s data_collection/tests -v
python scripts/test_audit_inchikey_overlap.py -v
```

They cover stereo/isotope distinctions, tautomer and alternative-SMILES matches,
reference selection, exclusion of the conformation reference, quarantine,
byte-preserving output, input immutability, overwrite refusal, changed-input
refusal, and chunk integrity/resume.

## Broader audit and provenance

`build_holdout_registry.py` hashes all retained local validation/test files and
extracts the actual GEOM downstream validation and 1,000-molecule benchmark
references. Stored metadata keys that disagree with text-derived identities are
recorded rather than silently overwritten. The graph index contains only
train/validation entries; benchmark identities come from `molecules.csv`.

```bash
python data_collection/build_holdout_registry.py \
  --source-root /mnt/weka/fgeikyan/fsq \
  --geom-manifests /mnt/weka/fgeikyan/3dmolgen_bigdata/bindgpt_redesigned_geom_fsq/manifests \
  --output data_collection/runs/NEW_REGISTRY

python data_collection/audit_from_registry.py \
  --registry data_collection/runs/holdout_registry_v3/registry.json \
  --output data_collection/runs/all_holdouts_audit_v1 --workers 32
```

The current registry lists 43 references: 42 molecule-exclusion references and
one separately treated nabla conformation reference. PoseBench, Platinum Diverse
and exact GEOM-XL identities remain explicitly pending. The registry includes
GEOM-revisited as a reserved reference, not an evaluated Table 6 source.

## Current runs

- Slurm **289650**: completed `submit_nabla_revision.sh`; output `runs/nabla_holdouts_v1/merged_train.csv`, 189,925,494 rows. Excluded 288,741 overlaps and quarantined 13 failures.
- Slurm **289654**: completed independent full-file verification: output SHA-256, byte size, retained/quarantined row counts and count conservation all passed.
- Slurm **289653**: `submit_global_audit.sh`, output `runs/all_holdouts_audit_v1`.

Slurm jobs run independently of the chat. Inspect `progress.json` and job logs while
running; do not open a live SQLite audit database from another host. A completed
filter produces `merged_train.csv`, `report.json`, `quarantined.csv`, and
`conversion_failures.csv`. The audit produces summary and detailed match CSVs.

No checkpoint, packed training index, or model training configuration is changed
automatically. Rebuild those only after the final data policy and corpus manifest
are frozen. Do not call a nabla-only correction globally leakage-free.

## Published evidence and manuscript wording

Small reports for the completed nabla correction are in
[evidence/nabla_inchikey_revision](evidence/nabla_inchikey_revision/README.md).
The 128 GB dataset, SQLite databases, molecule ledgers and generated run files
are excluded from this code commit. The file verifier checks bytes and counts;
a fresh molecular audit of the final release remains a separate requirement.

[FILTERING_METHODS.md](FILTERING_METHODS.md) contains the completed-results
paragraph, section-by-section manuscript changes, and a final-release template
whose pending counts and guarantees must be filled only after validation.

## Slurm configuration

Launch from the repository root after activating the pinned Python environment.
The launchers use `SLURM_SUBMIT_DIR` as the checkout by default; override it with
`COORDTOKEN_ROOT`. `COORDTOKEN_PYTHON` selects a Python executable (default: `python`).
The nabla audit accepts `COORDTOKEN_SOURCE_ROOT`; output paths and registry/audit
paths can be overridden with `COORDTOKEN_OUTPUT`, `COORDTOKEN_REGISTRY`, and
`COORDTOKEN_AUDIT` where applicable. Check the partition and memory directives
for your cluster before submission. Example:

```bash
export COORDTOKEN_ROOT="$PWD"
export COORDTOKEN_PYTHON=/path/to/pinned/environment/bin/python
export COORDTOKEN_SOURCE_ROOT=/path/to/fsq
sbatch scripts/submit_inchikey_nabla_all.sh
# After audit completion:
sbatch data_collection/submit_nabla_revision.sh
# After filtering completion:
sbatch data_collection/submit_verify_nabla_revision.sh
```

Source names are recovered from merged row-name prefixes and sibling `val/` and
`test/` CSV filenames. Preserve that layout and dataset naming for these tools.
The filter fails on unknown prefixes; it does not silently drop records. Inputs
must remain immutable; size/mtime checks are not a replacement for a frozen
snapshot. References have SHA-256 fingerprints, and filtering records hashes
for every input chunk and the final output. Resume requires the identical code,
inputs, policy and environment recorded by that run.
