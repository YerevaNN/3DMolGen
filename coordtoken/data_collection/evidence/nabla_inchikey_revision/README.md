# Completed nabla InChIKey correction evidence

Reports copied from the completed 2026-09-25 run. Input: 190,214,248 rows.
Selected references: nabla validation, scaffold test, and structure test.
All training sources were eligible, including nabla.

- `report.json`: 288,741 overlap rows excluded, 13 conversion failures quarantined,
  189,925,494 retained; per-source totals and output SHA-256.
- `independent_verification.json`: a separate full-file read confirms the hash,
  byte size and record counts. This is not an independent chemical-identity audit.
- `manifest.json`: exact filter code fingerprint, inputs and selected policy.
- `audit_manifest.json`, `audit_reference_summary.json`, `audit_summary.json`:
  provenance and aggregate results from the precursor audit of all four nabla
  references. That audit also includes the conformation test; the filter selected
  only the other three. Its all-reference totals are NOT the removal count.

Run-local paths and versions are retained as provenance, not portable defaults.
The filter and audit Python files in this commit match the completed-run code
fingerprints. The Slurm launchers were made configurable for other checkouts.
CSV data, molecule ledgers and SQLite databases are not part of this commit.
This correction is intermediate: global holdout, scaffold, geometry and structural
quality gates remain pending. See ../../FILTERING_METHODS.md.
