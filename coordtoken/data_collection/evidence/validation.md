# Validation performed

The migration was checked on September 16, 2026 using Python 3.10.14 in the
observed `titan` environment. No full 190m-data run was performed.

- Parsed every packaged Python file and checked all recovered shell/Slurm
  launchers with `bash -n`.
- Compared all 84 imported source/evidence files with the originals. Differences
  are limited to text whitespace. Every recovered Python file has an identical
  abstract syntax tree to its source; both file hashes are recorded separately.
- Ran the complete deduplicate → GEOM filter → merge path on temporary CSVs
  containing within-file duplicates, duplicates across datasets/splits,
  different conformers of the same SMILES, and a held-out reference SMILES.
  Verified test ownership, removal of held-out SMILES across splits, preservation
  of distinct conformers, dataset-prefixed names, sorted merging, and exactly
  three expected training rows.
- Repeated the fixture replay and entered separately at the filter and merge
  stages. All four paths produced identical merged-file bytes.
- Exercised execution from another working directory, spaces in file paths,
  config-relative paths, preflight without writes, SHA-256 input/output records,
  refusal to overwrite outputs, overlap protection, missing inputs, and an
  incorrect expected row count. Confirmed that the source fixtures were unchanged
  and that a count mismatch produced a failed run manifest.
- Rebuilt the actual GEOM-revisited reference from the available pickle:
  23,404 enriched-text rows and 1,041 base SMILES, matching the historical log.
  The resulting checksum and input-pickle checksum are in `available_inputs.json`.
- Checked the staged diff for whitespace errors and excluded datasets, checkpoints,
  bytecode, and credentials from the commit.

The synthetic replay checks establish small-fixture behavior, not the correctness
or byte identity of an unavailable historical 190m-row snapshot. The original
splitting algorithms retain their historical nondeterminism and documented
nablaDFT overlap exception; they were not silently replaced during this migration.
