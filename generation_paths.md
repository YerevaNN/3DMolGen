# Generation paths for the section 2 FSQ rows

Each directory holds `generation_results.pickle` — a `{canonical_smiles: [RDKit Mol, ...]}`
dict with `2k` conformers per molecule over the 1,000-molecule `distinct_smi` test set —
plus `generation_results.txt` (the stats line) and `logs.txt`. Each pickle is ~150 MB.

Rows were matched to `outputs/geom_eval.csv` by their exact
`smiles_mismatch`/`mol_parse_fail` signature rather than by directory name.

| Setting | Size | COV-R | COV-P | MAT-R | MAT-P | mismatch / parse-fail | Generation directory |
|---|---|---:|---:|---:|---:|---|---|
| big data, 0 sft | 0.6B | 58.3 | 40.4 | 0.73 | 0.93 | 5527 / 26 | `/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/gen_results_user/20260613_071804_qwen06b_bigdata_step70534_fsq_distinct` |
| sft only | 0.6B | 67.3 | 62.4 | 0.64 | 0.71 | 920 / 2 | `/mnt/weka/fgeikyan/3DMolGen/outputs/gen_results_user/20260614_033610_qwen06b_step28320_6e_fsq_distinct_full` |
| big data + sft | 0.6B | 69.1 | 67.0 | 0.61 | 0.65 | 630 / 7 | `/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/gen_results_user/20260614_024445_qwen06b_sft_6e_step28320_fsq_distinct` |
| big data, 0 sft | 1.7B | 60.0 | 43.2 | 0.71 | 0.91 | 5379 / 4 | `/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/gen_results_user/20260619_032611_qwen17b_bigdata_step47023_fsq_distinct` |
| sft only | 1.7B | 68.1 | 64.9 | 0.62 | 0.67 | 85 / 1 | `/mnt/weka/fgeikyan/3DMolGen/outputs/gen_results_user/20260612_090931_qwen17b_step28320_fsq_distinct_full_ropefix_bs256` |
| big data + sft | 1.7B | 69.3 | 68.4 | 0.60 | 0.63 | 203 / 4 | `/mnt/weka/fgeikyan/3dmolgen_bigdata/outputs/gen_results_user/20260622_092610_qwen17b_sft_6e_step28320_fsq_distinct` |

## Notes

- The two **sft only** rows live in the other repo, `/mnt/weka/fgeikyan/3DMolGen/`, not in
  `3dmolgen_bigdata`.
- The 1.7B sft-only directory is named `..._ropefix_bs256`. Siblings from the same day
  (`_check3`, `_full_ropefix`, plain `_full`, and the `20260611_*` runs) are earlier attempts
  at the same step-28320 checkpoint; only the `bs256` run reproduces the 68.1 / 64.9 in the
  table, so treat the others as superseded.
- Intermediate SFT epochs (1e–5e) exist for both sizes in the same two directories, if the
  per-epoch curves are ever wanted as generations rather than as `geom_eval.csv` metrics.
- The binned rows of the section 2 table have no generation directory here; those numbers
  came from outside this repo.
- Companion machine-readable version: `generation_paths.csv`.
