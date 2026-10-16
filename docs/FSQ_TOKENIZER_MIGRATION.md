# New CoordToken integration: file map

Inspection date: 2026-10-16. This is a proposed implementation plan; the code snapshot still uses the legacy all-token FSQ model. Paths below are relative to the repository root. The new model emits **one 12-bit code per atom plus one transmitted molecular register code**, with no codes for bonds, rings, branches, dots, or padding.

## Representation contract

Port the exact tested model and quantizer from the selected checkpoint's frozen CoordToken source. New checkpoints contain `model`, `config`, `vocab`, and `manifest`; they are not the old Lightning `state_dict`/`hyper_parameters` format. Require strict state loading, a supported configuration, the exact vocabulary, and an explicit checkpoint SHA. The current `strict=False` loader can accept an incompatible checkpoint without providing a valid model.

Keep old checkpoints/data accessible through an explicit legacy format. Add a distinct new format/version rather than changing the meaning of existing streams. A proposed atom-pair serialization is `[C]<11>([O]<33>)<REG><7>`: two atom codes and one register code, with punctuation preserved and uncoded. The marker spelling is a design proposal, not a feature already implemented. Use the same contract for preprocessing, LM tokenization, generation, and decoding.

On decode, scatter atom codes into graph-token positions; the register occupies the appended register slot after the padded graph width. Non-atom/padded geometry vectors must be **zero vectors**, not the quantized vector for code ID 0. ID 0 is valid molecular geometry. The decoder still receives graph identities for structural context. Preserve coordinate centering, atom order, attention masks, positional embeddings, FP32 quantization/output, and the supported bf16 inference path. An absent or duplicate register is an invalid new-format record, not a reason to drop the register or infer it.

## Core implementation and data path

| File | Required change |
|---|---|
| `src/molgen3D/data_processing/smiles_encoder_decoder_fsq.py` | Add a versioned new-model adapter/loader; atom masks; sparse atom-code serialization/parsing; separate register handling; strict ID/count validation; correct dense reconstruction for decode. Extend `FSQRepresentation` to expose graph tokens, atom positions, atom codes/latents and register code/latent explicitly. Retain legacy dispatch. Explicit paths must fail if unavailable rather than silently fall back to an old checkpoint. |
| **New** `src/molgen3D/data_processing/coordtoken_model.py` | Port the selected checkpoint's exact `Config`/`Tokenizer` with encoder/decoder registers and masked geometry bottleneck; use package-relative imports. Do not approximate the new model using the old `MolFSQModel`. |
| **New** `src/molgen3D/data_processing/coordtoken_quantizer.py` | Port the frozen quantizer or establish numerical equivalence against it. The existing `fsq.py` is a smaller implementation; 4096 IDs alone do not prove equivalent quantization. Preserve legacy `fsq.py` for old models. |
| `src/molgen3D/data_processing/preprocess_big_data_fsq.py` | Select the new adapter explicitly; emit format/checkpoint metadata; batch atom masks/register outputs correctly; distinguish graph-token length from serialized LM length. Add or call a streaming path for the final coordinate CSV, since the current grouped-molecule pipeline is not automatically a reader for that final CSV. |
| **New** `src/molgen3D/data_processing/preprocess_coordtoken_csv.py` | Recommended dedicated streaming CSV-to-versioned-JSONL encoder for the cleaned corpus; preserve row identity and split, bound GPU batches, fail/report unsupported records, and record counts/checksum provenance. Never re-split or silently drop rows here. |
| `src/molgen3D/data_processing/utils.py` | Expose format/checkpoint/version options through `get_encoder()` and its encoder configuration; avoid legacy defaults in a new-format run. |
| `src/molgen3D/data_processing/smiles_encoder_decoder.py` | Make graph extraction register-aware (prefer structured parsing), and correct/test the existing `decode_cartesian_v2` sanitize/stereo/explicit-H behavior. Keep the stored representation's known bracket ambiguity documented; changing the decoder cannot restore lost source information. |
| `scripts/run_fsq_bigdata_preprocessing.sbatch`, `scripts/submit_fsq_bigdata_preprocessing.sh` | Pass explicit new checkpoint/format/input/output settings; write to a separate encoded dataset directory. Add a CSV launcher if using the new streaming entry point. |
| `scripts/shuffle_jsonl_dataset.py`, `scripts/run_fsq_shuffle_dataset.sbatch` | Check preservation of every encoded record and new metadata; update destinations and input manifests for the new corpus. The shuffle algorithm itself need not change if it treats records opaquely. |

Existing legacy FSQ codes cannot be converted by deleting punctuation codes: atom and register codes must be recomputed from the retained coordinate-bearing records with the new checkpoint. Re-encode all used train/validation/test/ICL/SFT inputs under the same versioned contract. Reuse the already defined molecular splits.

## Autoregressive training and inference

| File | Required change or verification |
|---|---|
| **New** `src/molgen3D/training/tokenizers/get_Qwen3_coordtoken.py` | Reproducibly build a versioned LM tokenizer from the existing assets: keep `<0>` through `<4095>` as atomic tokens and add a dedicated register marker. Test marker/code round trips and IDs in context. The existing `get_Qwen3.py` builds general delimiters, not this new FSQ contract. |
| **New generated directory** `src/molgen3D/training/tokenizers/Qwen3_tokenizer_coordtoken_atom12_reg1/` | Commit tokenizer assets (`tokenizer.json`, `tokenizer_config.json`, `added_tokens.json`, `special_tokens_map.json`, and the necessary base vocabulary/merges/template) with format and checkpoint provenance. Preserve the existing legacy assets. |
| `src/molgen3D/config/paths.yaml` | Add separate tokenizer, encoded-data, and checkpoint aliases for atom-plus-register runs. Preserve the legacy aliases so old experiments remain runnable. |
| `src/molgen3D/config/pretrain/qwen3_06b_fsq_bigdata_from_base.toml`, `qwen3_17b_fsq_bigdata_from_base.toml`, `qwen3_06b_fsq_bigdata_sft.toml`, `qwen3_17b_fsq_bigdata_sft.toml` | Prefer new versioned copies. Set new data/tokenizer aliases and output names; recompute packed-token/step budgets and scheduler milestones. Same molecule count does not imply the same LM token count. Do not treat an old geometry-trained LM checkpoint as an unchanged resume. |
| `src/molgen3D/training/pretraining/torchtitan_runner.py` | Add validation of format/tokenizer/checkpoint compatibility and new marker IDs; verify embedding/output-vocabulary capacity for the register marker. Generic Qwen attention/FSQ-free LM training does not need a new architecture. |
| `src/molgen3D/training/pretraining/dataprocessing/{dataloader.py,sequence_packing.py,count_tokens.py}` | Verify that serialization, labels and masks retain atom/register codes, including the final register before the conformer terminator. Recompute token counts with the new tokenizer. Change packing logic only where these checks expose assumptions about the old format. |
| `scripts/launch_torchtitan_qwen3.sh`, `scripts/submit_qwen3_17b_fsq_bigdata.sh`, `scripts/submit_qwen3_06b_fsq_bigdata_sft.sh`, `scripts/submit_qwen3_17b_fsq_bigdata_sft.sh` | Select new TOMLs/output directories; log format and checkpoint fingerprints. Retain explicit node/GPU/global-batch accounting. |
| `src/molgen3D/evaluation/inference.py` | Load the new adapter with the matching checkpoint; fail if loading fails (currently it logs and continues with `None`). Validate the generated format/register before graph comparison and decode; keep legacy mode explicit. |
| `scripts/run_qwen3_06b_fsq_bigdata_inference.sbatch`, `scripts/submit_qwen3_06b_fsq_bigdata_inference.sh` | Select matching new LM, LM tokenizer, coordinate tokenizer, and versioned results paths; remove accidental legacy checkpoint fallback for these runs. |
| `pyproject.toml`, `setup.sh`, `docs/pretraining_runbook.md` | Pin/document a working inference/training dependency set and the new preprocessing/training/evaluation commands. In particular ensure the copied quantizer's `einops` dependency and new model's PyTorch attention APIs are available. |

## Additional paths if those experiments will use the new tokenizer

These must either remain explicitly pinned legacy experiments or migrate together; do not silently relabel historical results.

| Files | Change |
|---|---|
| `src/molgen3D/evaluation/{conformer_interpolation.py,generated_conformer_interpolation.py,baseline_tokenizer_refinement.py,setup3_tokenizer_refinement.py}` | Update code/latent layout and per-atom versus register policies. Existing interpolation/bit-editing APIs assume token-aligned arrays. Decide whether the molecular register is interpolated/perturbed and report that choice. |
| `scripts/run_conformer_interpolation_stage1.sbatch`, `run_conformer_interpolation_stage1_eval.sbatch`, `run_generated_interpolation_setup3.sbatch`, `run_baseline_{prepare,refine,iterate}.sbatch`, `run_setup3_refine{,_iterate,_eval,_posebusters}.sbatch` | Route versioned model/format settings into those evaluation paths; retain old outputs and old checkpoints. |
| `src/molgen3D/experiments/bindgpt_fsq/{config.py,fsq_frozen.py,sequences.py,vocabulary.py,tokenizer_prep.py}` | Change the frozen fingerprint, code-count and code/graph layout contracts, union vocabulary/register token, and tokenizer audits. Both interleaved and graph-prefix/code-suffix serializers currently use graph-token counts rather than atom counts. |
| `src/molgen3D/experiments/bindgpt_fsq/{dataset.py,decode.py,generate.py,cli.py,train.py,model.py}` | Update/audit cached data validation, generation lengths/constraints, decoding and checkpoint metadata to agree with the new contract. Generic language-model layers can stay the same. |
| `bindgpt_geom_fsq/configs/`, `bindgpt_redesigned_geom_fsq/configs/`, their scripts and reports | Use a separately named experiment/config/cache version; rerun tokenizer audits and token-budget accounting instead of overwriting frozen legacy experiments. |

## Verification before training

Update `tests/data_processing/test_fsq_preprocessing.py` with explicit legacy and atom-plus-register cases; add strict-load and standalone-versus-ported-model reconstruction tests, register placement after padded widths, mixed lengths, ID0, missing/extra/out-of-range codes, branches/rings/disconnected graphs, and atom order. Add tests for the LM tokenizer and inference loader's failure path. Extend `tests/evaluation/test_conformer_interpolation.py`, `test_setup3_tokenizer_refinement.py`, and `tests/experiments/test_bindgpt_fsq_{pipeline,tokenizers}.py` if those paths migrate. Include explicit-H and stereo graph-to-RDKit regression cases.

Run a small encode/decode/reconstruction comparison against the frozen CoordToken model using training/development molecules, then a short LM training/save/resume/generate/decode smoke test. Verify the register is trained as a target token and survives generation. Once that passes, encode the chosen full corpus, verify row counts and split membership, recount packed tokens, and launch training. Official test scores must not select implementation or training hyperparameters.
