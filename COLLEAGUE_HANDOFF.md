# 3DMolGen + CoordToken: one-checkout handoff

Branch: **`coordtoken-integration-handoff`** in `YerevaNN/3DMolGen`.

```bash
git clone --branch coordtoken-integration-handoff --single-branch https://github.com/YerevaNN/3DMolGen.git
cd 3DMolGen
```

The new coordinate-tokenizer adapter, CSV-to-Qwen preparation, versioned Qwen tokenizer, packed-loader checks, Qwen checkpoint export and generation decoder are implemented here. The frozen tokenizer training source and final data-cleaning pipeline are included as ordinary tracked files. No submodule is required. This branch was checked against remote `main` on October 17; the previous handoff commit was not an ancestor of `main`. This update stays on the handoff branch.

**Representation:** `<317>[C]<42>(=[O]<901>)[O]<0>` — one molecular code first, then one geometry code after each atom. All codes share the 4,096-token alphabet. No `<REG>` marker; no geometry codes after punctuation. These numbers illustrate the format, not a particular encoded geometry.

**Selected checkpoint:** `/mnt/weka/fgeikyan/CoordToken/artifacts/releases/coordtoken-atom12-reg1-mixed-epoch2-20261017/model.pt`.
SHA-256: `29d487cf3783978c801c5a237e3c3c86cc23ab058aa7edea7eff0aa707c4bc4f`.
The read-only copy is the two-epoch mixed-corpus model selected by validation. Weights are external, not committed to Git. Its directory includes config/result evidence and frozen source.

| Purpose | Entry point |
|---|---|
| New representation, preparation/training/inference commands | [Integration guide](docs/FSQ_TOKENIZER_MIGRATION.md) |
| Strict encoder/decoder | `src/molgen3D/data_processing/coordtoken.py` |
| Frozen tokenizer model and training/evaluation source | [Tokenizer source](coordtoken/docs/UPDATED_TOKENIZER.md) |
| Final cleaning/splitting pipeline and release evidence | [Final data pipeline](coordtoken/data_collection/FINAL_PIPELINE.md) |
| Integration validation scope and results | [Checks](docs/coordtoken_integration_validation.json) |
| Existing Qwen environment and legacy training details | [Training environment](docs/CURRENT_TRAINING_HANDOFF.md) |

Reuse coordinate-bearing CSVs and the established splits. Encode new integer streams with this checkpoint: old geometry integers have different meanings and cannot be migrated by dropping punctuation codes. Existing legacy paths remain explicitly legacy; the new entry points do not silently switch them.

No production Qwen training, full-corpus conversion, or new downstream experiment is claimed by this source update. CPU parity, packed targets, tiny-Qwen save/resume, and constrained generation are checked; full GPU/distributed production execution remains a launch-time validation step. Legacy interpolation/refinement/BindGPT experiments have not been relabeled as new-tokenizer experiments.

Source provenance: frozen model snapshot `43fa947` is recorded in `docs/coordtoken_snapshot_provenance.json`; the added data snapshot and documentation overrides are recorded in `docs/coordtoken_final_snapshot_provenance.json`.
