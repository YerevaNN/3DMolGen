# 3DMolGen and updated CoordToken: single-checkout handoff

Branch: `coordtoken-integration-handoff` in `YerevaNN/3DMolGen`.

```bash
git clone --branch coordtoken-integration-handoff --single-branch https://github.com/YerevaNN/3DMolGen.git
cd 3DMolGen
```

This branch includes the current 3DMolGen training/inference code and the updated atom-plus-register CoordToken source in one checkout. The tokenizer files are ordinary tracked files; no submodule or second clone is needed. **The 3DMolGen adapter still uses the legacy tokenizer; integration is the next implementation step.**

| What to work on | Location |
|---|---|
| 3DMolGen model training, preprocessing and inference | `src/molgen3D/`, `scripts/`, `tests/` |
| Existing training environment and launch instructions | [Current training handoff](docs/CURRENT_TRAINING_HANDOFF.md) |
| Updated coordinate tokenizer, checkpoints and representation contract | [Updated tokenizer handoff](coordtoken/docs/UPDATED_TOKENIZER.md) |
| Exact new model and quantizer | `coordtoken/research/mixed_epoch2_20261014_v3/model.py`, `quantizer.py` |
| Initial tokenizer training, tuning, packing and metrics | `coordtoken/research/final_and_nabla_20261012/` |
| Tokenizer second-epoch continuation | `coordtoken/research/mixed_epoch2_20261014_v3/` |
| Tokenizer whole-test evaluator | `coordtoken/research/mixed_epoch2_test_20261015/` |
| Files to change for 3DMolGen integration | [Migration map](docs/FSQ_TOKENIZER_MIGRATION.md) |

The new tokenizer emits one geometry code per atom plus one transmitted molecular register code. Punctuation has no geometry code. Port the exact model and quantizer, load the new checkpoint strictly, and preserve atom/register placement. The `<REG>` marker in the migration map is a proposed downstream serialization, not an already implemented format.

Coordinate-bearing strings can be reused. Generate new FSQ integer streams with the selected new tokenizer checkpoint; old-model integers cannot be converted by deleting punctuation codes. Retain the established data splits.

## External inputs and launch paths

Datasets and weights remain on shared storage. Their paths and checkpoint hashes are listed in the linked handoffs. The frozen tokenizer scripts retain original absolute cluster paths and output guards: inspect and configure a separate run before submission. Merely cloning this branch does not redirect those launchers to the bundled source or make the environments interchangeable. Keep frozen files unchanged as reference; adapt a separate implementation for integration. In particular, `coordtoken/train.py` is the legacy trainer, while the updated trainers are in `coordtoken/research/`.

## Source provenance

- 3DMolGen baseline: `b85fac7de524b351743907d740144e1b2d8cc78c`.
- CoordToken snapshot: `43fa947e0d52ed42f94aef5a2e98bf39f71e05a7`, copied byte-for-byte under `coordtoken/`.
- Per-file hashes: [snapshot provenance](docs/coordtoken_snapshot_provenance.json).

The combined branch packages the published source and documentation. It does not launch jobs, change the live working repositories, or claim the migration is complete.
