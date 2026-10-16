# Updated atom-code CoordToken

The frozen implementation emits one 12-bit geometry code per atom and one transmitted molecular register code. SMILES punctuation remains graph context and has no geometry code. The selected model uses width 1024, eight encoder and eight decoder layers, and dropout 0.05. This branch publishes the source used for the completed runs; 3DMolGen integration is described in [FSQ_TOKENIZER_MIGRATION.md](FSQ_TOKENIZER_MIGRATION.md) and is not implemented here.

## Start here

- Exact model and quantizer: `research/mixed_epoch2_20261014_v3/model.py` and `quantizer.py`.
- Initial training, data packing, tuning and metrics: `research/final_and_nabla_20261012/`.
- Second-epoch continuation, fresh shuffle, optimizer/RNG restoration and health checks: `research/mixed_epoch2_20261014_v3/`.
- Whole-test evaluator and paired comparison: `research/mixed_epoch2_test_20261015/`.
- `train.py` remains the legacy all-token trainer. The new trainer is `train_run.py` in the frozen directories above.

The frozen scripts retain their actual cluster paths and checksum guards. They depend on external packed data, indices, run configuration and checkpoints; this is not a self-contained download-and-launch package. Do not run historical launchers against existing output directories. Configure a separate experiment when moving machines; retain the frozen source for reference. Original launchers use `/mnt/weka/fgeikyan/tokenizer/conda_envs/train/bin/python`; the tested environment includes PyTorch 2.6 with CUDA 12.4, NumPy and einops. Distributed training requires the cluster GPU environment. Source hashes are recorded in `tokenizer_source_snapshot_20261016.json`; the epoch-two case is copied to `mixed_epoch2_case.json`.

## Checkpoints on the shared filesystem

All paths below are relative to `/mnt/weka/fgeikyan/CoordToken/`. Weights and datasets are not uploaded to Git.

| Training data / checkpoint | Path | SHA256 |
|---|---|---|
| CoordToken, epoch 1 | `logs/final_and_nabla_20261012/selected/mixed_full.pt` | `c3ee867ab22d5893a8a3fbba894657279301ce2add16e1e862966b2046118a41` |
| CoordToken, epoch 2 | `logs/mixed_epoch2_20261014/run/model.pt` | `29d487cf3783978c801c5a237e3c3c86cc23ab058aa7edea7eff0aa707c4bc4f` |
| CASF-filtered data, epoch 1 | `logs/paper_runs_20261013/new_paper/run/model.pt` | `d42d533b8ec30e8fc3d6ca699eed00b37d9aae60e04cc0f1a577ea21c18c1159` |

## Loading and representation

Use `Config` and `Tokenizer` from the frozen model, construct the config from checkpoint `config` excluding the informational `parameters` field, and load checkpoint `model` with `strict=True`. Use the checkpoint's exact `vocab`. These are not legacy Lightning checkpoints. Checkpoint loading should only be used with trusted artifacts.

Inputs are graph-token IDs (padding -1), centered coordinates and an atom mask. `model(tokens, coords, atoms)` returns reconstructed coordinates and code indices. The dense index tensor has padded graph width plus one register position. Serialize indices only at atom positions plus the last register position. Discarded positions are -1; code ID 0 is valid and is not a padding vector. `model.decode(tokens, indices, atoms)` reconstructs from this dense layout. Use evaluation mode and the documented precision path for comparisons.

The register is trained as a transmitted code and must reach the decoder. Its slot is after the padded graph width, including in mixed-length batches. Coordinate-bearing input strings do not need recollection or regeneration. Existing old-model FSQ integer streams do need encoding with the new checkpoint because both code layout and learned code meanings changed.

## Training and evaluation

The mixed-data runs use eight GPUs, global batch 512 and peak learning rate 5e-5. The second epoch restores the first epoch's optimizer and model, uses a new seed-124 order of all 189,792,028 training rows, immediately restores peak learning rate, then applies cosine decay to zero over the final 20%. The protocol files describe the exact schedules; do not assume every tuning run used this continuation schedule.

Reported reconstruction RMSD is the mean per-conformer heavy-atom RMSD in the centered coordinate frame without rotational alignment or symmetry matching. Test evaluation is separate from tuning. This source publication does not launch training, evaluation or dataset encoding.
