# Current training handoff

Combined checkout: [start here](../COLLEAGUE_HANDOFF.md). The updated tokenizer source is bundled under `coordtoken/`; the adapter integration remains to be done.

Baseline snapshot date: 2026-10-16. Repository: `YerevaNN/3DMolGen`, branch `big_data_training_with_fsq`. This preserves the current all-SMILES-token FSQ implementation before the atom-plus-register migration. **The new CoordToken checkpoint is not yet supported by this snapshot.** See [the migration map](FSQ_TOKENIZER_MIGRATION.md).

## What is included

Training and inference source, all four 0.6B/1.7B big-data pretraining/SFT TOMLs, Slurm launchers, preprocessing and shuffle tools, token-count/compute-budget helpers, existing Qwen tokenizer assets, refinement and BindGPT experiment source, tests, and local TorchData source with its upstream license. Large data, trained weights, generated samples, run outputs, chat backups, and scratch drafts remain external. The snapshot does not change model behavior.

## Start here

1. Clone this branch and read `docs/pretraining_runbook.md`.
2. Inspect `setup.sh`, `pyproject.toml`, and `docs/training_environment_snapshot_20261016.json` before choosing an environment. The setup script installs PyTorch 2.9.1, whereas the launcher's existing external environment contains a PyTorch 2.11 development build and TorchTitan 0.2.1. The JSON records observed package metadata, not a tested lockfile. The FSQ unit-test environment uses PyTorch 2.6.0. These environments are not interchangeable by assumption.
3. Set `PYTHON_BIN`, `TRAINING_VENV_SITE_PACKAGES`, and `TRAINING_SHARED_LIB` explicitly for the target machine. The launcher currently searches shared cluster paths and prepends the external site-packages directory ahead of `src`. Consequently, the installed TorchData 0.11.0 may take precedence over the checked-in 0.10.0 source; preserve/inspect that import order rather than assuming which copy runs.
4. Review `src/molgen3D/config/paths.yaml` and the selected TOML. Resolve training data, tokenizer assets, base model, output folders, and checkpoint paths before submission. Absolute paths describe this cluster, not portable defaults.
5. The launcher's import preflight checks TorchTitan, Qwen support, and project modules before starting distributed training. A full GPU training smoke test has not been rerun for this backup commit.

## Current launch entry points

Run from the repository root. These commands launch the **old representation** and must not be used with new atom-plus-register data until migration is complete.

```bash
# 0.6B big-data pretraining: launcher default is 3 nodes x 8 GPUs.
export TRAIN_TOML="$PWD/src/molgen3D/config/pretrain/qwen3_06b_fsq_bigdata_from_base.toml"
sbatch scripts/launch_torchtitan_qwen3.sh

# 1.7B big-data pretraining: also 3 nodes x 8 GPUs by default.
bash scripts/submit_qwen3_17b_fsq_bigdata.sh

# SFT wrappers: 1 node x 8 GPUs.
bash scripts/submit_qwen3_06b_fsq_bigdata_sft.sh
bash scripts/submit_qwen3_17b_fsq_bigdata_sft.sh
```

Choose the matching TOML and distributed parallelism settings when overriding node count. These are autoregressive Qwen training jobs; they are separate from the eight-GPU coordinate-tokenizer training jobs in the CoordToken repository.

Preprocessing: `scripts/submit_fsq_bigdata_preprocessing.sh` and `scripts/run_fsq_bigdata_preprocessing.sbatch` call `molgen3D.data_processing.preprocess_big_data_fsq`. Read their input format and environment overrides first. `scripts/shuffle_jsonl_dataset.py` / `scripts/run_fsq_shuffle_dataset.sbatch` handle JSONL shuffling. `scripts/run_qwen3_06b_fsq_bigdata_inference.sbatch` and its submission wrapper run the current inference path.

## External assets

| Asset | Current location or definition |
|---|---|
| Legacy all-token FSQ checkpoint | `/mnt/weka/fgeikyan/fsq/new_checkpoints/full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt` |
| Legacy encoded big-data training | `data:fsq_big_data_train` in `src/molgen3D/config/paths.yaml`; currently `/mnt/weka/fgeikyan/fsq/3dbigdata_fsq_4096_shuffled/processed_strings/train` |
| Legacy GEOM FSQ partitions | `/mnt/weka/fgeikyan/fsq/geom_fsq_4096/processed_strings/{train,valid,test}` |
| Qwen base weights | Selected TOML's `base_model_tag` and `initial_load_path`; e.g. `/mnt/weka/vtarasov/Qwen3-0.6B-Base/patched_for_titan/` |
| Current Qwen tokenizer assets | Tracked `src/molgen3D/training/tokenizers/Qwen3_tokenizer_fsq_4096/` |
| Clean CoordToken coordinate CSVs | `/mnt/weka/fgeikyan/3dmolgen_bigdata/data_handoff/merged_deduplicated_20261006/merged_{train,val,test}.csv` |
| Separate CASF-filtered training corpus | `/mnt/weka/vtarasov/dedup_casf285_plinder23_20261007/protomer/final/merged_train_clean_shuffled.csv` |
| New completed CoordToken epoch-2 weights | `/mnt/weka/fgeikyan/CoordToken/logs/mixed_epoch2_20261014/run/model.pt` |
| Epoch-2 weights SHA256 | `29d487cf3783978c801c5a237e3c3c86cc23ab058aa7edea7eff0aa707c4bc4f` |
| New CASF-trained tokenizer weights | `/mnt/weka/fgeikyan/CoordToken/logs/paper_runs_20261013/new_paper/run/model.pt` |
| CASF-trained weights SHA256 | `d42d533b8ec30e8fc3d6ca699eed00b37d9aae60e04cc0f1a577ea21c18c1159` |
| New tokenizer implementation | `/mnt/weka/fgeikyan/CoordToken/research/mixed_epoch2_20261014_v3/{model.py,quantizer.py}` |

The new checkpoints and model source are pointers for the next integration, not part of the legacy loader. Select the appropriate corpus/checkpoint pair for the intended paper; do not silently substitute one for the other. Data and weight locations require shared-storage access or a separate transfer. Git alone does not supply them. Historical locations above are configured paths, not a fresh full-content audit.

## Validation of this snapshot

89 existing tests passed in the coordinate-tokenizer Python 3.10 environment: FSQ preprocessing, interpolation, refinement, compute-matching, PoseBench RMSD aggregation, and BindGPT pipeline/tokenizers. Python syntax and shell syntax were checked for the selected source files. This establishes a backed-up baseline, not completed migration or a newly verified distributed Qwen run.

```bash
PYTHONPATH=src python -m pytest -q \
  tests/data_processing/test_fsq_preprocessing.py \
  tests/evaluation/test_conformer_interpolation.py \
  tests/evaluation/test_setup3_tokenizer_refinement.py \
  tests/test_compute_matching_epochs.py \
  tests/test_posebench_rescore_unaligned.py tests/experiments
```
