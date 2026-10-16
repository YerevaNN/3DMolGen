# Scratch BindGPT on GEOM-DRUGS: Original Coordinate Tokens vs. FSQ Tokens

## Purpose

Run a controlled comparison of two BindGPT-style autoregressive models trained **from random
initialization on GEOM-DRUGS only**:

1. `BindGPT-Coord`: the original BindGPT coordinate tokenizer.
2. `BindGPT-FSQ`: the same language model with only the coordinate representation replaced by the
   existing frozen CoordToken/FSQ tokenizer.

Both trained models are evaluated on two tasks without further training:

1. **Unconditional 3D molecule generation**, using the BindGPT/MolDiff-style GEOM-DRUGS protocol.
2. **SMILES-conditioned conformer generation**, using the project's regular GEOM-DRUGS
   recall/precision COV and AMR protocol.

The experiment tests whether the learned FSQ coordinate representation improves an otherwise
matched BindGPT model. It is not a reproduction of the published BindGPT result because the
published model first used Uni-Mol-scale pretraining.

## Immutable Experiment Decisions

- Do not load the public BindGPT model weights.
- Do not pretrain on Uni-Mol, the 208M-conformer corpus, the internal 190M-conformer corpus, or any
  other external molecular corpus.
- Train both autoregressive models from random initialization.
- Train only on conformers whose molecular identities belong to the frozen GEOM-DRUGS training
  split.
- Use the same train/validation/test molecular identities for both tokenizers.
- Use all eligible GEOM-DRUGS conformers for each training molecule; do not reduce the main run to
  MCF's maximum of 20 conformers per molecule.
- Keep the transformer architecture, training examples, example order, optimizer, learning-rate
  schedule, number of epochs, decoding budgets, and evaluation code fixed across tokenizers.
- Freeze the FSQ encoder and decoder. No FSQ parameter may receive gradients during BindGPT
  training.
- Do not use MMFF, UFF, xTB, RDKit conformer optimization, energy reranking, or rejection based on
  reference RMSD in either primary evaluation.
- Save raw generations before parsing, filtering, or metric computation.

Any departure from these decisions creates a separate experiment and must use a distinct run name.

## Scientific Questions

### Primary question

With identical GEOM-DRUGS data exposure and the same scratch-trained BindGPT architecture, does
replacing BindGPT's scalar coordinate tokens with the frozen FSQ representation improve generation?

### Unconditional-generation endpoints

- molecular validity and connectivity
- QED, synthetic accessibility, and Lipinski statistics
- agreement with held-out GEOM-DRUGS distributions for local geometry, bonds, and rings
- raw-attempt efficiency and valid yield

### Conformer-generation endpoints

- Recall COV and AMR
- Precision COV and AMR
- the full Recall-COV and Precision-COV curves as functions of RMSD threshold
- coordinate-decoding success and valid-conformer yield

## Experimental Conditions

| Run ID | LM initialization | LM training data | SMILES tokenizer | Coordinate tokenizer | Coordinate decoder |
| --- | --- | --- | --- | --- | --- |
| `bindgpt_coord_geom_scratch` | random | GEOM-DRUGS train only | BindGPT character-level | original BindGPT integer/fraction tokens | deterministic scalar decoding |
| `bindgpt_fsq_geom_scratch` | random | GEOM-DRUGS train only | identical BindGPT character-level | frozen CoordToken/FSQ checkpoint | frozen CoordToken/FSQ decoder |

Everything except the coordinate representation must be matched. In particular, `BindGPT-FSQ`
must not switch to a different SMILES tokenizer, pretrained text model, transformer size, or data
pipeline.

## Inputs Required From the Executor

The person running the experiment should fill and commit a machine-readable experiment config with
these values before launching training:

| Field | Required value |
| --- | --- |
| `geom_root` | path to the unmodified GEOM-DRUGS source data |
| `split_source` | exact repository/file used to define molecular splits |
| `split_sha256` | SHA-256 of the frozen split artifact |
| `bindgpt_tokenizer_revision` | immutable Hugging Face commit or local tokenizer hash |
| `fsq_checkpoint` | exact CoordToken/FSQ checkpoint path |
| `fsq_checkpoint_sha256` | SHA-256 of the FSQ checkpoint |
| `fsq_config` | codebook sizes, number of levels/codebooks, latent shape, and decoder configuration |
| `fsq_code_revision` | Git commit used for FSQ encoding and decoding |
| `training_code_revision` | Git commit used for the two BindGPT runs |
| `evaluation_code_revision` | Git commit used for all metrics |
| `rdkit_version` | exact RDKit version used for parsing and RMSD |
| `pytorch_transformers_versions` | exact PyTorch and Transformers versions |
| `slurm_cluster` | cluster name and GPU model |

Do not use a floating branch, `latest` checkpoint, or mutable dataset path in the final run.

## FSQ Checkpoint Provenance

No large-corpus pretraining is permitted for the **BindGPT language models**. The supplied FSQ
checkpoint is a separate learned component and therefore needs its own provenance audit.

Before training, record:

- datasets used to train the FSQ checkpoint
- whether any GEOM-DRUGS validation or test conformers were used
- whether exact validation/test molecular identities occurred with different conformers
- whether the FSQ model was selected using GEOM-DRUGS test performance
- the coordinate normalization and canonicalization used during FSQ training

Preferred condition: the FSQ checkpoint was trained without any molecule from the frozen
GEOM-DRUGS validation or test sets. If that is not true, do not hide it. Label the FSQ result as
`tokenizer-overlap` and avoid a clean held-out-generalization claim. Training a clean FSQ checkpoint
on GEOM-DRUGS training identities only is the correct follow-up.

## Dataset and Split

### Split definition

Use the exact molecule-level split used by the project's regular GEOM-DRUGS conformer benchmark.
The intended MCF/GeoMol-style split contains:

- 243,473 training molecules
- 30,433 validation molecules
- 1,000 test molecules used for final conformer evaluation

Treat these counts as a verification target, not as a substitute for the actual identity lists.
The split artifact itself is authoritative.

If the existing project evaluator uses a different frozen identity list, stop and reconcile the
difference before training. Do not train on one split and silently evaluate on another.

### Molecular identity

Create a manifest with one row per molecular identity and at least these columns:

```text
source_molecule_id
split
source_smiles
standardized_isomeric_smiles
inchi
inchikey
num_source_conformers
num_eligible_conformers
exclusion_reason
```

Use standardized isomeric canonical SMILES and full InChIKey for overlap checks. Splitting must be
by molecule, never by individual conformer.

### Leakage assertions

The preprocessing job must terminate with an error unless all of the following hold:

```text
train_inchikeys intersect validation_inchikeys = empty
train_inchikeys intersect test_inchikeys = empty
validation_inchikeys intersect test_inchikeys = empty
```

Also check standardized isomeric SMILES. Save collision tables even when the collision count is
zero.

### Conformer inclusion

For every molecular identity in the training split:

- include every conformer that passes the shared deterministic preprocessing pipeline
- preserve the source conformer ID
- preserve atom identity and stereochemistry
- remove explicit hydrogens for the primary experiment
- require a single connected molecular component
- do not energy-rerank or RMSD-deduplicate conformers unless the existing source pipeline already
  does so
- apply exactly the same exclusions to both tokenizer variants

The primary claim is based on matched examples. Therefore, create the final ordered training
manifest once and consume the exact same rows in the exact same epoch order for both models.

### Validation and test use

- Use validation identities for loss monitoring, checkpoint diagnostics, and unconditional
  reference-distribution metrics.
- Use the 1,000 test identities only for final metrics after training and decoding settings are
  frozen.
- Do not choose temperatures, checkpoints, FSQ settings, or postprocessing using test COV/AMR.

## Shared Sequence Construction

Every training example represents one molecule-conformer pair:

```text
<LIGAND>{randomized_smiles}<XYZ>{coordinate_representation}<EOS>
```

Requirements:

- `<LIGAND>`, `<XYZ>`, and `<EOS>` semantics are identical across variants.
- Randomized SMILES must preserve the molecular graph and stereochemistry.
- Reorder the coordinate array to match the atom order in the randomized SMILES.
- Use the same randomized SMILES realization for the paired Coord and FSQ examples.
- Center coordinates using one shared deterministic rule. Prefer the rule already used by the
  BindGPT GEOM pipeline; record it explicitly.
- Do not apply random rotations. BindGPT reports no rotation augmentation for GEOM-DRUGS training.
- Never truncate a sequence silently. Count and report examples that exceed context length.

Build and hash a paired-example manifest containing the source conformer ID, randomized-SMILES
seed, atom permutation, centering transform, and the sequence length under both tokenizers.

## Tokenizer A: Original BindGPT Coordinates

Use the public BindGPT tokenizer files as the source of truth, but do not load public model weights.

The expected representation is:

- character-level SMILES tokens
- `<XYZ>` separator
- atom count followed by coordinates aligned with SMILES atom order
- six tokens per atom coordinate triplet: integer and fractional tokens for each of `x`, `y`, and
  `z`

The executor must add unit tests showing that the local implementation exactly round-trips a set of
public BindGPT examples and matches the public tokenizer's token IDs. Do not reimplement numeric
rounding from memory if the public tokenizer can be invoked directly.

Report coordinate quantization error on at least 10,000 validation conformers:

- coordinate MAE
- aligned heavy-atom RMSD after encode/decode
- 50th, 90th, 95th, and 99th percentiles
- sequence tokens per atom and per molecule

## Tokenizer B: CoordToken/FSQ Coordinates

Keep the complete prefix through `<XYZ>` identical to Tokenizer A. Replace only the original
coordinate suffix with FSQ code tokens.

Required behavior:

1. Load the frozen FSQ encoder/decoder checkpoint.
2. Apply exactly the coordinate normalization expected by that checkpoint.
3. Encode the conformer into its ordered discrete FSQ codes.
4. Map each code or code tuple to a reserved LM token without collisions with SMILES or special
   tokens.
5. Emit sufficient shape/length metadata for deterministic decoding if the FSQ architecture
   requires it.
6. Decode generated FSQ tokens back to one coordinate triplet per prompted atom.
7. Invert normalization and centering consistently.

The experiment must document the precise serialization. For example, state whether there is one
token per atom, multiple codebook tokens per atom, or a global latent sequence. Do not call the
representation "FSQ tokens" without this mapping.

Required tokenizer-only validation:

- exact expected number of decoded atoms
- deterministic encode/decode with the checkpoint in evaluation mode
- no gradient flow into the FSQ model
- reconstruction statistics on the same 10,000 validation conformers used for Tokenizer A
- token count and compression ratio relative to original BindGPT coordinates
- invalid or undecodable FSQ sequence rate

## Language-Model Architecture

Use the BindGPT 108M GPT-NeoX configuration for both variants:

- decoder-only GPT-NeoX
- 15 transformer layers
- hidden size 768
- 12 attention heads
- feed-forward/intermediate size 3,072
- rotary positional embeddings
- maximum sequence length 2,048
- untied input and output embeddings, matching the released configuration

Initialize all LM parameters from scratch. Use one shared initialization seed for corresponding
transformer blocks. Vocabulary-dependent embedding/output matrices will have different shapes and
must be independently initialized with the same initialization rule.

Record exact parameter counts. Report both total parameters and transformer-body parameters so
vocabulary-size differences are visible.

## Primary Training Protocol

Use the GEOM supervised-fine-tuning hyperparameters reported by BindGPT, adapted to scratch
training without changing the optimizer recipe:

- epochs: 1 over the frozen full training-example manifest
- optimizer: AdamW
- maximum learning rate: `5e-4`
- warmup: 100 optimizer steps
- schedule: cosine decay after warmup
- weight decay: `1e-2`
- gradient clipping: global norm `1.0`
- precision: bfloat16 computation
- token loss weights: `1.0` for every non-padding token
- SMILES randomization: enabled
- random coordinate rotations: disabled

### Matching rule

The primary comparison is matched by **molecule-conformer examples**, not by tokens or wall time.

- same examples per optimizer update
- same global example batch size
- same number and order of optimizer updates
- same examples over the epoch
- same learning-rate value at the corresponding optimizer update

Use packing and gradient accumulation as necessary, but do not let shorter FSQ sequences cause the
FSQ model to see more training examples. Report tokens processed and GPU-hours separately; these
are efficiency outcomes, not matching variables.

If memory prevents the original-token model from using the chosen global example batch, lower the
global example batch for **both** models. Do not use different effective example batches.

### Scratch-training safeguard

One GEOM epoch was originally a fine-tuning recipe, not a published scratch-training recipe. Save
checkpoints at 25%, 50%, 75%, and 100% of the epoch and inspect validation loss without looking at
test metrics.

If both models are still clearly improving at one epoch, run a predeclared secondary continuation
to three total epochs for both models. Use the same data ordering seeds and report both:

- `1ep`: primary BindGPT-recipe comparison
- `3ep`: scratch-convergence comparison

Do not extend only the worse model. Do not select different epoch counts for the headline pair.

### Seeds

Use training seeds `17`, `23`, and `42` for the final comparison. Run seed `17` first as the pipeline
validation run. Launch the remaining seeds only after both seed-17 variants pass all integrity
checks.

For each seed, match:

- transformer-body initialization
- data order
- randomized-SMILES choices
- dropout RNG policy as closely as possible

## Training Integrity Checks

Every run must log:

- examples and non-padding tokens processed
- optimizer steps and learning rate
- total and coordinate-suffix loss
- validation total and coordinate-suffix loss
- gradient norm
- GPU-hours, peak allocated memory, and throughput
- sequence-length histogram
- percentage of sequences near or over context length
- FSQ encoder/decoder parameter checksum before and after training

Terminate the run if:

- an FSQ parameter changes
- the paired example counts differ
- an example from a validation or test identity enters training
- any sequence is truncated without a recorded exclusion
- NaN/Inf loss occurs and cannot be recovered from the last valid checkpoint

## Evaluation 1: Unconditional 3D Molecule Generation

### Scope

This follows the public BindGPT generation interface and the MolDiff-style metrics adopted in the
BindGPT GEOM table. BindGPT's complete evaluation repository is not public, so this should be called
**BindGPT/MolDiff-style**, not an exact reproduction of every unpublished implementation detail.

### Generation prompt and sampling

Prompt each model with:

```text
<LIGAND>
```

Primary sampling configuration:

- stochastic autoregressive sampling
- temperature `1.0`
- top-p `1.0`
- top-k `50`, matching the default behavior of the public Transformers example
- maximum total sequence length `400`, matching the public BindGPT example
- no beam search
- no classifier-free guidance
- no energy scoring or reranking
- generation seeds `2023`, `2024`, and `2025`

Generate until the run contains at least 1,000 successfully parsed, connected molecules for each
generation seed. Preserve every failed and successful raw attempt. Report the number of attempts
needed to reach 1,000 valid connected molecules.

The primary metric population is the first 1,000 valid connected molecules in deterministic sample
order. Additionally report validity/connectivity over all raw attempts up to that point. This keeps
the BindGPT table population comparable while exposing rejection efficiency.

### Parsing

For each raw sequence:

1. Require exactly one `<LIGAND>` and one `<XYZ>` section in the expected order.
2. Parse and sanitize the generated SMILES.
3. Reject disconnected graphs for the connected-molecule table, but count them in raw validity.
4. Require the coordinate decoder to return exactly one triplet per generated atom.
5. Attach the generated conformer without coordinate optimization.
6. Save the raw string, token IDs, parsed SMILES, canonical SMILES, coordinates, and failure reason.

Use a closed failure-reason vocabulary such as:

```text
missing_special_token
smiles_parse_failure
sanitization_failure
disconnected
coordinate_decode_failure
atom_count_mismatch
nonfinite_coordinate
sequence_length_limit
other
```

### Reference distribution

Compute reference statistics once from the held-out GEOM-DRUGS split using the same preprocessing
and metric code as generated molecules.

Because the BindGPT text describes comparison with the GEOM validation distribution while the
released MolDiff notebook defaults to the test distribution, compute both at negligible additional
cost:

- **primary:** the frozen 1,000-molecule GEOM-DRUGS test set
- **secondary compatibility table:** GEOM-DRUGS validation set

Never combine validation and test references.

### Required unconditional metrics

Report the BindGPT table fields:

| Group | Metric | Direction |
| --- | --- | --- |
| Drug-likeness | mean QED | higher |
| Drug-likeness | mean normalized SA score | higher |
| Drug-likeness | mean Lipinski count | higher |
| 3D structure | JS distance/divergence of bond-length distributions | lower |
| 3D structure | JS distance/divergence of bond-angle distributions | lower |
| 3D structure | JS distance/divergence of dihedral-angle distributions | lower |
| Bonds | JS of number of bonds per atom | lower |
| Bonds | JS of bond-type frequencies | lower |
| Bonds | JS of bond-pair frequencies | lower |
| Bonds | JS of bond-triplet frequencies | lower |
| Rings | JS of number of rings | lower |
| Rings | JS of number of `n`-sized rings | lower |
| Rings | number/frequency of intersecting rings | lower or match reference |

Also report these diagnostics, even though they are not all present in the BindGPT headline table:

- raw SMILES validity
- sanitization success
- connectivity
- coordinate-decode success
- valid-connected yield
- uniqueness among valid molecules
- novelty relative to GEOM training molecules
- atom-count and molecular-weight distributions
- wall time and GPU time to obtain 1,000 valid connected molecules

Pin the MolDiff evaluation code or port it into a non-notebook script. The MolDiff reference
configuration samples 1,000 molecules. Its released analysis code computes held-out distribution
statistics and should be treated as the compatibility implementation. Record whether the reported
quantity is SciPy's Jensen-Shannon **distance** or its squared divergence; do not label one as the
other.

### Unconditional result table

| Model | Seed | Raw validity | Connected yield | QED | SA | Lipinski | JS bond length | JS bond angle | JS dihedral | JS bond/atom | JS bond types | JS bond pairs | JS bond triplets | JS rings | JS ring sizes | Intersecting rings | GPU time / 1K valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BindGPT-Coord | 2023 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| BindGPT-FSQ | 2023 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

Produce one table per training seed and an aggregate table with mean, standard deviation, and the
three individual values. Do not average away failed training runs.

## Evaluation 2: Regular GEOM-DRUGS Conformer Protocol

### Evaluation set

Use the frozen 1,000-molecule GEOM-DRUGS test set from the molecule-level split. For each test
molecule, retain all eligible reference conformers used by the existing project evaluator.

Before evaluating the new checkpoints, run the current evaluator on one archived model/output with
known metrics. The evaluator should reproduce the archived values within numerical tolerance. This
calibration protects against changes in atom mapping, symmetry handling, or aggregation.

### Conditional generation

For each test molecule with `K` reference conformers:

1. Construct one deterministic canonical isomeric SMILES prompt with atom order recorded.
2. Prompt the model with `<LIGAND>{SMILES}<XYZ>`.
3. Generate `2K` valid coordinate conformers.
4. Use the same primary decoding settings for both tokenizers.
5. Do not optimize, energy-rank, RMSD-filter, or diversity-filter the outputs.

Primary decoding settings:

- temperature `1.0`
- top-p `1.0`
- top-k `50`
- independent stochastic samples
- no beam search

Allow at most `4K` raw attempts to obtain `2K` decodable conformers. Record raw attempts and every
failure. If fewer than `2K` valid conformers are obtained, mark the molecule as a generation
failure and report it; do not silently drop the molecule. A run with more than 1% failed test
molecules is not eligible for the headline COV/AMR table until the failure policy is explicitly
resolved for both methods.

Use generation seeds `2023`, `2024`, and `2025` for each trained checkpoint. These are sampling
replicates, not substitutes for independent training seeds.

### RMSD

Use the project's existing RMSD implementation. It must be:

- heavy-atom only
- alignment-based
- symmetry-aware with equivalent-atom permutations
- computed between conformers of the same molecular graph and stereochemistry
- reported in angstroms

The preferred implementation is the already validated RDKit-based best-alignment evaluator used by
the project. Pin RDKit because symmetry and sanitization behavior may change across releases.

### Metrics

Let `C_r` be the reference conformers and `C_g` the generated conformers for one molecule.

```text
AMR-R = mean over R in C_r of min over G in C_g RMSD(R, G)
COV-R(delta) = fraction of R in C_r with min_G RMSD(R, G) < delta

AMR-P = mean over G in C_g of min over R in C_r RMSD(G, R)
COV-P(delta) = fraction of G in C_g with min_R RMSD(G, R) < delta
```

Use `delta = 0.75 A` for the headline GEOM-DRUGS COV values. Preserve the strict/inclusive threshold
convention of the current project evaluator and record it.

For each metric, compute the value per molecule and report the mean and median across the 1,000
molecules:

- Recall COV at `0.75 A`
- Recall AMR
- Precision COV at `0.75 A`
- Precision AMR

Also save the complete per-molecule table and stratify by:

- heavy-atom count
- rotatable-bond count
- number of reference conformers
- ring count

### Full coverage curves

Compute Recall-COV and Precision-COV for thresholds from `0.00 A` through `2.00 A` in increments of
`0.05 A`. Save:

- macro-average curve across molecules
- median curve across molecules
- per-molecule curves or sufficient nearest-RMSD arrays to reconstruct them
- area under the coverage curve over `[0, 2.0 A]` as a secondary summary

Do not replace the `0.75 A` headline with curve AUC; report both.

### Conformer result table

| Model | Training seed | Sampling seed | COV-R mean | COV-R median | AMR-R mean | AMR-R median | COV-P mean | COV-P median | AMR-P mean | AMR-P median | Decode yield | Failed molecules |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BindGPT-Coord | 17 | 2023 |  |  |  |  |  |  |  |  |  |  |
| BindGPT-FSQ | 17 | 2023 |  |  |  |  |  |  |  |  |  |  |

Report aggregate uncertainty across training seeds. Sampling-seed variability should be reported
separately so it is not confused with training variability.

## Optional Decoding Sweep

The primary comparison must use the fixed decoding settings above. After those results are sealed,
an optional validation-only sweep may test:

- temperature: `0.7`, `0.85`, `1.0`, `1.15`
- top-p: `0.9`, `0.95`, `1.0`
- top-k: `0`, `50`

Select one setting per tokenizer using validation data only, then run it once on test and label it
`tokenizer-specific tuned decoding`. Keep the fixed-decoding comparison as the causal headline.

## Statistical Analysis

For conformer metrics:

- compare tokenizers on paired test molecules
- bootstrap molecules with 10,000 paired resamples
- report the mean paired difference and 95% confidence interval
- keep training seeds separate, then summarize the distribution across seeds

For unconditional metrics:

- report all three generation seeds
- bootstrap generated molecules for scalar property means
- for distribution distances, bootstrap both generated molecules and reference molecules when
  computationally feasible

Do not claim an FSQ improvement based on one favorable seed when the direction is inconsistent.

## Slurm Execution Plan

### Job graph

```text
prepare_and_hash_data
        |
        +--> tokenizer_roundtrip_coord
        +--> tokenizer_roundtrip_fsq
                    |
                    v
          build_paired_manifests
                    |
          +---------+---------+
          |                   |
          v                   v
 train_coord_seed17     train_fsq_seed17
          |                   |
          +---------+---------+
                    v
          integrity_and_val_check
                    |
          +---------+---------+
          |                   |
          v                   v
      seeds23/42          seeds23/42
          |                   |
          +---------+---------+
                    v
       unconditional_generation
                    |
       conditional_2K_generation
                    |
              metric_jobs
                    |
              aggregate_report
```

### Submission policy

- Use Slurm dependencies so evaluation cannot start before its checkpoint and manifests are
  complete.
- Put each training seed in its own job.
- Use job arrays for test-molecule conditional sampling and metric shards.
- Do not allow multiple array tasks to overwrite one output file.
- Requeue only from an atomic checkpoint with a complete optimizer/scheduler/RNG state.
- Capture `scontrol show job`, `nvidia-smi`, module list, environment lock file, and Git status in
  every run directory.

### Resource discovery pilot

Before full training, run both tokenizers on the same 10,000-example shard to measure:

- maximum sequence length
- examples/sec and tokens/sec
- peak GPU memory
- feasible microbatch size
- gradient-accumulation setting needed for the shared global example batch

The pilot is for resource sizing only. Do not use pilot test metrics to change the scientific
protocol.

### Slurm template

The executor should adapt this skeleton to the actual repository entry points:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=bindgpt-fsq-geom
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Load the cluster-specific CUDA/conda environment here.
# Invoke the committed training entry point with one immutable YAML config.
# Never encode scientific settings only in this shell script.

srun python -m <training_entrypoint> --config <frozen_config_yaml>
```

Exact partition, account, time, memory, launcher, and distributed-training flags are intentionally
left to the cluster operator.

## Required Artifact Layout

```text
bindgpt_geom_fsq/
  manifests/
    molecules.csv
    train_examples.parquet
    validation_examples.parquet
    test_molecules.parquet
    overlap_report.json
    hashes.json
  configs/
    coord_seed17.yaml
    fsq_seed17.yaml
    ...
  tokenizer_audit/
    coord_reconstruction.json
    fsq_reconstruction.json
    sequence_lengths.parquet
  checkpoints/
    coord/seed_17/...
    fsq/seed_17/...
  generations/
    unconditional/<model>/<train_seed>/<sample_seed>/...
    conformer/<model>/<train_seed>/<sample_seed>/<shard>/...
  metrics/
    unconditional/...
    conformer/...
    per_molecule/...
  logs/
  environment/
  final_report.md
```

Each generation record should contain:

```text
run_id
checkpoint_sha256
train_seed
sampling_seed
source_or_prompt_smiles
raw_token_ids
raw_decoded_text_or_fsq_codes
parsed_canonical_smiles
coordinates
attempt_index
validity_flags
failure_reason
generation_wall_time
```

Use Parquet/JSONL plus SDF exports rather than SDF alone; SDF does not preserve all failed attempts
or token-level provenance.

## Preflight Checklist

- [ ] Exact GEOM source and split artifacts are hashed.
- [ ] Train/validation/test molecular identity intersections are empty.
- [ ] Both variants consume the same ordered training examples.
- [ ] Public BindGPT tokenizer round-trip tests pass.
- [ ] FSQ checkpoint/config/code hashes are recorded.
- [ ] FSQ training overlap is documented.
- [ ] FSQ parameters are frozen and checksum-protected.
- [ ] Both coordinate decoders return the prompt atom order.
- [ ] No sequence is silently truncated.
- [ ] The existing GEOM evaluator reproduces one archived result.
- [ ] Training and evaluation environments are locked.
- [ ] Seed-17 pilot completes for both variants before other seeds launch.

## Completion Criteria

The experiment is complete only when all of the following are delivered:

1. Scratch-trained Coord and FSQ checkpoints for the same declared seeds.
2. Dataset, split, example, tokenizer, checkpoint, and code hashes.
3. Full training curves and compute/throughput measurements.
4. Raw unconditional generations and the BindGPT/MolDiff-style table.
5. Raw conditional generations for every test molecule.
6. Per-molecule Recall/Precision COV and AMR values.
7. Full Recall-COV and Precision-COV threshold curves.
8. Decode/validity failure analysis for both tokenizers.
9. Paired uncertainty estimates across molecules and variation across training seeds.
10. A short report that states the FSQ checkpoint's overlap status and lists every protocol
    deviation.

## Interpretation Rules

- A better unconditional distribution score does not by itself show better conformer coverage.
- Higher Precision COV with lower Recall COV is a tradeoff, not an unqualified win.
- Shorter FSQ sequences and lower GPU cost are valid systems results, but they are distinct from
  representation-quality results.
- If the FSQ tokenizer saw test molecules during its own training, the result can demonstrate
  end-to-end performance but not clean held-out generalization.
- If one scratch-trained model is clearly under-converged, report that fact and use the predeclared
  three-epoch continuation for both models.
- The public pretrained/fine-tuned BindGPT checkpoint may be shown as contextual reference only; it
  is not part of this controlled no-pretraining comparison.

## Primary Sources

- [BindGPT paper](https://arxiv.org/abs/2406.03686)
- [BindGPT project page](https://bindgpt.github.io/)
- [Public GEOM-fine-tuned BindGPT model card](https://huggingface.co/insilicomedicine/bindgpt_finetuned)
- [MCF paper and GEOM conformer protocol](https://arxiv.org/abs/2311.17932)
- [MCF code and processed split downloads](https://github.com/apple/ml-mcf)
- [MolDiff code and unconditional evaluation pipeline](https://github.com/pengxingang/MolDiff)

## Known Reproducibility Limitation

BindGPT's complete training and evaluation repository remains unavailable at the linked location.
The paper and public model card specify the architecture, sequence format, main GEOM training
hyperparameters, generation interface, and headline metrics, but not every preprocessing and
evaluation implementation detail. This plan therefore pins public BindGPT behavior where available
and uses the released MolDiff evaluator for the unconditional compatibility table. Every local
choice that cannot be recovered from public artifacts must be written into the frozen experiment
config before test evaluation.
