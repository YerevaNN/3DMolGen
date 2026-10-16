# BindGPT-Cartesian vs BindGPT-CartesianToken on GEOM-DRUGS

Two BindGPT 108M GPT-NeoX language models trained **from random initialization on
GEOM-DRUGS only**, differing solely in how atomic coordinates are tokenized:

| Run id | Coordinate tokenizer | Coordinate decoder |
| --- | --- | --- |
| `bindgpt_coord_geom_scratch` | original BindGPT integer + fraction tokens | deterministic scalar decoding |
| `bindgpt_fsq_geom_scratch` | frozen CoordToken/FSQ checkpoint | frozen CoordToken/FSQ decoder |

**Start here: [`docs/experiment_description.md`](docs/experiment_description.md)** --
what the experiment does, what is matched, and how to read the results.

Original plan: [`docs/experiment_plan.md`](docs/experiment_plan.md). Decisions
that override it: [`docs/implementation_answers.md`](docs/implementation_answers.md).
Code: [`src/molgen3D/experiments/bindgpt_fsq/`](../src/molgen3D/experiments/bindgpt_fsq).

## Sequence format

```
<LIGAND> {canonical isomeric SMILES} <XYZ> {repeated graph + geometry} <EOS>
```

The left side is the plain canonical isomeric SMILES in BindGPT's element-level
tokens. The right side repeats the molecular graph in the FSQ tokenizer's
bracketed-atom vocabulary with each arm's geometry interleaved in its native
position:

```
coord   ... <XYZ> [C] -1 .608 -1 .880 0 .149 [C] -1 .344 -0 .501 0 .764 [C@@H] ... 1 ...
fsq     ... <XYZ> [C] <977> [C] <2726> [C@@H] <547> 1 <1344> [C] <1624> ...
```

`coord` spends six tokens on every atom-bearing graph token and nothing on the
others; `fsq` spends one code token on **every** graph token, atom or not,
because that is what the frozen tokenizer emits. Repeating the graph is a
deliberate departure from published BindGPT's coordinate-only suffix — the FSQ
codes cannot be lifted out of the graph without destroying the representation —
and both arms pay the cost, so delimiters, graph vocabulary and graph-token loss
treatment stay identical. Measured on GEOM: `coord` averages 248 tokens per
example, `fsq` 136.

Both arms share one 5,530-token union vocabulary, including a `.` compatibility
token for disconnected rows retained by the existing SFT pipeline, so embedding shapes, output
shapes and parameter counts are identical and one seed gives bitwise-identical
initial weights (verified: same `init_parameter_sha256`). During generation an
arm mask forbids the other arm's geometry tokens.

## Pinned inputs

| Field | Value |
| --- | --- |
| `fsq_checkpoint` | `/mnt/weka/fgeikyan/fsq/new_checkpoints/full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt` |
| `fsq_config` | `d_model=1024`, 8 encoder + 8 decoder layers, `levels=[2]*12` → 4,096 codes, 202,011,535 parameters, epoch 0 end / `global_step=185757` |
| `bindgpt_tokenizer` | `insilicomedicine/bindgpt_finetuned`, tokenizer and config files only, hashed in `tokenizer_audit/bindgpt_smiles_tokens.json` |
| `geom_train_source` | all 5,745,689 rows of `/data/vtarasov/geom_cartesian_v3/processed_strings/train`, aligned row-for-row with `/mnt/weka/fgeikyan/fsq/geom_fsq_4096/processed_strings/train`; no identity-overlap filtering is added |
| `geom_valid_source` | all 716,818 rows of `/data/vtarasov/geom_cartesian_v3/processed_strings/valid` |
| `test_set` | `/mnt/weka/fgeikyan/tokenizer/distinct_smi_filtered_13.pickle` — 1,000 molecules, 106,777 reference conformers (GeoMol / Torsional-Diffusion split) |
| `fsq_checkpoint_sha256` | `270d4cab25f723662005163357f413338e2836c5ef08ae0a9aa3cca1d0c538ff` |
| `moldiff_evaluator` | vendored at commit `db62fa1b`, see `vendor/moldiff/PROVENANCE.json` |

The full decimal `geom_cartesian_v3` corpus is readable. Its six training
shards contain exactly 5,745,689 rows and align with the historical FSQ shards.
Extraction verifies canonical SMILES row-by-row. We add no molecular-identity
exclusions: overlaps and disconnected SMILES are retained, matching the
existing SFT data policy. Both arms therefore consume the same historical
training population.

## Running it

```bash
cd bindgpt_geom_fsq/scripts
./submit_all.sh                    # preparation, manifests, tokenizer audit
START=examples ./submit_all.sh     # example arrays, packing, training, evaluation
```

Run the second command after the manifests job completes. Shard counts only
exist after extraction, and the cluster compute nodes do not provide `sbatch`,
so the example fan-out is intentionally submitted from the login node. Use
`START=pack ./submit_all.sh` to reuse already-built examples.

Runs on the `premium` partition (H100, 7-day walltime) by default; override with
`PARTITION=...`. Slurm dependencies gate every stage on its inputs, each training
seed is its own job, conditional sampling and example building run as job arrays
that never share an output file, and `gpu09` is excluded everywhere (it wedges
CUDA jobs while still reporting healthy). `common.sh` captures `scontrol`, `nvidia-smi`,
module list, `pip freeze` and git status into `environment/` for every job.

Individual stages, if you want them by hand:

```bash
export PYTHONPATH=$PWD/src:$PWD/bindgpt_geom_fsq/vendor
CLI="python -m molgen3D.experiments.bindgpt_fsq.cli"

$CLI prepare-tokenizers                     # pin public files, build union vocab
$CLI extract --split train                  # decimal conformers out of the corpus
$CLI manifests --workers 64                 # identity/overlap audit; no overlap filtering
$CLI audit-tokenizers --conformers 10000    # both coordinate tokenizers
$CLI build-examples --split train --shard 0 # paired coord + fsq examples
$CLI pack --split train                     # memory-mappable token blobs
$CLI train --config configs/fsq_seed17.yaml
$CLI generate-unconditional --config ... --checkpoint ...
$CLI generate-conditional  --config ... --checkpoint ...
$CLI metrics-unconditional --attempts ... --tag ... --both-references
$CLI metrics-conformer --generations ... --tag ...
$CLI report
```

## Integrity gates

The pipeline stops rather than producing a quietly wrong result when:

- the extracted GEOM train count differs from the corpus's recorded total;
- train/validation/test identity intersections are always reported but never
  removed, matching the existing SFT data policy;
- the two arms' example counts differ, at build time or after packing;
- any packed sequence exceeds the 2,048-token context — the builder excludes
  over-length conformers and records them, the trainer never truncates;
- the loss goes non-finite;
- a frozen FSQ parameter moves, checked by SHA-256 over all 202M parameters
  before and after every stage that touches the tokenizer.

## Results

Both tables and the full provenance land in `final_report.md`, written by
`$CLI report` from whatever metric files exist. The FSQ tokenizer's overlap
status is stated at the top of that report: the pinned tokenizer was trained on
a merged corpus whose GEOM component is this train split, so
`manifests/overlap_report.json` → `fsq_tokenizer_audit` **is** the
FSQ-training / GEOM-test identity audit. If it is not empty, the report labels
the run `tokenizer-overlap` and the result shows end-to-end performance rather
than clean held-out generalization.

## Directory layout

```
bindgpt_geom_fsq/
  docs/            the plan and the implementation decisions
  configs/         one frozen YAML per arm per seed
  manifests/       molecules.csv, graph_index.jsonl, overlap_report.json, hashes.json
  tokenizer_audit/ pinned public files, union tokenizer, both round-trip audits
  data/            extracted/ examples/ packed/
  checkpoints/     <arm>/seed_<n>/step-*-{025,050,075,100}pct and -final
  generations/     unconditional/ and conformer/, every raw attempt preserved
  metrics/         unconditional/ and conformer/, plus per-molecule tables
  vendor/moldiff/  pinned MolDiff evaluator with PROVENANCE.json
  logs/  environment/  final_report.md
```
