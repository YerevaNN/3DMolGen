# BindGPT + FSQ Implementation Answers

This memo answers the ten implementation questions raised about the proposed GEOM-DRUGS
experiment. It records current decisions without replacing the longer experiment plan.

## 1. FSQ Serialization

Preserve the native order of the FSQ graph and coordinate tokens. Do not extract, reorder, or move
FSQ codes into an artificial atom-only block.

Use a paired sequence with:

- canonical SMILES on the left; and
- the graph repeated on the right with its geometric representation interleaved.

Schematic FSQ example:

```text
[SMILES]CC[S@+](O-)...[/SMILES]
[CONFORMER][C]<2710>[C]<2687>[S@+]<3004>(<932>[O-]<2104>)<2843>...[/CONFORMER]
```

The Coord arm must use the same paired outer format and repeat the graph on the right as well. Its
right side interleaves the original BindGPT scalar coordinate tokens after atom-bearing graph
tokens. This prevents graph repetition from being unique to the FSQ arm.

This is not the literal published BindGPT sequence format, because published BindGPT has a
coordinate-only suffix. It is a controlled adaptation required to preserve the native FSQ
representation. The repeated graph, delimiters, graph vocabulary, and graph-token loss treatment
must be identical between arms.

For the primary experiment, let both models generate the complete right-hand stream. Record a
failure if its graph disagrees with the left-hand SMILES. A forced-copy right-side graph can be a
later ablation, not an undocumented correction.

## 2. Language-Model Architecture

Use the public BindGPT GPT-NeoX architecture, not Qwen and not an approximate Qwen flavor.

Instantiate it from configuration without loading pretrained weights:

```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("insilicomedicine/bindgpt_finetuned")
config.vocab_size = len(experiment_tokenizer)
model = AutoModelForCausalLM.from_config(config)
```

Expected configuration:

- GPT-NeoX decoder;
- 15 layers;
- hidden size 768;
- 12 attention heads;
- intermediate size 3,072;
- rotary positional embeddings;
- context length 2,048; and
- untied input/output embeddings.

Initialize both arms from scratch with the same initialization seed. A shared union vocabulary is
preferred so both models have identical embedding shapes and total parameter counts. It should
contain the shared graph tokens, BindGPT scalar-coordinate tokens, and FSQ code tokens. Apply
arm-specific valid-token masks during generation.

The unreleased original BindGPT trainer is not required. Hugging Face Transformers with
FlashAttention 2 and DeepSpeed/FSDP is sufficient to reproduce the public architecture and reported
training recipe.

## 3. Exact Qwen GEOM Split

The Qwen was trained on the original GEOM drugs split, not revisited. Check out the codebase for it.

## 4. FSQ Training and Test Overlap

Do not retrain FSQ.

Use the existing frozen production FSQ checkpoint. Its training corpus was explicitly cleaned to
exclude all benchmark test identities, including the GEOM-DRUGS test set.

The two BindGPT language models receive no external pretraining. The FSQ tokenizer is an existing
pretrained, test-clean component and is intentionally part of the method.

Archive the existing exclusion manifest or reproduce the identity audit:

```text
FSQ training InChIKeys intersect GEOM test InChIKeys = empty
FSQ training canonical isomeric SMILES intersect GEOM test SMILES = empty
```

This is a provenance check, not a request for new FSQ training.

## 5. Pinned FSQ Checkpoint

`last-v2.ckpt` should be it. try to find if the current codebase uses it, but i'm pretty sure it corresponds. we should have somewhere a normal name for it, but anyways.

## 6. Unconditional Evaluation

Confirmed in scope, even though it requires new code.

Implement:

- unconditional prompting and generation;
- parsing of the left and repeated right molecular graphs;
- graph-consistency and coordinate-decoding failures;
- raw-attempt preservation;
- QED, normalized SA, and Lipinski metrics;
- bond-length, bond-angle, and dihedral distributions;
- bond, bond-pair, bond-triplet, ring, and ring-size distributions; and
- comparison with the frozen held-out GEOM reference distribution.

Port or pin the released MolDiff evaluator. SciPy's `jensenshannon` returns Jensen-Shannon
**distance**, so report JS distance rather than squared divergence.

BindGPT does not publish the exact identity list behind its stated GEOM validation reference, while
MolDiff's released notebook defaults to test. Use the held-out reference set already used by the
existing Qwen unconditional protocol if one exists. Otherwise freeze either validation or test
before generation and document the choice. Do not claim an exact reproduction of unpublished
BindGPT evaluation code.

## 7. GEOM Coverage Threshold

Use only the regular GEOM-DRUGS headline threshold:

```text
delta = 0.75 A
```

Preserve the existing evaluator's strict `< 0.75 A` convention. Report Recall/Precision COV and
AMR, with mean and median across test molecules.

Do not introduce a new threshold grid, `0-2 A` curve, or coverage AUC in this experiment.

## 8. Batching and Packing

Stay close to BindGPT.

BindGPT documents variable-length sequences, padding/loss-enabled tokens, gradient accumulation,
and a target of at least 1.6M enabled tokens per optimizer step. It does not document fixed-block
packing.

Use:

- intact molecule-conformer examples;
- length bucketing;
- dynamic padding;
- attention and loss masks; and
- gradient accumulation toward the BindGPT enabled-token target, subject to cluster capacity.

Do not use `ChunkPacker` if it splits an example across 4,096-token block boundaries. Both arms see
the same molecule-conformer examples for the same number of epochs, while token counts, throughput,
and possibly optimizer-step counts are recorded as consequences of the representation.

## 9. Seeds

Remove the proposed multi-seed grid.

Use one fixed training seed and one fixed sampling seed, preferably the existing project defaults.
Use the same recorded seeds for both arms. Additional seeds are a follow-up only if the result is
marginal or training is unstable; they are not part of the initial experiment or a published
BindGPT/MCF requirement.

## 10. SMILES Randomization

Disable SMILES randomization for both arms.

Use canonical isomeric SMILES during training and evaluation. Do not apply atom-order augmentation.
This keeps both inputs matched and remains within the frozen FSQ decoder's training distribution.

## Remaining Unanswered Items

Two operational facts still need to be recovered:

1. the exact Qwen GEOM split, conformer cap, and relationship to the regular MCF/GeoMol test set;
2. the immutable production FSQ checkpoint that produced the intended existing results.

Everything else above is a fixed implementation decision.
