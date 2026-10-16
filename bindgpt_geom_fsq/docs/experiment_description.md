# What this experiment does

**One question.** Take the BindGPT architecture, train it from scratch on
GEOM-DRUGS, and change nothing except how atomic coordinates are written down.
Does the learned FSQ coordinate representation beat BindGPT's original scalar
coordinate tokens?

Two runs, identical in every other respect:

**Naming.** The two arms are called **Cartesian** and **CoordToken**, matching
the project's own terminology (`geom_cartesian_v3`, `encode_cartesian_v2`, and
the CoordToken/FSQ tokenizer). Internally, and in every artifact path, config
and metric filename, the keys remain `coord` and `fsq`.

| Run | Coordinates written as | Decoded by |
| --- | --- | --- |
| `bindgpt_coord_geom_scratch` | BindGPT integer + fraction tokens | arithmetic |
| `bindgpt_fsq_geom_scratch` | codes from the frozen FSQ tokenizer | the frozen FSQ decoder |

Neither language model is pretrained. Neither sees any corpus but GEOM-DRUGS.
The FSQ tokenizer is a separate pretrained component, frozen throughout.

---

## 1. The sequence format

Every training example is one molecule-conformer pair:

```
<LIGAND> {canonical isomeric SMILES} <XYZ> {molecular graph, geometry interleaved} <EOS>
```

Left of `<XYZ>`: the plain canonical isomeric SMILES in BindGPT's own
element-level tokens (`Cl` and `Br` are single tokens, not two characters),
with one compatibility token, `.`, added so the disconnected rows present in
the existing SFT data are retained rather than filtered.

Right of `<XYZ>`: the molecular graph **repeated**, with each arm's geometry
interleaved where that representation naturally puts it.

```
coord   <XYZ> [C] -1 .608 -1 .880 0 .149  [C] -1 .344 -0 .501 0 .764  [C@@H] ... 1 ...
fsq     <XYZ> [C] <977>                   [C] <2726>                  [C@@H] <547> 1 <1344> ...
```

`coord` spends six tokens on each atom (two per axis: an integer token and a
three-digit fraction token) and nothing on structural tokens like `(` or `=`.
`fsq` spends exactly one code token on **every** graph token, atom or not.

### Why the graph is repeated

Published BindGPT puts coordinates in a bare suffix after `<XYZ>`. That is
impossible for FSQ: its tokenizer emits one code per graph token including
non-atom tokens, and its decoder is a transformer that reads the entire token
sequence at once. Lifting the codes into an atom-only block would destroy the
representation.

A worked example makes the constraint concrete. Take `CN(CCO)c1cnccn1` — 11
heavy atoms, 15 graph tokens — from the validation split.

**Published BindGPT.** Coordinates are a bare float suffix; you recover which
number belongs to which atom by counting in SMILES order.

```text
<LIGAND>CN(CCO)c1cnccn1<XYZ>1.284 1.698 -0.254 0.733 0.369 -0.153 1.551 -0.800
-0.441 3.060 -0.572 -0.397 3.530 -0.099 0.842 -0.603 0.152 -0.070 -1.574 1.176
0.025 -2.852 0.911 0.152 -3.229 -0.366 0.184 -2.295 -1.377 0.066 -1.001 -1.131
-0.065<EOS>
```

33 floats for 11 atoms.

**Ours, BindGPT-Cartesian.** The graph is repeated after `<XYZ>`, and six scalar
tokens follow each *atom* token. `(`, `)` and the ring digits stand bare — they
carry no geometry.

```text
<LIGAND> C N ( C C O ) c 1 c n c c n 1 <XYZ>
[C] 1 .284 1 .698 -0 .254  [N] 0 .733 0 .369 -0 .153  (
[C] 1 .551 -0 .800 -0 .441  [C] 3 .060 -0 .572 -0 .397  [O] 3 .530 -0 .099 0 .842  )
[c] -0 .603 0 .152 -0 .070  1  [c] -1 .574 1 .176 0 .025  [n] -2 .852 0 .911 0 .152
[c] -3 .229 -0 .366 0 .184  [c] -2 .295 -1 .377 0 .066  [n] -1 .001 -1 .131 -0 .065  1
<EOS>
```

**Ours, BindGPT-CoordToken.** One code follows *every* graph token.

```text
<LIGAND> C N ( C C O ) c 1 c n c c n 1 <XYZ>
[C] <195>  [N] <3778>  ( <3702>  [C] <2316>  [C] <2735>  [O] <3732>  ) <1157>
[c] <3778>  1 <1737>  [c] <34>  [n] <3952>  [c] <2755>  [c] <3969>  [n] <12>  1 <2336>
<EOS>
```

**Why that cannot be a suffix: 15 codes for 11 atoms.** Four of them —
`( <3702>`, `) <1157>`, `1 <1737>`, `1 <2336>` — belong to non-atom tokens.
There is no atom to hang them on, so "one triplet per atom in SMILES order"
cannot express the representation. Separately, the decoder is a transformer over
the whole interleaved sequence, so even the atom codes are not independently
decodable. Repeating the graph is the only layout the tokenizer can express, and
paying the identical cost in the Cartesian arm is what keeps the two comparable.

The token-count consequence, measured over the training set: the Cartesian arm
averages 248 tokens per example against CoordToken's 136, a 1.83x difference
that shows up as tokens processed and GPU-hours, never as a difference in
examples seen.

So instead of bending FSQ to fit the published layout, **both** arms repeat the
graph. The repetition is a cost, and it is paid equally. Delimiters, graph
vocabulary, and graph-token loss weighting are byte-identical across arms; only
the geometry tokens differ. This is a deliberate, documented departure from
published BindGPT and is why the run is not a BindGPT reproduction.

### One vocabulary, two masks

Both arms share a single 5,530-token vocabulary:

| block | size | notes |
| --- | ---: | --- |
| specials | 5 | `<PAD> <EOS> <UNK> <LIGAND> <XYZ>` |
| SMILES | 254 | BindGPT's 253 tokens plus `.` for existing disconnected SFT rows |
| graph | 192 | the FSQ tokenizer's bracketed-atom vocabulary |
| coordinate integers | 200 | `0`..`99`, `-0`..`-99` — shared with SMILES |
| coordinate fractions | 1,000 | `.000`..`.999` |
| FSQ codes | 4,096 | `<0>`..`<4095>` |

Sharing the vocabulary means both models have identical embedding shapes,
identical output-head shapes, identical parameter counts, and — from one seed —
**bitwise-identical initial weights**. Verified: both arms report the same
`init_parameter_sha256`. The transformer body is therefore matched by
construction, not by careful copying.

At generation time an arm mask forbids the other arm's geometry tokens: `coord`
cannot emit FSQ codes, `fsq` cannot emit fraction tokens. Integers stay legal in
both, because BindGPT's tokenizer merges adjacent SMILES ring-closure digits
into them (`...c2ccccc21)` gives the single token `21`) — and a coordinate always
costs an integer *and* a fraction, so denying fractions already makes scalar
coordinates unspellable.

---

## 2. The model

BindGPT's public 108M GPT-NeoX configuration, read from the pinned public config
and instantiated with **no pretrained weights**:

- 15 layers, hidden 768, 12 heads, intermediate 3,072
- rotary embeddings (`rotary_pct=0.25`), no parallel residual
- 2,048-token context, untied input/output embeddings

| | parameters |
| --- | ---: |
| transformer body | 106,319,616 |
| input embedding | 4,247,040 |
| output embedding | 4,247,040 |
| **total** | **114,813,696** |

The body is the 106M BindGPT reports; the total is larger because the union
vocabulary is 5,530 tokens rather than BindGPT's 1,287. Both counts are reported
so the vocabulary effect stays visible.

---

## 3. Data

**Test set.** The frozen 1,000-molecule GEOM-DRUGS benchmark set — the
GeoMol / Torsional-Diffusion split, 106,777 reference conformers, mean 106.8 per
molecule. This is the same set the project's published COV/MAT numbers use, and
the same set ET-Flow, MCF-S and MCF-B report on.

**Training set.** All 5,745,689 conformers in
`/data/vtarasov/geom_cartesian_v3/processed_strings/train`. These six decimal
shards align row-for-row with the historical
`geom_fsq_4096/processed_strings/train` shards. Extraction asserts the total and
checks canonical SMILES on every aligned row.

We add no train/validation/test identity exclusions. Overlap is audited and
reported, never used as a filter; disconnected SMILES are retained as they were
in the existing SFT dataset. Both arms consume exactly the historical training
population, so runs are directly comparable by data rows.

Validation is all 716,818 rows from the corresponding decimal `valid` split.
The extracted count is asserted, so a partial scan fails the job.

**Conformer selection.** Conformers were capped at 30 per molecule and ranked by
energy when that corpus was built. That predates this experiment, cannot be
undone without the raw tree, and applies identically to both arms.

**Splitting is by molecular identity, never by conformer.** Identity is keyed on
standardized isomeric canonical SMILES and full InChIKey. All conformers of one
molecule share the coordinate-stripped graph string, so that string is the exact
grouping key and RDKit runs once per molecule rather than once per conformer.

**Training-data policy.** Every parseable source row is retained, including
disconnected SMILES and all train/validation/test identity intersections.
Overlap and fragment counts are audit metadata only. A row can still fail both
arms together if it cannot be rendered in either representation (for example,
an unparseable graph, coordinates outside ±99 Å, a graph over the frozen FSQ
tokenizer's 218-token ceiling, or a sequence over the 2,048-token context); every
such representation failure is counted.

---

## 4. What is matched, and what is merely measured

**Matched — the comparison rests on these:**

- the same molecule-conformer examples, built in one pass so an exclusion in
  either arm excludes the conformer from both;
- the same example order, from one seeded permutation of the global example
  index, depending only on `(n_examples, seed, epoch)` and never on length;
- the same examples in the same optimizer step — a global batch is a fixed slice
  of that permutation, and length bucketing happens *only* when splitting a
  global batch into micro-batches, so padding shrinks but membership never moves;
- the same optimizer, schedule, learning rate at each step, epoch count, and
  decoding settings;
- bitwise-identical initial weights.

**Measured, not matched — these are outcomes, not variables:**
tokens processed, padding, throughput, GPU-hours, peak memory.

---

## 5. Training recipe

BindGPT's GEOM supervised-fine-tuning recipe, applied from random init:

| | |
| --- | --- |
| epochs | 1 over the frozen example manifest |
| optimizer | AdamW, lr 5e-4, weight decay 1e-2 |
| schedule | 100 warmup steps, then cosine decay |
| clipping | global norm 1.0 |
| precision | bfloat16 |
| token loss weight | 1.0 for every non-padding token |
| global example batch | 2,048 examples (≈2,800 steps/epoch) |
| SMILES randomization | **off** |
| coordinate rotation | **off** |
| seeds | one training seed (17), one sampling seed (2023) |

Randomization is off because canonical isomeric SMILES keeps the two arms
matched and stays inside the frozen FSQ decoder's training distribution;
randomized atom order would penalize the CoordToken arm for a reason unrelated to
representation quality.

Uniform token weighting is exact rather than approximate: the global count of
*scored* tokens (which excludes both padding and the one position per row that
next-token loss drops) is all-reduced before the backward pass, so after DDP
averages gradients across ranks the result is precisely the global token mean.

Checkpoints are saved at 25%, 50%, 75% and 100% of the epoch, so
under-convergence is visible from validation loss without touching test metrics.

---

## 6. Evaluation

### Evaluation 1 — unconditional generation (BindGPT/MolDiff-style)

Prompt each model with `<LIGAND>` alone. Sample at temperature 1.0, top-p 1.0,
top-k 50, no beams, no guidance, no energy reranking, until 1,000 valid connected
molecules exist. Every raw attempt is written out before any parsing.

Metrics come from the **released MolDiff evaluator**, vendored at commit
`db62fa1b` with per-file hashes, plus its notebook's JS step ported into a
script. Bin widths and the trimming rule are MolDiff's, unchanged. The reported
number is SciPy's `jensenshannon` — the Jensen-Shannon **distance**, not the
squared divergence.

Reported: mean QED, normalized SA and Lipinski count; JS distance of bond-length,
bond-angle and dihedral distributions; JS of bond types, bond pairs, bond
triplets, bonds-per-atom, ring counts and ring sizes; intersecting-ring atoms;
plus raw validity, connectivity, coordinate-decode success, uniqueness and the
GPU time to reach 1,000 molecules.

Reference distribution: the frozen 1,000-molecule test set is primary, the GEOM
validation split is a secondary table. The two are never combined.

**Sanity check on the metric code:** reference-against-itself gives exactly 0.0
on every JS distance, and two disjoint 300-molecule GEOM samples give 0.02–0.17.
That second number is the finite-sample noise floor — differences below it mean
nothing.

### Evaluation 2 — the regular GEOM-DRUGS conformer protocol

For each of the 1,000 test molecules with `K` references, prompt with
`<LIGAND>{SMILES}<XYZ>` and generate `2K` conformers, allowing at most `4K`
attempts. No optimization, no energy ranking, no RMSD or diversity filtering.

RMSD is the project's existing evaluator: heavy-atom only, alignment-based,
symmetry-aware, strict `<` threshold. That code already reproduces four
published models on their own released conformers to 0.3–1.2 COV points.

Reported at the single headline threshold **δ = 0.75 Å**: Recall COV and AMR,
Precision COV and AMR, each as mean and median over molecules, plus the full
per-molecule table stratified by heavy-atom count, rotatable bonds, reference
count and ring count. No threshold grid, no coverage curve, no AUC.

A molecule that fails to yield `2K` conformers is reported as a generation
failure, never silently dropped. Above 1% failed molecules the headline table is
not eligible until the failure policy is resolved for both arms.

---

## 7. The tokenizers, measured

Both coordinate tokenizers audited on the same 10,000 held-out validation
conformers.

| | coord (BindGPT scalars) | fsq (frozen checkpoint) |
| --- | ---: | ---: |
| geometry tokens, mean | 198.8 | 86.5 |
| geometry tokens, max | 583 | 266 |
| **compression vs coord** | 1.00× | **2.30×** |
| coordinate MAE | 0.00025 Å | 0.0334 Å |
| aligned RMSD, mean | 0.00050 Å | 0.0733 Å |
| RMSD p90 / p99 | 0.00054 / 0.00057 Å | 0.094 / 0.127 Å |
| undecodable rate | 0.0% | 0.0% |
| conformers excluded | 0 out of range | 0 over 218 tokens |

The coord tokenizer is checked against the **public BindGPT tokenizer itself**,
not against a specification: 10,000 coordinate sequences and 10,000 SMILES, with
**0 mismatches** in either.

**Read the FSQ row carefully.** 0.0733 Å is a *floor* on the CoordToken arm's AMR — no
amount of language-model accuracy decodes below it. That is roughly 10% of the
0.75 Å coverage threshold, so it should not by itself decide COV. But if the two
arms land within ~0.07 Å of each other on AMR, the reconstruction floor is a live
confound and the AMR comparison cannot carry a conclusion. An FSQ win larger than
that margin is real.

Sequence lengths for whole training examples, measured on GEOM: `coord` averages
248 tokens (max 938), `fsq` averages 136 (max 572).

---

## 8. The FSQ checkpoint

| | |
| --- | --- |
| path | `.../full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt` |
| SHA-256 | `270d4cab25f723662005163357f413338e2836c5ef08ae0a9aa3cca1d0c538ff` |
| architecture | `d_model=1024`, 8 encoder + 8 decoder layers, `levels=[2]*12` → 4,096 codes |
| parameters | 202,011,535 |
| state | epoch 0 end, `global_step=185757` |
| training corpus | the merged multi-dataset corpus; identity overlap is measured from the readable GEOM rows and is not filtered |

Frozen throughout: no gradient reaches it, and a SHA-256 over all 202M
parameters is compared before and after every stage that touches it.

**Overlap status — measured.** The `manifests` stage audits identity
intersections at full scale on both keys and writes the collision tables. It
never filters them: the experiment matches the existing SFT data policy. Actual
result on this corpus:

| pair | InChIKey | canonical SMILES |
| --- | ---: | ---: |
| train ∩ benchmark test | **22** | 11 |
| train ∩ validation | 586 | 241 |
| validation ∩ benchmark test | 5 | 1 |

So 22 of the 1,000 test molecules (2.2%) have conformers in training. The run is
labelled `tokenizer-overlap` and describes end-to-end performance, not clean
held-out generalization. It is the *same* leakage the Qwen SFT baseline has,
which is the point — but say "same leakage as the SFT baseline", never "clean
split". The per-molecule table is saved, so re-check the headline numbers with
those 22 molecules dropped before publishing anything.

---

## 9. What would make this experiment wrong, and how it stops

The pipeline halts rather than producing a quietly wrong number when:

- the extracted GEOM train count differs from the corpus's recorded total;
- the two arms' example counts differ, at build time or after packing;
- a packed sequence exceeds the 2,048-token context — the builder excludes
  over-length conformers and records them; the trainer never truncates;
- a rank receives no examples, or the global batch does not divide across ranks;
- the loss goes non-finite;
- a frozen FSQ parameter moves.

---

## 10. Honest limits

- **This is not a BindGPT reproduction.** The published model was pretrained at
  Uni-Mol scale first; these are scratch runs. The sequence layout repeats the
  graph, which published BindGPT does not. Published BindGPT is contextual
  reference only.
- **The unconditional evaluation is MolDiff-*style*.** BindGPT's evaluation
  repository is not public; the MolDiff evaluator is pinned as the compatibility
  implementation, and every local choice is written down.
- **The FSQ tokenizer is 202M pretrained parameters** trained on a large
  multi-dataset corpus. The language models get no pretraining, but the
  coordinate channel does. That is the method, not a leak — but it means the
  result is about *this* FSQ checkpoint, not about FSQ as an idea.
- **The Cartesian arm is gradient-clipped far more than the CoordToken arm.** Pre-clip
  gradient norms against the recipe's 1.0 threshold, post-warmup, by phase:

  | steps | coord clipped | coord median | fsq clipped | fsq median |
  | --- | ---: | ---: | ---: | ---: |
  | 100-1k | 24.7% | 0.83 | 18.0% | 0.74 |
  | 1k-2k | **96.0%** | 1.43 | 0.0% | 0.55 |
  | 2k-end | **71.4%** | 1.55 | 0.0% | 0.30 |
  | overall | **64.5%** | 1.20 | 6.0% | 0.54 |

  Neither arm is unstable — both losses fall monotonically to their minima and
  the CoordToken arm's validation loss (0.993) sits just above its training loss
  (0.928), so there is no overfitting either. But the Cartesian arm spends most of
  training with its gradient rescaled, by a factor around 1/1.4 to 1/1.55 in the
  second half, while the CoordToken arm is never clipped after step 1,000.

  This is the matched recipe faithfully applied: BindGPT specifies global-norm
  clipping at 1.0 and both arms get exactly that, so the difference is a
  property of the two representations, not a confound we introduced. It still
  limits one conclusion: if coord underperforms, "coord was effectively trained
  at a lower learning rate" is a competing explanation this run cannot rule out.
  The clean follow-up is to rerun both arms with a clip threshold high enough
  that neither binds and check the direction holds. Do not quote a coord-vs-FSQ
  gap as purely representational without it.

- **One seed.** A marginal or direction-inconsistent result is a reason to add
  seeds 23 and 42, not to report the one that looks better.
- **Better distribution scores do not imply better coverage**, higher precision
  with lower recall is a tradeoff and not a win, and shorter sequences plus lower
  GPU cost are systems results distinct from representation quality.

---

## 11. Where things live

```
bindgpt_geom_fsq/
  docs/            this file, the original plan, the implementation decisions
  configs/         one frozen YAML per arm per seed — all scientific settings
  manifests/       molecules.csv, overlap_report.json, hashes.json
  tokenizer_audit/ pinned public files, union tokenizer, both audits, comparison
  data/            extracted/ examples/ packed/
  checkpoints/     <arm>/seed_<n>/step-*-{025,050,075,100}pct and -final
  generations/     every raw attempt, successful or not
  metrics/         both tables plus per-molecule results
  vendor/moldiff/  pinned MolDiff evaluator with PROVENANCE.json
  final_report.md  both result tables, overlap status, protocol deviations
```

Code: `src/molgen3D/experiments/bindgpt_fsq/`. Contracts:
`tests/experiments/` (49 tests). Job graph:
`bindgpt_geom_fsq/scripts/submit_all.sh`, running on the `premium` partition with
`gpu09` excluded everywhere. Submission has two login-side phases: the first
builds the extracted data, manifests, and tokenizer audit; after manifests
finish, `START=examples` sizes and submits the example arrays and chains all
remaining jobs. This split is required because shard counts are data-dependent
and compute nodes do not provide the `sbatch` client.

## Results

_To be filled in once both arms have trained and been evaluated._
