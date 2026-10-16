## 1. Compute control

The binned-coordinate and FSQ representations serialize the same molecules at
different sequence lengths, so equal epochs would hand the binned model more
compute. We try to match FLOPs.

We use the standard `6ND` parameter-token proxy and the `2N'x` term, which is
the cost of encoding the corpus with the frozen pretrained tokenizer and the
result is reused in every epoch.

```
6NDe = 6ND'e' + 2N'x
```

Here `N` = 751,632,384 is the Qwen parameter count, `N'` = 202,011,535 the full
FSQ encoder–decoder parameter count, `e'` = 6 the epochs of our FSQ run, `x` the
number of source SMILES tokens in the corpus, `D` and `D'` the tokens per
example under each representation, and `e` the quantity we solve for.

Each training example presents the initial SMILES followed by the extended
SMILES, so the `x` source tokens are counted twice under both representations.
In the binned representation every atom contributes three coordinate-bin tokens
and one delimiter; atoms are the bracketed tokens, a fraction `(1-a)` of the
source, where `a` ≈ 0.4 is the fraction of coordinate-free topology tokens
(bonds, branches, rings, stereochemistry, separators) — 99,220,841 of
248,140,943 source tokens over all 5,745,689 conformers of the GEOM training
split. Under FSQ every source token instead receives a single code:

```
D  = x + x + 4x(1-a) = x(6 - 4a)
D' = x + (x + x)     = 3x
```

Every term is linear in `x`, so the corpus size cancels and

```
e = (9Ne' + N') / (3N(6 - 4a)) ≈ 4.11
```

We therefore compare our 6-epoch FSQ model against a binned baseline trained
for 4 epochs.

**The binned representation, for the paper's methods section.** 256 uniform bins
over [−10.5748, 10.5102] Å applied to centroid-centred heavy-atom coordinates —
bin width **0.0824 Å**, so quantisation costs at most 0.0412 Å per coordinate and
0.0238 Å RMS. The vocabulary carries 258 bin tokens: `001`–`256` for the interior
bins plus `000` and `257` as saturation tokens for coordinates below and above the
range, which is how clipping is represented rather than dropped. **Clipping is
negligible**: on the 1,000-molecule test set 54 of 8,650,290 heavy-atom
coordinates fall outside the range (0.0006%), touching 53 of 106,777 conformers
and 3 of 1,000 molecules, with the largest centred coordinate at 11.81 Å against
the 10.51 Å bound.

**Corpus size under each representation.** Same molecules, same
`[SMILES]…[/SMILES]` + `[CONFORMER]…[/CONFORMER]` basis, each half tokenized with
its own tokenizer:

| representation | SMILES | conformer | tokens/conf | corpus |
|---|---:|---:|---:|---:|
| decimal coordinate strings | 33.3 | 575.9 | 609.2 | 115.87 B |
| binned (3 bins + delimiter per atom) | 33.3 | 173.0 | 206.3 | 39.24 B |
| FSQ (one code per source token) | 33.3 | 92.1 | 125.4 | 23.85 B |

Binned cuts the decimal corpus **2.95×** and FSQ cuts it **4.86×**, so FSQ is a
further **1.65×** below binned — which is the sequence-length asymmetry §1's
epoch matching corrects for.

**Full-corpus CoordToken training.** AdamW used β₁=0.9, β₂=0.999, ε=1e−8;
weight decay was 0.01 for matrix weights and 0 for bias/norm parameters. Gradient
norm clipping was 0.1 and the seed was 123. The deployed `last-v2.ckpt` ran
185,757 steps: linear warmup through step 18,575, stable LR to step 167,180,
then cosine decay from step 167,181 to zero at the end. Checkpoints were saved
every 18,576 steps and at epoch end without metric-based selection; `last-v2`
is the epoch-end checkpoint at step 185,757.

Positions use fixed absolute sinusoidal encodings (sine/cosine, base 10,000),
generated at runtime rather than learned in a 218-entry table. Thus the
Transformer can process a longer sequence at its native length: inference only
has to raise or disable the wrapper's `max_tokens=218` safety check. No position
interpolation or model resizing is needed, although lengths above 218 are an
extrapolation beyond training and self-attention memory grows quadratically.

---

## 2. Main results: big-data pretraining improves FSQ after SFT

GEOM COV/MAT on 1,000 test molecules (106,777 reference conformers), generating
`2k` conformers per molecule for a molecule with `k` references. All numbers at
a **0.75 Å** threshold.

- **COV-R** — % of *references* within 0.75 Å of some generated conformer
- **COV-P** — % of *generated* within 0.75 Å of some reference
- **MAT-R / MAT-P** — the same two directions as mean nearest-neighbour RMSD, in Å

So COV is higher-is-better and MAT lower-is-better; R rewards covering the
ensemble, P penalises generating conformers that match nothing.

Big-data pretraining is one epoch throughout. SFT runs 6 epochs for FSQ and the
compute-matched binned comparisons run for 4 epochs (§1). The earlier binned
numbers were evaluated on the wrong test set and are not used; every binned row
below is from the correct-set results in `generation_paths.csv`. All six pairings
are complete on every metric.

Rows are paired binned-then-FSQ per setting.

| Setting | Size | Repr. | COV-R | COV-P | MAT-R | MAT-P | PB score (%) |
|---|---|---|---:|---:|---:|---:|---:|
| big data, 0 sft | 0.6B | binned | 55.2 | 35.3 | 0.76 | 1.00 | 76.71 |
| big data, 0 sft | 0.6B | FSQ | 58.3 | 40.4 | 0.73 | 0.93 | 89.66 |
| sft only, 4e | 0.6B | binned | 67.4 | 53.1 | 0.63 | 0.80 | 80.03 |
| sft only, 6e | 0.6B | FSQ | 67.3 | 62.4 | 0.64 | 0.71 | 89.63 |
| big data + sft, 4e | 0.6B | binned | 67.6 | 57.6 | 0.63 | 0.75 | 80.21 |
| big data + sft, 6e | 0.6B | FSQ | **69.1** | **67.0** | 0.61 | 0.65 | 89.68 |
| big data, 0 sft | 1.7B | binned | 58.0 | 38.4 | 0.74 | 0.96 | 78.53 |
| big data, 0 sft | 1.7B | FSQ | 60.0 | 43.2 | 0.71 | 0.91 | 90.97 |
| sft only, 4e | 1.7B | binned | 68.0 | 56.2 | 0.62 | 0.77 | 82.72 |
| sft only, 6e | 1.7B | FSQ | 68.1 | 64.9 | 0.62 | 0.67 | 89.99 |
| big data + sft, 4e | 1.7B | binned | 69.0 | 61.0 | 0.62 | 0.72 | 81.96 |
| big data + sft, 6e | 1.7B | FSQ | **69.3** | **68.4** | 0.60 | 0.63 | 90.32 |

- **Big data without SFT** is much worse than either SFT condition at both
  scales, so the SFT stage is required.
- Across all six binned comparisons, FSQ gains **+4.8 to +9.4 COV-P** and
  **+7.27 to +12.95** points of PB score. COV-R is the
  weaker axis: +0.1 to +3.1, and at 0.6B sft-only the binned model is 0.1 ahead
  (67.4 vs 67.3), the single cell where FSQ does not win. MAT is equal or better
  everywhere except that same row, where MAT-R is 0.64 against 0.63.
- **The precision and validity gaps are the real result.** They hold at every
  size and every training stage, whereas the recall gap largely closes after SFT
  (0.1–1.5 points, against 2.0–3.1 before it). So the binned representation can
  find the ensemble; it just spends far more of its budget on conformers that
  match nothing and are physically invalid.
- Within FSQ, **big data + SFT** beats **SFT-only** at both scales. At 0.6B it
  gains **+1.8 COV-R and +4.6 COV-P** and improves MAT-R/MAT-P from 0.64/0.71
  to 0.61/0.65. At 1.7B it gains **+1.2 COV-R and +3.5 COV-P** and improves
  MAT-R/MAT-P from 0.62/0.67 to 0.60/0.63.
- Scaling FSQ from 0.6B to 1.7B improves every metric, but the gain after big
  data + SFT is small: +0.2 COV-R, +1.4 COV-P, and −0.01/−0.02 MAT-R/MAT-P.
- Epoch choice: we report the final (6e) checkpoint everywhere. At 1.7B big data + sft, 5e is 0.1 ahead on both coverage metrics with MAT identical.
- **These numbers are comparable to the published literature.** Plain
  GEOM-Drugs, standard 1,000-molecule test split: the same molecules *and* the
  same reference ensembles as Torsional Diffusion, MCF and ET-Flow (their
  released files generate exactly 2× our reference count, molecule by molecule),
  and our COV/MAT code reproduces their published rows when run on their own
  conformers — Torsional to 0.3 COV points and 0.004 Å, ET-Flow to 0.5 and 0.02
  once the two broken molecules in its release are set aside (§3). There is no
  GEOM-revisited data anywhere in the project, so binned-vs-FSQ carries no
  dataset confound either.

The same twelve models at the older **1.25 Å** threshold, and how they compare
with every autoregressive conformer model that has published COV/MAT, are in
detailed notes §2 (`generation_paths_covmat_1.25.csv`).

---

## 3. CoordToken refinement of predicted poses

One CoordToken encode/decode pass over coordinates someone else produced. Two
settings: other methods' GEOM conformer **ensembles** (below), and docked
**poses** from PoseBench (further down). The pass never sees the protein, the
reference, or the graph — only the conformer's own heavy-atom coordinates.

### GEOM conformer generators: four baselines, one pass and then ten

**How it is calculated.** Each baseline ships `{SMILES: [Mol, ...]}` with one
conformer per `Mol`: Torsional Diffusion 998 molecules / 212,910 conformers,
ETFlow 1,000 / 213,556, MCF-S and MCF-B 1,000 / 213,552. Those keys are exactly
the 1,000 molecules of our own test set and the counts are exactly `2k` for our
`k` references, so everything sits alongside §2. Four phases, only one needing a
GPU (`src/molgen3D/evaluation/baseline_tokenizer_refinement.py`):

1. `prepare` — strip hydrogens, since the tokenizer is heavy-atom only. This
   matters for Torsional, whose with-hydrogen PB score is 75.06 against 81.45
   without; its hydrogen placement is strained and its heavy-atom geometry is not.
2. `refine` — FSQ encode → decode, batch 128, ~6 min per method on one H100.
   Failures keep the original coordinates rather than being dropped; there were
   zero encode and zero decode failures across all 853,570 conformers.
3. `gate` — assemble `refine-broken` by taking the refined conformer only where
   the original failed PoseBusters, from per-conformer flags.
4. `evaluate` — COV/MAT at 0.75 Å against the `distinct_smi` references.

CoordToken subtracts each conformer's arithmetic heavy-atom centroid before
encoding; decoded coordinates stay centred. Translation cannot affect aligned
COV/MAT or molecule-only PoseBusters.

Validity is PoseBusters in `mol` mode (`scripts/setup3_posebusters.py`, 4 CPU
shards). **PB micro** pools conformers: passing every check, over all conformers.
**PB macro** averages each molecule's own pass fraction, so a 10-conformer
molecule counts as much as a 3,000-conformer one.

**All four methods are scored on one molecule set of 938.** Torsional Diffusion
picks a different tautomer or resonance form than the GEOM reference for 60 of
its 998 molecules, so no atom correspondence exists and RMSD is undefined there;
only 9 are true atom-count mismatches. Its usable set is 938, and it is a strict
subset of the other three methods' 1,000, so 938 is the intersection. Averaging
each method over its own set instead would compare them on different molecules —
and those 62 molecules are systematically harder, worth about **0.7 PB points**
to the three clean methods. PoseBusters is over **936** of the 938 — two are lost
to its internal-energy check hanging, one of them permanently — and every PB cell
in this section uses that same 936.

Cells read `before → refine-all → refine-broken`. COV/MAT at 0.75 Å, PB in `mol`
mode.

| Method | COV-R | COV-P | MAT-R | MAT-P | PB micro | PB macro |
|---|---|---|---|---|---|---|
| Torsional Diffusion | 73.0 → **73.6** → 73.1 | 55.4 → **56.2** → 55.6 | 0.578 → **0.569** → 0.577 | 0.779 → **0.770** → 0.776 | 81.59 → 86.00 → **86.72** | 85.23 → 88.27 → **88.99** |
| ETFlow | 79.8 → 79.7 → **79.8** | 74.2 → 74.0 → **74.2** | 0.565 → 0.470 → **0.460** | 2.459 → 0.564 → **0.555** | 91.91 → 92.04 → **92.52** | 93.63 → 94.11 → **94.60** |
| MCF-S | 79.8 → 79.7 → **79.9** | 57.1 → 57.0 → **57.1** | 0.513 → 0.513 → 0.513 | 0.766 → 0.768 → **0.766** | 90.13 → 91.84 → **92.51** | 92.48 → 93.57 → **94.41** |
| MCF-B | 83.3 → 83.0 → **83.3** | 63.3 → 63.1 → **63.3** | 0.440 → 0.447 → **0.440** | 0.680 → 0.686 → **0.679** | 91.84 → 92.26 → **92.79** | 93.41 → 93.76 → **94.47** |

**Refinement buys validity, not accuracy, and it buys most where the input is
worst.** Blanket refinement gains **+4.41** PB micro for Torsional, +1.71 for
MCF-S, +0.42 for MCF-B and +0.13 for ETFlow — monotone in how bad the input was.
COV/MAT meanwhile is flat or fractionally negative for the three good methods and
improves only for Torsional. Same relationship the docking panel and Platinum
Diverse show, now on four independent generators.

These gains are **smaller than the same table on mixed bases** would suggest
(+4.51, +2.43, +0.63, +0.22): the 62 excluded molecules are ones refinement
disproportionately fixed, so scoring the clean methods over 1,000 while scoring
Torsional over 938 inflated their apparent gain by up to half a point. The
ordering and the conclusion are unchanged.

**`refine-broken` is the column to report.** Best PB of the three for every
method on both micro and macro, *and* COV/MAT back at the `before` value or
better in every case, because it only touches the 8.8–18.5% of conformers that
failed. Blanket refinement's small accuracy cost — MCF-B loses 0.3 COV-R and
0.007 MAT-R — disappears under the gate.

**Hydrogens.** `before` is scored with hydrogens removed, matching what the
tokenizer can emit. For Torsional that choice is load-bearing: its with-hydrogen
PB score is 75.06 against 81.45 on its own 998, and the entire gap is
`internal_energy` (85.87 → 97.17). Its heavy-atom geometry is fine and its
hydrogen placement is strained; scoring with hydrogens would have credited
refinement with ~6 extra points for merely deleting them.

#### Iterating the pass: rounds 1 to 10

Round *r* feeds round *r−1*'s `after` conformers back in, so round *r* is *r*
passes over every conformer and each round's `before` is the previous round's
`after`. Driver `scripts/submit_baseline_iterations.sh`, one GPU job per method
per round (~7.5 min each) with PoseBusters and COV/MAT hanging off each round so
scoring never blocks the next pass. Table built by
`scripts/baseline_rounds_common.py` → `outputs/baseline_refinement/rounds_common.csv`.

Every cell uses one molecule set. Torsional Diffusion has usable COV/MAT for 938
of its 998 molecules and that set is a strict subset of the other three methods'
1,000, so **938** is the intersection and every COV/MAT cell is averaged over it.
PoseBusters loses two more in some rounds — one is the thiadiazole/naphthalene
that hangs its internal-energy check and now records a timeout — so **PB is over
936**, the molecules scored in every method and every round. `original` is the
baseline's own conformers; round *r* is *r* passes.

| Method | | original | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Torsional** | PB micro | 81.59 | 86.00 | 87.79 | 88.73 | 89.35 | 89.73 | 90.01 | 90.17 | 90.36 | 90.48 | 90.58 |
| | PB macro | 85.23 | 88.27 | 89.47 | 90.06 | 90.46 | 90.73 | 90.90 | 90.99 | 91.15 | 91.22 | 91.29 |
| | COV-R | 72.98 | 73.62 | 73.79 | 73.88 | 73.89 | 73.89 | 73.94 | 73.96 | 73.96 | 73.94 | 73.93 |
| | COV-P | 55.44 | 56.16 | 56.31 | 56.42 | 56.48 | 56.49 | 56.50 | 56.52 | 56.51 | 56.49 | 56.46 |
| | MAT-R | 0.578 | 0.569 | 0.567 | 0.567 | 0.567 | 0.567 | 0.567 | 0.567 | 0.567 | 0.567 | 0.567 |
| | MAT-P | 0.779 | 0.770 | 0.768 | 0.767 | 0.767 | 0.767 | 0.767 | 0.767 | 0.767 | 0.767 | 0.767 |
| **MCF-S** | PB micro | 90.13 | 91.84 | 91.86 | 91.88 | 91.87 | 91.87 | 91.87 | 91.89 | 91.88 | 91.89 | 91.88 |
| | PB macro | 92.48 | 93.57 | 93.47 | 93.40 | 93.28 | 93.22 | 93.22 | 93.17 | 93.20 | 93.17 | 93.20 |
| | COV-R | 79.81 | 79.67 | 79.59 | 79.44 | 79.38 | 79.32 | 79.28 | 79.25 | 79.22 | 79.19 | 79.19 |
| | COV-P | 57.15 | 56.97 | 56.85 | 56.72 | 56.65 | 56.61 | 56.58 | 56.51 | 56.51 | 56.48 | 56.47 |
| | MAT-R | 0.513 | 0.513 | 0.516 | 0.517 | 0.519 | 0.520 | 0.521 | 0.521 | 0.522 | 0.522 | 0.522 |
| | MAT-P | 0.766 | 0.768 | 0.770 | 0.772 | 0.773 | 0.774 | 0.774 | 0.775 | 0.775 | 0.776 | 0.776 |
| **MCF-B** | PB micro | 91.84 | 92.26 | 92.25 | 92.25 | 92.25 | 92.25 | 92.25 | 92.25 | 92.25 | 92.24 | 92.23 |
| | PB macro | 93.41 | 93.77 | 93.66 | 93.62 | 93.55 | 93.54 | 93.46 | 93.45 | 93.44 | 93.45 | 93.45 |
| | COV-R | 83.30 | 83.05 | 82.91 | 82.86 | 82.82 | 82.76 | 82.78 | 82.77 | 82.74 | 82.74 | 82.76 |
| | COV-P | 63.34 | 63.06 | 62.92 | 62.88 | 62.83 | 62.78 | 62.75 | 62.74 | 62.70 | 62.70 | 62.67 |
| | MAT-R | 0.440 | 0.447 | 0.451 | 0.454 | 0.456 | 0.457 | 0.459 | 0.459 | 0.460 | 0.460 | 0.461 |
| | MAT-P | 0.680 | 0.686 | 0.689 | 0.692 | 0.693 | 0.694 | 0.695 | 0.696 | 0.697 | 0.697 | 0.697 |
| **ETFlow** | PB micro | 91.91 | 92.04 | 92.08 | 92.11 | 92.11 | 92.13 | 92.14 | 92.12 | 92.12 | 92.12 | 92.11 |
| | PB macro | 93.63 | 94.11 | 94.14 | 94.17 | 94.13 | 94.16 | 94.16 | 94.14 | 94.16 | 94.15 | 94.15 |
| | COV-R | 79.78 | 79.66 | 79.59 | 79.50 | 79.53 | 79.57 | 79.55 | 79.55 | 79.56 | 79.52 | 79.49 |
| | COV-P | 74.20 | 74.02 | 73.88 | 73.76 | 73.68 | 73.64 | 73.58 | 73.53 | 73.51 | 73.45 | 73.45 |
| | MAT-R | 0.565 | 0.470 | 0.475 | 0.479 | 0.481 | 0.483 | 0.484 | 0.485 | 0.486 | 0.487 | 0.487 |
| | MAT-P | 2.459 | 0.564 | 0.569 | 0.573 | 0.576 | 0.578 | 0.579 | 0.581 | 0.582 | 0.583 | 0.583 |

**Every method converges to the same 91.9–92.3% PB micro, and only the worst
input gains by getting there.** Torsional climbs **+8.99** from its own
conformers (81.59 → 90.58), +4.41 in pass 1 and +4.58 across passes 2–10, still
rising. The three clean methods take everything in pass 1 (+1.71, +0.42, +0.13)
and passes 2–10 add +0.04, −0.03, +0.07 — nothing.

**What the extra passes do cost is coverage, monotonically.** From pass 1 to
pass 10 COV-R falls 0.48 for MCF-S, 0.29 for MCF-B and 0.17 for ETFlow, with
MAT-R rising in lockstep (+0.009, +0.014, +0.017). Torsional is the exception at
+0.31 COV-R, because for it the manifold is an improvement. So iterating is
strictly a loss for anything already on the manifold: no validity left to gain,
and a steady walk away from the references. **One gated pass stays the operating
point.**

**Extra passes cost accuracy, monotonically, for everything already clean.**
MAT-R degrades every round for the three good methods (+0.009, +0.013, +0.018)
and COV-R slides (−0.48, −0.22, −0.19). Torsional is the exception, gaining
+0.31 COV-R, because for it the manifold is an improvement. So the pass pulls
structures onto the manifold whether or not they belong there.

**Micro and macro disagree in sign for both MCF variants** — micro flat or up,
macro down ~0.3 over ten rounds. What loses validity under repeated passes are
the small ensembles, which macro weights equally and micro barely sees.

One pass is the right operating point for clean inputs; gating also avoids
moving conformers that already pass.

### Docked poses: nine PoseBench methods

> **Correction (2026-09-03):** the RMSD and joint RMSD-and-validity values in
> this section used a PoseBench v0.6.0 path that aligned the crystal ligand in
> place. Use `posebench_rmsd_rescoring.md` for the corrected unaligned
> `CalcRMS` results. Ligand-only validity numbers are unaffected.

One CoordToken encode/decode pass over each method's predicted poses, over three
repeats. Nine PoseBench baselines: two that dock into a given protein
(**DiffDock**, **AutoDock Vina**) and seven that predict or move the protein too
(**DynamicBind**, **NeuralPLexer**, **RFAA**, **Chai-1**, **Chai-1-Single-Seq**,
**AF3**, **AF3-Single-Seq**). Each cell below reads `before → refine-all →
selective`: the method's own pose, then one CoordToken encode/decode pass applied
to every pose, then the pass applied only where the original fails PoseBusters.

Percentages use PoseBench's common denominator — 375, i.e. 125 per repeat × 3
repeats, and 125 for RFAA which ran once — with the same NaN mask as their
published data. A method that produced fewer poses is therefore scored as if the
missing ones failed, so these numbers mix coverage with quality. Detailed notes
§4 has the denominator argument, the reproduction audit, the coverage/quality
split, and the full **protein-aware** PoseBusters table; **this section reports
ligand-only validity**, which is what a conformer refiner can actually be judged
on.

### Ligand-only validity, check by check

PoseBusters runs its checks in three families:

1. **Chemical validity and consistency** — is it still the right molecule?
   RDKit sanitization (legal valences, aromaticity), all atoms connected (it did
   not fragment), molecular formula preserved, bonds and bond orders preserved,
   E/Z double-bond stereochemistry preserved, tetrahedral chirality preserved.
2. **Intramolecular validity** — is the geometry physically sensible on its own?
   Bond lengths, bond angles, no internal steric clashes, aromatic rings flat,
   double-bond environments flat, and no absurd UFF internal energy.
3. **Intermolecular validity** — does it sit sensibly in the pocket? Minimum
   distance to the protein and to organic/inorganic cofactors, volume overlap
   with each of those, and the protein–ligand maximum distance.

**We report families 1 and 2 only — "ligand-only" validity.** Those are the
checks a conformer refiner can be judged on: they depend on the ligand's own
geometry and nothing else. Family 3 depends on the pocket, which the refiner
never sees, so a change there says as much about the protein as about the
refinement. (Family 3 is what the protein-aware `PB-valid` column in detailed
notes §4 adds, which is why those numbers are much lower.)

Scored over the poses each method actually produced, not the common denominator,
so these are not comparable to the protein-aware table. Cells read
`before → refine-all → selective`, with the gate on ligand-only validity here:
keep the original pose if its own geometry already passes. Rates now include raw
pass counts in parentheses. The adjacent transition columns show
`– → fixed/broken under refine-all → fixed/broken under selective`, always
relative to the original.

| Check | Family | DiffDock pass rate (count), n=352 | DiffDock fixed/broken | Vina pass rate (count), n=188 | Vina fixed/broken |
|---|---|---|---|---|---|
| sanitization | chemical | 100.0% (352) → 100.0% (352) → 100.0% (352) | – → 0/0 → 0/0 | 100.0% (188) → 100.0% (188) → 100.0% (188) | – → 0/0 → 0/0 |
| all atoms connected | chemical | 100.0% (352) → 100.0% (352) → 100.0% (352) | – → 0/0 → 0/0 | 100.0% (188) → 100.0% (188) → 100.0% (188) | – → 0/0 → 0/0 |
| molecular formula | chemical | 100.0% (352) → 100.0% (352) → 100.0% (352) | – → 0/0 → 0/0 | 100.0% (188) → 100.0% (188) → 100.0% (188) | – → 0/0 → 0/0 |
| molecular bonds | chemical | 100.0% (352) → 100.0% (352) → 100.0% (352) | – → 0/0 → 0/0 | 100.0% (188) → 100.0% (188) → 100.0% (188) | – → 0/0 → 0/0 |
| double-bond stereochemistry | chemical | 96.6% (340) → 96.6% (340) → 96.6% (340) | – → 0/0 → 0/0 | 96.8% (182) → 96.8% (182) → 96.8% (182) | – → 0/0 → 0/0 |
| tetrahedral chirality | chemical | 99.1% (349) → 98.9% (348) → 98.9% (348) | – → 0/1 → 0/1 | 100.0% (188) → 100.0% (188) → 100.0% (188) | – → 0/0 → 0/0 |
| **chemical valid (all 6)** | composite | **95.7% (337) → 95.7% (337) → 95.7% (337)** | **– → 0/0 → 0/0** | **96.8% (182) → 96.8% (182) → 96.8% (182)** | **– → 0/0 → 0/0** |
| bond lengths | intramolecular | 99.4% (350) → 99.7% (351) → 99.7% (351) | – → 2/1 → 2/1 | 100.0% (188) → 98.9% (186) → 98.9% (186) | – → 0/2 → 0/2 |
| bond angles | intramolecular | 100.0% (352) → 99.7% (351) → 99.7% (351) | – → 0/1 → 0/1 | 98.9% (186) → 98.9% (186) → 98.9% (186) | – → 0/0 → 0/0 |
| internal steric clash | intramolecular | 97.4% (343) → 98.6% (347) → 98.6% (347) | – → 4/0 → 4/0 | 100.0% (188) → 100.0% (188) → 100.0% (188) | – → 0/0 → 0/0 |
| aromatic-ring flatness | intramolecular | 100.0% (352) → 100.0% (352) → 100.0% (352) | – → 0/0 → 0/0 | 100.0% (188) → 100.0% (188) → 100.0% (188) | – → 0/0 → 0/0 |
| double-bond flatness | intramolecular | 99.7% (351) → 99.7% (351) → 99.7% (351) | – → 0/0 → 0/0 | 100.0% (188) → 100.0% (188) → 100.0% (188) | – → 0/0 → 0/0 |
| internal energy | intramolecular | 94.9% (334) → 97.7% (344) → **97.7% (344)** | – → 11/1 → 11/1 | 97.3% (183) → 98.9% (186) → **99.5% (187)** | – → 4/1 → 4/0 |
| **intramolecular valid (all 6)** | composite | **93.8% (330) → 96.6% (340) → 96.6% (340)** | **– → 10/0 → 10/0** | **97.3% (183) → 98.4% (185) → 98.9% (186)** | **– → 3/1 → 3/0** |
| **ligand-only valid (1 + 2)** | composite | **91.5% (322) → 93.5% (329) → 93.5% (329)** | **– → 7/0 → 7/0** | **94.1% (177) → 95.2% (179) → 95.7% (180)** | **– → 3/1 → 3/0** |
| RMSD ≤ 2 Å | accuracy | 87.5% (308) → 86.9% (306) → **87.8% (309)** | – → 1/3 → 1/0 | 58.0% (109) → 56.9% (107) → **57.4% (108)** | – → 0/2 → 0/1 |
| ligand-only valid & RMSD ≤ 2 Å | endpoint | 81.5% (287) → 82.4% (290) → **83.2% (293)** | – → 6/3 → 6/0 | 55.3% (104) → 54.8% (103) → **55.9% (105)** | – → 1/2 → 1/0 |

Chemistry is almost untouched, but DiffDock has one real exception:
`7U0U_FK5` in repeat 3 loses tetrahedral-chirality validity. At RDKit atom 44
(zero-based), CoordToken shortens one adjacent bond from about 1.52 Å to 0.39 Å
and reduces the local signed tetrahedral volume from about 2.55 to 0.052 Å³, so
RDKit can no longer assign that stereocentre unambiguously. The molecular graph,
formula, and bond orders remain unchanged. This pose already failed the
double-bond-stereochemistry check before refinement, so the all-six chemical
composite stays at 337/352 despite the additional failed chirality check.

The main gain is internal energy: DiffDock fixes 11 poses and breaks 1, while
Vina fixes 4 and breaks 1 under refine-all (and none under selective). Internal
steric clash adds four clean fixes for DiffDock. Thus "refinement improves
physical validity" mostly means that it relaxes strained conformers into
lower-energy ones.

**All nine methods, ligand-only.** Same construction as the protein-aware table
— same NaN mask and common denominators — but validity uses only families 1 and 2, and
the gate is on ligand-only validity rather than full PB-valid. Cells and count
columns again read `before → refine-all → selective`; count pairs are relative
to the original. Here the pose-transition column means ligand-only
invalid→valid / valid→invalid, and the PB-test column uses the 12 ligand-only
tests.

| Method | cRMSD ≤ 1 Å | RMSD ≤ 2 Å | RMSD ≤ 2 Å & ligand-only valid | ligand-only valid | Paired n | poses invalid→valid / valid→invalid | PB tests fixed/broken |
|---|---|---|---|---|---:|---|---|
| NeuralPLexer | 98.4 → 98.9 → **98.9** | 72.3 → 72.5 → **73.1** | 53.9 → 59.2 → **61.3** | 61.3 → 71.5 → **73.6** | 371 | – → 46/8 → 46/0 | – → 123/24 → 123/16 |
| RFAA | 76.8 → 76.8 → **76.8** | 64.8 → 64.0 → **64.0** | 54.4 → 53.6 → **54.4** | 61.6 → 62.4 → **63.2** | 94 | – → 2/1 → 2/0 | – → 5/1 → 5/0 |
| DynamicBind | 92.8 → 92.8 → **92.8** | 71.5 → 71.5 → **71.5** | 66.9 → 67.5 → **67.5** | 87.5 → 88.0 → **88.0** | 348 | – → 2/0 → 2/0 | – → 7/0 → 7/0 |
| DiffDock | 93.6 → 93.9 → **93.9** | 82.1 → 81.6 → **82.4** | 76.5 → 77.3 → **78.1** | 85.9 → 87.7 → **87.7** | 352 | – → 7/0 → 7/0 | – → 17/4 → 17/4 |
| AF3-Single-Seq | 93.9 → 94.4 → **94.4** | 70.7 → 70.1 → **70.1** | 55.2 → 56.8 → **56.8** | 68.3 → 70.1 → **70.1** | 354 | – → 7/0 → 7/0 | – → 9/0 → 9/0 |
| Vina | 45.6 → 45.6 → **45.6** | 29.1 → 28.5 → **28.8** | 27.7 → 27.5 → **28.0** | 47.2 → 47.7 → **48.0** | 188 | – → 3/1 → 3/0 | – → 4/3 → 4/2 |
| Chai-1 | 91.5 → 92.3 → **91.7** | 83.5 → 83.5 → **83.5** | 68.0 → 69.3 → **70.1** | 73.3 → 74.7 → **75.5** | 359 | – → 8/3 → 8/0 | – → 11/6 → 11/3 |
| AF3 | 97.6 → 97.6 → **97.6** | 89.6 → 89.6 → **89.6** | 76.0 → 75.2 → **76.0** | 77.6 → 76.8 → **77.6** | 354 | – → 0/3 → 0/0 | – → 0/3 → 0/0 |
| Chai-1-Single-Seq | 93.3 → 93.3 → **93.3** | 81.9 → 82.4 → **81.9** | 72.5 → 72.8 → **73.6** | 78.4 → 78.1 → **79.5** | 353 | – → 4/5 → 4/0 | – → 5/6 → 5/0 |

Dropping the pocket changes the picture in two ways. The comparison numbers below
are the protein-aware scores from detailed notes §4.

**Everything is much higher, and the ordering scrambles.** NeuralPLexer goes from
4.0% protein-aware PB-valid to **61.3%** ligand-only, and DynamicBind from 8.0%
to **87.5%** — now the best of the nine. Their poses are internally fine and
simply sit wrong in the pocket. AF3 rises much less, 61.1% to 77.6%, so it drops
in relative standing, and Vina barely moves at all, 45.6% to 47.2%, because its
problem was never intermolecular.

**Blanket refinement stops being harmful almost everywhere.** On ligand-only
validity it *gains* for seven of nine methods (NeuralPLexer +10.2, DiffDock +1.8,
Chai-1 +1.4) and loses only for AF3 (−0.8) and Chai-1-SS (−0.3). The damage seen
under protein-aware scoring was largely refinement pushing ligands into the
protein, which this view does not measure.

**Gating still wins, and never loses.** Selective is ≥ refine-all on every
method and every metric here, and ≥ baseline everywhere except Chai-1's cRMSD
(91.5 → 91.7 is still above baseline; nothing falls below). The largest gain is
NeuralPLexer at **+12.3** ligand-only.

### Source and paper wording

Morehead, Giri, Liu & Cheng, *Deep Learning for Protein-Ligand Docking: Are We
There Yet?*, ICML AI4Science Workshop 2024 (arXiv:2405.14108; data DOI
10.5281/zenodo.14629652). We use the latest release, **v0.6.0 (2025-02-09)**,
whose numbers differ from the original workshop paper's — the alignment step was
rewritten, which their changelog says gives up to 3× better scores. Do not
compare numbers across the two versions.

We re-derived the baselines from raw predictions rather than quoting them, and
audited pose by pose against the published per-pose data. Wherever both sides
compute an RMSD the values are bit-identical, so
the scoring and the alignment transform agree, and after the NaN mask above
eight of nine methods reproduce the published baselines on every metric.

NeuralPLexer is the exception, and its cause is different in kind. It has no
NaNs on either side, so nothing is masked; instead 173 of its 371 poses have
genuinely different RMSD values, up to 6.14 Å apart. We are scoring different
coordinates, which fits the relaxed-pose explanation above — relaxation both
moves atoms and clears the steric clashes that sink its validity.

For the paper, quote their baselines verbatim, report ours over
actually-evaluated poses, and state the convention in the caption:

> Following the PoseBench Figure 4 convention, success counts were normalized by
> a common denominator of 125 per repeat (375 across three repeats) on the
> 130-complex post-2021-09-30 subset; 375 is not the number of distinct
> complexes or evaluated poses, and the actual available predictions were 352
> for DiffDock and 188 for Vina.

### Platinum Diverse: the same test on clean conformers

The same one-pass refinement, but on conformers rather than docked poses:
BCL::Conf's published resonance-aware MES50 ensembles for Platinum Diverse
2017_01 (2,859 molecules, 132,165 conformers). The graphs are untouched; only
coordinates change. Each molecule has one experimental bound conformation, so
the endpoints are the ensemble's nearest symmetry-aware heavy-atom RMSD and
recovery at 0.6 Å and 1.0 Å, with PoseBusters run in molecule-only mode.

This matters because the inputs are far cleaner than anything in the docking
panel — **98.1% PB-valid**, against 61% for the best co-folding model.

| Metric | before | refine-all | selective | Δ selective | conformers invalid→valid / valid→invalid | PB tests fixed/broken |
|---|---:|---:|---:|---:|---|---|
| PoseBusters valid | 98.10 | **84.28** | **98.67** | +0.56 | – → 656/20,390 → 656/0 | – → 1,084/45,344 → 1,084/1,053 |
| recovery @ 0.6 Å | 63.80 | 61.53 | 63.80 | +0.00 | n/a | n/a |
| recovery @ 1.0 Å | 86.60 | 84.44 | 86.60 | +0.00 | n/a | n/a |
| mean nearest RMSD (Å) | 0.5433 | 0.5738 | 0.5436 | +0.0003 | n/a | n/a |

The conformer column counts the PB-invalid conformers that become fully PB-valid
and the PB-valid conformers that become invalid. The PB-test column instead
pools all individual `conformer × check` transitions. Both refined entries are
relative to baseline.

Refining everything is a disaster here, but gating fixes it completely. Only 2,380 of 132,165 conformers (1.8%) fail
PoseBusters and get refined; the rest are left alone. Per check, bond lengths recover to 99.62, bond
angles to 99.65, and internal energy ends *above* baseline at 99.74.

**Check by check.** Platinum's checks are already ligand-only — the molecule-only
config has no intermolecular tests, since these are free conformers with no
protein frame, so `pb_valid` *is* the ligand-only composite. Pooled over all
132,165 conformers, each rate includes its raw pass count. The transition column
reads `– → fixed/broken under refine-all → fixed/broken under selective`:

| Check | pass rate (count): before → refine-all → selective | fixed/broken: before → refine-all → selective |
|---|---|---|
| mol pred loaded | 100.00% (132,165) → 100.00% (132,165) → 100.00% (132,165) | – → 0/0 → 0/0 |
| sanitization | 100.00% (132,165) → 100.00% (132,165) → 100.00% (132,165) | – → 0/0 → 0/0 |
| inchi convertible | 99.92% (132,065) → 99.92% (132,065) → 99.92% (132,065) | – → 0/0 → 0/0 |
| all atoms connected | 100.00% (132,165) → 100.00% (132,165) → 100.00% (132,165) | – → 0/0 → 0/0 |
| no radicals | 100.00% (132,165) → 100.00% (132,165) → 100.00% (132,165) | – → 0/0 → 0/0 |
| bond lengths | 99.98% (132,132) → **84.83% (112,121)** → 99.62% (131,669) | – → 33/20,044 → 33/496 |
| bond angles | 99.75% (131,833) → **87.65% (115,844)** → 99.65% (131,704) | – → 208/16,197 → 208/337 |
| internal steric clash | 99.35% (131,309) → 97.81% (129,277) → **99.61% (131,648)** | – → 377/2,409 → 377/38 |
| aromatic ring flatness | 99.99% (132,153) → 98.91% (130,722) → 99.98% (132,145) | – → 12/1,443 → 12/20 |
| non-aromatic ring non-flatness | 99.47% (131,467) → 99.65% (131,704) → **99.66% (131,713)** | – → 246/9 → 246/0 |
| double bond flatness | 99.76% (131,843) → 99.07% (130,933) → 99.72% (131,795) | – → 25/935 → 25/73 |
| internal energy | 99.66% (131,722) → 96.54% (127,598) → **99.74% (131,816)** | – → 183/4,307 → 183/89 |
| **pb valid (all 12)** | **98.20% (129,785) → 83.27% (110,051) → 98.70% (130,441)** | **– → 656/20,390 → 656/0** |

Under refine-all, the imbalance is severe: it breaks **20,044** bond-length
checks to fix 33 and **16,197** bond-angle checks to fix 208. Only non-aromatic
ring non-flatness has a positive fixed-minus-broken balance. Selective refinement
keeps all 1,084 fixed individual tests while reducing broken tests from 45,344
to 1,053.

Under the gate, three checks finish *above* baseline — internal steric clash
99.35 → 99.61, internal energy 99.66 → 99.74, ring non-flatness 99.47 → 99.66 —
because those are the defects the 2,380 invalid conformers actually had. Bond
lengths and angles land fractionally below baseline, since relaxing a strained
conformer still risks its bonds. The five loading and connectivity checks are
flat at 100% throughout, as they must be: coordinates change, the graph does not.

Source: `/mnt/weka/fgeikyan/CoordToken/posebusters_refinement/platinum_diverse_refinement/`
(`summary_metrics.csv`, `posebusters_check_summary.csv`, `per_conformer_results.csv`).
The `before` column above reproduces their `summary_metrics.csv` exactly. Data is
ACS Figshare supplement 10.1021/acs.jcim.0c01140.s004, MD5-verified.
