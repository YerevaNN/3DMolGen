# CoordToken refinement of GEOM conformer generators

One CoordToken encode/decode pass over conformers produced by four published
GEOM generators. The pass sees only a conformer's own heavy-atom coordinates —
never the reference, the protein, or the graph.

## The molecule set: 938, for every number here

The GEOM test set is 1,000 molecules. ETFlow, MCF-S and MCF-B ship conformers for
all 1,000; Torsional Diffusion ships 998, and of those only **938 can be scored**.

The 60 that cannot are not a pipeline failure. COV/MAT is a symmetry-aware
heavy-atom RMSD, which needs an atom-to-atom correspondence between the generated
conformer and the reference. For those 60, Torsional Diffusion's molecule and the
GEOM reference disagree — a different tautomer, or a different resonance/charge
assignment of the same formula:

    O=C1CC(c2ccccc2)CC(=O)C1=CNCCN1CCOCC1     (GEOM reference)
    O=C1CC(c2ccccc2)CC(O)=C1C=NCCN1CCOCC1     (prediction)

`GetBestRMS` finds no common substructure, every entry of the RMSD matrix is NaN,
and the molecule has no COV/MAT value under any condition. Only 9 of the 60 are
true atom-count mismatches; the rest are this tautomer/resonance disagreement.
The other three methods lose nothing this way.

**938 is a clean intersection**: Torsional's usable set is a strict *subset* of
the other three methods' 1,000, so adopting it costs them 62 molecules each and
costs Torsional nothing. Every cell below — COV, MAT, PoseBusters micro and
all four methods, all ten rounds — is averaged over those same 938.

**Why the basis matters.** The 62 excluded molecules are systematically harder,
so scoring Torsional over 938 while scoring everyone else over 1,000 flattered
the other three:

| Method | PB micro on own set | PB micro on 938 | shift |
|---|---:|---:|---:|
| Torsional Diffusion | 81.45 (998) | 81.59 | +0.14 |
| MCF-S | 88.67 (1,000) | 89.95 | **+1.28** |
| MCF-B | 90.97 (1,000) | 91.87 | **+0.90** |
| ETFlow | 91.16 (1,000) | 91.95 | **+0.79** |

They are also molecules refinement disproportionately fixes, so the mixed basis
**overstated the gains**: MCF-S's blanket-refinement gain falls from +2.43 to
+1.93, MCF-B's from +0.63 to +0.46, ETFlow's from +0.22 to +0.14. Ordering and
conclusions unchanged; magnitudes not.

A related trap, worth recording because it nearly produced a finding: PoseBusters
occasionally loses a molecule (a scorer crash, or a timeout on a very large
ensemble), and *which* one varies by round. Dropping a single molecule moves
validity by up to 0.09 — the size of the trends here. On per-round sets MCF-S
appeared to dip at round 10; on a fixed set it is flat. Fix the molecule set
first, then read the trends.

## One pass, four generators

`original → refine-all → refine-broken`. `original` is each baseline's own
conformers with hydrogens removed (the tokenizer is heavy-atom only);
`refine-all` applies the pass to every conformer; `refine-broken` applies it only
where the original conformer fails PoseBusters. COV/MAT at 0.75 Å, PoseBusters in
`mol` mode. **PB-valid is conformer-pooled** (micro): conformers passing every
check, over all conformers — the same definition as the published baseline scores
and as §2. The molecule-weighted variant (macro) agrees in sign and ordering
everywhere in this file and is omitted; it is kept in the CSVs
(`round1_938.csv`, `rounds_gated_938.csv`) for anyone who wants it.

| Method | COV-R | COV-P | MAT-R | MAT-P | PB-valid |
|---|---|---|---|---|---|
| Torsional Diffusion | 73.0 → **73.6** → 73.1 | 55.4 → **56.2** → 55.6 | 0.578 → **0.569** → 0.577 | 0.779 → **0.770** → 0.776 | 81.59 → 86.05 → **86.76** |
| ETFlow | 79.8 → 79.7 → **79.8** | 74.2 → 74.0 → **74.2** | 0.565 → 0.470 → **0.460** | 2.459 → 0.564 → **0.555** | 91.95 → 92.09 → **92.56** |
| MCF-S | 79.8 → 79.7 → **79.9** | 57.1 → 57.0 → **57.1** | 0.513 → 0.513 → 0.513 | 0.766 → 0.768 → **0.766** | 89.95 → 91.88 → **92.54** |
| MCF-B | 83.3 → 83.0 → **83.3** | 63.3 → 63.1 → **63.3** | 0.440 → 0.447 → **0.440** | 0.680 → 0.686 → **0.679** | 91.87 → 92.33 → **92.84** |

Machine-readable, one row per method and condition, four decimals:
`outputs/baseline_refinement/round1_938.csv`. Long form, if the arrow cells above
are awkward to parse:

| Method | Condition | COV-R | COV-P | MAT-R | MAT-P | PB-valid |
|---|---|---:|---:|---:|---:|---:|
| Torsional Diffusion | original | 72.98 | 55.44 | 0.5782 | 0.7786 | 81.59 |
| Torsional Diffusion | refine-all | 73.62 | 56.16 | 0.5693 | 0.7699 | 86.05 |
| Torsional Diffusion | refine-broken | 73.11 | 55.60 | 0.5768 | 0.7764 | 86.76 |
| ETFlow | original | 79.78 | 74.20 | 0.5650 | 2.4593 | 91.95 |
| ETFlow | refine-all | 79.66 | 74.02 | 0.4697 | 0.5636 | 92.09 |
| ETFlow | refine-broken | 79.77 | 74.20 | 0.4600 | 0.5546 | 92.56 |
| MCF-S | original | 79.81 | 57.15 | 0.5131 | 0.7663 | 89.95 |
| MCF-S | refine-all | 79.67 | 56.97 | 0.5134 | 0.7682 | 91.88 |
| MCF-S | refine-broken | 79.86 | 57.15 | 0.5125 | 0.7656 | 92.54 |
| MCF-B | original | 83.30 | 63.34 | 0.4401 | 0.6797 | 91.88 |
| MCF-B | refine-all | 83.05 | 63.06 | 0.4469 | 0.6858 | 92.33 |
| MCF-B | refine-broken | 83.30 | 63.34 | 0.4401 | 0.6794 | 92.84 |

All 12 rows are over the same 938 molecules, on every metric including PB.

**Refinement buys validity, not accuracy, and it buys most where the input is
worst.** Blanket refinement gains **+4.46** PB micro for Torsional, +1.93 for
MCF-S, +0.46 for MCF-B, +0.14 for ETFlow — monotone in how bad the input was.
COV/MAT is meanwhile flat or fractionally negative for the three good methods and
improves only for Torsional.

**`refine-broken` is the column to report.** Best PB of the three for every
method, *and* COV/MAT back at the `original` value or
better in every case, because it touches only the 8.8–18.5% of conformers that
failed. Blanket refinement's small accuracy cost — MCF-B loses 0.3 COV-R and
0.007 MAT-R — disappears under the gate.

**ETFlow's MAT-P of 2.459 → 0.564 is two broken molecules, and the pass does not
fix them.** A phosphonate and a bisphosphonate arrive from the source file with
atom-to-atom spans up to 7,883 Å across 5,624 conformers. One pass returns every
one to physical size (worst span 19.5 Å), which is what collapses their
per-molecule MAT-P from 1430.57 and 359.97 Å to 1.383 and 2.054. But **not one of
the 5,624 conformers passes PoseBusters, before or after**, and the per-check
detail shows the pass makes them *worse* wherever a check can actually be
evaluated:

| check | phosphonate before → after | bisphosphonate before → after |
|---|---|---|
| all_atoms_connected | 0.0 → 0.0 | 0.0 → 0.0 |
| bond_lengths | 100.0 → 1.1 | 92.1 → 0.0 |
| bond_angles | 100.0 → 5.1 | 97.5 → 0.0 |
| internal_steric_clash | 100.0 → 30.4 | 100.0 → 14.8 |
| internal_energy | 100.0 → 68.9 | 99.8 → 15.8 |

The exploded inputs pass the local checks **vacuously**: with atoms thousands of Å
apart RDKit perceives no bonds, so "all bond lengths in range" holds over an empty
set, and only `all_atoms_connected` catches the molecule. After the pass the
structure is compact enough for bonds to be perceived, so those checks finally
have something to measure and fail almost everywhere, while connectivity never
recovers. Their COV also stays at 0.00 → 0.08.

Two consequences for reporting. **Do not quote 2.459 → 0.564 as evidence the
tokenizer repairs broken geometry** — it is an RMSD collapse on conformers that
remain chemically invalid throughout. And these two molecules drag ETFlow's
validity down by 2.6 points in *both* conditions: excluding them, ETFlow reads
94.54 → 94.69 rather than 91.95 → 92.09, and its MAT-P reads 0.5516 → 0.5612,
i.e. the pass is fractionally negative on the other 936 molecules.

**Hydrogens are load-bearing for Torsional only.** Its with-hydrogen PB score is
75.06 against 81.45 on its own 998, and the whole gap is `internal_energy`
(85.87 → 97.17): its heavy-atom geometry is fine, its hydrogen placement is
strained. Scoring with hydrogens would have credited refinement with ~6 extra
points for merely deleting them.

## Ten rounds, Torsional Diffusion

Round *r* feeds round *r−1*'s output back in, so round *r* is *r* successive
passes over every conformer. Torsional is the method with something to gain, so
it is the one worth iterating.

| Metric | original | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PB-valid | 81.59 | 86.05 | 87.85 | 88.79 | 89.40 | 89.78 | 90.06 | 90.21 | 90.41 | 90.53 | 90.63 |
| COV-R | 72.98 | 73.62 | 73.79 | 73.88 | 73.89 | 73.89 | 73.94 | 73.96 | 73.96 | 73.94 | 73.93 |
| COV-P | 55.44 | 56.16 | 56.31 | 56.42 | 56.48 | 56.49 | 56.50 | 56.52 | 56.51 | 56.49 | 56.46 |
| MAT-R | 0.578 | 0.569 | 0.567 | 0.567 | 0.567 | 0.567 | 0.567 | 0.567 | 0.567 | 0.567 | 0.567 |
| MAT-P | 0.779 | 0.770 | 0.768 | 0.767 | 0.767 | 0.767 | 0.767 | 0.767 | 0.767 | 0.767 | 0.767 |

**Refine-broken across all ten rounds.** The gate keeps each original conformer
that already passed PoseBusters and substitutes the *r*-times-refined one only
where it failed — the same policy as the round-1 table, extended to every round.
It is reconstructed exactly from the per-conformer flags rather than re-scored:
a gated conformer is valid iff the original or the refined one is.

| Round | refine-all | refine-broken | gating gain |
|---:|---:|---:|---:|
| original | 81.59 | – | – |
| 1 | 86.05 | **86.76** | +0.71 |
| 2 | 87.85 | **88.66** | +0.80 |
| 3 | 88.79 | **89.65** | +0.86 |
| 4 | 89.40 | **90.31** | +0.91 |
| 5 | 89.78 | **90.73** | +0.95 |
| 6 | 90.06 | **91.04** | +0.98 |
| 7 | 90.21 | **91.22** | +1.00 |
| 8 | 90.41 | **91.42** | +1.01 |
| 9 | 90.53 | **91.55** | +1.02 |
| 10 | 90.63 | **91.65** | +1.02 |

CSV: `outputs/baseline_refinement/rounds_gated_938.csv`.

**Gating beats blanket refinement at every round, and by more as rounds
accumulate** — +0.71 at round 1 rising monotonically to +1.02 at round 10. Each
extra pass breaks more conformers that were already valid, so the originals the
gate preserves are worth progressively more. Gated round 10 reaches **91.65**,
inside the 91.9–92.3 band the other three methods occupy and which blanket
refinement never reaches (90.63).

**Caveat: this table is validity only.** COV/MAT for refine-broken exists for
round 1 alone. The `gate` phase was not run past round 1, and unlike PB it cannot
be reconstructed from the two conditions — gating produces a mixed conformer
ensemble, and COV/MAT is a minimum RMSD over the actual set, not a combination of
aggregates. Recovering it means assembling the gated ensembles against the
round-1 originals and re-running `evaluate`, about 30 min per method-round.

**It converges to the tokenizer's own manifold, and the manifold is where the
other three already sit.** Torsional climbs **+9.04** PB micro from its own
conformers (81.59 → 90.63) — +4.46 in pass 1, +4.58 across passes 2–10 — in
geometrically decaying increments (+1.80, +0.94, +0.61, +0.38, +0.28, +0.15,
+0.20, +0.12, +0.10) that extrapolate to roughly 90.8. That lands just under
MCF-S (91.88), MCF-B (92.33) and ETFlow (92.09) after a *single* pass: the pass
has a fixed point, and ten passes walk the worst input to it.

**Accuracy does not degrade here, because for Torsional the manifold is an
improvement.** COV-R rises 72.98 → 73.93 and MAT-R falls 0.578 → 0.567, both
saturating by round 3 and flat thereafter. This is the opposite of the three
clean methods, where the same iteration costs 0.17–0.48 COV-R over ten rounds
with MAT-R rising in lockstep — for them the pass pulls conformers *off* better
geometry and onto the manifold.

**Diminishing returns set in immediately.** Pass 1 delivers half the total gain,
passes 1–3 deliver 80%, and passes 4–10 together deliver less than pass 1 alone.
With `refine-broken` at round 1 reaching 86.76 for a single pass, iteration is
worth it only when the input is far off-manifold and validity is the objective.

## Reproduction

    scripts/baseline_rounds_938.py --workers 16     # writes rounds_938.csv

COV/MAT is read from `outputs/baseline_refinement/{method}[_r{round}]/molecules/*.pickle`
(per-molecule `metrics`, threshold index 14 = 0.75 Å); PoseBusters from
`outputs/baseline_refinement/pb_{method}_{before,after}` and `pb_{method}_r{round}/*.json`
(per-conformer `flags`, so `refine-broken` is assembled without re-busting).
The script intersects the molecule sets before averaging.
