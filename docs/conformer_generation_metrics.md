---
tags: [metrics, evaluation, conformer, coverage, matching, rmsd, geom-drugs]
type: reference
status: active
---

> **TL;DR:** COV (Coverage) measures "did we find the conformer?" (binary); MAT (Matching) measures "how close were we?" (continuous RMSD). Both have Recall (reference→generated) and Precision (generated→reference) variants.

| Field | Value |
|-------|-------|
| **Metrics** | COV-R, COV-P, MAT-R, MAT-P |
| **Comparison Method** | RMSD (Root Mean Square Deviation) |
| **Typical Threshold** | 0.5Å for coverage |
| **Benchmark Dataset** | GEOM-Drugs |

**Reference this doc when:** evaluating conformer generation, interpreting HP sweep results, comparing to baselines, understanding recall vs precision tradeoffs

---

# Conformer Generation Metrics

## The Core Task

3DMolGen predicts 3D molecular conformers from SMILES strings:

```
Input:  "CCO" (SMILES string - flat 2D topology)
           ↓
       [Language Model]
           ↓
Output: Multiple 3D "snapshots" of the molecule
```

A single molecule isn't rigid—it's a chain of atoms that can rotate around bonds. Each valid "pose" is called a **conformer**:

```
Same molecule, different conformers:

   Conformer A        Conformer B        Conformer C
      ○                  ○                    ○
     /                  /                      \
    ○                  ○───○                    ○
     \                                         /
      ○                                       ○
```

## RMSD: The Distance Measure

**RMSD** (Root Mean Square Deviation) measures how different two 3D structures are. Imagine overlaying two conformers optimally and measuring how far apart corresponding atoms are on average.

| RMSD Value | Interpretation |
|------------|----------------|
| < 0.5Å | Excellent match |
| < 1.25Å | Good match |
| > 2.0Å | Poor match |

## The Two Metric Families

```
┌─────────────────────────────────────────────────────────────────┐
│                     REFERENCE SET                               │
│              (Ground-truth conformers from GEOM)                │
│                                                                 │
│      [R1]      [R2]      [R3]      [R4]      [R5]              │
│       ●         ●         ●         ●         ●                │
└─────────────────────────────────────────────────────────────────┘
                              ↕ Compare via RMSD
┌─────────────────────────────────────────────────────────────────┐
│                     GENERATED SET                               │
│               (Model's predicted conformers)                    │
│                                                                 │
│    [G1]    [G2]    [G3]    [G4]    [G5]    [G6]    [G7]        │
│     ◆       ◆       ◆       ◆       ◆       ◆       ◆          │
└─────────────────────────────────────────────────────────────────┘
```

---

## COV (Coverage) — "Did we find it?"

Binary question per conformer: Is there a match within threshold (typically 0.5Å)?

### Coverage Recall (COV-R)

**Question:** "What fraction of ground-truth conformers did we recover?"

```
Reference conformers:  ●   ●   ●   ●   ●
Has close match?       ✓   ✓   ✗   ✓   ✗

COV-R = 3/5 = 0.60
```

**Interpretation:** Of all the real conformers that exist, how many did we find?

### Coverage Precision (COV-P)

**Question:** "What fraction of our generated conformers were correct?"

```
Generated conformers:  ◆   ◆   ◆   ◆   ◆   ◆   ◆
Matches reference?     ✓   ✗   ✓   ✗   ✓   ✓   ✗

COV-P = 4/7 = 0.57
```

**Interpretation:** Of everything we generated, how much was actually valid?

### Coverage Patterns

| Pattern | Meaning |
|---------|---------|
| High COV-R, Low COV-P | Found most references but generated lots of junk |
| Low COV-R, High COV-P | Precise but missing some conformers |
| Both high | Ideal performance |

---

## MAT (Matching) — "How close were we?"

Continuous measure: Average RMSD to closest match. Unlike COV, this captures *how close* the matches are, not just whether they're within threshold.

### Matching Recall (MAT-R)

**Question:** "For each reference, how close was the nearest generated conformer?"

```
For each REFERENCE ●, find closest GENERATED ◆:

    ●───────0.3Å───────◆
    ●───────0.8Å───────◆
    ●───────0.4Å───────◆
    ●───────1.5Å───────◆

MAT-R = average of these distances (lower raw RMSD = better)
```

**Note:** In evaluation code, this is often normalized so higher = better.

### Matching Precision (MAT-P)

**Question:** "For each generated conformer, how close was the nearest reference?"

```
For each GENERATED ◆, find closest REFERENCE ●:

    ◆───────0.2Å───────●
    ◆───────0.5Å───────●
    ◆───────1.2Å───────●
    ◆───────0.4Å───────●

MAT-P = average of these distances
```

### Matching Patterns

| Pattern | Meaning |
|---------|---------|
| MAT-P > MAT-R | Generated conformers are close to *something* in reference, but not covering all references equally |
| MAT-R > MAT-P | Good coverage of references, but some generations are far from any reference |

---

## Error Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| `smiles_mismatch` | Generated SMILES tokens don't match input prompt | 0 |
| `mol_parse_fail` | RDKit couldn't parse output into valid molecule | 0 |
| `no_eos` | Model didn't produce end-of-sequence token (hit max_length) | 0 |

---

## Temperature/Sampling Tradeoffs

Higher temperature increases diversity:

```
Temperature Effect:

         COV-R (Recall)              COV-P (Precision)
         ↑ higher = better           ↑ higher = better

temp 0.8 ████████░░ ~0.65           temp 0.8 ██████████ ~0.59
temp 1.0 █████████░ ~0.67           temp 1.0 ████████░░ ~0.56
temp 1.2 ██████████ ~0.68           temp 1.2 ███████░░░ ~0.53
```

**Pattern:** Higher temperature → Better recall (more diverse = finds more references), worse precision (more "wrong" conformers too).

---

## Comparison to Related Work

These metrics are standard in conformer generation papers:

| Method | Typical COV-R | Typical MAT-R |
|--------|---------------|---------------|
| RDKit ETKDG | ~0.4-0.5 | ~0.8-1.0 RMSD |
| GeoMol | ~0.7-0.8 | ~0.5-0.7 RMSD |
| Torsional Diffusion | ~0.8-0.9 | ~0.4-0.6 RMSD |
| 3DMolGen (this project) | ~0.65-0.68 | ~0.63-0.67 |

*Values vary by threshold and dataset split.*

---

## Quick Reference

```
COV = "Did we find it?"  (binary, threshold-based)
      Recall:    "What fraction of ground truth did we recover?"
      Precision: "What fraction of our guesses were correct?"

MAT = "How close?"       (continuous RMSD)
      Recall:    "Average distance to closest generated, per reference"
      Precision: "Average distance to closest reference, per generated"
```

## See Also

- `src/molgen3D/evaluation/inference.py` — Generation pipeline
- `src/molgen3D/evaluation/posebusters_check.py` — Validation scoring
- `outputs/eval_sweep_results_extraction/` — HP sweep results
