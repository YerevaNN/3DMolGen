# GRPO Reward Design (GEOM-Drugs aligned) — How it works and what each term is meant to do

This document explains the **design intent** of the GRPO reward function used for 2D→3D conformer generation on GEOM-Drugs.
The reward is computed **per prompt**, where the policy generates a **set of K rollouts** (completions), and we score each rollout with components that collectively target the benchmark’s goals.

---

## 0) Setup and notation

For one prompt (one molecule):

- We have **K rollouts**: generated conformers \(y_1, \dots, y_K\).
- We load **M ground-truth conformers** \(g_1, \dots, g_M\) for the same canonical SMILES (bounded by `max_ground_truths`).

We compute the distance matrix (RMSD):

\[
D_{ij} = \text{RMSD}(y_i, g_j)
\]

We define a validity indicator:

\[
v_i \in \{0,1\}
\]

where \(v_i=1\) means rollout \(i\) is considered valid (graph-matched, parseable, finite RMSD if gated), and invalid rollouts receive a fixed floor reward \(r_{\text{floor}}<0\).

---

## 1) Validity gate (hard constraint before reward shaping)

### What it checks
A rollout is considered valid if it passes:
1. **Graph match / same molecule**: the SMILES extracted from the generated conformer corresponds to the same molecular graph as the prompt’s canonical SMILES.
2. **Parseable conformer**: the completion contains a `[CONFORMER]...[/CONFORMER]` block and decoding produces an RDKit Mol.
3. (Optional) **Finite RMSD**: if the `hard_rmsd_gate` is enabled, rollouts without finite RMSD to any GT are invalidated.

### Expected effect
- Prevents the policy from “gaming” rewards with malformed outputs.
- Forces learning to happen **only on chemically consistent generations**.
- Creates a strong pressure to produce valid format + correct graph early in training.

If invalid:
\[
r_i = r_{\text{floor}}
\]

If valid:
\[
r_i = \lambda_{\text{qual}} r^{\text{qual}}_i + \lambda_{\text{smcov}} r^{\text{smcov}}_i + \lambda_{\text{match}} r^{\text{match}}_i
\]

---

## 2) Reward term #1: Dense quality \(r^{\text{qual}}\) (AMR-P proxy)

### Definition
For a valid rollout \(i\), we take its **best** (minimum) RMSD among GT conformers:

\[
d_i = \min_j D_{ij}
\]

Then we shape it into a smooth reward:

\[
r^{\text{qual}}_i = \exp\left(-\frac{d_i}{\sigma}\right)
\]

(Your implementation uses this exact form; \(\sigma\) controls how quickly reward decays with RMSD.)

### Design intent
This is the “make conformers good” term:
- It gives **dense learning signal** every step (not sparse thresholding).
- It directly improves “how close is at least one generated conformer to some reference”.

### Expected effect on behavior
- Pulls generated conformers toward *some* plausible ground-truth modes.
- Improves distributions of `min RMSD` metrics (`geom/d_min_*` style plots).
- If this dominates too much, the policy can collapse to a few easy modes (diversity risk).

---

## 3) Reward term #2: Smooth marginal coverage \(r^{\text{smcov}}\) (set-aware, pre-threshold coverage)

This term is **group-aware**: it is not just about “is this rollout close”, but “does this rollout add new coverage given the other rollouts.”

### Step 1: Convert distances to soft coverage strength
\[
K_{ij} = \exp\left(-\left(\frac{D_{ij}}{\rho}\right)^2\right)
\]

- If \(D_{ij}\) is small → \(K_{ij}\approx 1\)
- If \(D_{ij}\) is large → \(K_{ij}\approx 0\)

### Step 2: Define a *set-level* soft coverage objective
For a fixed GT conformer \(g_j\), the probability-like “covered by at least one rollout” quantity is:

\[
\text{soft\_cov}_j(S) = 1 - \prod_{i\in S}(1-K_{ij})
\]

Where \(S\) is the set of all K rollouts.
Averaging across GTs defines a set score:

\[
F(S) = \frac{1}{M}\sum_{j=1}^M \left(1 - \prod_{i\in S}(1-K_{ij})\right)
\]

### Step 3: Per-rollout credit = marginal contribution to the set score
Instead of assigning \(F(S)\) to everyone (which would give identical rewards), each rollout receives its marginal contribution:

\[
r^{\text{smcov}}_i = F(S) - F(S\setminus\{i\})
\]

This has a closed form:

\[
r^{\text{smcov}}_i = \frac{1}{M}\sum_{j=1}^M K_{ij}\prod_{k\neq i}(1-K_{kj})
\]

### Design intent
This term is meant to improve **coverage with small K** by encouraging complementary rollouts:
- If rollout \(i\) covers GT \(j\) but others already cover \(j\), then \(\prod_{k\neq i}(1-K_{kj})\) is small → \(i\) gets little credit.
- If rollout \(i\) covers a GT that others do not, the product term is near 1 → \(i\) gets strong credit.

### Expected effect on behavior
- Encourages rollouts to spread across distinct GT modes (anti-redundancy).
- Provides a **dense, pre-threshold** signal even when \(D_{ij}\) is not yet under the hard \(\delta\) threshold.
- Helps fight mode collapse *if its magnitude is comparable to the other terms* (scaling matters).

---

## 4) Reward term #3: Hard unique matching \(r^{\text{match}}\) (COV-R under \(\delta\) proxy)

This term targets the GEOM “recall/coverage under threshold” type metric in a more discrete way.

### Eligibility under threshold
An edge \((i,j)\) is eligible if:
\[
v_i = 1 \quad\text{and}\quad D_{ij} < \delta
\]

### Max-cardinality one-to-one matching
We solve a bipartite matching problem between rollouts and GTs:
- One rollout can match at most one GT.
- One GT can be matched by at most one rollout.
- We aim for **max number of matches** (coverage), and use distance as a tie-break (min cost).

Let \(\mathcal{M}\) be the set of matched pairs \((i,j)\).

### Per-rollout match reward (shaped)
For matched rollouts:
\[
r^{\text{match}}_i =
\max\left(0,\ 1-\frac{D_{ij}}{\delta}\right)
\quad\text{if }(i,j)\in \mathcal{M}
\]
Otherwise:
\[
r^{\text{match}}_i = 0
\]

### Design intent
This is the “hard coverage” term:
- Encourages generating **multiple distinct conformers** that each match different GTs under \(\delta\).
- Enforces uniqueness by construction (one-to-one matching).

### Expected effect on behavior
- Increases the count of GT conformers that are covered under \(\delta\) when K is limited.
- Promotes diversity *only insofar as it helps produce more distinct under-\(\delta\) hits*.
- Can become sparse early if few rollouts ever cross the \(\delta\) threshold (smcov is meant to help before that).

---

## 5) Final combined reward

For each rollout \(i\):

- If invalid:
\[
r_i = r_{\text{floor}}
\]

- If valid:
\[
r_i = \lambda_{\text{qual}} r^{\text{qual}}_i
    + \lambda_{\text{smcov}} r^{\text{smcov}}_i
    + \lambda_{\text{match}} r^{\text{match}}_i
\]

### What each coefficient does
- \(\lambda_{\text{qual}}\): prioritizes “make at least one conformer very close”.
- \(\lambda_{\text{smcov}}\): prioritizes “make the set cover more GT modes (softly)”.
- \(\lambda_{\text{match}}\): prioritizes “hit as many distinct GT conformers under \(\delta\) as possible”.

---

## 6) Intuition: why these 3 together

- **Quality** makes outputs geometrically plausible and improves RMSD distribution.
- **Smooth coverage** provides dense, set-aware shaping so rollouts become complementary even before they cross the hard threshold.
- **Hard matching** directly targets the “unique hits under \(\delta\)” notion needed for recall/coverage metrics in GEOM-Drugs, especially when K is small.

In short:
- \(r^{\text{qual}}\): *“be good”*
- \(r^{\text{smcov}}\): *“be non-redundant as a set (softly)”*
- \(r^{\text{match}}\): *“be non-redundant as a set under the benchmark threshold”*