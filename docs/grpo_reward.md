# GRPO Reward v3 (GEOM‑Drugs aligned)

This document describes the **current** “reward_v3” implementation you pasted (the refactored/profiling version). It is written from the *reward-design* perspective: what each term is trying to incentivize, how validity gating works, and how to interpret/tune the key hyperparameters.

---

## 1) What the reward is optimizing

For each prompt (a molecule), the policy generates **K rollouts** (conformers). Let:

- Prompt molecule (ground truth set): canonical SMILES `s`
- Generated rollouts: $y_1,\dots,y_K$
- Reference conformers for this molecule: $g_1,\dots,g_M$ (loaded from GEOM‑Drugs, capped by `max_ground_truths`)
- RMSD distance matrix:  
  $$
  D_{i,j} = \text{RMSD}(y_i, g_j)
  $$
  (computed only for valid rollouts; invalid rows remain $+\infty$)

The per-rollout total reward is:
$$
r_i = 
\begin{cases}
r_{\text{floor}} & \text{if rollout } i \text{ is invalid}\\
\lambda_{\text{qual}}\,r^{\text{qual}}_i + \lambda_{\text{smcov}}\,r^{\text{smcov}}_i + \lambda_{\text{match}}\,r^{\text{match}}_i & \text{otherwise}
\end{cases}
$$

So the model is pushed to:
1) produce **chemically consistent** conformers for the right graph,  
2) make each rollout **close** to some reference,  
3) make the **set** of rollouts cover **many different** references (diversity/coverage under small K).

---

## 2) Validity gating (what counts as “eligible” for reward)

A rollout is treated as valid only if it passes **all** of these:

### 2.1 Conformer tag + RDKit parse
- The completion must contain `"[CONFORMER]" ... "[/CONFORMER]"`.
- `decode_cartesian_v2()` must succeed (RDKit molecule created).

### 2.2 Graph match gate (hard)
- The rollout is decoded to a SMILES string (via `strip_smiles(conformer_text)`).
- `same_molecular_graph(canonical_smiles, generated_smiles)` must be **True**.

If graph mismatch, the rollout gets `r_floor` (no point rewarding geometry for the wrong molecule).

### 2.3 PoseBusters gate (configurable but hard when enabled)
- If `grpo.posebusters.mode` is anything other than `"off"`, every **base-valid** rollout is checked with PoseBusters before geometry rewards are computed.
- Rollouts that fail the selected PoseBusters suite (or cause PoseBusters to error) are marked invalid and receive `r_floor`.  
- The metrics `pose/checked_rate`, `pose/pass_rate`, and `pose/error_rate` respectively track how many base-valid samples reached PoseBusters, how many of those passed, and what fraction errored.

### 2.4 Finite RMSD gate (hard)
Even if PoseBusters (or the base gate) passes, the rollout must have a **finite** $\min_j D_{i,j}$. If all RMSDs fail / are `inf`, the rollout is treated as invalid and gets `r_floor`.

> Note on `hard_rmsd_gate`: in the current code, *finite RMSD is always required* for validity; the knob mainly controls logging/stats messaging. If you intended "graph-valid but no RMSD" to still be considered valid (with e.g. $r^{\mathrm{qual}}=0$), that requires a code change.

---

## 3) The three reward components

### 3.1 Quality term $r^{\mathrm{qual}}$: "be close to *some* ground truth"
For each rollout $i$, define:
$$
d_i = \min_j D_{i,j}
$$
Then:
$$
r^{\text{qual}}_i = \exp(-d_i/\sigma)
$$
- **Effect:** pushes each rollout toward *any* plausible conformer in the reference set.  
- **$\sigma$** controls how quickly reward decays with RMSD (smaller $\sigma$ = harsher).

---

### 3.2 Smooth marginal coverage $r^{\mathrm{smcov}}$: "don't all crowd the same GT"
This is the **group-aware** term. It rewards rollouts that contribute **new** coverage of the reference set *given what the other rollouts already cover*.

Define a soft "hit kernel":
$$
K_{i,j} = \exp\!\left( -(D_{i,j}/\rho)^2 \right)
$$
Soft probability reference $j$ is covered by the **set**:
$$
\text{soft\_cov}_j = 1 - \prod_{i=1}^{K} (1 - K_{i,j})
$$
Marginal contribution of rollout $i$ for reference $j$:
$$
\Delta_{i,j} = K_{i,j}\prod_{i'\neq i}(1 - K_{i',j})
$$
Rollout reward:
$$
r^{\text{smcov}}_i = \frac{1}{M}\sum_{j=1}^{M}\Delta_{i,j}
$$

- **Intuition:** if a reference is already covered by other rollouts, the product term becomes small, so you get **little** marginal credit for also covering it. If it is *not* covered, you get **more** credit.
- **$\rho$** controls the softness radius (larger $\rho$ = more forgiving distances contribute to coverage).

---

### 3.3 Hard matching bonus $r^{\mathrm{match}}$: "unique coverage under $\delta$"
Define eligible edges if:
$$
D_{i,j} < \delta
$$
Then compute a **max-cardinality one-to-one matching** (Hungarian assignment). This uses `scipy.optimize.linear_sum_assignment`. 

Matched rollout shaping:
$$
r^{\text{match}}_i = \max\left(0,\ 1-\frac{D_{i,j}}{\delta}\right)
$$

- **Effect:** directly rewards "unique hits" under a hard threshold $\delta$.
- **$\delta$** should usually be aligned to the evaluation threshold you care about.

---

## 4) What gets logged (and how to interpret it)

Metrics are emitted to W&B (when a `wandb.run` is active) under these keys:

### Validity & gating
- `gate/graph_match_rate`, `gate/rdkit_parse_rate`, `gate/base_valid_rate`, `gate/final_valid_rate`
- `pose/checked_rate`, `pose/pass_rate`, `pose/error_rate`

### Geometry quality
- `rmsd/d_min_mean`, `rmsd/d_min_p50`, `rmsd/d_min_p90`
- `rmsd/frac_under_delta`, `rmsd/frac_under_2delta`

### Coverage & utilization
- `cov/refs_hit_mean`, `cov/refs_hit_p50`, `cov/cov_ratio_mean`
- `cov/unique_nearest_refs_mean`, `cov/nearest_collision_rate_mean`
- `cov/valid_rollouts_mean`

### Matching diagnostics
- `match/num_matched_mean`, `match/max_possible_mean`, `match/efficiency_mean`
- `match/matched_dist_p50`, `match/matched_dist_p90`
- `match/eligible_edge_density`

### Smooth marginal coverage
- `smcov/soft_cov_mean`
- `smcov/pct_gt_cov_gt_0p1`, `smcov/pct_gt_cov_gt_0p5`
- `smcov/corr_with_refs_hit`

### Reward decomposition
- `reward/total_mean`, `reward/total_std`
- `reward/comp_qual_mean`, `reward/comp_smcov_mean`, `reward/comp_match_mean`
- `reward/comp_smcov_frac`

### Optional diversity proxy
- `div/pairwise_rmsd_p50`: median pairwise RMSD among PoseBusters+RMSD-valid rollouts (logged only when pairwise logging is enabled)

---

## 5) Hyperparameters: meaning + practical tuning guidance

### 5.1 Reward-shape hyperparameters

**`sigma` (quality sharpness)**  
- Smaller = only very close conformers get meaningful $r^{\mathrm{qual}}$.  
- Bigger = broader gradient signal but may tolerate sloppier geometry.  
- Typical: 0.25–0.45.

**`rho` (coverage softness radius)**  
- Larger = broader "soft hits", stronger set-level shaping from farther RMSDs.  
- Too large can make everything look covered (weak uniqueness pressure).  
- Typical: 0.6–1.0.

**`delta` (hard coverage threshold)**  
- Too low → almost no eligible edges → `r_match` mostly 0.  
- Too high → matching becomes easy but less meaningful.  
- Typical: 0.75–0.85.

**`lambda_*` (component weights)**  
- `lambda_qual`: "find a plausible conformer"
- `lambda_smcov`: "spread out across references"
- `lambda_match`: "hard unique hits under $\delta$"

A stable pattern: keep `lambda_qual` and `lambda_match` ~1, tune `lambda_smcov` (e.g., 2 → 8).

**`r_floor` (invalid penalty)**  
Typical: -0.5 to -1.0.

**`max_ground_truths`**  
Controls CPU cost $O(KM)$. Higher = richer target set but slower reward.

---

### 5.2 GRPO training hyperparameters (clip/LR intuition)

In TRL’s GRPO, the key stability controls are the ratio clipping parameters in `GRPOConfig`:
- `epsilon`: ratio clipping range  
- `delta`: “target clipping” parameter  
- `epsilon_high`: alternative upper clipping used in TR‑DPO style clipping

Practical rule:
- **Higher reward weights / higher variance rewards → lower LR and/or tighter clipping.**

Suggested starting region:
- `learning_rate`: **1e‑5 to 3e‑5**
- clip range (your `epsilon_*`): **2e‑4 to 6e‑4**
- `max_grad_norm`: 0.5–1.0
- `temperature`: 0.8–1.2

---

## 6) How to verify smcov is helping (not inflating)

When increasing `lambda_smcov` or `rho`, you want **at least two** of these to improve:

- **Hard coverage improves:** `match/efficiency_mean` and `cov/refs_hit_mean` trend up
- **Quality improves:** `rmsd/d_min_p50` / `rmsd/d_min_mean` trend down
- **No collapse:** optional `div/pairwise_rmsd_p50` does not fall sharply

If only `reward/comp_smcov_mean` rises while the others don’t move, smcov is likely being exploited as an easy soft term.

---

## 7) Implementation notes

- Matching uses Hungarian assignment via `linear_sum_assignment` and is disabled if SciPy is missing. 
- Reward is computed per prompt group (set-aware), then mapped back to the flat batch.
- Optional profiling can log per-section CPU times to W&B.

---
