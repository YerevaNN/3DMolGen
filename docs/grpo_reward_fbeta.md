# GRPO Reward (F-beta coverage)

This page documents the **F-beta reward** implementation used in `grpo_reward_fbeta.py`. It focuses on the F‑beta formulation (COV‑R / COV‑P driven), the gating logic, and the metrics logged to W&B.

---

## 1) What the reward is optimizing (plain English)

For each prompt (a molecule), the policy generates **K rollouts** (conformer blocks). Let:

- Prompt molecule: canonical SMILES `s`
- Generated rollouts: `y_1, ..., y_K`
- Reference conformers for this molecule: `g_1, ..., g_M` (from GEOM‑Drugs, capped by `max_ground_truths`)
- RMSD distance matrix:

```math
D_{i,j} = \mathrm{RMSD}(y_i, g_j)
```

The reward is **set‑level** and balances:
1) **Precision**: generated conformers are close to some GT.  
2) **Recall**: GT conformers are covered by at least one generated conformer.  
3) **Completion & diversity**: generate enough **valid** and **unique** conformers.

---

## 1.1 Symbols & key hyperparameters

**Symbols**
- `K`: number of conformer blocks *considered per completion* (from `target_conformers`).
- `M`: number of ground‑truth conformers (capped by `max_ground_truths`).
- `D_{i,j}`: RMSD between generated conformer `i` and GT conformer `j`.

**Key hyperparameters**
- `fbeta_delta` (`\delta`): RMSD threshold for “covered”.
- `fbeta_beta` (`\beta`): F‑beta tradeoff (>\!1 favors recall, <\!1 favors precision).
- `fbeta_gamma` (`\gamma`): completion exponent.
- `fbeta_dup_rmsd_tau` (`\tau`): duplicate RMSD threshold for uniqueness.
- `fbeta_warmup_lambda` (`\lambda_{warm}`): warmup bonus weight.
- `fbeta_warmup_sigma` (`\sigma`): warmup soft‑precision scale.
- `fbeta_recall_ref_sample`: number of GT conformers sampled for recall.

---

## 2) Validity gating (what counts as “eligible”)

Each completion is parsed into at most `target_conformers` blocks (this is `K`):

```text
[CONFORMER] ... [/CONFORMER]
```

Each block is valid only if it passes **all** of:

### 2.1 Decode + RDKit conformer
- Decode (binned or unbinned) succeeds.
- RDKit molecule exists with ≥1 conformer.

### 2.2 Graph match gate (hard)
- `same_molecular_graph(canonical_smiles, generated_smiles)` must be **True**.
- If mismatch, the block is invalid and excluded.

### 2.3 PoseBusters gate (optional but hard if enabled)
- If `grpo.posebusters.mode != "off"`, all base-valid blocks are checked.
- Failing blocks are invalid and excluded.

Let `t_valid` be the number of valid blocks after all gates.

If `t_valid == 0`, the completion gets **0 reward**.

---

## 2.4 Generation-time logits processor (rollout control)

During rollout generation, we attach a **conformer control logits processor**
and a **stopping criterion** (see `logits_constraints.py`) when
`grpo.enable_conformer_logits_processor` is enabled.

### What it enforces
- **Forces** a `[CONFORMER]` tag at the start and **after every** `[/CONFORMER]`
  until `target_conformers` blocks are generated.
- **Bans SMILES start tokens** (e.g., `[SMILES]`) and **pad tokens** during
  conformer generation to avoid drifting into non‑conformer text.
- **Never bans** conformer start/end tags (they are explicitly unbanned).
- **Stops generation** once each sequence contains `target_conformers`
  `[/CONFORMER]` tags.

### How it works (high‑level)
- The processor counts start/end tag occurrences in each sequence.
- If the sequence just ended a conformer and has not reached `target_conformers`,
  it **hard‑forces** the next token to be the next `[CONFORMER]` tag token.
- Banned tokens are set to `-inf` in the logits.
- If a row becomes all `-inf` (e.g., after constraints), it falls back to a safe
  conformer token so sampling can continue.

### Key knobs
- `grpo.target_conformers`: number of conformer blocks to force and stop at.
- `grpo.enable_conformer_logits_processor`: enable/disable constraints.

---

## 2.5 GRPO gating options (config switches)

These knobs control *when* a completion is allowed to contribute reward:

- **`fbeta_min_valid_to_score`**  
  Minimum number of valid conformers required for any non‑zero reward.
  If `t_valid < fbeta_min_valid_to_score`, reward is forced to `0`.

- **`fbeta_drop_if_valid_lt`**  
  A stricter gate applied after reward computation. If `t_valid` is below this,
  reward is **zeroed out**, regardless of F‑beta. (Default: `3`.)

- **`grpo.posebusters.*`**  
  Enables PoseBusters gating (`mode: basic|geometry|full`). When enabled, any
  block that fails PoseBusters is treated as invalid and excluded from `t_valid`.

- **Graph match gate (hard)**  
  If `same_molecular_graph(prompt_smiles, generated_smiles)` is false, the block
  is invalid. This prevents rewarding conformers for the wrong molecule.

- **Decode/RDKit parse gate (hard)**  
  If decode fails or the RDKit conformer is missing, the block is invalid.

These gates collectively determine `t_valid`, which drives both:
1) the **completion factor** and **uniq_frac**, and  
2) whether the completion gets **any reward at all**.

---

## 3) Coverage metrics used by the reward

### 3.1 Precision coverage (COV‑P)

Let:

```math
d_i = \min_j D_{i,j}
```

Then (with `fbeta_delta = \delta`):

```math
\mathrm{COV\!-\!P} = \frac{1}{K}\sum_{i=1}^{K} \mathbb{1}[d_i < \delta]
```

This measures what fraction of generated conformers are within `delta` of **some** GT.

### 3.2 Recall coverage (COV‑R)

Let:

```math
\tilde{d}_j = \min_i D_{i,j}
```

Then (with `fbeta_delta = \delta`):

```math
\mathrm{COV\!-\!R} = \frac{1}{M}\sum_{j=1}^{M} \mathbb{1}[\tilde{d}_j < \delta]
```

This measures what fraction of GT conformers are covered by **any** generated conformer.

> Implementation note: COV‑R can be computed on a **random subset** of GT conformers
> (size `fbeta_recall_ref_sample`) for speed. COV‑P always uses the full GT set.

---

## 4) F‑beta core reward

The core F‑beta score is:

```math
\mathrm{F}_\beta =
\frac{(1+\beta^2)\,\mathrm{COV\!-\!P}\,\mathrm{COV\!-\!R}}
{\beta^2\,\mathrm{COV\!-\!P} + \mathrm{COV\!-\!R} + \varepsilon}
```

(`\varepsilon` is a small constant to avoid divide‑by‑zero.)

This is the standard F‑beta harmonic tradeoff between precision and recall.
`fbeta_beta` controls the tradeoff (>\!1 favors recall, <\!1 favors precision).

---

## 5) Warmup precision bonus

During early training, a **soft precision** term keeps gradients alive even when
COV values are low. It is computed from the min RMSD per generated conformer,
using `fbeta_warmup_sigma = \sigma`:

```math
\mathrm{soft\_p} = \frac{1}{K} \sum_{i=1}^{K}
\exp\!\left(-\frac{d_i^2}{\sigma^2}\right)
```

---

## 6) Completion + diversity scaling

The final reward is scaled by:

### 6.1 Completion factor

```math
\mathrm{completion} =
\left(\frac{t_{\mathrm{valid}}}{K}\right)^\gamma
```

Where `K = target_conformers` and `\gamma = fbeta_gamma`.

### 6.2 Uniqueness fraction

Valid conformers are deduplicated: a conformer counts as a **duplicate** if its
RMSD to any earlier valid conformer is `< \tau` (where `\tau = fbeta_dup_rmsd_tau`).

```math
\mathrm{uniq\_frac} = \frac{t_{\mathrm{uniq}}}{t_{\mathrm{valid}}}
```

---

## 7) Final reward formula

```math
r =
(\mathrm{F}_\beta + \lambda_{\mathrm{warm}} \cdot \mathrm{soft\_p})
\cdot \mathrm{completion} \cdot \mathrm{uniq\_frac}
```

Where `\lambda_{warm} = fbeta_warmup_lambda`.

### Hard floor rules

- If `t_valid < fbeta_min_valid_to_score`: reward = 0
- If `t_valid < fbeta_drop_if_valid_lt`: reward = 0

---

## 8) Hyperparameters (what they do)

**Core thresholds**
- `fbeta_delta` (`\delta`): RMSD threshold for “covered”. Lower = stricter.
- `fbeta_beta` (`\beta`): tradeoff in F‑beta. `>1` favors **recall** (COV‑R), `<1` favors **precision** (COV‑P).

**Coverage sampling**
- `fbeta_recall_ref_sample`: number of GT conformers sampled for recall. Lower = faster but noisier COV‑R estimate.

**Completion & diversity**
- `target_conformers` (`K`): max conformer blocks parsed per completion.
- `fbeta_gamma` (`\gamma`): exponent on completion factor; higher = stronger penalty for missing valid blocks.
- `fbeta_dup_rmsd_tau` (`\tau`): RMSD threshold for considering two valid conformers “duplicates”. Lower = harsher uniqueness.

**Warmup**
- `fbeta_warmup_lambda` (`\lambda_warm`): weight on soft precision bonus.
- `fbeta_warmup_sigma` (`\sigma`): scale for soft precision; smaller = more punitive to larger RMSD.

**Decoding**
- `fbeta_use_binned_decoder`: decode binned conformer tokens.
- `fbeta_binned_ranges`, `fbeta_binned_bin_size`: define the cartesian binning grid.

**Gates & compute**
- `max_ground_truths`: max GT conformers loaded.
- `rmsd_workers`: CPU workers for RMSD.
- `grpo.posebusters.*`: controls PoseBusters gating.

---

## 9) Example with numbers (simple intuition)

Assume:
- `COV‑R = 0.40`, `COV‑P = 0.70`, `\beta = 1.5`

Then:

```math
\mathrm{F}_\beta = \frac{(1+2.25)\cdot 0.70\cdot 0.40}{2.25\cdot 0.70 + 0.40}
 \approx 0.461
```

If `soft_p = 0.70`, `\lambda_{warm}=0.3`, then:

```math
\mathrm{F}_\beta + \lambda_{warm}\cdot \mathrm{soft\_p}
 \approx 0.461 + 0.21 = 0.671
```

Now suppose:
- `t_valid = 6`, `K=8`, `\gamma=1.5`
- `uniq_frac = 4/6 = 0.667`

Then:

```math
\mathrm{completion} = (6/8)^{1.5} \approx 0.650
```

Final reward:

```math
r \approx 0.671 \cdot 0.650 \cdot 0.667 \approx 0.29
```

**Effect summary:**
- Increasing `\delta` raises both COV‑R/P (easier to be “covered”).
- Increasing `\beta` shifts reward toward **recall**.
- Increasing `\gamma` penalizes missing valid conformers more strongly.
- Lower `\tau` penalizes duplicates more strongly.
- Increasing `\lambda_{warm}` helps early training but can over‑reward low‑quality conformers if too large.

---

## 10) Metrics logged to W&B

These are emitted by `grpo_reward_fbeta.py` when `wandb.run` is active and `step % log_every_steps == 0`:

### Reward summary
- `reward/final_mean`, `reward/final_std`
- `reward/fbeta_mean`
- `reward/cov_r_mean`, `reward/cov_p_mean`
- `reward/completion_mean`, `reward/uniq_frac_mean`
- `reward/t_valid_mean`, `reward/t_valid_p50`, `reward/t_valid_p90`
- `reward/nonzero_frac`

### Parsing & gates
- `parse/n_blocks_found_mean`, `parse/n_blocks_parsed_mean`
- `parse/n_blocks_found_total`, `parse/n_blocks_parsed_total`
- `parse/fraction_with_ltK_blocks`, `parse/fraction_empty_completion`
- `gate/decode_fail_rate`, `gate/rdkit_fail_rate`
- `gate/smiles_mismatch_rate`, `gate/posebusters_fail_rate`
- `gate/valid_rate`

### RMSD stats
- `rmsd/gen_min_mean`, `rmsd/gen_min_p50`, `rmsd/gen_min_p90`
- `rmsd/ref_min_mean`, `rmsd/ref_min_p50`, `rmsd/ref_min_p90`

### Collapse diagnostics
- `collapse/uniq_frac_mean`
- `collapse/percent_rollouts_all_same`

---

## 10.1 Metrics intuition with a simple example

Assume one batch has **100 completions**, `K=8` target conformers each.
Suppose the model produces an average of **6 valid conformers per completion**,
and those valid conformers are **mostly unique** (average uniq fraction 0.75).

**Completion‑level metrics**
- `reward/t_valid_mean ≈ 6.0`  
  On average, each completion had 6 valid blocks out of 8.
- `reward/completion_mean ≈ (6/8)^{\gamma}`  
  If `\gamma=1`, that would be `0.75`. If `\gamma=1.5`, it’s smaller.
- `reward/uniq_frac_mean ≈ 0.75`  
  Roughly 75% of valid conformers are unique after RMSD deduplication.

**Coverage metrics**
Suppose that, across all valid conformers:
- **70%** are within `\delta` of some GT conformer → `reward/cov_p_mean ≈ 0.70`.
- **45%** of GT conformers are covered by at least one generated conformer
  (or by the sampled recall set) → `reward/cov_r_mean ≈ 0.45`.

These feed into `reward/fbeta_mean`, which will sit between COV‑R and COV‑P
depending on `fbeta_beta` (>\!1 pushes it toward recall).

**RMSD metrics**
If the distribution of min RMSDs looks like:
- median 0.9 Å, 90th percentile 1.7 Å
then:
- `rmsd/gen_min_p50 ≈ 0.9`, `rmsd/gen_min_p90 ≈ 1.7`

**Parsing/gating**
If 800 conformer blocks were parsed and 20 failed decode:
- `gate/decode_fail_rate = 20 / 800 = 0.025`

If 5% had wrong SMILES graphs:
- `gate/smiles_mismatch_rate ≈ 0.05`

**Collapse diagnostics**
If some completions generate the *same* conformer repeatedly:
- `collapse/percent_rollouts_all_same` rises.
  For example, if 10 out of 100 completions have all valid conformers
  identical, the metric is `0.10`.

These examples should help anchor what “good” looks like:
high `cov_p`, improving `cov_r`, low gate failure rates, and low collapse.

## 11) Implementation notes

- COV‑P uses **all** GT conformers; COV‑R may use a **sampled subset** for speed.
- Uniqueness is computed by RMSD pairwise against already‑accepted valid conformers.
- All reward components are computed **per completion** (not per block), then averaged.
