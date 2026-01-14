# GRPO Training Metrics Guide

This guide explains the metrics emitted by the GRPO reward function for conformer generation. They cover validity gating, RMSD quality, coverage, reward shaping, and optional diagnostics so you can monitor both model correctness and reward-signal health.

## Overview

Metrics fall into eight families:
1. **Validity & Gating** – SMILES parsing, graph alignment, final survival rate
2. **PoseBusters** – Geometry health when PoseBusters is enabled
3. **Geometry Quality** – RMSD statistics and thresholds
4. **Coverage & Utilization** – How many references each batch touches
5. **Smooth Coverage Diagnostics** – Sigmoid-based coverage proxies
6. **Matching** – Hungarian assignment success and match quality
7. **Reward Components** – Contribution and variance of each reward term
8. **Diversity & Additional** – Optional pairwise RMSDs and other rollup stats

---

## 1. Validity & Gating Metrics

These describe how many completions pass the structural gates.

### `gate/graph_match_rate`
- **Definition**: Fraction of completions whose decoded SMILES matches the canonical prompt SMILES.
- **Interpretation**: Should remain >0.9 for conditioned generation. Drops mean the model is ignoring the prompt or emitting malformed SMILES.

### `gate/rdkit_parse_rate`
- **Definition**: Fraction of completions that RDKit can parse after extracting the `[CONFORMER]` block.
- **Interpretation**: >0.99 is healthy; sustained dips point to formatting issues in the text decoder.

### `gate/base_valid_rate`
- **Definition**: Joint success of graph match ∧ RDKit parse.
- **Interpretation**: Mirrors the lowest of the two upstream gates. Large gaps imply the pairwise combination is revealing extra issues (e.g., parses succeed but the graph drifts).

### `gate/final_valid_rate`
- **Definition**: Fraction of rollouts that survive PoseBusters (if enabled) plus the finite-RMSD gate.
- **Interpretation**: Aim for ≥0.75. When `grpo.hard_rmsd_gate=false`, non-finite RMSD rows still count here but will receive the reward floor, so track `rmsd/frac_under_delta` to see if geometry is actually usable.

---

## 2. PoseBusters Metrics

Only active when PoseBusters is on; otherwise these stay at 0.

### `pose/checked_rate`
- **Definition**: Checked rollouts divided by the base-valid count.
- **Interpretation**: Equals 1.0 when every eligible rollout reached PoseBusters. Lower values mean PoseBusters was skipped due to mode, errors, or missing molecules.

### `pose/pass_rate`
- **Definition**: Pass rate conditioned on PoseBusters being run.
- **Interpretation**: ≥0.95 indicates strong local geometry; <0.8 suggests clashes or chemistry failures that gate off most rewards.

### `pose/error_rate`
- **Definition**: Fraction of PoseBusters attempts that errored.
- **Interpretation**: Should stay at 0. Non-zero values usually mean PoseBusters lacks dependencies or encountered malformed molecules; those rollouts are forced invalid.

---

## 3. Geometry Metrics

These summarize best-RMSD quality for valid rollouts.

### `rmsd/d_min_mean`
- **Definition**: Mean of $d_i = \min_j \mathrm{RMSD}(y_i, g_j)$ over PoseBusters-valid rollouts with finite RMSD.
- **Interpretation**: ≤0.8 Å (for δ≈0.75–0.85) means good geometric fidelity; >1.0 Å signals the model is missing targets.

### `rmsd/d_min_p50`
- **Definition**: Median of the same $d_i$ distribution.
- **Interpretation**: <δ indicates most generations are usable; compare with the mean to judge skew.

### `rmsd/d_min_p90`
- **Definition**: 90th percentile of $d_i$.
- **Interpretation**: Keeps the tail in check. Values ≤2δ mean only a few outliers are bad.

### `rmsd/frac_under_delta`
- **Definition**: Fraction of valid rollouts with $d_i < \delta$.
- **Interpretation**: A proxy for “COV-R recall.” >0.5 shows most rollouts land within threshold.

### `rmsd/frac_under_2delta`
- **Definition**: Fraction with $d_i < 2\delta$.
- **Interpretation**: Indicates how many samples are recoverable with looser post-processing. Near 1.0 is ideal.

---

## 4. Coverage & Utilization Metrics

These look at discrete hit counts per prompt group.

### `cov/refs_hit_mean` and `cov/refs_hit_p50`
- **Definition**: Mean/median number of references that have at least one rollout under δ.
- **Interpretation**: Higher = broader coverage. Compare mean vs median to ensure coverage isn’t dominated by a single prompt.

### `cov/cov_ratio_mean`
- **Definition**: `refs_hit / M`, where $M$ is the number of reference conformers loaded.
- **Interpretation**: Normalized coverage fraction. Aim for ≥0.5 when K≈M; lower values mean the sampler revisits only a few references.

### `cov/unique_nearest_refs_mean`
- **Definition**: Average number of distinct references that are the *nearest* one for some valid rollout.
- **Interpretation**: Scales with diversity; values near 1–2 despite many valid rollouts indicate collapse.

### `cov/nearest_collision_rate_mean`
- **Definition**: $1 - \text{unique\_nearest} / \text{valid\_rollouts}$.
- **Interpretation**: Collision rates ≤0.3 imply rollouts spread across references; >0.6 means many rollouts share the same nearest neighbor.

### `cov/valid_rollouts_mean`
- **Definition**: Average number of PoseBusters-valid rollouts per prompt.
- **Interpretation**: Should track `K * gate/final_valid_rate`. Drops mean gating is discarding whole prompt groups.

### `covdiff/*` ratios
- `covdiff/cover_ratio_mean`, `covdiff/covered_ratio_mean`: Fraction of references whose best RMSD is <δ (the two match today because both read the same intermediate array).
- `covdiff/unique_cover_ratio_mean`, `covdiff/unique_covered_ratio_mean`: Fraction of references that are uniquely covered (best <δ, second-best ≥δ). Also duplicates currently but kept for future experiments.
- **Interpretation**: Use these to monitor whether the discrete coverage behavior that feeds the smcov reward remains high even if smooth proxies look good.

---

## 5. Smooth Coverage Diagnostics

These come directly from `compute_smooth_coverage_reward`.

### `bestcov/soft_cov_mean`
- **Definition**: Mean of $1 - \prod_i (1 - K_{i,j})$ across references, where $K_{i,j} = \sigma((\delta - D_{i,j}) / \rho)$.
- **Interpretation**: Dense soft coverage; should move with the smcov component. >0.7 indicates most references have a confident nearby rollout.

### `bestcov/pct_gt_cov_gt_0p1` and `bestcov/pct_gt_cov_gt_0p5`
- **Definition**: Fraction of references whose soft coverage exceeds 0.1 or 0.5.
- **Interpretation**: Helps distinguish broad-but-weak coverage (0.1) from strong hits (0.5).

### `bestcov/corr_with_refs_hit`
- **Definition**: Pearson correlation between per-prompt `refs_hit` and `soft_cov_mean`.
- **Interpretation**: Positive correlation means the smooth term aligns with actual discrete hits; near zero indicates the smooth reward may be inflated by distant rollouts.

---

## 6. Matching Metrics

### `match/num_matched_mean`
- **Definition**: Mean number of rollout↔reference matches found by the Hungarian solver.
- **Interpretation**: Goes up as more rollouts land under δ.

### `match/max_possible_mean`
- **Definition**: Mean of $\min(\text{valid\_rollouts}, \text{refs\_hit})$ per prompt.
- **Interpretation**: Upper bound for achievable matches; use with efficiency.

### `match/efficiency_mean`
- **Definition**: `num_matched / max_possible`.
- **Interpretation**: Near 1.0 means matching is saturated; low values indicate either sparse eligibility or solver failures.

### `match/matched_dist_p50` and `match/matched_dist_p90`
- **Definition**: Percentiles of RMSD values among matched pairs.
- **Interpretation**: `p50 < 0.5 Å` and `p90 < δ` mean matches are tight. Larger tails mean the solver is forced to accept marginal alignments.

### `match/eligible_edge_density`
- **Definition**: Eligible edges (`distance < δ`) divided by `K*M`.
- **Interpretation**: A dense graph (>0.1) gives the matching term room to work; extremely sparse graphs (<0.02) mean quality collapses regardless of solver.

---

## 7. Reward Components & Stability

### `reward/total_mean`, `reward/total_std`
- **Definition**: Mean and std of combined rewards over valid rollouts.
- **Interpretation**: Monitor for sign flips or exploding variance. Std much larger than mean implies unstable scaling.

### `reward/comp_qual_mean`, `reward/comp_smcov_mean`, `reward/comp_match_mean`
- **Definition**: Component contributions after multiplying by their λ’s and averaging over valid rollouts.
- **Interpretation**: Should roughly track λ-weighting. If one term dwarfs the others, consider retuning weights.

### `reward/comp_qual_std`, `reward/comp_smcov_std`, `reward/comp_match_std`
- **Definition**: Std dev of each weighted component over valid rollouts.
- **Interpretation**: Highlights which component introduces most variance into the policy gradient.

### `reward/qual_group_std_mean`, `reward/smcov_group_std_mean`, `reward/match_group_std_mean`, `reward/group_std_mean`
- **Definition**: Average within-prompt std for each component and for the combined reward.
- **Interpretation**: These mirror TRL’s `scale_rewards` behavior; high group std means strong per-prompt variance even after gating.

### `reward/bestcov_rank_corr_mean`
- **Definition**: Mean Spearman correlation between per-rollout smcov contributions and total rewards (computed within each prompt group).
- **Interpretation**: Positive values confirm the smooth coverage signal is aligned with the final reward ordering.

### `reward/comp_smcov_frac`
- **Definition**: `reward/comp_smcov_mean / (reward/total_mean + 1e-8)`.
- **Interpretation**: Fractional contribution of smcov to the overall reward. Values near 0 or 1 indicate imbalance.

---

## 8. Diversity & Additional Metrics

### `div/pairwise_rmsd_p50`
- **Definition**: Median RMSD among PoseBusters-valid rollouts (sampled) when `enable_pairwise_rmsd_logging` is true.
- **Interpretation**: >1.0 Å = healthy diversity; <0.5 Å suggests the sampler produces near-duplicates. Logged only every `pairwise_rmsd_log_every` steps.

---

## Quick Reference: Target Values

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| `gate/final_valid_rate` | > 0.8 | 0.5-0.8 | 0.3-0.5 | < 0.3 |
| `gate/graph_match_rate` | > 0.9 | 0.7-0.9 | 0.5-0.7 | < 0.5 |
| `rmsd/d_min_p50` | < 0.5 Å | 0.5-1.0 Å | 1.0-1.5 Å | > 1.5 Å |
| `rmsd/d_min_p90` | < 1.0 Å | 1.0-2.0 Å | 2.0-3.0 Å | > 3.0 Å |
| `match/efficiency_mean` | > 0.7 | 0.5-0.7 | 0.3-0.5 | < 0.3 |
| `match/matched_dist_p50` | < 0.5 Å | 0.5-0.75 Å | 0.75-1.0 Å | > 1.0 Å |
| `bestcov/soft_cov_mean` | > 0.7 | 0.5-0.7 | 0.3-0.5 | < 0.3 |
| `bestcov/pct_gt_cov_gt_0p5` | > 0.7 | 0.5-0.7 | 0.3-0.5 | < 0.3 |
| `div/pairwise_rmsd_p50` | 0.5-2.0 Å | 0.3-0.5 Å | 0.1-0.3 Å | < 0.1 Å |
| `rmsd/frac_under_delta` | > 0.5 | 0.3-0.5 | 0.1-0.3 | < 0.1 |

---

## Training Monitoring Strategy

### During Training (Real-time Dashboard)

**Primary Metrics (Monitor Continuously):**
1. `gate/final_valid_rate` – Are we generating usable conformers?
2. `rmsd/d_min_p50` – Typical geometric accuracy.
3. `match/efficiency_mean` – Coverage quality under the matching term.
4. `train/loss` – TRL objective.
5. `train/ratio_mean` – Policy stability (should stay near 1.0).

**Secondary Metrics (Check Periodically):**
6. `rmsd/d_min_p90` – Tail behavior.
7. `match/matched_dist_p50` – Quality of successful matches.
8. `reward/comp_qual_mean` – Quality contribution magnitude.
9. `bestcov/soft_cov_mean` – Smooth coverage health.

### Warning Signs

**🚨 Training Issues:**
- `gate/final_valid_rate` dropping → check `gate/graph_match_rate` and `pose/*` to find the failing gate.
- `train/ratio_mean` far from 1 or `train/approx_kl` > 0.1 → policy updates too aggressive.
- `train/clip_ratio` > 0.5 → trust region too small or gradients too spiky.

**🚨 Quality Issues:**
- `rmsd/d_min_p50` stuck or rising → geometry not improving.
- `match/efficiency_mean` < 0.4 → few unique matches being found.
- `div/pairwise_rmsd_p50` < 0.3 Å → mode collapse among valid rollouts.
- `match/matched_dist_p90` > δ → solver is forced to use borderline matches (investigate RMSD or δ).

**🚨 Coverage Issues:**
- `bestcov/soft_cov_mean` low while `reward/comp_smcov_mean` high → smcov reward being exploited.
- `cov/refs_hit_mean` stagnant → sampler not exploring new references even if overall quality improves.
- `rmsd/frac_under_delta` < 0.2 → very few rollouts fall under δ, so both matching and smcov collapse.

---

## Metric Relationships

- **Validity → Geometry**: High `gate/final_valid_rate` is a prerequisite for meaningful RMSD stats.
- **Geometry → Matching**: Lower `rmsd/d_min_p50` increases eligible edges and boosts `match/efficiency_mean`.
- **Matching → Coverage**: Higher `match/num_matched_mean` typically raises `bestcov/soft_cov_mean` and `covdiff/cover_ratio_mean`.
- **Reward Coupling**: `reward/comp_qual_mean + reward/comp_smcov_mean + reward/comp_match_mean ≈ reward/total_mean`. Watch `reward/bestcov_rank_corr_mean` to confirm smcov aligns with total reward.
- **Diversity vs Coverage**: Healthy `div/pairwise_rmsd_p50` supports both discrete (`cov/*`) and smooth (`bestcov/*`) coverage metrics.

---

## Notes

- All metrics are computed per batch and logged to W&B when `wandb.run` is active.
- Pairwise RMSD metrics require `enable_pairwise_rmsd_logging` and obey `pairwise_rmsd_log_every`.
- Metrics with `train/` prefixes still come from TRL’s trainer; this guide focuses on reward-side logging.
- Percentiles (p50/p90) are more robust to outliers than means—use both.
- Compare trends over time rather than single steps; the reward pipeline mean-centers per prompt, so transient dips are normal when prompts change.