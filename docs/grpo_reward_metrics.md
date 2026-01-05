# GRPO Reward Metrics Reference

This document summarizes every custom metric emitted by `grpo_reward_v3.py`. All
metric keys are logged under these names in W&B. Use this guide to understand
what each metric measures and how to interpret its values during training.

## A. Validity & Gating

| Key | Meaning | Typical Interpretation |
| --- | --- | --- |
| `gate/graph_match_rate` | Fraction of completions whose decoded SMILES matches the canonical SMILES. | **0.98–1.00:** perfect conditioning; **0.9–0.97:** occasional prompt drift to investigate; **<0.9:** model likely ignoring the conditional SMILES. |
| `gate/rdkit_parse_rate` | Fraction of completions that RDKit can parse. | **>0.99:** healthy; **0.95–0.99:** some malformed conformer tags; **<0.95:** check generation formatting immediately. |
| `gate/base_valid_rate` | `graph_mask ∧ parsed_mask`. | Usually mirrors the smaller of the two rates above; if it drops **>5 pp** below either input metric, look for combined parsing issues. |
| `gate/final_valid_rate` | Fraction that survives PoseBusters + finite RMSD gate. | **≥0.75:** acceptable; **0.50–0.75:** geometry or RMSD noise creeping in; **<0.5:** rewards will be mostly floor and should be debugged before training further. |

## B. PoseBusters

| Key | Meaning | Typical Interpretation |
| --- | --- | --- |
| `pose/checked_rate` | PoseBusters checks divided by base-valid count. | **1.0:** all eligible rollouts were checked; **0.5–0.99:** PoseBusters skipped some samples (often due to runtime errors or disabled mode); **0:** PoseBusters disabled. |
| `pose/pass_rate` | Pass rate over checked samples. | **0.95–1.0:** geometry is solid; **0.8–0.95:** mild atom/angle issues; **<0.8:** conformers likely collapsing or invalid—the reward is mostly gated off. |
| `pose/error_rate` | Fraction of PoseBusters calls that errored. | Should stay **0**; even **0.05** means 1 in 20 rollouts fails due to PoseBusters configuration or bad inputs. |

## C. RMSD Quality

| Key | Meaning | Typical Interpretation |
| --- | --- | --- |
| `rmsd/d_min_mean` | Mean best RMSD per rollout. | **≤0.8 Å (for δ≈0.85):** strong quality; **0.8–1.0 Å:** borderline; **>1.0 Å:** generations are missing the target geometry. |
| `rmsd/d_min_p50`, `rmsd/d_min_p90` | Median / 90th percentile of best RMSDs. | **p50 ≤ δ** indicates most rollouts are usable; **p90 ≤ 2δ** means tails are under control. |
| `rmsd/frac_under_delta` | Fraction with RMSD < `delta`. | **≥0.6:** majority match well; **0.3–0.6:** mixed quality; **<0.3:** matches are rare. |
| `rmsd/frac_under_2delta` | Fraction with RMSD < 2×`delta`. | Values **≥0.9** mean most rollouts are recoverable; if this also tanks, the model needs more supervision. |

## D. Coverage & Reference Utilization

| Key | Meaning | Typical Interpretation |
| --- | --- | --- |
| `cov/refs_hit_mean` | Mean count of references hit (<`delta`). | **M (max)** is perfect; values around **0.3M–0.5M** suggest partial coverage; **≪0.3M** means the model only touches a few conformers. |
| `cov/refs_hit_p50` | Median reference hits. | Helps ensure coverage isn’t dominated by a few prompts; keep **p50 close to mean**. |
| `cov/cov_ratio_mean` | Normalized coverage = `refs_hit / M`. | **≥0.5** means at least half of the conformer set is reached; **<0.3** signals coverage collapse. |
| `cov/unique_nearest_refs_mean` | Unique nearest reference count. | Should scale with valid rollouts; if it stays **near 1–2** while `valid_rollouts` is large, the sampler is collapsing. |
| `cov/nearest_collision_rate_mean` | `1 - unique_refs / valid_rollouts`. | **≤0.3** healthy diversity; **0.3–0.6** mild collapse; **>0.6** severe collapse (most rollouts share the same nearest reference). |
| `covdiff/cover_ratio_mean` | Fraction of GT references that are covered (<δ) by *any* rollout (averaged per prompt). | Track this to ensure the new coverage difference reward still hits a large portion of the reference ensemble. |
| `covdiff/unique_cover_ratio_mean` | Fraction of GT references that are uniquely covered by exactly one rollout. | This approximates the “difference reward” mass—if it trends to zero the coverage term is no longer informative. |
| `cov/valid_rollouts_mean` | Average valid rollouts per prompt. | Expect **≈K * final_valid_rate**; sudden drops mean gating is rejecting entire batches. |

## E. Matching Diagnostics

| Key | Meaning | Typical Interpretation |
| --- | --- | --- |
| `match/num_matched_mean` | Average number of rollout↔reference matches found by the bipartite solver. | Goes up when more rollouts land within `delta`. |
| `match/max_possible_mean` | Upper bound `min(valid_rollouts, refs_hit)` per group. | Provides context for the previous metric. |
| `match/efficiency_mean` | `num_matched / max_possible`. | Values near 1.0 mean matching is saturated; low values indicate either quality issues or solver failures. |
| `match/matched_dist_p50`, `match/matched_dist_p90` | Percentiles of distances among matched pairs. | **p50 < 0.5δ, p90 < δ** indicates matches are tight; if p90 > δ the solver is picking the minimum among poor fits. |
| `match/eligible_edge_density` | Eligible edges (`distance < δ`) normalized by `K*M`. | **>0.1** means a reasonably dense matching graph; **<0.02** makes matching extremely sparse. |

## F. Smooth Coverage Diagnostics

| Key | Meaning | Typical Interpretation |
| --- | --- | --- |
| `smcov/soft_cov_mean` | Average soft coverage (smooth kernel) over the reference set. | Higher means references are broadly covered even if no exact match exists. |
| `smcov/pct_gt_cov_gt_0p1`, `smcov/pct_gt_cov_gt_0p5` | Fraction of references with soft coverage exceeding 0.1 / 0.5. | Used to see how many references receive strong signal versus just weak contributions. |
| `smcov/corr_with_refs_hit` | Pearson correlation between `refs_hit` and `smcov` contribution per group. | Positive correlation means the smooth coverage reward aligns with actual discrete hits. |

## G. Reward Decomposition

| Key | Meaning | Typical Interpretation |
| --- | --- | --- |
| `reward/total_mean`, `reward/total_std` | Final reward mean/std. | **Mean ≥0.4** indicates strong positive signal (given λ’s); std should be **≤ mean** for stable learning. |
| `reward/comp_qual_mean`, `reward/comp_smcov_mean`, `reward/comp_match_mean` | Component contributions over valid rollouts. | Expect roughly proportional to λs; if one term dwarfs the others by ×10, consider retuning weights. |
| `reward/comp_smcov_frac` | Smooth coverage fraction. | **0.3–0.6** is typical with λ_smcov=8. Values near **1.0** mean the other terms contribute little; near **0** means coverage is ineffective. |

## H. Diversity

| Key | Meaning | Typical Interpretation |
| --- | --- | --- |
| `div/pairwise_rmsd_p50` | Median RMSD between valid rollout pairs. | **>1.0 Å:** healthy diversity; **0.5–1.0 Å:** partial collapse; **<0.5 Å:** generations are nearly identical. |

### Notes

- Metrics default to `0.0` when no data exists for the step (e.g., if a percentile cannot be computed). Watch the gating metrics first—if `final_valid_rate` is near zero, downstream fields will be zero by construction.
- Pairwise RMSD metrics (`div/*`) only update when `enable_pairwise_rmsd_logging` is true and the current step satisfies `pairwise_rmsd_log_every`.
- PoseBusters metrics remain zero if `grpo.enable_posebusters` is false.

For deeper debugging, combine these metrics with the per-step logs printed by `reward_function` in `run.log`.

