# GRPO Training Metrics Guide

This guide explains the **20 essential metrics** tracked during GRPO training for conformer generation. These metrics provide comprehensive monitoring of model performance, training stability, and generation quality.

## Overview

The metrics are organized into 6 categories:
1. **Validity Metrics** (3) - Whether outputs pass quality gates
2. **Geometry Metrics** (3) - RMSD accuracy to reference conformers
3. **Matching/Coverage Metrics** (5) - How well references are covered
4. **Reward Components** (3) - Breakdown of reward signal
5. **Coverage Metrics** (2) - Soft coverage statistics
6. **Diversity & Additional** (4) - Conformer diversity and batch-level stats

---

## 1. Validity Metrics

These metrics track whether generated conformers pass quality gates.

### `validity_rate`
- **Definition**: Final validity rate after all gates (graph match + optional RMSD gate)
- **Range**: 0.0 to 1.0
- **Formula**: `valid_rollouts / total_rollouts`
- **Interpretation**:
  - **> 0.8**: Excellent - model generates mostly valid conformers
  - **0.5 - 0.8**: Good - acceptable validity rate
  - **< 0.5**: Poor - many invalid outputs, training may be unstable
- **What to watch**: Should increase over training. If it drops, check `graph_match_rate` and `finite_rmsd_rate` to identify the failure point.

### `graph_match_rate`
- **Definition**: Fraction of rollouts where generated SMILES matches the canonical molecular graph
- **Range**: 0.0 to 1.0
- **Meaning**: First validity gate - ensures conformers correspond to the correct molecule structure
- **Interpretation**:
  - **> 0.9**: Excellent - model rarely generates wrong molecules
  - **0.7 - 0.9**: Good - occasional graph mismatches
  - **< 0.7**: Poor - model frequently generates incorrect molecular structures
- **What to watch**: If low, the model is generating chemically invalid structures. This should be very high (>0.9) for a well-trained model.

### `finite_rmsd_rate`
- **Definition**: Fraction of rollouts that have finite RMSD values (can be computed against references)
- **Range**: 0.0 to 1.0
- **Meaning**: Second validity gate - indicates conformers are geometrically valid enough to compute RMSD
- **Interpretation**:
  - **> 0.8**: Good - most conformers are geometrically reasonable
  - **0.5 - 0.8**: Acceptable - some conformers have geometric issues
  - **< 0.5**: Poor - many conformers are geometrically invalid
- **What to watch**: If `graph_match_rate` is high but `finite_rmsd_rate` is low, conformers pass graph check but have geometric problems (e.g., distorted bonds, clashes).

---

## 2. Geometry Metrics

These metrics measure how close generated conformers are to reference conformations.

### `geom/d_min_p50`
- **Definition**: 50th percentile (median) of minimum RMSD values
- **Units**: Angstroms (Å)
- **Formula**: For each valid rollout, `d_i = min_j RMSD(rollout_i, ref_j)`, then take median
- **Interpretation**:
  - **< 0.5 Å**: Excellent - typical conformers are very close to references
  - **0.5 - 1.0 Å**: Good - typical conformers are reasonably accurate
  - **1.0 - 1.5 Å**: Acceptable - some room for improvement
  - **> 1.5 Å**: Poor - typical conformers are far from references
- **What to watch**: Primary metric for geometric quality. Should decrease over training. Compare with `d_min_p90` to see if there are outliers.

### `geom/d_min_p90`
- **Definition**: 90th percentile of minimum RMSD values
- **Units**: Angstroms (Å)
- **Meaning**: Worst-case geometric accuracy (90% of conformers are better than this)
- **Interpretation**:
  - **< 1.0 Å**: Excellent - even worst conformers are reasonably close
  - **1.0 - 2.0 Å**: Good - worst conformers are acceptable
  - **> 2.0 Å**: Poor - worst conformers are far from references
- **What to watch**: Identifies outliers. If `p50` is good but `p90` is high, you have a long tail of poor conformers. Gap between `p50` and `p90` indicates consistency.

### `geom/d_min_mean`
- **Definition**: Average of minimum RMSD values
- **Units**: Angstroms (Å)
- **Interpretation**:
  - Should be close to `d_min_p50` if distribution is symmetric
  - If `mean > p50`: Distribution is right-skewed (some very bad conformers)
  - If `mean < p50`: Distribution is left-skewed (some very good conformers)
- **What to watch**: Compare with `p50` to understand distribution shape. Large difference indicates outliers.

---

## 3. Matching/Coverage Metrics

These metrics track how well generated conformers cover reference conformations.

### `match/match_efficiency`
- **Definition**: Fraction of possible matches achieved: `num_matched / max_possible_matches`
- **Range**: 0.0 to 1.0
- **Meaning**: How efficiently the matching algorithm covers references
- **Interpretation**:
  - **> 0.7**: Excellent - matching algorithm is finding most possible matches
  - **0.5 - 0.7**: Good - reasonable coverage efficiency
  - **< 0.5**: Poor - many potential matches are missed
- **What to watch**: Should increase over training. Low values indicate the matching algorithm isn't finding good assignments, possibly due to poor conformer quality or suboptimal matching.

### `match/num_matched`
- **Definition**: Average number of matched pairs per prompt group
- **Meaning**: Absolute count of successful matches
- **Interpretation**:
  - Higher is better - more references are being covered
  - Depends on `num_generations` (K) and number of references (M)
  - Should increase as model improves
- **What to watch**: Track alongside `match_efficiency`. If `num_matched` increases but `efficiency` stays same, you're generating more valid conformers but not improving matching quality.

### `match/dist_p50`
- **Definition**: 50th percentile (median) RMSD of matched pairs
- **Units**: Angstroms (Å)
- **Meaning**: Typical quality of successful matches
- **Interpretation**:
  - **< 0.5 Å**: Excellent - typical matches are very close
  - **0.5 - 0.75 Å**: Good - typical matches are within threshold
  - **> 0.75 Å**: Warning - typical matches are at the threshold edge (may fail if threshold is strict)
- **What to watch**: Should decrease over training. Compare with `match/eligible_dist_p50` - if `matched_dist_p50 < eligible_dist_p50`, matching is selecting better pairs (good!).

### `match/dist_p90`
- **Definition**: 90th percentile RMSD of matched pairs
- **Units**: Angstroms (Å)
- **Meaning**: Worst-case match quality (90% of matches are better)
- **Interpretation**:
  - **< 0.75 Å**: Good - worst matches are still within threshold
  - **≈ 0.75 Å**: Acceptable - worst matches are at threshold
  - **> 0.75 Å**: Warning - some matches exceed threshold (shouldn't happen if matching is correct)
- **What to watch**: Should be ≤ delta (default 0.75 Å). If higher, there may be a bug in matching logic.

### `match/refs_hit`
- **Definition**: Average number of unique references covered per prompt group
- **Meaning**: Coverage breadth - how many different reference conformations are matched
- **Interpretation**:
  - Higher is better - more diverse coverage
  - Maximum is min(K, M) where K=num_generations, M=num_references
  - Should increase as model improves
- **What to watch**: Track alongside `match_efficiency`. If both increase, you're covering more references more efficiently.

---

## 4. Reward Components

These metrics show the contribution of each reward term to the final reward signal.

### `reward/component_quality`
- **Definition**: Weighted quality reward: `λ_qual * mean(r_qual)`
- **Formula**: `λ_qual * exp(-d_i / sigma)` where `d_i = min_j RMSD(rollout_i, ref_j)`
- **Range**: 0.0 to λ_qual (typically 0.0 to 1.0)
- **Meaning**: Contribution of the quality term (AMR-P proxy) to final reward
- **Interpretation**:
  - Higher is better - indicates conformers are close to references
  - Should increase over training as geometric accuracy improves
  - Exponential decay means small RMSD improvements yield large reward gains
- **What to watch**: Primary driver of reward when conformers are close to references. Should correlate with `geom/d_min_p50`.

### `reward/component_smcov`
- **Definition**: Weighted smooth coverage reward: `λ_smcov * mean(r_smcov)`
- **Formula**: Group-aware coverage using smooth kernel `exp(-(D/rho)²)`
- **Range**: 0.0 to λ_smcov (typically 0.0 to 1.0)
- **Meaning**: Contribution of smooth marginal coverage term
- **Interpretation**:
  - Higher is better - indicates diverse coverage across the group
  - Rewards rollouts that cover references not yet well-covered by other rollouts
  - Should increase as model learns to generate diverse conformers
- **What to watch**: Should correlate with `coverage/soft_mean`. If low, model may be generating similar conformers (mode collapse).

#### New coverage-difference explainer (current default)
- **Definition**: Same metric, but the underlying `r_smcov` now equals `r_diff + r_depth + r_prec` where  
  `r_diff = (# uniquely covered refs)/M`,  
  `r_depth = unique_quality_weight/M * Σ unique_refs (1 - d_win/δ)`,  
  `r_prec = precision_weight * sigmoid((δ - min_j D_{i,j}) / ρ)`.
- **Meaning**: Most of the weight comes from the hard-δ difference reward; the sigmoid tail just keeps each rollout near at least one reference.
- **What to watch**: Should track the newly logged `covdiff/cover_ratio_mean` and `covdiff/unique_cover_ratio_mean`. If those fall while `component_smcov` rises, the shaping weights are too large.

### `reward/component_match`
- **Definition**: Weighted matching reward: `λ_match * mean(r_match)`
- **Formula**: `λ_match * (1 - D[i,j]/delta)` for matched pairs, 0 otherwise
- **Range**: 0.0 to λ_match (typically 0.0 to 1.0)
- **Meaning**: Contribution of hard matching bonus (COV-R proxy)
- **Interpretation**:
  - Higher is better - indicates more successful matches
  - Should increase as `match/num_matched` increases
  - Rewards unique one-to-one matches under δ threshold
- **What to watch**: Should correlate with `match/match_efficiency`. If low, model isn't generating conformers that match references.

---

## 5. Coverage Metrics

These metrics track soft coverage of reference conformations.

### `coverage/soft_mean`
- **Definition**: Average soft coverage per reference conformer
- **Range**: 0.0 to 1.0
- **Formula**: `1 - ∏(1 - K[i,j])` where `K = exp(-(D/rho)²)` is a smooth kernel
- **Meaning**: Probability that each reference is "covered" by at least one rollout (smooth version)
- **Interpretation**:
  - **> 0.7**: Excellent - most references are well-covered
  - **0.5 - 0.7**: Good - reasonable coverage
  - **< 0.5**: Poor - many references are poorly covered
- **What to watch**: Should increase over training. Lower than `match_efficiency` because it's a smooth (softer) metric.

### `coverage/pct_gt_0.5`
- **Definition**: Fraction of references with soft coverage > 0.5
- **Range**: 0.0 to 1.0
- **Meaning**: How many references receive "good" coverage (above 50% threshold)
- **Interpretation**:
  - **> 0.7**: Excellent - most references are well-covered
  - **0.5 - 0.7**: Good - majority of references are covered
  - **< 0.5**: Poor - less than half of references are well-covered
- **What to watch**: Should increase over training. More interpretable than `soft_mean` - directly tells you how many refs are "good enough".

---

## 6. Diversity & Additional Metrics

### `diversity/pairwise_mean`
- **Definition**: Average RMSD between pairs of generated conformers
- **Units**: Angstroms (Å)
- **Meaning**: Conformer diversity - how different generated conformers are from each other
- **Interpretation**:
  - **> 1.0 Å**: Good diversity - conformers explore different conformations
  - **0.5 - 1.0 Å**: Moderate diversity
  - **< 0.5 Å**: Low diversity - conformers are similar (possible mode collapse)
- **What to watch**: Should be reasonably high (>0.5 Å) to ensure diverse exploration. If very low, model may be generating similar conformers. However, if too high (>2.0 Å), conformers may be too scattered.

### `fraction_under_delta`
- **Definition**: Fraction of valid rollouts with `d_i < delta` (default δ=0.75 Å)
- **Range**: 0.0 to 1.0
- **Meaning**: Proportion of conformers that are "close enough" to references
- **Interpretation**:
  - **> 0.5**: Good - majority of conformers are within threshold
  - **0.3 - 0.5**: Acceptable - reasonable fraction within threshold
  - **< 0.3**: Poor - most conformers are far from references
- **What to watch**: Should increase over training. Correlates with `match/num_matched` - more conformers under threshold means more potential matches.

### `match/eligible_dist_p50`
- **Definition**: 50th percentile RMSD of all eligible edges (before matching)
- **Units**: Angstroms (Å)
- **Meaning**: Typical quality of all potential matches (not just selected ones)
- **Interpretation**:
  - Compare with `match/dist_p50`:
    - If `matched_dist_p50 < eligible_dist_p50`: Matching algorithm is selecting better pairs (good!)
    - If `matched_dist_p50 ≈ eligible_dist_p50`: Matching is selecting average-quality pairs
    - If `matched_dist_p50 > eligible_dist_p50`: Matching is selecting worse pairs (unlikely, indicates bug)
- **What to watch**: Diagnostic metric. Use to verify matching algorithm is working correctly.

### `reward/matched_total`
- **Definition**: Total number of matched pairs across entire batch
- **Meaning**: Batch-level absolute count of successful matches
- **Interpretation**:
  - Higher is better - more total matches across batch
  - Depends on batch size and number of prompts
  - Should increase as model improves
- **What to watch**: Track alongside `match/num_matched` (per-group average). If batch size changes, `matched_total` will change but `num_matched` should be stable.

---

## Quick Reference: Target Values

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| `validity_rate` | > 0.8 | 0.5-0.8 | 0.3-0.5 | < 0.3 |
| `graph_match_rate` | > 0.9 | 0.7-0.9 | 0.5-0.7 | < 0.5 |
| `finite_rmsd_rate` | > 0.8 | 0.5-0.8 | 0.3-0.5 | < 0.3 |
| `geom/d_min_p50` | < 0.5 Å | 0.5-1.0 Å | 1.0-1.5 Å | > 1.5 Å |
| `geom/d_min_p90` | < 1.0 Å | 1.0-2.0 Å | 2.0-3.0 Å | > 3.0 Å |
| `match/match_efficiency` | > 0.7 | 0.5-0.7 | 0.3-0.5 | < 0.3 |
| `match/dist_p50` | < 0.5 Å | 0.5-0.75 Å | 0.75-1.0 Å | > 1.0 Å |
| `coverage/soft_mean` | > 0.7 | 0.5-0.7 | 0.3-0.5 | < 0.3 |
| `coverage/pct_gt_0.5` | > 0.7 | 0.5-0.7 | 0.3-0.5 | < 0.3 |
| `diversity/pairwise_mean` | 0.5-2.0 Å | 0.3-0.5 Å | 0.1-0.3 Å | < 0.1 Å |
| `fraction_under_delta` | > 0.5 | 0.3-0.5 | 0.1-0.3 | < 0.1 |

---

## Training Monitoring Strategy

### During Training (Real-time Dashboard)

**Primary Metrics (Monitor Continuously):**
1. `validity_rate` - Are we generating valid conformers?
2. `geom/d_min_p50` - Typical geometric accuracy
3. `match/match_efficiency` - Coverage quality
4. `train/loss` - Training objective (from TRL)
5. `train/ratio_mean` - Policy stability (from TRL, should be ~1.0)

**Secondary Metrics (Check Periodically):**
6. `geom/d_min_p90` - Worst-case accuracy
7. `match/dist_p50` - Typical match quality
8. `reward/component_quality` - Quality reward contribution
9. `coverage/soft_mean` - Soft coverage

### Warning Signs

**🚨 Training Issues:**
- `validity_rate` dropping → Check `graph_match_rate` and `finite_rmsd_rate`
- `train/ratio_mean` far from 1.0 → Policy updates too aggressive/conservative
- `train/approx_kl` > 0.1 → Policy diverging too fast (from TRL)
- `train/clip_ratio` > 0.5 → Too many updates clipped, may need larger trust region

**🚨 Quality Issues:**
- `geom/d_min_p50` not decreasing → Model not improving geometric accuracy
- `match/match_efficiency` low → Not covering references well
- `diversity/pairwise_mean` < 0.3 Å → Possible mode collapse
- `match/dist_p90` > 0.75 Å → Some matches exceed threshold (check matching logic)

**🚨 Coverage Issues:**
- `coverage/soft_mean` low → References not well-covered
- `match/refs_hit` not increasing → Not exploring diverse conformations
- `fraction_under_delta` low → Most conformers far from references

---

## Metric Relationships

Understanding how metrics relate helps diagnose issues:

- **Validity → Geometry**: `validity_rate` must be high for geometry metrics to be meaningful
- **Geometry → Matching**: Lower `d_min_p50` → more eligible edges → higher `match_efficiency`
- **Matching → Coverage**: Higher `match_efficiency` → better `coverage/soft_mean`
- **Reward Components**: `component_quality` + `component_smcov` + `component_match` ≈ total reward signal
- **Diversity vs Coverage**: High `diversity/pairwise_mean` helps `coverage/soft_mean` by exploring more conformations

---

## Notes

- All metrics are computed per batch and logged to WandB
- Metrics with `train/` prefix are from TRL's GRPOTrainer (not covered here)
- Percentiles (p50, p90) are more robust than means for skewed distributions
- Compare metrics over time to track training progress
- Use percentiles together (p50 + p90) to understand distribution shape

