# Sampling Hyperparameter Sweep - Round 1

## Sweep Matrix

| Config | Temperature | Main Param | Jobs per Test Set |
|--------|-------------|------------|-------------------|
| top_p_sweep1 | 0.8 | top_p=0.9 | 1 |
| top_p_sweep2 | 1.0 | top_p=0.9 | 1 |
| top_p_sweep3 | 1.2 | top_p=0.9 | 1 |
| min_p_sweep1 | 0.8 | min_p=0.1 | 1 |
| min_p_sweep2 | 1.0 | min_p=0.1 | 1 |
| min_p_sweep3 | 1.2 | min_p=0.1 | 1 |
| top_k_sweep1 | 0.8 | top_k=50 | 1 |
| top_k_sweep2 | 1.0 | top_k=50 | 1 |
| top_k_sweep3 | 1.2 | top_k=50 | 1 |

**Total: 9 jobs per test set**

## Summary

This sweep explores three sampling strategies across three temperature values:
- **top_p sweep**: top_p=0.9 at temperatures [0.8, 1.0, 1.2]
- **min_p sweep**: min_p=0.1 at temperatures [0.8, 1.0, 1.2]
- **top_k sweep**: top_k=50 at temperatures [0.8, 1.0, 1.2]


## Eval Submission Script

```bash
./scripts/run_hp_sweep_eval.sh
```

## Round 1 Results

### Coverage & Matching Metrics

| Config | Temp | Param | COV-R ↑ | COV-P ↑ | MAT-R ↓ | MAT-P ↓ | Runtime |
|--------|------|-------|---------|---------|---------|---------|---------|
| top_p_sweep1 | 0.8 | top_p=0.9 | 0.6439 | 0.5883 | 0.6726 | 0.7414 | 223m |
| top_p_sweep2 | 1.0 | top_p=0.9 | 0.6549 | 0.5770 | 0.6589 | 0.7534 | 219m |
| top_p_sweep3 | 1.2 | top_p=0.9 | 0.6633 | 0.5616 | 0.6474 | 0.7708 | 223m |
| min_p_sweep1 | 0.8 | min_p=0.1 | 0.6516 | 0.5897 | 0.6630 | 0.7399 | 226m |
| min_p_sweep2 | 1.0 | min_p=0.1 | 0.6623 | 0.5723 | 0.6505 | 0.7601 | 226m |
| min_p_sweep3 | 1.2 | min_p=0.1 | 0.6764 | 0.5563 | 0.6334 | 0.7779 | 227m |
| top_k_sweep1 | 0.8 | top_k=50 | 0.6691 | 0.5741 | 0.6359 | 0.7595 | 195m |
| top_k_sweep2 | 1.0 | top_k=50 | 0.6781 | 0.5448 | 0.6311 | 0.7880 | 199m |
| **top_k_sweep3** | **1.2** | **top_k=50** | **0.6824** | 0.5299 | **0.6235** | **0.8087** | 206m |

### Generation Quality

| Config | Temp | SMILES Mismatch | Parse Fail | No EOS | Total Conformers |
|--------|------|-----------------|------------|--------|------------------|
| top_p_sweep1 | 0.8 | 169 | 2 | 0 | 213,385 |
| top_p_sweep2 | 1.0 | 199 | 5 | 0 | 213,352 |
| top_p_sweep3 | 1.2 | 234 | 5 | 0 | 213,317 |
| min_p_sweep1 | 0.8 | 161 | 0 | 0 | 213,395 |
| min_p_sweep2 | 1.0 | 219 | 2 | 0 | 213,335 |
| min_p_sweep3 | 1.2 | 225 | 4 | 0 | 213,327 |
| top_k_sweep1 | 0.8 | 226 | 0 | 0 | 213,330 |
| top_k_sweep2 | 1.0 | 291 | 2 | 0 | 213,263 |
| top_k_sweep3 | 1.2 | 477 | 10 | 0 | 213,069 |

### Round 1 Observations

1. **Temperature effect**: Higher temperature consistently improves COV-R (diversity) at the cost of COV-P (precision) across all methods.
2. **Best COV-R**: `top_k_sweep3` (temp=1.2, top_k=50) achieves highest COV-R at 0.6824.
3. **Best COV-P**: `min_p_sweep1` (temp=0.8, min_p=0.1) achieves highest COV-P at 0.5897.
4. **Error rates**: Higher temperature increases SMILES mismatches. `top_k` at temp=1.2 has notably higher error rate (477 mismatches).
5. **Runtime**: `top_k` configs run ~15% faster than `top_p`/`min_p` (~195-206m vs ~219-227m).


---

# Sampling HP Sweep - Round 2 (Fixed Temperature, Vary Parameters)

## Design Rationale

Round 1 tested how temperature scaling affects each method with fixed parameter values.
Round 2 tests the inherent characteristics of each method at neutral temperature (1.0),
isolating the effect of the truncation/filtering mechanism itself.

## Sweep Matrix

| Config | Temperature | Main Param | Rationale |
|--------|-------------|------------|-----------|
| top_p_r2_1 | 1.0 | top_p=0.8 | Tighter nucleus |
| top_p_r2_2 | 1.0 | top_p=0.9 | Baseline (same as R1) |
| top_p_r2_3 | 1.0 | top_p=0.95 | Looser nucleus |
| min_p_r2_1 | 1.0 | min_p=0.05 | More permissive |
| min_p_r2_2 | 1.0 | min_p=0.1 | Baseline (same as R1) |
| min_p_r2_3 | 1.0 | min_p=0.15 | Stricter filtering |
| top_k_r2_1 | 1.0 | top_k=20 | Narrow vocabulary |
| top_k_r2_2 | 1.0 | top_k=50 | Baseline (same as R1) |
| top_k_r2_3 | 1.0 | top_k=100 | Broader vocabulary |

**Total: 9 jobs per test set**

## Summary

- **top_p sweep**: temp=1.0 with top_p=[0.8, 0.9, 0.95]
- **min_p sweep**: temp=1.0 with min_p=[0.05, 0.1, 0.15]
- **top_k sweep**: temp=1.0 with top_k=[20, 50, 100]

## Inference Command

```bash
python -m molgen3D.evaluation.inference \
    --device a100 \
    --grid_run_inference \
    --binned \
    --test_set distinct
```

## Eval Submission Script

```bash
./scripts/run_hp_sweep_round2_eval.sh
```

## Round 2 Results

### Coverage & Matching Metrics

| Config | Param Value | COV-R ↑ | COV-P ↑ | MAT-R ↓ | MAT-P ↓ | Runtime |
|--------|-------------|---------|---------|---------|---------|---------|
| top_p_r2_1 | top_p=0.80 | 0.6381 | **0.5944** | 0.6778 | 0.7349 | 217m |
| top_p_r2_2 | top_p=0.90 | 0.6538 | 0.5740 | 0.6609 | 0.7583 | 218m |
| top_p_r2_3 | top_p=0.95 | 0.6648 | 0.5667 | 0.6426 | 0.7635 | 227m |
| min_p_r2_1 | min_p=0.05 | 0.6680 | 0.5632 | 0.6414 | 0.7678 | 224m |
| min_p_r2_2 | min_p=0.10 | 0.6631 | 0.5698 | 0.6472 | 0.7590 | 224m |
| min_p_r2_3 | min_p=0.15 | 0.6599 | 0.5793 | 0.6537 | 0.7526 | 225m |
| top_k_r2_1 | top_k=20 | 0.6337 | 0.5676 | 0.6817 | 0.7627 | 194m |
| top_k_r2_2 | top_k=50 | 0.6799 | 0.5534 | 0.6254 | 0.7814 | 195m |
| **top_k_r2_3** | **top_k=100** | **0.6821** | 0.5407 | **0.6187** | **0.7935** | 196m |

### Generation Quality

| Config | Param Value | SMILES Mismatch | Parse Fail | No EOS | Total Conformers |
|--------|-------------|-----------------|------------|--------|------------------|
| top_p_r2_1 | top_p=0.80 | 178 | 2 | 0 | 213,376 |
| top_p_r2_2 | top_p=0.90 | 221 | 1 | 0 | 213,334 |
| top_p_r2_3 | top_p=0.95 | 224 | 0 | 0 | 213,332 |
| min_p_r2_1 | min_p=0.05 | 211 | 3 | 0 | 213,342 |
| min_p_r2_2 | min_p=0.10 | 209 | 0 | 0 | 213,347 |
| min_p_r2_3 | min_p=0.15 | 172 | 0 | 0 | 213,384 |
| top_k_r2_1 | top_k=20 | 292 | 1 | 0 | 213,263 |
| top_k_r2_2 | top_k=50 | 300 | 3 | 0 | 213,253 |
| top_k_r2_3 | top_k=100 | 286 | 6 | 0 | 213,264 |

### Round 2 Observations

1. **top_p effect**: Tighter nucleus (0.8) gives better precision (COV-P=0.5944), looser (0.95) gives better recall (COV-R=0.6648).
2. **min_p effect**: More permissive (0.05) slightly improves COV-R; stricter (0.15) improves COV-P and reduces errors.
3. **top_k effect**: Larger vocabulary (top_k=100) achieves best COV-R (0.6821), smallest (top_k=20) has worst COV-R (0.6337).
4. **Best overall**: `top_k_r2_3` (top_k=100) matches Round 1's best COV-R while maintaining reasonable error rates.
5. **Error rates**: `min_p=0.15` has lowest error rate (172 mismatches), `top_k` methods have higher error rates (~286-300).

---

## Combined Analysis

### Best Configurations by Metric

| Metric | Best Config | Value | Notes |
|--------|-------------|-------|-------|
| **COV-R** (diversity) | top_k_sweep3 | 0.6824 | temp=1.2, top_k=50 |
| **COV-P** (precision) | top_p_r2_1 | 0.5944 | temp=1.0, top_p=0.8 |
| **MAT-R** | top_k_r2_3 | 0.6187 | temp=1.0, top_k=100 |
| **MAT-P** | top_k_sweep3 | 0.8087 | temp=1.2, top_k=50 |
| **Lowest errors** | min_p_sweep1 | 161 | temp=0.8, min_p=0.1 |

### Key Takeaways

1. **top_k dominates for diversity**: Both rounds show top_k achieving highest COV-R values, especially with larger k or higher temperature.
2. **top_p best for precision**: Tighter nucleus sampling (top_p=0.8) achieves best COV-P.
3. **min_p is most robust**: Lowest error rates, stable performance across parameter ranges.
4. **Temperature vs parameter**: Temperature has larger effect on diversity/precision trade-off than the method's primary parameter.
5. **Recommended configs**:
   - For diversity: `top_k=50-100` at temp=1.0-1.2
   - For precision: `top_p=0.8` at temp=1.0
   - For robustness: `min_p=0.1-0.15` at temp=0.8-1.0

---

# Sampling HP Sweep - Round 3 (Extended Temperature Range)

## Design Rationale

Round 1-2 covered temperatures [0.8, 1.0, 1.2]. Round 3 extends the temperature range to explore:
- **Lower temperatures** (0.6, 0.7): More deterministic sampling
- **Higher temperatures** (1.3, 1.5): More exploratory sampling

Focus on `top_k` only, testing three k values (30, 50, 70) to fill gaps around the k=50 baseline.

## Sweep Matrix

| Config | Temperature | top_k | Rationale |
|--------|-------------|-------|-----------|
| top_k_r3_30_t06 | 0.6 | 30 | Tight k, very low temp |
| top_k_r3_30_t07 | 0.7 | 30 | Tight k, low temp |
| top_k_r3_30_t13 | 1.3 | 30 | Tight k, high temp |
| top_k_r3_30_t15 | 1.5 | 30 | Tight k, very high temp |
| top_k_r3_50_t06 | 0.6 | 50 | Baseline k, very low temp |
| top_k_r3_50_t07 | 0.7 | 50 | Baseline k, low temp |
| top_k_r3_50_t13 | 1.3 | 50 | Baseline k, high temp |
| top_k_r3_50_t15 | 1.5 | 50 | Baseline k, very high temp |
| top_k_r3_70_t06 | 0.6 | 70 | Wide k, very low temp |
| top_k_r3_70_t07 | 0.7 | 70 | Wide k, low temp |
| top_k_r3_70_t13 | 1.3 | 70 | Wide k, high temp |
| top_k_r3_70_t15 | 1.5 | 70 | Wide k, very high temp |

**Total: 12 jobs per test set**

## Summary

- **top_k values**: [30, 50, 70]
- **Temperatures**: [0.6, 0.7, 1.3, 1.5]
- **Model**: `qw600_pre_binned_filtered` at step `4e`

## Inference Command

```bash
python -m molgen3D.evaluation.inference \
    --device a100 \
    --grid_run_inference \
    --binned \
    --test_set distinct
```

## Eval Submission Script

```bash
./scripts/run_hp_sweep_round3.sh
```

## Round 3 Results

### Coverage & Matching Metrics

| Config | Temp | top_k | COV-R ↑ | COV-P ↑ | MAT-R ↓ | MAT-P ↓ | Runtime |
|--------|------|-------|---------|---------|---------|---------|---------|
| top_k_r3_30_t06 | 0.6 | 30 | 0.6296 | **0.5942** | 0.6829 | 0.7336 | 195m |
| top_k_r3_30_t07 | 0.7 | 30 | 0.6364 | 0.5831 | 0.6787 | 0.7452 | 195m |
| top_k_r3_30_t13 | 1.3 | 30 | 0.6609 | 0.5293 | 0.6513 | 0.8081 | 205m |
| top_k_r3_30_t15 | 1.5 | 30 | 0.6626 | 0.5039 | 0.6512 | 0.8461 | 216m |
| top_k_r3_50_t06 | 0.6 | 50 | 0.6382 | 0.5883 | 0.6661 | 0.7390 | 195m |
| top_k_r3_50_t07 | 0.7 | 50 | 0.6558 | 0.5814 | 0.6514 | 0.7499 | 197m |
| top_k_r3_50_t13 | 1.3 | 50 | 0.6803 | 0.5191 | 0.6268 | 0.8252 | 203m |
| top_k_r3_50_t15 | 1.5 | 50 | 0.6729 | 0.4860 | 0.6386 | 0.8683 | 240m |
| top_k_r3_70_t06 | 0.6 | 70 | 0.6476 | 0.5885 | 0.6573 | 0.7394 | 203m |
| top_k_r3_70_t07 | 0.7 | 70 | 0.6591 | 0.5767 | 0.6454 | 0.7555 | 193m |
| **top_k_r3_70_t13** | **1.3** | **70** | **0.6873** | 0.5127 | **0.6175** | 0.8295 | 204m |
| top_k_r3_70_t15 | 1.5 | 70 | 0.6732 | 0.4803 | 0.6304 | **0.8795** | 229m |

### Generation Quality

| Config | Temp | top_k | SMILES Mismatch | Parse Fail | No EOS | Total Conformers |
|--------|------|-------|-----------------|------------|--------|------------------|
| top_k_r3_30_t06 | 0.6 | 30 | **159** | 2 | 0 | 213,395 |
| top_k_r3_30_t07 | 0.7 | 30 | 187 | 2 | 0 | 213,367 |
| top_k_r3_30_t13 | 1.3 | 30 | 705 | 23 | 0 | 212,828 |
| top_k_r3_30_t15 | 1.5 | 30 | 1,679 | 95 | 0 | 211,782 |
| top_k_r3_50_t06 | 0.6 | 50 | 176 | 0 | 0 | 213,380 |
| top_k_r3_50_t07 | 0.7 | 50 | 181 | 0 | 0 | 213,375 |
| top_k_r3_50_t13 | 1.3 | 50 | 719 | 32 | 0 | 212,805 |
| top_k_r3_50_t15 | 1.5 | 50 | 1,931 | 119 | 0 | 211,506 |
| top_k_r3_70_t06 | 0.6 | 70 | 177 | 2 | 0 | 213,377 |
| top_k_r3_70_t07 | 0.7 | 70 | 204 | 3 | 0 | 213,349 |
| top_k_r3_70_t13 | 1.3 | 70 | 773 | 31 | 0 | 212,752 |
| top_k_r3_70_t15 | 1.5 | 70 | 2,095 | 122 | 1 | 211,338 |

### Round 3 Observations

1. **New best COV-R**: `top_k_r3_70_t13` (k=70, temp=1.3) achieves **0.6873**, surpassing R1's best (0.6824).
2. **Temperature sweet spot at 1.3**: Maximizes diversity without the catastrophic error spike of temp=1.5.
3. **Error explosion at temp=1.5**: SMILES mismatches jump 10x (from ~170-200 at temp=0.6-0.7 to ~1,700-2,100 at temp=1.5).
4. **Low temp precision**: k=30 at temp=0.6 achieves COV-P=0.5942, matching R2's best (top_p=0.8).
5. **Diminishing returns at temp=1.5**: COV-R drops slightly vs temp=1.3 while errors explode.
6. **top_k scaling**: Larger k improves COV-R at all temperatures; effect strongest at temp=1.3.

---

## Combined Analysis (All Rounds)

### Best Configurations by Metric

| Metric | Best Config | Value | Round | Notes |
|--------|-------------|-------|-------|-------|
| **COV-R** (diversity) | top_k_r3_70_t13 | **0.6873** | R3 | k=70, temp=1.3 |
| **COV-P** (precision) | top_p_r2_1 | **0.5944** | R2 | top_p=0.8, temp=1.0 |
| **COV-P** (alt) | top_k_r3_30_t06 | 0.5942 | R3 | k=30, temp=0.6 |
| **MAT-R** | top_k_r3_70_t13 | **0.6175** | R3 | k=70, temp=1.3 |
| **Lowest errors** | top_k_r3_30_t06 | **159** | R3 | k=30, temp=0.6 |

### Temperature-Quality Trade-off Summary

| Temp Range | COV-R | COV-P | Avg Errors | Recommendation |
|------------|-------|-------|------------|----------------|
| 0.6-0.7 | 0.63-0.66 | 0.58-0.59 | ~170-200 | Production (low error) |
| 0.8-1.0 | 0.65-0.68 | 0.54-0.59 | ~160-300 | Balanced |
| 1.2-1.3 | 0.68-0.69 | 0.51-0.53 | ~480-780 | Research (max diversity) |
| 1.5 | 0.66-0.67 | 0.48-0.50 | ~1,700-2,100 | Avoid (poor trade-off) |

### Key Takeaways (Updated)

1. **top_k with temp=1.3 is optimal for diversity**: k=70 at temp=1.3 sets new COV-R record (0.6873).
2. **Low temperature for precision/reliability**: temp=0.6-0.7 with tight k (30) matches best COV-P while having lowest errors.
3. **Avoid temp=1.5**: Error rate explodes (10x) with marginal or negative diversity gain.
4. **Recommended production configs**:
   - **Max diversity**: k=70, temp=1.3 (COV-R=0.6873, ~773 errors)
   - **Max precision**: top_p=0.8, temp=1.0 (COV-P=0.5944, ~178 errors)
   - **Best balanced**: k=30, temp=0.6 (COV-R=0.6296, COV-P=0.5942, 159 errors)
   - **Low error + good diversity**: k=50, temp=1.0 (COV-R=0.6781, ~291 errors)

---

# Sampling HP Sweep - Round 4 (Fill Gaps)

## Design Rationale

Rounds 1-3 left gaps in top_p and min_p temperature coverage:
- **top_p**: p=0.9 has temps [0.8, 1.0, 1.2], but p=0.8 and p=0.95 only have temp=1.0
- **min_p**: min_p=0.1 has temps [0.8, 1.0, 1.2], but min_p=0.05 and min_p=0.15 only have temp=1.0

Round 4 fills these gaps to enable complete temperature curves for all parameter values.

## Sweep Matrix

| Config | Temperature | Main Param | Rationale |
|--------|-------------|------------|-----------|
| top_p_r4_08_t08 | 0.8 | top_p=0.8 | Fill gap: p=0.8 low temp |
| top_p_r4_08_t12 | 1.2 | top_p=0.8 | Fill gap: p=0.8 high temp |
| top_p_r4_95_t08 | 0.8 | top_p=0.95 | Fill gap: p=0.95 low temp |
| top_p_r4_95_t12 | 1.2 | top_p=0.95 | Fill gap: p=0.95 high temp |
| min_p_r4_05_t08 | 0.8 | min_p=0.05 | Fill gap: min_p=0.05 low temp |
| min_p_r4_05_t12 | 1.2 | min_p=0.05 | Fill gap: min_p=0.05 high temp |
| min_p_r4_15_t08 | 0.8 | min_p=0.15 | Fill gap: min_p=0.15 low temp |
| min_p_r4_15_t12 | 1.2 | min_p=0.15 | Fill gap: min_p=0.15 high temp |

**Total: 8 jobs per test set**

## Summary

- **top_p sweep**: p=[0.8, 0.95] at temps=[0.8, 1.2]
- **min_p sweep**: min_p=[0.05, 0.15] at temps=[0.8, 1.2]
- **Model**: `qw600_pre_binned_filtered` at step `4e`

## Inference Command

```bash
./scripts/run_hp_sweep_round4.sh
```

## After Inference Completes

Create eval script with the generated directory names, then run extraction script.

## Round 4 Results

*Pending - run inference first*

