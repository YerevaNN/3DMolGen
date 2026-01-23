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

## Eval Results
TBD..


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
./scripts/run_hp_sweep_eval_round2.sh
```

## Results

TBD..

