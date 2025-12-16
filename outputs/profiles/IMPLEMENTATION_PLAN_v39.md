# Qwen LP v39 Implementation Plan

## Critical Findings

### 1. Greedy Sampling is WRONG for Qwen3
**From Qwen3-0.6B official docs:**
> DO NOT use greedy decoding - causes performance degradation and endless repetitions

**Recommended for Qwen3 (non-thinking mode):**
- Temperature: 0.7
- Top P: 0.8
- Top K: 20

**Current smoke tests use:** `greedy` (do_sample=False) - THIS IS WRONG!

### 2. Placeholder Mismatch
**Current placeholder:** `0.0000,0.0000,0.0000` (23 chars, 22 tokens)
**Median actual coordinate:** 21 chars

```
Coordinate length distribution (Qwen NO LP, 1000 samples):
  21 chars: 30.8% ← MODE/MEDIAN
  22 chars: 30.7%
  20 chars: 19.8%
  23 chars:  9.3% ← current placeholder matches only 9.3%!
```

### 3. Position Drift is the Root Cause
- 35.9% of coordinates have POSITIVE drift (shorter than 22-token template)
- This creates "extra" FREE positions where junk tokens appear
- Junk examples: ` _______,`, `\n`, backticks

## Implementation Tasks

### Task 1: Add Qwen3-Recommended Sampling Config
File: `src/molgen3D/config/sampling_config.py`

```python
# Add this config for Qwen3 models
qwen3_sampling_config = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
)

# Update sampling_configs dict:
sampling_configs = {
    ...
    "qwen3": qwen3_sampling_config,
}
```

### Task 2: Create LP v39 with Better Placeholder
File: `src/molgen3D/evaluation/qwen_constraint_logit_processor.py`

**Changes:**
1. Update `COORD_PLACEHOLDER` from 4 decimals to 3 decimals:
```python
# OLD: COORD_PLACEHOLDER = "0.0000,0.0000,0.0000"  # 22 tokens
# NEW:
COORD_PLACEHOLDER = "0.000,0.000,0.0000"  # ~20 tokens, matches median
```

2. Simplify blocking to match generic LP (just angle brackets):
```python
# Remove parentheses from structural_chars
structural_chars = set('<>')  # NOT set('<>()[]')
```

3. Keep lookahead but reduce range:
```python
LOOKAHEAD_RANGE = 2  # Reduce from 4 to 2
```

4. Update VERSION:
```python
VERSION = "v3.9_shorter_placeholder_minimal_blocking"
```

### Task 3: Run Comparison Tests

```bash
# Test 1: Qwen LP v39 with Qwen3 sampling
python scripts/logit_processor/run_logit_processor_smoke.py \
  --model-alias qwen3_06b_pre --model-step 40k \
  --tokenizer-name qwen3_0.6b_custom --processor-type qwen \
  --sampling-config qwen3 \
  --sample-size 100 --batch-size 32 \
  --json-report outputs/smoke/qwen_lp_v39_qwen3sampling_100.json \
  --submit h100

# Test 2: Qwen NO LP with Qwen3 sampling (baseline)
python scripts/logit_processor/run_logit_processor_smoke.py \
  --model-alias qwen3_06b_pre --model-step 40k \
  --tokenizer-name qwen3_0.6b_custom --no-logit-processor \
  --sampling-config qwen3 \
  --sample-size 100 --batch-size 32 \
  --json-report outputs/smoke/qwen_no_lp_qwen3sampling_100.json \
  --submit h100
```

## Expected Outcomes

### With Better Placeholder (3 decimals → 20 tokens):
- Position drift reduced: 35.9% → ~10% with positive drift
- Fewer junk tokens at coordinate boundaries
- Better pass rate

### With Qwen3 Sampling:
- Qwen docs say greedy causes "performance degradation and endless repetitions"
- Proper sampling (temp=0.7, top_p=0.8) should improve quality

### Target:
- Pass rate: ≥99.4% (match or beat no-LP)
- Performance: ≤450s for 1000 samples (match or beat no-LP)

## Data References

| Metric | Source File |
|--------|-------------|
| Coordinate length distribution | `outputs/smoke/qwen_no_lp_1000samples_h100.json` |
| Decimal precision stats | `outputs/smoke/qwen_lp_v38_1000samples_h100.json` |
| Performance comparison | `outputs/profiles/ANALYSIS_WITH_EVIDENCE.md` |
| Tokenization analysis | `outputs/profiles/TOKENIZATION_DEEP_DIVE.md` |

## Source Code References

| File | Purpose |
|------|---------|
| `src/molgen3D/evaluation/qwen_constraint_logit_processor.py` | Qwen LP implementation |
| `src/molgen3D/evaluation/constraint_logit_processor.py` | Generic LP (reference) |
| `src/molgen3D/config/sampling_config.py` | Sampling configurations |
| `scripts/logit_processor/run_logit_processor_smoke.py` | Smoke test runner |
