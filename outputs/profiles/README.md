# Qwen vs LLaMA LP Performance Analysis

**Date:** 2025-12-13
**Hardware:** H100 (8x 80GB HBM3)
**Models:** Qwen 0.6B, LLaMA 380M
**Version:** Analysis v2.0 (Corrected with Evidence)

---

## TL;DR - Key Findings

### The Core Result

| Model | LP Effect | Why? |
|-------|-----------|------|
| **LLaMA** | **3.2x FASTER** with LP | Model generates garbage without LP (18.9% pass rate) |
| **Qwen** | **1.12x SLOWER** with LP | Model already generates valid output (99.4% pass rate) |

### Why Can't Qwen LP Be Faster?

1. **Qwen already produces valid output without LP** (99.4% pass rate)
2. **LP adds overhead** (masking operations, template building)
3. **Position drift creates junk tokens** (3.8% of coordinates corrupted)
4. **Tokenization difference** (22 vs 16 tokens per coordinate placeholder)

### Why Is LLaMA LP Faster?

1. **LLaMA generates garbage without LP** (18.9% pass rate)
2. **LP forces valid structure** (99.8% pass rate)
3. **Shorter outputs** (doesn't generate garbage until EOS)
4. **Massive token savings** → 3.2x speedup

---

## Files in This Directory

### ⭐ START HERE (Evidence-Based Analysis)

| Priority | File | Description |
|----------|------|-------------|
| 1 | **`ANALYSIS_WITH_EVIDENCE.md`** | Complete analysis with data citations (NEW) |
| 2 | **`TOKENIZATION_DEEP_DIVE.md`** | Detailed tokenization statistics (NEW) |
| 3 | **`TRACE_COMPARISON.md`** | Kernel-level profiling comparison |

### Supporting Documentation

| File | Description |
|------|-------------|
| `qwen_lp_profile_h100.json.gz` | 226MB compressed Chrome trace (Qwen, 64 samples) |
| `llama_lp_profile_h100.json.gz` | 84MB compressed Chrome trace (LLaMA, 64 samples) |
| `JUNK_TOKEN_EVIDENCE.md` | Evidence of junk token generation |
| `LLAMA_VS_QWEN_HYPOTHESIS.md` | Original hypothesis and validation plan |
| `PERFETTO_GUIDE.md` | How to use Perfetto UI and trace_processor_shell |
| `compare_traces.sh` | Automated trace comparison script |

### Legacy Files (Initial Analysis - Partially Incorrect)

| File | Status |
|------|--------|
| `ANALYSIS.md` | ❌ Initial analysis (wrong conclusions) |
| `ANALYSIS_CORRECTED.md` | ⚠️ Partially corrected |
| `FINAL_ANALYSIS.md` | ⚠️ Needs evidence citations |
| `SUMMARY.md` | ⚠️ Based on incorrect analysis |

---

## Key Evidence (from 1000-sample smoke tests)

### 1. Pass Rate Comparison

**Source:** `outputs/smoke/*.json`

| Configuration | Pass Rate | Time (1000 samples) | Time/Sample |
|---------------|-----------|---------------------|-------------|
| Qwen WITH LP | 99.6% | 500.82s | 0.50s |
| Qwen WITHOUT LP | **99.4%** | 446.28s | 0.45s |
| LLaMA WITH LP | 99.8% | 140.45s | 0.14s |
| LLaMA WITHOUT LP | **18.9%** | 455.16s | 0.46s |

**Key insight:** Qwen generates valid output 99.4% of the time WITHOUT LP. LLaMA only 18.9%.

### 2. Tokenization Comparison

**Source:** `TOKENIZATION_DEEP_DIVE.md`

| Coordinate Block | Qwen Tokens | LLaMA Tokens |
|-----------------|-------------|--------------|
| `<0.0000,0.0000,0.0000>` | 22 | 16 |
| `<0.5,1.2,0.3>` (short) | 13 (drift +9) | 13 (drift +3) |
| `<1.234,-2.345,0.987>` | 19 (drift +3) | 13 (drift +3) |

**Qwen position drift:** 0 to +9 tokens (high variance → junk tokens)
**LLaMA position drift:** 0 to +3 tokens (low variance → minimal junk)

### 3. Junk Token Statistics

**Source:** `ANALYSIS_WITH_EVIDENCE.md` Section 3

| Model | Junk Coordinates | Total | Percentage |
|-------|------------------|-------|------------|
| Qwen LP | 61 | 1,619 | 3.8% |
| LLaMA LP | 12 | 1,619 | 0.7% |

**Qwen junk examples:** `<5.799,-0.2748,-0.9587abra>`, `<6.9467,-1.369,0.8415amen>`
**LLaMA junk examples:** `<3.0644,0.625,0.9841C>` (single char only)

### 4. Performance Breakdown (64 samples, profiled)

**Source:** `qwen_lp_64samples_h100_profiled.json`, `llama_lp_64samples_h100_profiled.json`

| Metric | Qwen LP | LLaMA LP | Ratio |
|--------|---------|----------|-------|
| Total time | 115.46s | 32.87s | 3.51x |
| Tokens generated | 38,195 | 28,536 | 1.34x |
| Time per token | 3.02ms | 1.15ms | 2.62x |
| Output chars | 48,815 | 48,664 | 1.00x |

**Same output length, 3.5x time difference!**

---

## Conclusions

### For Qwen (well-trained model)

1. **Skip LP for production** - 99.4% pass rate without LP
2. **Use LP only for 100% guarantee** - Accept 12% slowdown
3. **Tokenization limits LP efficiency** - Can't fix without retraining

### For LLaMA (undertrained model)

1. **LP is essential** - 18.9% → 99.8% pass rate
2. **LP provides massive speedup** - 3.2x faster due to fewer tokens
3. **Keep using generic LP** - Working as intended

---

## Reproduction Commands

```bash
# Qwen LP smoke test (1000 samples)
python scripts/logit_processor/run_logit_processor_smoke.py \
  --model-alias qwen3_06b_pre --model-step 40k \
  --tokenizer-name qwen3_0.6b_custom --processor-type qwen \
  --sample-size 1000 --batch-size 32 \
  --json-report outputs/smoke/qwen_lp_test.json \
  --submit h100

# Qwen baseline (no LP)
python scripts/logit_processor/run_logit_processor_smoke.py \
  --model-alias qwen3_06b_pre --model-step 40k \
  --tokenizer-name qwen3_0.6b_custom --no-logit-processor \
  --sample-size 1000 --batch-size 32 \
  --json-report outputs/smoke/qwen_no_lp_test.json \
  --submit h100
```

---

## Quick Reference

### Trace Analysis Commands
```bash
# Top 50 operations by time
echo "SELECT name, COUNT(*) as count, SUM(dur)/1e6 as ms FROM slices GROUP BY name ORDER BY ms DESC LIMIT 50;" | \
  trace_processor_shell outputs/profiles/qwen_lp_profile_h100.json.gz 2>&1 | grep -A 55 "name"

# Compare traces
./outputs/profiles/compare_traces.sh
```

### Visual Analysis (Perfetto UI)
1. Transfer: `scp <node>:~/3DMolGen-new/3DMolGen/outputs/profiles/*.json.gz ~/Downloads/`
2. Open https://ui.perfetto.dev
3. Load trace file

---

## Technical Details

### Model Configuration
| Model | Parameters | Vocab Size | Architecture |
|-------|------------|------------|--------------|
| Qwen 0.6B | ~600M | 151,673 | 24 layers, 16 heads |
| LLaMA 380M | ~380M | 128,330 | 24 layers |

### Hardware
- **GPU:** H100 (SM 9.0, 80GB HBM3)
- **Node:** YerevaNN cluster (SLURM partition: h100)

---

## References

- **Primary analysis:** `ANALYSIS_WITH_EVIDENCE.md` (START HERE)
- **Tokenization details:** `TOKENIZATION_DEEP_DIVE.md`
- **Profiling traces:** `*.json.gz` files
- **Source code:** `../../src/molgen3D/evaluation/qwen_constraint_logit_processor.py`
- **Smoke test data:** `../smoke/*.json`
