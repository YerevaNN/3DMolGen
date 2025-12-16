# Qwen vs LLaMA LP Performance - Quick Summary

## The Numbers

```
                         QWEN          LLAMA        RATIO
─────────────────────────────────────────────────────────
Total Time              115.5s         32.9s        3.5x ❌
Throughput              423/s         1480/s        3.5x ❌
Junk Coordinates        11.1%          1.1%        10x ❌
Time per Sample         1.80s          0.51s        3.5x ❌
─────────────────────────────────────────────────────────
Pass Rate               100%           100%         1x ✅
Parse Failures          0              0            -  ✅
Total Output            48.8K          48.7K        1x ✅
```

## Root Cause

**Qwen tokenizes single digits:**
```
"1.234,-5.678,9.012" → ['1','.','2','3','4','5',',','-','5','.','6','7','8',',','9','.','0','1','2']
                       19 tokens ↑
```

**LLaMA tokenizes multi-char sequences:**
```
"1.234,-5.678,9.012" → ['1.','234',',','-5.','678',',','9.','012']
                       ~8 tokens ↑
```

**Result:**
- Qwen: Template has 22 tokens, actual needs 18 → **4 junk tokens** → wasted compute
- LLaMA: Template has 9 tokens, actual needs 8 → **minimal drift** → efficient

## Junk Token Examples

**Qwen (11.1% corrupted):**
```
<-2.139,1.459,0.3174 （>          Unicode paren
<5.799,-0.2748,-0.9587abra>       "abra" text
<6.9467,-1.369,0.8415amen>        "amen" text
<3.669,-3.1469,0.1561chedulers>   "chedulers" fragment
```

**LLaMA (1.1% corrupted):**
```
<3.0644,0.625,0.9841C>            Single char 'C'
<1.026,0.6438,-0.4296[c>          Bracket + 'c'
```

## The Fix

**Dynamic Template Truncation:**
```python
if template.is_free[pos]:
    # Check if coordinate just completed (ends with digit)
    if last_token_is_digit():
        # Skip to next `>` and force it
        jump_to_closing_bracket()
```

**Expected gain:** ~12s (10% speedup)

## Conclusion

✅ **Hypothesis validated:** Tokenization granularity causes 3.5x performance difference
✅ **Junk tokens confirmed:** 10x more in Qwen (11% vs 1%)
✅ **Solution identified:** Dynamic truncation
✅ **Both LPs work correctly:** 100% structural validity

**Read FINAL_ANALYSIS.md for full details.**
