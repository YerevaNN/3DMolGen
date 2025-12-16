# LLaMA vs Qwen LP Performance Mystery

## The Core Problem

**LLaMA (m380) behavior:**
- Without LP: ~18s for 64 samples
- With LP: ~15s for 64 samples
- **Result: 1.2x SPEEDUP** ✅ (LP forces tokens, reduces sampling)

**Qwen behavior:**
- Without LP: 24s for 64 samples
- With LP: 57s for 64 samples (current profile)
- **Result: 2.4x SLOWDOWN** ❌ (LP adds massive overhead)

**This is backwards.** The LP should speed up generation by:
1. Forcing COPY tokens (no sampling needed)
2. Blocking tokens in FREE positions (smaller effective vocab)
3. Early termination when template is done

## Why Qwen is Different: Tokenization Hypothesis

### LLaMA Tokenization (Hypothesis)
```python
# Coordinate: "1.2345,-0.6789,2.1234"
# Possible tokenization (multi-char digit sequences):
['1.2345', ',', '-0.6789', ',', '2.1234']  # 5 tokens
```

### Qwen Tokenization (Known)
```python
# Coordinate: "1.2345,-0.6789,2.1234"
# Single-digit tokenization:
['1', '.', '2', '3', '4', '5', ',', '-', '0', '.', '6', '7', '8', '9', ',', '2', '.', '1', '2', '3', '4']  # 21 tokens
```

### The Problem: Position Drift

**Template uses placeholder:** `0.0000,0.0000,0.0000` (22 tokens in Qwen)

**Actual generated coords:** May tokenize to 18-25 tokens depending on:
- Number of decimal places (3 vs 4)
- Negative signs (`-` as separate token or merged with digit)
- Number magnitude (fewer digits for small numbers)

**What happens with drift:**

```
Position 0-17:  Model generates "1.234,-5.678,9.012" (18 tokens of valid coords)
Position 18:    Template says FREE (expects 4 more tokens to reach 22)
                → Model generates junk token (maybe ',', maybe '0', maybe '.')
Position 19:    Template says FREE (expects 3 more tokens)
                → Model generates more junk
Position 20:    Template says FREE (expects 2 more tokens)
                → Model generates more junk
Position 21:    Template says FREE (expects 1 more token)
                → Model generates more junk
Position 22:    Template forces '>' (COPY position)
```

**Cost of each junk token:**
- Full forward pass through 24 layers (~5-10ms)
- Sampling overhead (~0.5ms)
- Memory operations (~1ms)
- **Total: ~7-12ms per junk token**

**With 3-5 junk tokens per coord block × 3-5 atoms per molecule:**
- 10-25 junk tokens per molecule
- 64 molecules × 15 junk tokens = **960 extra forward passes**
- 960 × 10ms = **9.6 seconds wasted**

This partially explains the 33s overhead!

## Why LLaMA Doesn't Have This Problem

**Hypothesis:** LLaMA tokenizer encodes digit sequences as multi-character tokens.

Example:
```
Placeholder "0.0000,0.0000,0.0000" → ['0.', '0000', ',', '0.', '0000', ',', '0.', '0000']  # 8 tokens
Actual coord "1.2345,-6.789,0.123" → ['1.', '2345', ',', '-', '6.', '789', ',', '0.', '123']  # 9 tokens
```

**Difference:** ±1 token instead of ±4 tokens

**Result:** Minimal junk tokens → minimal overhead → LP actually speeds up by forcing COPY tokens

## Additional Qwen-Specific Overhead

### 1. More Kernel Launches per Token

From current profile:
- 2.9M kernel launches for 96K tokens = **30 launches/token**
- Expected baseline: ~20 launches/token
- **Extra 10 launches/token × 96K tokens × 4.65μs = 4.5s overhead**

### 2. More Memory Operations

From current profile:
```
elementwise_kernel (direct_copy): 7,441ms (139,272 launches)
```

- 139K copies for 96K tokens = **1.45 copies/token**
- More copies = more memory bandwidth used
- Could indicate PyTorch creating intermediate tensors during masking

### 3. Masking Overhead Scales with Vocab Size

Qwen vocab: 152,064 tokens (padded for tensor cores)
LLaMA vocab: ~32,000 tokens

For each FREE position:
```python
scores[b, self.blocked_mask] = float('-inf')
```

**Qwen:** Writing to ~1,200 positions out of 152K vocab (0.8% of vocab)
**LLaMA:** Writing to ~400 positions out of 32K vocab (1.25% of vocab)

Absolute numbers:
- Qwen: 1,200 writes per FREE position
- LLaMA: 400 writes per FREE position

**But** if Qwen has MORE FREE positions due to finer tokenization:
- Qwen: 60 FREE positions per coord block × 3 atoms = 180 FREE per molecule
- LLaMA: 20 FREE positions per coord block × 3 atoms = 60 FREE per molecule

Total masking writes per molecule:
- Qwen: 180 × 1,200 = 216,000 writes
- LLaMA: 60 × 400 = 24,000 writes

**Qwen does 9x more masking writes per molecule!**

## Experimental Validation: Profile LLaMA

Run the exact same profiling setup with m380/LLaMA:

```bash
python scripts/logit_processor/run_logit_processor_smoke.py \
  --model-alias m380_conf_v2 \
  --model-step 2e \
  --tokenizer-name llama3_chem_v1 \
  --processor-type generic \
  --attention sdpa \
  --sample-size 64 \
  --batch-size 32 \
  --parallel-templates \
  --profile \
  --profile-output outputs/profiles/llama_lp_profile_h100.json \
  --json-report outputs/smoke/llama_lp_64samples_h100_profiled.json \
  --submit h100
```

### What to Compare

| Metric | Qwen (Current) | LLaMA (To Measure) | Expected Difference |
|--------|----------------|--------------------|--------------------|
| Total time | 57s | ~15-20s | **3x faster** |
| Kernel launches | 2.9M (30/token) | ~1.5M (20/token) | **2x fewer** |
| Elementwise copies | 139K | ~60K | **2x fewer** |
| FREE positions per coord | ~20 | ~6-8 | **3x fewer** |
| Masking writes per molecule | 216K | 24K | **9x fewer** |

### Specific Queries to Run on Both Traces

#### 1. Count FREE position operations
```bash
# Estimate FREE positions by counting masking operations
echo "SELECT COUNT(*) as mask_ops FROM slices WHERE name LIKE '%elementwise%' AND dur < 50000;" | \
  trace_processor_shell <trace> 2>&1 | grep -A 5 "mask_ops"
```

#### 2. Average forward pass time
```bash
# Find attention kernel frequency
echo "SELECT AVG(dur)/1e6 as avg_fwd_ms, COUNT(*) as count FROM slices WHERE name LIKE '%fmha%';" | \
  trace_processor_shell <trace> 2>&1 | grep -A 5 "avg_fwd"
```

#### 3. Template building overhead
```bash
# Timeline analysis - find template building phase
echo "SELECT dur/1e9 as duration_sec FROM slices WHERE name LIKE '%build%' OR name LIKE '%template%' ORDER BY dur DESC LIMIT 5;" | \
  trace_processor_shell <trace> 2>&1 | grep -A 10 "duration"
```

## Potential Solutions (Ranked by Feasibility)

### 1. **Dynamic Template Truncation** (Most Promising)

Modify the LP to detect when a coordinate block is complete:

```python
def __call__(self, input_ids, scores):
    # ... existing code ...

    if template.is_free[pos]:
        # Check if we've just completed a valid coordinate
        last_3_tokens = tokenizer.decode(input_ids[b, -3:])
        if matches_coord_pattern(last_3_tokens):  # e.g., ends with ">\[" or ">="
            # Jump to next COPY position (skip remaining FREE positions)
            next_copy_pos = find_next_copy_position(template, pos)
            if next_copy_pos:
                pos = next_copy_pos
                # Now force the COPY token
                scores[b, :] = float('-inf')
                scores[b, template.ref_ids[pos]] = 0.0
                return scores

    # ... rest of existing logic ...
```

**Pros:**
- Eliminates junk token generation
- No change to template building
- Should recover most of the lost performance

**Cons:**
- Requires pattern detection (fragile)
- Need to handle edge cases (what if pattern is ambiguous?)

### 2. **Adaptive Placeholder** (Medium Effort)

Build template with Qwen-specific placeholder that matches actual tokenization:

```python
# Instead of "0.0000,0.0000,0.0000"
# Use "1.234,-5.678,9.012" (typical actual coords)

COORD_PLACEHOLDER_QWEN = "1.234,-5.678,9.012"  # 21 tokens in Qwen, matches typical actual
```

**Pros:**
- Reduces position drift by ~50%
- Simple to implement (one-line change)

**Cons:**
- Still not perfect (some molecules will drift)
- Need different placeholder per tokenizer (maintenance burden)

### 3. **Vectorized Batch Masking** (You Tried This)

Pre-compute all masking operations for entire sequence:

```python
# Build full mask tensor upfront: [batch, seq_len, vocab]
full_mask = build_full_mask_tensor(templates, vocab_size)

def __call__(self, input_ids, scores):
    pos = cur_len - self._prev_lens
    # Single vectorized operation
    scores = torch.where(full_mask[:, pos, :], float('-inf'), scores)
    return scores
```

**Why it doesn't help:**
- Memory explosion: batch × max_seq_len × vocab = 32 × 2000 × 152K = **9.7 GB**
- Can't pre-compute because COPY positions depend on ref_ids (which vary per sample)
- Synchronization overhead of moving large tensors

### 4. **Accept the Slowdown** (Pragmatic)

If 2.4x slowdown is acceptable for 100% structural validity:

**Arguments for:**
- Production correctness >> speed
- Retrying failures would be even slower
- Qwen might just be inherently slower for this task

**Arguments against:**
- LLaMA proves it CAN be faster with LP
- 33s overhead for 64 samples is significant at scale
- Should understand WHY before accepting

## Critical Questions for Profiling Comparison

Once you have both traces:

1. **Token count mismatch:**
   - How many tokens does Qwen generate per molecule vs LLaMA?
   - Are there "junk" tokens visible in the Qwen outputs?

2. **Forward pass frequency:**
   - Does Qwen have MORE forward passes per valid output token?
   - Count fmha_cutlass kernels and divide by actual molecule count

3. **Masking overhead:**
   - Time spent in elementwise kernels as % of total
   - Qwen should be much higher if masking is the issue

4. **Memory bandwidth:**
   - Compare Memcpy operations
   - Qwen might be memory-bound while LLaMA is compute-bound

## Prediction

**I predict the LLaMA trace will show:**
- ✅ Fewer total tokens per molecule (tighter tokenization)
- ✅ Fewer kernel launches per molecule (less masking)
- ✅ Lower ratio of elementwise_kernel to fmha_cutlass
- ✅ Faster end-to-end time despite similar model size

**If true, this confirms:** Qwen's fine-grained tokenization is fundamentally incompatible with position-based constraint enforcement.

**Next step:** Dynamic template truncation (#1 above) to detect coordinate completion and skip junk tokens.

## Action Items

1. **Run LLaMA profile** (same settings as Qwen)
2. **Compare traces side-by-side** in Perfetto UI
3. **Count tokens per molecule** in both outputs (manual inspection)
4. **Measure FREE position frequency** via queries above
5. **Prototype dynamic truncation** if hypothesis confirmed

---

**Bottom Line:** Your instinct is correct - this is a fundamental tokenization mismatch, not a simple performance bug. The profiling comparison will confirm the root cause and guide the fix.
