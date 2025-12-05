# Constrained Conformer Generation - Pre-computed Mask

This document explains the pre-computed mask approach for constrained conformer generation.

## Problem Statement

We have a language model that generates **enriched SMILES** for molecules:

- **Input (prompt):** Canonical SMILES describing molecular topology (e.g., `CC=O`)
- **Output (target):** Enriched SMILES where each atom has 3D coordinates (aka CONFORMER):
  ```
  [C]<x,y,z>[C]<x,y,z>=[O]<x,y,z>
  ```

The constraint: **topology tokens must be copied exactly** while **coordinate content is freely generated**.

```
Topology (COPY):  [C]  [C]  =  [O]  (bonds, atoms, rings, parens)
Coordinates (FREE):  <x,y,z>  <x,y,z>  <x,y,z>  (numbers, commas, signs)
```

## Design Overview

### Core Concept

Pre-compute a **reference skeleton** with placeholder coordinates, tokenize once, derive a **COPY/FREE mask** from token positions. At generation time, use pure position-based lookup.

### Why Pre-computed Mask?

| Approach | Speed | Complexity |
|----------|-------|------------|
| State machine (v1) | ~255s for 64 samples | High - per-token Python iteration |
| Pre-computed mask (v2) | ~15s for 64 samples | Low - position lookup |

The v2 approach eliminates per-token Python conditionals in the hot path.

## How It Works

### Step 1: Build Reference Skeleton

For SMILES `CC=O`, build:
```
[C]<0.0000,0.0000,0.0000>[C]<0.0000,0.0000,0.0000>=[O]<0.0000,0.0000,0.0000>
```

The placeholder `0.0000,0.0000,0.0000` tokenizes similarly to actual coordinates (~14 tokens).
The reason we use placeholder coordinates with 4 decimal places is because of the training data and training process.
The model usually generates coordinates with 4 decimal places, rare instances of 3 decimal places are also seen.
Thus we know that the placeholder coordinates will tokenize to similar number of tokens as the actual coordinates.

### Step 2: Derive COPY/FREE Mask

Tokenize the skeleton and mark each token:

```
Token:     [C]   ]<    0    .    000   0    ,    ...   >    [C]   ...
Position:   0     1    2    3     4    5    6    ...  14    15    ...
is_free:    F     F    T    T     T    T    T    ...   F     F    ...
```

- `<` token: COPY (fixed)
- Content inside `<...>`: FREE
- `>` token: COPY (fixed)
- All structural tokens: COPY (fixed)

### Step 3: Logit Processing

At each generation step:

```python
pos = current_length - prompt_length

if pos >= template.seq_len:
    # Done - force EOS
    force_token(eos_token_id)
elif template.is_free[pos]:
    # FREE - block only special tokens (and angle brackets!)
    block_tokens(blocked_mask)
else:
    # COPY - force exact token
    force_token(template.ref_ids[pos])
```

## Code Walkthrough: Core Implementation Details

This section explains the non-trivial parts of the implementation in `constraint_logit_processor.py`.

### Building the COPY/FREE Mask from Character Positions

The key challenge: we need a **token-level** mask (FREE or COPY per token), but coordinates are defined at the **character level** (everything between `<` and `>`). The solution uses HuggingFace's `offset_mapping` to bridge these two levels.

#### What `return_offsets_mapping=True` does

When tokenizing, this option returns `(start_char, end_char)` indices for each token, pointing back into the original string:

```python
encoding = tokenizer.encode_plus(
    ref_str,
    add_special_tokens=False,
    return_offsets_mapping=True,  # Returns character spans for each token
)
ref_ids = encoding["input_ids"]           # Token IDs
offset_mapping = encoding["offset_mapping"]  # [(start, end), ...] per token
```

**Example** for `ref_str = "[C]<1,2,3>[O]<4,5,6>"`:

| Token | Offset | Characters |
|-------|--------|------------|
| `[C]` | (0, 3) | chars 0-2 |
| `<` | (3, 4) | char 3 |
| `1` | (4, 5) | char 4 |
| `,` | (5, 6) | char 5 |
| `2` | (6, 7) | char 6 |
| `,` | (7, 8) | char 7 |
| `3` | (8, 9) | char 8 |
| `>` | (9, 10) | char 9 |
| `[O]` | (10, 13) | chars 10-12 |
| ... | ... | ... |

#### Step 1: Build character-level mask

First, we mark each character as FREE (inside coordinates) or not:

```python
char_is_free = []
in_coord = False
for char in ref_str:
    if char == '<':
        char_is_free.append(False)  # '<' itself is COPY
        in_coord = True
    elif char == '>':
        char_is_free.append(False)  # '>' itself is COPY
        in_coord = False
    else:
        char_is_free.append(in_coord)  # True only if inside <...>
```

**Result for `"[C]<1,2,3>[O]<4,5,6>"`:**

| Index | Char | `char_is_free` |
|-------|------|----------------|
| 0 | `[` | False |
| 1 | `C` | False |
| 2 | `]` | False |
| 3 | `<` | False |
| 4 | `1` | **True** |
| 5 | `,` | **True** |
| 6 | `2` | **True** |
| 7 | `,` | **True** |
| 8 | `3` | **True** |
| 9 | `>` | False |
| 10 | `[` | False |
| ... | ... | ... |

#### Step 2: Lift to token-level mask

Now we convert character-level to token-level. A token is FREE **only if ALL its characters are FREE**:

```python
is_free = []
for start_char, end_char in offset_mapping:
    if start_char >= end_char:
        # Empty span (e.g., some special tokens) → COPY
        is_free.append(False)
    else:
        # Slice the character mask for this token's span
        token_chars = char_is_free[start_char:end_char]
        # Token is FREE only if every character in span is FREE
        is_free.append(all(token_chars))
```

**Why `all()`?** If a token spans both coordinate content AND a bracket (e.g., a hypothetical `3>` token covering chars 8-10), we must mark it COPY to preserve structure. Only tokens entirely within `<...>` get FREE status.

**Final result:**

| Token | Span | Characters' Freedom | `is_free` |
|-------|------|---------------------|-----------|
| `[C]` | (0,3) | F, F, F | **False** |
| `<` | (3,4) | F | **False** |
| `1` | (4,5) | T | **True** |
| `,` | (5,6) | T | **True** |
| `2` | (6,7) | T | **True** |
| `,` | (7,8) | T | **True** |
| `3` | (8,9) | T | **True** |
| `>` | (9,10) | F | **False** |
| `[O]` | (10,13) | F, F, F | **False** |

### The `__call__` Method: Runtime Logit Processing

At each generation step, the processor modifies logits based on position:

```python
def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    batch_size, cur_len = input_ids.shape

    for b in range(batch_size):
        template = self.templates[b]

        # First call: record starting length (handles variable prompt padding)
        if self._prev_lens[b] is None:
            self._prev_lens[b] = cur_len

        # Position = how many tokens generated so far (0-indexed)
        pos = cur_len - self._prev_lens[b]

        if pos >= template.seq_len:
            # Past expected length → force EOS
            scores[b, :] = float('-inf')
            scores[b, self.eos_token_id] = 0.0

        elif template.is_free[pos]:
            # FREE position → block only dangerous tokens (angle brackets, special tags)
            scores[b, self.blocked_mask] = float('-inf')

        else:
            # COPY position → force exact reference token
            scores[b, :] = float('-inf')
            scores[b, template.ref_ids[pos]] = 0.0

    return scores
```

**Key insight:** The position `pos` indexes into the pre-computed `is_free` and `ref_ids` arrays. No per-token parsing or state machine needed at runtime - just array lookups.

### Why `_prev_lens` Tracks Starting Length

Different sequences in a batch may have different prompt lengths due to padding. By recording `cur_len` on the first call for each sequence, we correctly compute `pos` as the number of **generated** tokens (not total tokens).

```
Sequence A: [PAD][PAD][PROMPT...]  → _prev_lens[0] = 50
Sequence B: [PROMPT............]  → _prev_lens[1] = 48

After generating 5 tokens:
  Sequence A: cur_len=55, pos = 55-50 = 5
  Sequence B: cur_len=53, pos = 53-48 = 5
```

Both sequences are at generation position 5, despite different total lengths.

## Critical Fix: Angle Bracket Blocking

### The Bug

When coordinates tokenize to **fewer tokens** than the placeholder, the model can generate `>` prematurely:

```
Template:    [c]<placeholder_14_tokens>([C]...
Actual:      [c]<actual_13_tokens>     ...

Position 14: Template says FREE (expecting more coord content)
             Model generates '>' to close coordinate
             This creates stray '>' in output!
```

### The Tokenizer Complication

The tokenizer has ~1,143 tokens containing `<` or `>`:
- `>` (token 29) - standalone
- `>(` (token 2284) - combined with parenthesis
- `>[` (token 31868) - combined with bracket
- etc.

If ANY of these tokens are generated in a FREE position, structure is corrupted.

### The Fix

Block ALL tokens containing `<` or `>` in FREE positions:

```python
vocab = tokenizer.get_vocab()
for token_str, token_id in vocab.items():
    if '<' in token_str or '>' in token_str:
        blocked_ids.add(token_id)
```

This ensures:
1. Model generates coordinate content (digits, `.`, `,`, `-`)
2. When template reaches COPY position for `>`, it forces the closing bracket
3. No stray angle brackets corrupt the structure

## Data Structures

```python
@dataclass
class PrecomputedTemplate:
    ref_ids: torch.LongTensor      # [seq_len] - reference token IDs
    is_free: torch.BoolTensor      # [seq_len] - True=FREE, False=COPY
    seq_len: int                   # expected sequence length
```

## Blocked Tokens in FREE Positions

| Category | Examples | Reason |
|----------|----------|--------|
| Angle brackets | `<`, `>`, `>(`, `>[`, etc. | Prevent coordinate bracket leakage |
| Special tags | `[CONFORMER]`, `[/CONFORMER]` | Prevent tag injection |
| Control tokens | BOS, EOS, PAD | Prevent premature termination |

Total: ~1,403 blocked tokens out of ~152k vocabulary.

## Performance

| Metric | With Processor | Without Processor |
|--------|----------------|-------------------|
| Time (64 samples) | ~15s | ~18s |
| Overhead | ~0% (faster due to shorter sequences) | baseline |
| Pass rate | 64/64 (100%) | variable |

## Files

- `src/molgen3D/evaluation/constraint_logit_processor.py` - Implementation
- `scripts/logit_processor/run_logit_processor_smoke.py` - Smoke test runner

## Usage

```python
from molgen3D.evaluation.constraint_logit_processor import (
    ConformerConstraintLogitsProcessor,
    build_templates_for_batch,
)

# Build templates for batch
templates = build_templates_for_batch(smiles_list, tokenizer)
prompt_lengths = [len(tokenizer.encode(f"[SMILES]{s}[/SMILES]")) for s in smiles_list]

# Create processor
processor = ConformerConstraintLogitsProcessor(
    templates, prompt_lengths,
    tokenizer=tokenizer,
    eos_token_id=tokenizer.eos_token_id,
)

# Use in generation
outputs = model.generate(
    input_ids=...,
    logits_processor=LogitsProcessorList([processor]),
    ...
)
```

## Running the smoke test (for v2 pre-computed mask)
Run the following commands to test the pre-computed mask processor:
Check `config/sampling_config.py` for the sampling configurations.

### Clean dataset
```bash
python scripts/logit_processor/run_logit_processor_smoke.py \
  --dataset clean \
  --sample-size 931 \
  --batch-size 128 \
  --sampling-config greedy \
  --json-report outputs/smoke/v2_precompute_simple_v2.json
```

### Distinct dataset
```bash
python scripts/logit_processor/run_logit_processor_smoke.py \
  --dataset distinct \
  --sample-size 1000 \
  --batch-size 128 \
  --sampling-config top \
  --json-report outputs/smoke/v2_precompute_simple_v2_distinct.json
```

# small sample of clean
```bash
python scripts/logit_processor/run_logit_processor_smoke.py \
  --dataset clean \
  --sample-size 32 \
  --batch-size 16 \
  --sampling-config greedy \
  --json-report outputs/smoke/v2_precompute_simple_v2_small.json
```

# small sample of clean with no logit processor
```bash
python scripts/logit_processor/run_logit_processor_smoke.py \
  --dataset clean \
  --sample-size 64 \
  --batch-size 32 \
  --sampling-config greedy \
  --no-logit-processor \
  --json-report outputs/smoke/v2_precompute_simple_v2_small_no_processor.json
```

# small sample of clean with top_p_sampling4
```bash
python scripts/logit_processor/run_logit_processor_smoke.py \
  --dataset distinct \
  --sample-size 1000 \
  --batch-size 128 \
  --sampling-config top_p_sampling4 \
  --json-report outputs/smoke/v2_precompute_simple_v2_distinct_top_p_sampling4.json
```

## Edge Cases with Non-Greedy Sampling

### Empirical Results

Testing on 1,000 distinct SMILES with `top_p_sampling4` (top_p=0.4):

| Metric | Value |
|--------|-------|
| Total samples | 1,000 |
| Pass (SMILES exact match + real coords) | 985 (98.5%) |
| Parse failures (`decode_cartesian_v2`) | 15 (1.5%) |

Full report: `outputs/smoke/v2_precompute_simple_v2_distinct_top_p_sampling4.json`

### Failure Pattern Analysis

The 15 failures fall into distinct categories:

| Pattern | Count | Example | Root Cause |
|---------|-------|---------|------------|
| Trailing comma | 12 | `1.094,1.2358,-1.0388,` | Extra `,` before `>` |
| Missing comma (decimal confusion) | 2 | `0.3014.7184,1.3211` | `.` generated instead of `,` |
| Trailing dash + atom leak | 1 | `1.1782,-` with `cc` | `-` and atom tokens in coords |

### Why These Failures Occur

In FREE positions, the logit processor blocks:
- All tokens containing `<` or `>` (angle brackets)
- Special tags (`[CONFORMER]`, `[SMILES]`, etc.)
- Control tokens (BOS, EOS, PAD)

However, it does NOT block:
- `,` (comma) - needed for coordinate separators
- `.` (decimal point) - needed for coordinate values
- `-` (minus sign) - needed for negative coordinates
- Digit tokens - needed for coordinate values

With **greedy sampling** (temperature=0), the model always picks the highest-probability token, which is well-trained to produce valid coordinate syntax. Result: 100% structural validity.

With **top_p sampling**, lower-probability tokens can be sampled in FREE positions:
- Extra `,` at end of coordinate block (before the forced `>`)
- `.` sampled where `,` was expected (decimal confusion)
- Rarely: atom tokens like `c` can appear if they're in the vocabulary

### Greedy vs Non-Greedy Trade-offs

| Sampling | Structural Validity | Coordinate Diversity | Use Case |
|----------|--------------------|--------------------|----------|
| Greedy | 100% | Deterministic | Production, validation |
| top_p=0.4 | ~98.5% | Higher variance | Exploration, multiple conformers |
| top_p=0.9 | Lower (untested) | Maximum variance | Research only |

### Potential Improvements (Future Work)

1. **Block comma at position N-1 within `<...>`**: If we know the coordinate block is about to end (template position approaching `>`), block `,` to prevent trailing commas.

2. **Syntactic state tracking**: Track whether we're expecting a digit, decimal, comma, or sign within the coordinate block. Block tokens that violate expected syntax.

3. **Post-processing cleanup**: Strip trailing `,` or `-` from coordinate blocks before parsing.

4. **Constrained character set**: In FREE positions, only allow: digits `0-9`, `.`, `,`, `-`. Block all alphabetic tokens.

Note: These improvements would add complexity to the logit processor. The current 98.5% pass rate with top_p=0.4 may be acceptable for many use cases.

## Limitations and Future Work

1. **Variable coordinate lengths:** If coordinates tokenize to fewer tokens than placeholder, model generates "filler" digits. This is benign but may produce slightly longer coordinates.

2. **Assumption:** Placeholder `0.0000,0.0000,0.0000` tokenizes similarly to actual coordinates. Data shows ~80% exact match, ~20% 1-3 fewer tokens.

3. **Future v2.1:** Could loosen `<` and `>` constraints if needed, or implement hybrid state machine + pre-computed mask for robustness.

4. **Non-greedy sampling edge cases:** See "Edge Cases with Non-Greedy Sampling" section above for detailed analysis of the ~1.5% failure rate with top_p sampling.
