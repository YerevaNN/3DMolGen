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

- `src/molgen3D/evaluation/constrained_logits_v2_precompute_mask.py` - Implementation
- `scripts/run_constrained_smoke_v2_precompute.py` - Smoke test runner

## Usage

```python
from molgen3D.evaluation.constrained_logits_v2_precompute_mask import (
    ConformerConstraintLogitsProcessorV2PrecomputeMask,
    build_templates_for_batch,
)

# Build templates for batch
templates = build_templates_for_batch(smiles_list, tokenizer)
prompt_lengths = [len(tokenizer.encode(f"[SMILES]{s}[/SMILES]")) for s in smiles_list]

# Create processor
processor = ConformerConstraintLogitsProcessorV2PrecomputeMask(
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

## Limitations and Future Work

1. **Variable coordinate lengths:** If coordinates tokenize to fewer tokens than placeholder, model generates "filler" digits. This is benign but may produce slightly longer coordinates.

2. **Assumption:** Placeholder `0.0000,0.0000,0.0000` tokenizes similarly to actual coordinates. Data shows ~80% exact match, ~20% 1-3 fewer tokens.

3. **Future v2.1:** Could loosen `<` and `>` constraints if needed, or implement hybrid state machine + pre-computed mask for robustness.
