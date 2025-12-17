# Constrained Conformer Generation - Qwen Allowlist LP v4.3

This document explains the Qwen Allowlist Logit Processor (v4.3) for constrained conformer generation.

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

Pre-compute a **reference skeleton** with placeholder coordinates, tokenize once, derive a **COPY/FREE mask** from token positions. At generation time, use pure position-based lookup with a **strict allowlist** for coordinate tokens.

### Why Pre-computed Mask + Allowlist?

| Approach | Pass Rate | Notes |
|----------|-----------|-------|
| No LP | 100% | Fast, but no structural guarantees |
| Blocklist (block dangerous tokens) | ~88% | Position drift causes malformed coords |
| **Allowlist (v4.3)** | ~100% | Only allows valid coordinate tokens |

The allowlist approach ensures that FREE positions can only generate valid coordinate characters (0-9, `.`, `,`, `-`), eliminating structural errors.

## How It Works

### Step 1: Build Reference Skeleton

For SMILES `CC`, build:
```
[C]<0.000,0.000,0.000>[C]<0.000,0.000,0.000>
```

The placeholder `0.000,0.000,0.000` (3 decimal places) matches training data format. This tokenizes to ~18 tokens per coordinate block.

### Step 2: Derive COPY/FREE Mask

Tokenize the skeleton and mark each token:

```
Token:     [C   ]<    0    .    000   0    ,    ...   >    [C   ...
Position:   0     1    2    3     4    5    6    ...  19    20   ...
is_free:    F     F    T    T     T    T    T    ...   F     F   ...
```

- Tokens inside `<...>` (coordinate content): **FREE**
- All other tokens (structure, brackets): **COPY**

### Step 3: Logit Processing with Smart Blocking

At each generation step:

```python
pos = current_length - prompt_length

if pos >= template.seq_len:
    # Done - force EOS
    force_token(eos_token_id)
elif template.is_free[pos]:
    # FREE - allow ONLY coordinate tokens (allowlist)
    allow_only(allowed_coord_tokens)
    # Smart blocking at special positions:
    if is_first_free_position:
        block(comma_only_tokens)  # No leading comma
    if is_last_free_before_close:
        block(comma_and_dash_tokens)  # No trailing punctuation
else:
    # COPY - force exact token
    force_token(template.ref_ids[pos])
```

## Coordinate Token Allowlist

Only 15 tokens are allowed in FREE positions:

| Token | ID | Description |
|-------|-----|-------------|
| `0`-`9` | 15-24 | Digit tokens |
| `.` | 13 | Decimal point |
| `,` | 11 | Coordinate separator |
| `-` | 12 | Minus sign |
| `,-` | 4999 | Merged comma-minus |
| `-.` | 14523 | Merged minus-period |

All other tokens are blocked in FREE positions, including:
- Alphabetic characters
- Special tags (`[CONFORMER]`, `[SMILES]`, etc.)
- Angle brackets (`<`, `>`)
- Control tokens (BOS, EOS, PAD)

## Smart Position Blocking

To prevent malformed coordinates, the LP applies additional blocking at critical positions:

### First FREE Position (after `<`)
- **Blocks:** Comma tokens (`,`, `,,`, etc.)
- **Allows:** Digits, minus (for negative coords), period
- **Reason:** Coordinates shouldn't start with a comma

### Last FREE Positions (LOOKAHEAD_RANGE=2 before `>`)
- **Blocks:** Comma AND dash tokens
- **Allows:** Only digits and period
- **Reason:** Prevents trailing punctuation like `1.234,>` or `1.234->`

## Code Walkthrough

### Building the COPY/FREE Mask from Character Positions

The key challenge: we need a **token-level** mask, but coordinates are defined at the **character level** (everything between `<` and `>`).

#### Step 1: Build character-level mask

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

#### Step 2: Lift to token-level mask

Using HuggingFace's `offset_mapping` to map tokens back to character positions:

```python
encoding = tokenizer.encode_plus(ref_str, return_offsets_mapping=True)
is_free = []
for start_char, end_char in encoding["offset_mapping"]:
    token_chars = char_is_free[start_char:end_char]
    # Token is FREE only if ALL its characters are FREE
    is_free.append(all(token_chars))
```

**Why `all()`?** If a token spans both coordinate content AND a bracket (e.g., `]<` covering structure and bracket), we mark it COPY to preserve structure.

### The `__call__` Method

```python
def __call__(self, input_ids, scores):
    for b in range(batch_size):
        template = self.templates[b]

        # Track starting length (handles variable prompt padding)
        if self._prev_lens[b] is None:
            self._prev_lens[b] = cur_len

        pos = cur_len - self._prev_lens[b]

        if pos >= template.seq_len:
            # Force EOS
            scores[b, :] = -inf
            scores[b, self.eos_token_id] = 0.0
        elif template.is_free[pos]:
            # Allow only coordinate tokens (strict allowlist)
            scores[b, ~self.allowed_mask] = -inf
            # Smart blocking at special positions
            if template.is_first_free[pos]:
                scores[b, self.comma_only_mask] = -inf
            if template.block_comma_dash[pos]:
                scores[b, self.comma_dash_mask] = -inf
        else:
            # Force exact token
            scores[b, :] = -inf
            scores[b, template.ref_ids[pos]] = 0.0

    return scores
```

## Data Structures

```python
@dataclass
class PrecomputedTemplate:
    ref_ids: torch.LongTensor        # [seq_len] - reference token IDs
    is_free: torch.BoolTensor        # [seq_len] - True=FREE, False=COPY
    seq_len: int                     # expected sequence length
    block_comma_dash: torch.BoolTensor  # [seq_len] - block comma/dash at this position
    is_first_free: torch.BoolTensor     # [seq_len] - first FREE position in coord block
```

## Performance

| Metric | With LP | Without LP |
|--------|---------|------------|
| Pass rate (greedy) | ~100% | ~100% |
| Pass rate (top_p=0.8) | ~100% | Variable |
| Overhead | Minimal | Baseline |

The LP ensures structural validity even with non-greedy sampling.

## Files

- `src/molgen3D/evaluation/qwen_logit_processor.py` - Implementation (v4.3)
- `scripts/logit_processor/run_logit_processor_smoke.py` - Smoke test runner

## Usage

```python
from molgen3D.evaluation.qwen_logit_processor import (
    QwenAllowlistLogitsProcessor,
    build_precomputed_template,
)

# Build templates for batch
templates = [build_precomputed_template(smi, tokenizer) for smi in smiles_list]
prompt_lengths = [len(tokenizer.encode(f"[SMILES]{s}[/SMILES]")) for s in smiles_list]

# Create processor
processor = QwenAllowlistLogitsProcessor(
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

## Inference Configuration

**All inference uses single GPU.**

When running inference:
- Local: Use `--device cuda:0`
- Slurm: Scripts request 1 GPU automatically (`gpus_per_node=1`)

## Running Smoke Tests

Check `config/sampling_config.py` for sampling configurations.

### Basic test (greedy)
```bash
python scripts/logit_processor/run_logit_processor_smoke.py \
  --dataset clean \
  --sample-size 64 \
  --batch-size 32 \
  --sampling-config greedy \
  --model-alias m600_qwen_pre \
  --model-step 2e \
  --tokenizer-name qwen3_0.6b_custom \
  --json-report outputs/smoke/qwen_v43_greedy.json
```

### With sampling (top_p)
```bash
python scripts/logit_processor/run_logit_processor_smoke.py \
  --dataset distinct \
  --sample-size 256 \
  --batch-size 64 \
  --sampling-config qwen3 \
  --model-alias m600_qwen_pre \
  --model-step 2e \
  --tokenizer-name qwen3_0.6b_custom \
  --json-report outputs/smoke/qwen_v43_topp.json
```

### Without logit processor (baseline)
```bash
python scripts/logit_processor/run_logit_processor_smoke.py \
  --dataset clean \
  --sample-size 64 \
  --batch-size 32 \
  --sampling-config greedy \
  --model-alias m600_qwen_pre \
  --model-step 2e \
  --tokenizer-name qwen3_0.6b_custom \
  --no-logit-processor \
  --json-report outputs/smoke/qwen_no_lp_baseline.json
```

### H100 submission
```bash
python scripts/logit_processor/run_logit_processor_smoke.py \
  --dataset clean \
  --sample-size 1000 \
  --batch-size 128 \
  --sampling-config qwen3 \
  --model-alias m600_qwen_pre \
  --model-step 2e \
  --tokenizer-name qwen3_0.6b_custom \
  --attention sdpa \
  --submit h100 \
  --json-report outputs/smoke/qwen_v43_h100.json
```

## Token Count Variability ("Drift")

### The Challenge

Different coordinate values tokenize to different numbers of tokens:

| Coordinate Block | Token Count | Notes |
|-----------------|-------------|-------|
| `<0.000,0.000,0.000>` (placeholder) | ~18 | Baseline |
| `<1.234,-0.567,0.891>` (typical) | ~18 | Usually matches |
| `<-1.23,4.56,-7.89>` (shorter) | ~16 | Fewer tokens due to merged `,-` |

When generated coordinates use fewer tokens than the placeholder, the model is still in FREE positions but has finished the coordinate content.

### How v4.3 Handles This

1. **Strict allowlist** - Only coordinate characters allowed, so "filler" content is valid (extra precision digits)
2. **Smart blocking** - Last 2 FREE positions block comma/dash to prevent trailing punctuation

Without these safeguards, the model might generate patterns like:
- `1.234,>` (trailing comma before forced `>`)
- `1.234,->` (trailing comma-dash)

## Limitations and Future Work

1. **Token count assumption**: The LP assumes placeholder and actual coordinates tokenize similarly. In practice, ~80% exact match, ~20% differ by 1-3 tokens.

2. **Conservative lookahead**: `LOOKAHEAD_RANGE=2` blocks the last 2 FREE positions before `>`. This is conservative to handle token count variability.

3. **Coordinate validation**: The LP validates token-level characters but doesn't enforce coordinate syntax (e.g., exactly 3 comma-separated values). Post-processing validation may be needed.

## Testing

Unit tests: `tests/inference/test_logit_processor.py`

```bash
pytest tests/inference/test_logit_processor.py -v
```

Real end-to-end testing via smoke runner:
```bash
python scripts/logit_processor/run_logit_processor_smoke.py --help
```
