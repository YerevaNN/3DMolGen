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

---

## Experimental Findings: Quality Impact Analysis (December 2025)

This section documents the experimental results comparing LP-constrained generation against unconstrained generation, and our decision to table further LP investigation.

### RMSD Evaluation Results

We ran full RMSD evaluation on the GEOM test set (~197,694 conformers) comparing LP vs baseline using the same `m600_qwen_pre` checkpoint at various training epochs:

| Experiment | Epochs | COV-R ↑ | COV-P ↑ | MAT-R ↓ | MAT-P ↓ | mismatch | parse_fail | no_eos |
|------------|--------|---------|---------|---------|---------|----------|------------|--------|
| 2.1 qwen06-pre | 1e | 66.3% | 47.4% | 0.62 Å | 0.93 Å | 1354 | 173 | 23 |
| 2.2 qwen06-pre | 2e | 68.1% | 49.7% | 0.60 Å | 0.91 Å | 687 | 137 | 44 |
| 2.3 qwen06-pre | 3e | 68.7% | 51.8% | 0.60 Å | 0.82 Å | 709 | 142 | 11 |
| 2.4 qwen06-pre | 4e | 68.3% | 52.4% | 0.59 Å | 0.82 Å | 710 | 119 | 9 |
| **qwen06-pre + LP** | 2e | 44.0% | 28.0% | 0.80 Å | 1.16 Å | **0** | 569 | 0 |

**Key observations:**
- ✅ **LP achieves its primary goal**: SMILES mismatch dropped from 687 → 0 (100% reduction)
- ✅ **LP eliminates EOS issues**: no_eos dropped from 44 → 0
- ❌ **LP significantly degrades coordinate quality**: COV-R dropped 35%, COV-P dropped 44%
- ❌ **LP increases parse failures**: 137 → 569 (4.15× increase)
- ⚠️ **Total failure count actually decreased** with LP (569 vs 868), but quality on successful generations plummeted

### The Distribution Shift Problem

#### Why LP Degrades Quality

The core issue: **chemical equivalence ≠ model familiarity**.

Without LP, the model can generate SMILES freely:
```
Input SMILES:     "CC(=O)O"  (acetic acid)
Model generates:  "C(C)(=O)O" (different but equivalent representation)
same_molecular_graph: ✅ PASS
Coordinates: Generated naturally for the representation the model chose → HIGH QUALITY
```

With LP, the model is forced to follow the exact input representation:
```
Input SMILES:     "CC(=O)O"
Model forced to:  "CC(=O)O"  (exact match)
same_molecular_graph: ✅ PASS (trivially)
Coordinates: Generated for a representation model isn't "comfortable" with → LOWER QUALITY
```

#### The Atom Ordering Problem

Even with canonical SMILES, the **atom traversal order** matters for coordinate generation. The model learned coordinate distributions conditioned on specific token patterns:

```python
# During training, model saw many examples like:
"[C][H3]<0.0,0.0,0.0>[C]<1.54,0.0,0.0>..."  # Pattern A (common)
"[C]<0.0,0.0,0.0>[C][H3]<1.54,0.0,0.0>..."  # Pattern B (rare)
```

Both represent the same molecule, but the model has strong priors for Pattern A. When LP forces Pattern B (or any non-preferred ordering), the hidden states evolve in ways the model wasn't trained for.

#### Error Compounding Over Sequence Length

```
           Without LP                    With LP
           ──────────                    ───────
Step 1:    natural token → natural       forced token → slight confusion
Step 2:    natural token → natural       forced token → more confusion
Step 3:    natural token → natural       forced token → hidden state drift
  ...         ...                           ...
Step N:    natural token → good coords   forced token → degraded coords
```

Each forced token pushes the hidden state further from the training distribution. By the time generation reaches coordinates, the model is operating in "alien territory."

### Hypotheses for Quality Degradation

1. **Representation Mismatch**: The model's "preferred" SMILES representation differs from the input. Forcing an exact match creates distribution shift.

2. **Hidden State Corruption**: Constrained tokens at early positions corrupt the hidden state, affecting downstream coordinate predictions even though coordinates are "free."

3. **Training-Inference Gap**: The model was trained on unconstrained generation. Constrained generation is fundamentally a different task the model wasn't optimized for.

4. **Coordinate Distribution Shift**: The model learned `P(coordinates | natural_SMILES_history)`. We're asking for `P(coordinates | forced_SMILES_history)` — a different distribution.

### Parse Failure Analysis

The 4× increase in parse failures (137 → 569) suggests LP exposes edge cases:
- Model generates malformed `<x,y,z>` strings under constraint
- Possible truncation or wrong number of coordinate blocks
- **Not yet investigated in detail** — would require sampling failed parses and categorizing error types

### Sampling Interaction

The experiments used `top_p=1.0` (full nucleus sampling). Sampling interacts with LP in a subtle way:

- **Constrained positions**: Sampling is irrelevant — LP forces exactly one token
- **Unconstrained positions (coordinates)**: Sampling draws from a corrupted distribution

Lower temperature or greedy decoding might reduce variance but won't fix the fundamental distribution shift.

### Decision: Table Further LP Investigation

**Given:**
- 20 days to deadline (as of December 2025)
- LP achieves structural correctness but degrades quality
- Full inference runs take 5-6 hours each
- RMSD evaluation takes 25-30 minutes

**Decision:** Table further LP investigation and document findings for future work.

**Rationale:**
- The quality degradation is too severe (~35-44% drop in coverage metrics)
- Fixing this would likely require training with LP constraints (significant effort)
- Current baseline without LP achieves acceptable results
- Time is better spent on other priorities

### Future Investigation Starting Points

If LP-based inference is revisited, consider these approaches:

#### 1. Probability Logging Diagnostic (Low Cost, High Value)

Add diagnostic hooks to measure distribution shift:

```python
# At each constrained position:
logits = model(input_ids)
probs = F.softmax(logits, dim=-1)

forced_token_id = constraint_template[position]
natural_token_id = probs.argmax()

# Key metrics:
forced_prob = probs[forced_token_id].item()
natural_prob = probs[natural_token_id].item()
prob_ratio = forced_prob / natural_prob  # How much model "disagrees"
```

**Analysis to run:**
- Per-token heatmap: Which SMILES tokens cause biggest disagreement?
- Positional analysis: Does disagreement grow over sequence length?
- Correlation: Do molecules with high disagreement have worse RMSD?

#### 2. Canonicalization Alignment

Ensure input SMILES use the exact same canonicalization as training data:
- Verify RDKit canonicalization settings match `encode_cartesian_v2()`
- Test if different canonical forms produce different quality results

#### 3. Training with LP Constraints

Train the model with LP constraints during training (not just inference):
- Model learns to generate coordinates under structural constraints
- Significant effort: requires modifying training pipeline

#### 4. Soft Constraints

Instead of hard masking (force token to -inf), try soft constraints:
- Add penalty to non-matching tokens rather than blocking completely
- Allow model to "correct" if it strongly disagrees

#### 5. Checkpoint Comparison

Test LP with different training epochs (1e, 2e, 3e, 4e):
- Hypothesis: Better-trained models might handle constraints better
- 4e baseline had lowest MAT-R (0.59) — might be more robust

#### 6. Parse Failure Investigation

Sample 50-100 failed parses and categorize:
- Truncated coordinates?
- Wrong format (missing commas, invalid numbers)?
- Extra/wrong tokens?
- Fix any trivial bugs exposed by LP

### References

- RMSD evaluation results: `rmsd_results_temp.csv` (project root, gitignored)
- Baseline experiments: 2.1-2.4 in RMSD table
- LP experiment: "qwen06 - pre - logit processor" row
- Checkpoint used: `m600_qwen_pre` with `step-40000-hf` (2e)
