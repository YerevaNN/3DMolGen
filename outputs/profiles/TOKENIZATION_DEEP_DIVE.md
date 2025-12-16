# Tokenization Deep Dive: Qwen vs LLaMA

**Date:** 2025-12-13
**Purpose:** Detailed analysis of tokenization differences and their impact on LP performance

---

## 1. Tokenizer Properties

### 1.1 Basic Characteristics

| Property | Qwen3_custom | LLaMA-3.2-chem |
|----------|-------------|-----------------|
| Vocabulary Size | 151,673 | 128,330 |
| Digit Tokenization | Single-character | Multi-character |
| Special Tokens | [CONFORMER], [/CONFORMER], etc. | Same |
| BPE Trained On | General text | Chemistry-adapted |

### 1.2 Digit Token Examples

**Qwen tokenization of digits:**
```python
"12345" → ['1', '2', '3', '4', '5']  # 5 tokens
"0.5"   → ['0', '.', '5']            # 3 tokens
"123"   → ['1', '2', '3']            # 3 tokens
```

**LLaMA tokenization of digits:**
```python
"12345" → ['123', '45'] or ['12', '345']  # 2 tokens (varies)
"0.5"   → ['0', '.', '5']                 # 3 tokens
"000"   → ['000']                          # 1 token (!)
```

**Key difference:** LLaMA has multi-digit tokens like `'000'`, `'234'`, `'123'` that compress repeated or sequential digits.

---

## 2. Coordinate Placeholder Analysis

### 2.1 Placeholder Definition

**Source:** `src/molgen3D/evaluation/qwen_constraint_logit_processor.py:40`

```python
COORD_PLACEHOLDER = "0.0000,0.0000,0.0000"
```

This placeholder is used to build the template. Its tokenization determines the number of FREE positions.

### 2.2 Placeholder Tokenization

**Full coordinate block: `<0.0000,0.0000,0.0000>`**

```
Qwen (22 tokens):
['<', '0', '.', '0', '0', '0', '0', ',', '0', '.', '0', '0', '0', '0', ',', '0', '.', '0', '0', '0', '0', '>']
  │                              │                              │                              │
  └── bracket (COPY)             └── comma (FREE)               └── comma (FREE)               └── bracket (COPY)

LLaMA (16 tokens):
['<', '0', '.', '000', '0', ',', '0', '.', '000', '0', ',', '0', '.', '000', '0', '>']
  │                         │                         │                         │
  └── bracket (COPY)        └── comma (FREE)          └── comma (FREE)          └── bracket (COPY)
```

**Difference:** Qwen tokenizes `0000` as 4 separate tokens, LLaMA as `000` + `0` = 2 tokens.

### 2.3 FREE Position Count

| Coordinate Block | Qwen FREE | LLaMA FREE | Difference |
|-----------------|-----------|------------|------------|
| `<0.0000,0.0000,0.0000>` | 20 | 14 | +6 |

(22 - 2 brackets = 20 FREE for Qwen, 16 - 2 brackets = 14 FREE for LLaMA)

---

## 3. Real Coordinate Tokenization

### 3.1 Short Coordinates (Fewer Digits)

| Coordinate | Qwen Tokens | LLaMA Tokens | Qwen Drift | LLaMA Drift |
|------------|-------------|--------------|------------|-------------|
| `<0.5,1.2,0.3>` | 13 | 13 | **+9** | +3 |
| `<-1.0,2.5,-0.7>` | 13 | 13 | **+9** | +3 |

**Problem:** Short coordinates have massive drift in Qwen (9 extra FREE positions!)

### 3.2 Medium Coordinates (Typical)

| Coordinate | Qwen Tokens | LLaMA Tokens | Qwen Drift | LLaMA Drift |
|------------|-------------|--------------|------------|-------------|
| `<1.234,-2.345,0.987>` | 19 | 13 | +3 | +3 |
| `<-3.14,2.71,-1.61>` | 16 | 13 | **+6** | +3 |

**Problem:** Even typical coordinates have variable drift in Qwen.

### 3.3 Long Coordinates (More Digits)

| Coordinate | Qwen Tokens | LLaMA Tokens | Qwen Drift | LLaMA Drift |
|------------|-------------|--------------|------------|-------------|
| `<12.345,-23.456,34.567>` | 22 | 13 | +0 | +3 |
| `<-1.2345,0.9876,-5.4321>` | 22 | 16 | +0 | +0 |

**Observation:** Long coordinates have minimal drift, but they're less common.

### 3.4 Drift Distribution

**Qwen drift range:** 0 to +9 tokens (high variance)
**LLaMA drift range:** 0 to +3 tokens (low variance)

The high variance in Qwen drift is the root cause of junk token generation.

---

## 4. Position Drift Mechanism

### 4.1 How Position Drift Creates Junk Tokens

**Scenario:** Template has 22 tokens, actual coordinate uses 15 tokens

```
Position:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
Template:  <  0  .  0  0  0  0  ,  0  .  0  0  0  0  ,  0  .  0  0  0  0  >
           ▼                                                           ▼
          COPY ←──────────── 20 FREE positions ───────────────────→  COPY

Actual:    <  1  .  2  3  4  ,  -  2  .  3  ,  0  .  9  >
           ▼                                           ▼
          COPY ←──────── 13 tokens (ends early!) ────→ ???

What happens at positions 15-21?
- Position 15-20: Template says FREE → model generates junk tokens
- Position 21: Template says COPY '>' → model finally outputs '>'

Result: <1.234,-2.3,0.9abcdef>
                      └──────┘
                      junk tokens from FREE positions
```

### 4.2 Why LLaMA Has Less Junk

LLaMA's multi-digit tokenization means:
1. Placeholder uses fewer tokens (16 vs 22)
2. Real coordinates use similar token counts
3. Drift is consistently low (+0 to +3)
4. Fewer "extra" FREE positions → less junk

---

## 5. Full Molecule Analysis

### 5.1 Sample Molecule Tokenization

**Molecule:** Formaldehyde (C=O, 2 atoms)
**Skeleton:** `[C]<0.0000,0.0000,0.0000>=[O]<0.0000,0.0000,0.0000>`

| Tokenizer | Total Tokens | FREE Positions | COPY Positions |
|-----------|--------------|----------------|----------------|
| Qwen | 47 | 40 | 7 |
| LLaMA | 35 | 28 | 7 |

**Same COPY positions** (structural tokens like `[C]`, `=`, `[O]`), but **12 more FREE positions** in Qwen.

### 5.2 Tokens Per Atom

For a molecule with N atoms:
- Qwen: ~22N coordinate tokens + structural tokens
- LLaMA: ~16N coordinate tokens + structural tokens

**Difference:** ~6N extra tokens for Qwen per molecule

### 5.3 Impact on Generation

**From profiled runs (64 samples, 1,619 total coordinates):**

| Metric | Qwen | LLaMA | Ratio |
|--------|------|-------|-------|
| Total generated tokens | 38,195 | 28,536 | 1.34x |
| Tokens per coordinate | 23.6 | 17.6 | 1.34x |
| Time (seconds) | 115.46 | 32.87 | 3.51x |

The 1.34x token ratio matches the theoretical difference from tokenization.

---

## 6. Junk Token Statistics

### 6.1 Junk Coordinate Detection

**Method:** Regex search for letters in coordinate blocks
```python
coord_pattern = r'<([^>]+)>'
junk_pattern = r'[a-zA-Z]'  # Any letter = junk
```

### 6.2 Results (64 samples)

| Model | Junk Coordinates | Total | Percentage |
|-------|------------------|-------|------------|
| Qwen LP | 61 | 1,619 | 3.8% |
| LLaMA LP | 12 | 1,619 | 0.7% |

**Qwen has 5x more junk coordinates than LLaMA.**

### 6.3 Junk Token Characteristics

**Qwen junk (multi-character, often full words):**
```
<5.799,-0.2748,-0.9587abra>        # "abra" (4 chars)
<6.9467,-1.369,0.8415amen>         # "amen" (4 chars)
<4.432,0.567,1.2637icineC>         # "icineC" (6 chars)
<3.669,-3.1469,0.1561chedulers>    # "chedulers" (9 chars!)
```

**LLaMA junk (single-character):**
```
<3.0644,0.625,0.9841C>             # "C" (1 char)
<4.0752,-1.801,0.0004c>            # "c" (1 char)
```

**Why the difference?**
- Qwen has more "extra" FREE positions from larger drift
- Model fills multiple positions → longer junk sequences
- LLaMA has fewer extra positions → single-char junk at most

---

## 7. Performance Impact Calculation

### 7.1 Junk Token Cost

**Qwen junk tokens:**
- 61 junk coordinates × ~3 junk tokens each = ~183 junk tokens
- Cost: 183 × 3.023ms = 0.55 seconds

**LLaMA junk tokens:**
- 12 junk coordinates × ~1 junk token each = ~12 junk tokens
- Cost: 12 × 1.152ms = 0.014 seconds

### 7.2 Extra Token Cost

**From tokenization difference (not junk, just more tokens):**
- Qwen: 38,195 tokens
- LLaMA: 28,536 tokens
- Difference: 9,659 extra tokens in Qwen

**If Qwen had same token count as LLaMA:**
- Expected time: 28,536 × 3.023ms = 86.3s (vs actual 115.5s)
- Extra time from more tokens: 29.2s

### 7.3 Per-Token Overhead

**Qwen per-token time:** 3.023ms
**LLaMA per-token time:** 1.152ms
**Ratio:** 2.62x

**Sources of per-token overhead:**
1. Model size (600M vs 380M): ~1.58x
2. Vocabulary size (152k vs 128k): ~1.18x
3. LP masking overhead: ~1.4x (40 vs 28 FREE positions)

**Combined:** 1.58 × 1.18 × 1.4 ≈ 2.61x ✓

---

## 8. Recommendations

### 8.1 Optimal Placeholder for Qwen

Current: `0.0000,0.0000,0.0000` (22 tokens)
Typical real coord: `1.234,-2.345,0.987` (19 tokens)

**Better placeholder:** `1.23,-4.56,7.89` (17 tokens)
- Closer to typical coordinate length
- Reduces drift for common cases
- Still has some drift for short coordinates

### 8.2 Dynamic Truncation

Detect when coordinate is complete and skip remaining FREE positions:

```python
def _coord_looks_complete(last_token: str) -> bool:
    """Check if last token ends a coordinate (digit before comma or bracket)."""
    return last_token and last_token[-1].isdigit()
```

When `_coord_looks_complete()` is True at a FREE position:
1. Look ahead to find next `>` (COPY position)
2. Skip remaining FREE positions
3. Force the `>` token

### 8.3 Alternative: Retrain with Better Tokenizer

Long-term solution: Train Qwen with multi-digit tokens
- Use LLaMA-style tokenizer with `'000'`, `'123'` tokens
- Would reduce drift to LLaMA levels
- Requires full retraining

---

## Appendix: Tokenization Code

### A.1 Reproduce Tokenization Analysis

```python
from transformers import AutoTokenizer

# Load tokenizers
qwen_tok = AutoTokenizer.from_pretrained(
    'src/molgen3D/training/tokenizers/Qwen3_tokenizer_custom',
    trust_remote_code=True
)
llama_tok = AutoTokenizer.from_pretrained(
    'src/molgen3D/training/tokenizers/Llama-3.2-chem-1B-v1',
    trust_remote_code=True
)

# Test coordinate
coord = '<0.0000,0.0000,0.0000>'

qwen_toks = qwen_tok.encode(coord, add_special_tokens=False)
llama_toks = llama_tok.encode(coord, add_special_tokens=False)

print(f'Qwen: {len(qwen_toks)} tokens')
print([qwen_tok.decode([t]) for t in qwen_toks])

print(f'LLaMA: {len(llama_toks)} tokens')
print([llama_tok.decode([t]) for t in llama_toks])
```

### A.2 Count Junk Coordinates

```python
import json
import re

with open('outputs/smoke/qwen_lp_64samples_h100_profiled.json') as f:
    data = json.load(f)

coord_pattern = r'<([^>]+)>'
junk_pattern = r'[a-zA-Z]'

junk_count = 0
total_coords = 0

for sample in data['passes']:
    text = sample.get('decoded_text', '')
    coords = re.findall(coord_pattern, text)
    for coord in coords:
        total_coords += 1
        if re.search(junk_pattern, coord):
            junk_count += 1
            print(f'  <{coord}>')

print(f'Junk: {junk_count}/{total_coords} ({100*junk_count/total_coords:.1f}%)')
```
