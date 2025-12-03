# v2 Pre-computed Mask Logit Processor Design

## Goal

Replace the v1.1 state machine with a pre-computed position-based mask for faster constrained conformer generation.

## Status
- Progress: 0%
- Started: 2025-12-03
- Blockers: None

## Background

### Performance baseline (v1.1 state machine)
- With logit processor: ~255 seconds (64 samples)
- Without logit processor: ~155 seconds (64 samples)
- **Overhead: ~100 seconds (65% slower)**

### Why v1.1 is slow
1. Per-token Python iteration in `_advance_state`
2. Python conditionals at every token
3. Looping through `blocked_in_coords` set for masking
4. Complex state tracking (segment index, coord window, etc.)

## Design

### Core Concept

Pre-compute a reference skeleton with coordinate placeholders, tokenize once, derive COPY/FREE mask from token positions. At generation time, use pure position-based lookup.

### Reference Skeleton

For SMILES `CC=O`, build:
```
[C]<0.0000,0.0000,0.0000>[C]<0.0000,0.0000,0.0000>=[O]<0.0000,0.0000,0.0000>
```

Using `0.0000,0.0000,0.0000` as placeholder (tokenizes similarly to actual coordinates).

### Mask Derivation

```
Position:  0           1    2-N      N+1  N+2         ...
Token:     [C]         <    (free)   >    [C]         ...
is_free:   False       False True... False False      ...
```

- `<` token: COPY (fixed)
- Content inside `<...>`: FREE
- `>` token: COPY (fixed)
- All structural tokens: COPY (fixed)

### Data Structures

```python
@dataclass
class PrecomputedTemplate:
    ref_ids: torch.LongTensor      # [seq_len] - reference token IDs
    is_free: torch.BoolTensor      # [seq_len] - True=FREE, False=COPY
    blocked_mask: torch.BoolTensor # [vocab_size] - blocked in FREE positions
```

### Logit Processor Logic

```python
def __call__(self, input_ids, scores):
    for b in range(batch_size):
        pos = cur_len - prompt_lengths[b]
        template = self.templates[b]

        if pos >= len(template.ref_ids):
            # Done - force EOS
            scores[b, :] = -inf
            scores[b, self.eos_token_id] = 0
        elif template.is_free[pos]:
            # FREE - block only special tokens
            scores[b, template.blocked_mask] = -inf
        else:
            # COPY - force exact token
            scores[b, :] = -inf
            scores[b, template.ref_ids[pos]] = 0

    return scores
```

### Key Assumptions

1. Coordinates are ~4 decimal places (`0.0000,0.0000,0.0000`)
2. Placeholder tokenizes similarly to actual coordinates
3. Model naturally generates appropriate coordinate content
4. `<|end_of_text|>` used for padding - must be blocked in FREE positions

### Blocked Tokens (FREE positions)

- `[CONFORMER]`, `[/CONFORMER]`
- `[SMILES]`, `[/SMILES]`
- `<|begin_of_text|>`, `<|end_of_text|>`
- EOS token, BOS token

### Performance Target

- Goal: Close to baseline (~155 seconds) or faster
- No Python loops in hot path
- Pre-computed tensors enable potential GPU acceleration

## Files

- `src/molgen3D/evaluation/constrained_logits_v2.py` - Logit processor
- `scripts/run_constrained_smoke_v2.py` - Smoke test runner
- Uses existing: `strip_smiles`, `same_molecular_graph` from `inference.py`

## Testing

```bash
# With logit processor (greedy)
python scripts/run_constrained_smoke_v2.py \
  --dataset clean --sample-size 64 --batch-size 64 \
  --sampling-config greedy \
  --json-report outputs/smoke/v2_with_processor.json

# Without logit processor (greedy, baseline)
python scripts/run_constrained_smoke_v2.py \
  --dataset clean --sample-size 64 --batch-size 64 \
  --sampling-config greedy \
  --no-logit-processor \
  --json-report outputs/smoke/v2_without_processor.json
```

Compare `time_taken` in JSON reports.

## Checklist

- [ ] Implement `constrained_logits_v2.py`
- [ ] Implement `run_constrained_smoke_v2.py` with validation using `strip_smiles`/`same_molecular_graph`
- [ ] Add greedy sampling config if not exists
- [ ] Run timing comparison
- [ ] Update analysis script if needed

## Notes

- v2.1 can loosen `<` and `>` constraints if needed
- Future: blend state machine + pre-computed mask for robustness
