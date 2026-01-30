# Logits Processor Testing Guide

## Overview

The `ConformerControlLogitsProcessor` is a sophisticated logits processor that:
1. **Forces conformer start tags** `[CONFORMER]` after each conformer end tag `[/CONFORMER]`
2. **Bans specific tokens** during generation (e.g., `[SMILES]`, EOS, PAD)
3. **Counts conformers** and stops forcing after reaching target_k
4. **Uses intelligent prefix matching** to force multi-token sequences token-by-token

## Key Features

### 1. Smart Token Forcing
- Detects when at the start (no conformers yet) or after a `[/CONFORMER]` tag
- Forces the next token to be from the `[CONFORMER]` sequence
- Uses prefix overlap detection to handle multi-token tags gracefully

### 2. Token Banning
- Prevents `[SMILES]` and `[/SMILES]` tokens from being generated (forces conformer-only output)
- Bans EOS and PAD tokens to prevent premature stopping
- Allows conformer tags to pass through

**Important**: Both opening `[SMILES]` and closing `[/SMILES]` tags must be banned, otherwise the model can generate invalid sequences like:
```
[SMILES]CCO[/SMILES][CONFORMER]...smiles...[/SMILES][CONFORMER]...
```
where `[/SMILES]` appears inside a conformer!

### 3. Conformer Counting
- Tracks how many `[CONFORMER]` and `[/CONFORMER]` tags have been generated
- Only forces conformer start when count < target_k
- Works with batched generation (tracks each sequence separately)

### 4. Stopping Criteria
- `ConformerCountStoppingCriteria` stops generation when all sequences reach target_k conformers
- Ensures exact conformer count is generated

## Testing

### Run the Test Script

```bash
# Quick test with target_k=4
bash scripts/run_logits_test.sh

# Or run directly with custom parameters
python scripts/test_logits_processor.py \
    --model_key m600_qwen_pre_4seq_binned \
    --checkpoint 4e \
    --tokenizer_key qwen3_0.6b_custom \
    --target_k 8 \
    --test_smiles "CCO" \
    --device cuda
```

### What the Test Does

The test script performs three tests:

**Test 1: Basic Generation**
- Generates conformers for a single SMILES
- Verifies target_k conformers are generated
- Checks that no `[SMILES]` tokens appear in generated portion

**Test 2: Batch Generation**
- Tests with 2 different SMILES in parallel
- Verifies both sequences generate target_k conformers
- Ensures batch processing works correctly

**Test 3: Different target_k Values**
- Tests with target_k = 2, 4, 6
- Verifies flexibility of target_k parameter

### Expected Output

For each test, you should see:

```
Prompt: [SMILES]CCO[/SMILES]

Generated output:
[SMILES]CCO[/SMILES][CONFORMER]...coords...[/CONFORMER][CONFORMER]...coords...[/CONFORMER]...(repeats target_k times)

Conformer count: 8 (target: 8) ✓
SMILES token in generated part: False ✓
```

### Success Criteria

✅ **PASS if:**
1. Number of `[/CONFORMER]` tags equals target_k
2. No `[SMILES]` tokens in generated portion
3. Generation stops after target_k conformers (not truncated)
4. All batch sequences reach target_k

❌ **FAIL if:**
1. Conformer count != target_k
2. `[SMILES]` tokens appear in generated portion
3. Generation stops early (hits max_new_tokens)
4. Generation contains invalid sequences

## How the Processor Works

### Initialization
```python
processor = ConformerControlLogitsProcessor(
    conformer_start_ids=[token_ids for "[CONFORMER]"],
    conformer_end_ids=[token_ids for "[/CONFORMER]"],
    banned_token_ids=[SMILES_START, EOS, PAD],
    target_k=8,
    force_hard=True,  # Set all other logits to -inf
)
```

### During Generation

At each step, the processor:

1. **Updates counts**: Scans new tokens to count conformer tags
2. **Checks state**: 
   - At start? (no conformers yet)
   - After end tag? (just finished a conformer)
3. **Decides action**:
   - If `count < target_k` AND (at_start OR after_end): Force `[CONFORMER]`
   - Always ban disallowed tokens
4. **Forces tokens**: 
   - Detects prefix overlap (e.g., if "[CON" already generated, force "F" next)
   - Sets logits to -inf for all except the next required token

### Example Flow

```
Step 1: Input = "[SMILES]CCO[/SMILES]"
        State: at_start=True, count=0
        Action: Force "[" from "[CONFORMER]"

Step 2: Input = "[SMILES]CCO[/SMILES]["
        State: prefix_overlap=1, count=0
        Action: Force "CONFORMER" token

Step 3: Input = "[SMILES]CCO[/SMILES][CONFORMER"
        State: prefix_overlap=continues...
        Action: Force "]" to complete "[CONFORMER]"

Step 4: Input = "[SMILES]CCO[/SMILES][CONFORMER]"
        State: conformer_start complete, count_start=1
        Action: Allow normal generation (coordinates)

Step N: Input = "...[/CONFORMER]"
        State: ended=True, count_end=1
        Action: Force "[CONFORMER]" again (count_end=1 < target_k=8)

... repeats until count_end=8 ...

Step M: Input = "...[/CONFORMER]" (8th time)
        State: ended=True, count_end=8
        Action: No forcing (count_end >= target_k), normal generation or stop
```

## Debugging

### Enable Verbose Logging

Add this to the processor to see internal state:

```python
# In ConformerControlLogitsProcessor.__call__
logger.debug(f"Row {row}: end_count={end_count}, start_count={start_count}, "
             f"at_start={at_start}, ended={ended}, should_force={should_force}")
```

### Common Issues

**Issue 1: Conformer count is less than target_k**
- Check max_new_tokens is sufficient
- Verify stopping criteria is being used
- Check if generation is hitting other stop conditions

**Issue 2: SMILES tokens appear in output**
- Verify banned_token_ids includes **both** `[SMILES]` and `[/SMILES]` token IDs
- Check if tokenizer.encode("[SMILES]") and tokenizer.encode("[/SMILES]") return correct IDs
- Common mistake: Only banning opening `[SMILES]` tag but not closing `[/SMILES]` tag

**Issue 3: Generation doesn't stop after target_k**
- Ensure ConformerCountStoppingCriteria is in stopping_criteria list
- Verify conformer_end_ids are correct

**Issue 4: Invalid sequences**
- Check if fallback logic is triggering (all logits are -inf)
- Verify conformer tag token IDs are correct

## Integration with Inference

Once tested, the processor can be integrated into the multi-conformer inference:

```python
from molgen3D.training.grpo.logits_constraints import (
    ConformerControlLogitsProcessor,
    ConformerCountStoppingCriteria,
)

# In generate_multiple_conformers():
processor = ConformerControlLogitsProcessor(
    conformer_start_ids=conformer_start_ids,
    conformer_end_ids=conformer_end_ids,
    banned_token_ids=banned_ids,
    target_k=num_conformers,
    force_hard=True,
)

stopper = ConformerCountStoppingCriteria(
    conformer_end_ids=conformer_end_ids,
    target_k=num_conformers,
)

outputs = model.generate(
    ...,
    logits_processor=[processor],
    stopping_criteria=[stopper],
)
```

## Performance Considerations

### Memory
- Processor maintains state per batch sequence
- Memory overhead: O(batch_size) for counters and cached lengths

### Speed
- Minimal overhead: O(new_tokens) per call to update counts
- Token forcing: O(1) per row per step
- Prefix matching: O(min(seq_len, pattern_len)) per row when forcing

### Batching
- Fully supports batched generation
- Each sequence in batch has independent state
- All sequences must reach target_k before stopping
