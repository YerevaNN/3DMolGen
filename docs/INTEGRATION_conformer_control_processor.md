# Integration of ConformerControlLogitsProcessor

## Overview

Updated `inference_multiconf.py` to use the sophisticated `ConformerControlLogitsProcessor` instead of a simple token-banning approach. This resolves issues with conformer tag generation and provides proper control over multi-conformer generation.

## Key Changes

### 1. Replaced Simple Processor with Sophisticated One

**Before**:
- Used simple `DisallowTokensLogitsProcessor` that just banned specific tokens
- Problem: Could ban conformer end tags, preventing proper generation
- Generated conformers one at a time in a loop

**After**:
- Uses `ConformerControlLogitsProcessor` from `molgen3D.training.grpo.logits_constraints`
- Intelligently forces `[CONFORMER]` tags at the right times
- Allows `[/CONFORMER]` tags to be generated naturally
- Tracks conformer counts internally
- Generates all conformers in one generation call

### 2. Added Stopping Criteria

**New**: `ConformerCountStoppingCriteria`
- Stops generation when exactly `target_k` conformers have been generated
- Works in coordination with the logits processor
- Prevents over-generation or truncation

### 3. Simplified Generation Loop

**Before**:
```python
for conf_idx in range(num_conformers):
    # Generate one conformer
    # Append to prompt
    # Generate next conformer
```

**After**:
```python
# Generate all conformers in one call
outputs = model.generate(
    ...,
    logits_processor=[ConformerControlLogitsProcessor(...)],
    stopping_criteria=[ConformerCountStoppingCriteria(...)],
)
# Extract all conformers from output
```

### 4. Removed conformers_per_batch Parameter

Since we now generate all conformers in one go, the `conformers_per_batch` parameter is no longer needed.

**CLI Before**:
```bash
python ... --conformers_per_batch 8 --conformer_multiplier 2
```

**CLI After**:
```bash
python ... --conformer_multiplier 2
```

## How It Works

### Step 1: Setup

```python
# Get token IDs as lists (for multi-token sequences)
conformer_start_ids = tokenizer.encode("[CONFORMER]", add_special_tokens=False)
conformer_end_ids = tokenizer.encode("[/CONFORMER]", add_special_tokens=False)
smiles_start_ids = tokenizer.encode("[SMILES]", add_special_tokens=False)
smiles_end_ids = tokenizer.encode("[/SMILES]", add_special_tokens=False)

# Setup banned tokens (BOTH SMILES tags + EOS/PAD)
banned_ids = set(smiles_start_ids)
banned_ids.update(smiles_end_ids)  # CRITICAL: Ban closing tag too!
if eos_token_id is not None:
    banned_ids.add(eos_token_id)
if pad_token_id is not None:
    banned_ids.add(pad_token_id)
```

### Step 2: Create Processor and Stopping Criteria

```python
logits_processor = ConformerControlLogitsProcessor(
    conformer_start_ids=conformer_start_ids,
    conformer_end_ids=conformer_end_ids,
    banned_token_ids=banned_ids,
    target_k=num_conformers,  # e.g., 20 for 2x of 10 ground truths
    force_hard=True,
)

stopping_criteria = ConformerCountStoppingCriteria(
    conformer_end_ids=conformer_end_ids,
    target_k=num_conformers,
)
```

### Step 3: Generate

```python
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=2500 * num_conformers,  # Scale by number of conformers
    generation_config=gen_config,
    logits_processor=[logits_processor],
    stopping_criteria=[stopping_criteria],
    use_cache=True,
)
```

### Step 4: Extract All Conformers

```python
# Find all [CONFORMER]...[/CONFORMER] pairs
conformer_strings = []
idx = 0
while True:
    conformer_start = decoded_output.find("[CONFORMER]", idx)
    if conformer_start == -1:
        break
    
    conformer_content_start = conformer_start + len("[CONFORMER]")
    conformer_end = decoded_output.find("[/CONFORMER]", conformer_content_start)
    
    if conformer_end == -1:
        break
    
    generated_conformer = decoded_output[conformer_content_start:conformer_end]
    conformer_strings.append(generated_conformer)
    
    idx = conformer_end + len("[/CONFORMER]")
```

## Expected Output Format

```
[SMILES]CCO[/SMILES][CONFORMER][C]<x,y,z>[C]<x,y,z>[O]<x,y,z>[/CONFORMER][CONFORMER][C]<x,y,z>[C]<x,y,z>[O]<x,y,z>[/CONFORMER]...
```

**Correct**:
- ✅ One initial `[SMILES]...[/SMILES]`
- ✅ N pairs of `[CONFORMER]...[/CONFORMER]`
- ✅ No extra SMILES tags in conformers
- ✅ All conformers properly closed

**Previous Issues** (now fixed):
- ❌ `[/SMILES]` appearing inside conformers → Fixed by banning both SMILES tags
- ❌ Missing `[/CONFORMER]` tags → Fixed by using proper processor that allows them
- ❌ Malformed sequences → Fixed by forcing tags at correct positions

## Files Modified

1. **`src/molgen3D/evaluation/inference_multiconf.py`**
   - Added imports for `ConformerControlLogitsProcessor` and `ConformerCountStoppingCriteria`
   - Removed simple `DisallowTokensLogitsProcessor` and `ForceTokenLogitsProcessor` classes
   - Updated `generate_multiple_conformers()` to use sophisticated processor
   - Simplified main loop (no more batching)
   - Removed `conformers_per_batch` parameter

2. **`scripts/test_multiconf_inference.sh`**
   - Removed `--conformers_per_batch` argument
   - Updated comments

3. **`src/molgen3D/training/grpo/logits_constraints.py`**
   - Fixed to ban both `[SMILES]` and `[/SMILES]` tags (was already done in previous fix)

## Usage

```bash
# Generate 2x ground truths for a small test
python src/molgen3D/evaluation/inference_multiconf.py \
    --device local \
    --test_set distinct \
    --conformer_multiplier 2 \
    --limit 5 \
    --binned

# Full run
python src/molgen3D/evaluation/inference_multiconf.py \
    --device local \
    --test_set distinct \
    --conformer_multiplier 2 \
    --binned
```

## Benefits

1. **Correctness**: Proper tag forcing and tracking ensures valid sequences
2. **Efficiency**: Single generation call per SMILES instead of iterative approach
3. **Simplicity**: Removed batching logic, cleaner code
4. **Robustness**: Sophisticated processor handles edge cases
5. **Flexibility**: Easy to adjust target_k per SMILES based on ground truth count

## Testing

Test with the logits processor test script first:

```bash
bash scripts/run_logits_test.sh
```

Then test the full inference:

```bash
bash scripts/test_multiconf_inference.sh
```

Check for:
- ✅ Correct number of conformers per SMILES
- ✅ No SMILES tags in conformers
- ✅ All conformers properly closed with `[/CONFORMER]`
- ✅ Valid molecule objects decoded successfully

## Performance Considerations

**Memory**: Generating all conformers at once requires more memory for KV cache. For very large target_k values (e.g., >50), may need to:
- Reduce batch size
- Use gradient checkpointing
- Split into multiple generation calls if needed

**Speed**: Single generation call is generally faster than multiple iterative calls due to:
- Better GPU utilization
- Less overhead from multiple forwards
- More efficient KV cache usage

**Quality**: Should be similar or better than iterative approach since:
- Model sees full context of previous conformers
- No prompt reconstruction artifacts
- More natural generation flow
