# Multi-Conformer Inference Design Notes

## Overview
New inference loop that generates multiple conformers per SMILES using logit processors to force conformer generation.

**Key Concept**: Each SMILES has a different number of ground truth conformers (k). The system generates a configurable multiple of k (e.g., 2k) for each SMILES.

## Key Design Decisions

### 1. Per-SMILES Target Generation
- Each SMILES has its own target conformer count: `target = multiplier × ground_truth_count`
- Example: SMILES_A with 10 ground truths → generate 20 conformers (if multiplier=2)
- Example: SMILES_B with 30 ground truths → generate 60 conformers (if multiplier=2)

### 2. Batched Generation per SMILES
- Each SMILES generates conformers in batches of `conformers_per_batch` (default 8)
- Each batch generates N conformers in one `model.generate()` call
- Output accumulates across batches: `[SMILES]...[/SMILES][CONFORMER]...[/CONFORMER]...[CONFORMER]...[/CONFORMER]...`
- Uses ConformerControlLogitsProcessor to force tags and ban SMILES tokens
- Continues in batches until reaching exact target for that SMILES
- Example: Target 20 conformers with batch=8 → 3 calls (8+8+4)

### 3. Main Loop Structure
- Iterates through SMILES one-by-one (not batching all prompts upfront)
- For each SMILES:
  - Batch 1: Generate N conformers from `[SMILES]...[/SMILES]`
  - Batch 2: Generate N more from accumulated output
  - Batch 3+: Continue until target reached
- Moves to next SMILES only after completing current one

### 3. Logit Processors
- **DisallowTokensLogitsProcessor**: Sets logits to -inf for disallowed tokens (SMILES start, EOS)
- **ForceTokenLogitsProcessor**: Forces specific token (available for future use)

## Potential Issues and Improvements

### Issue 1: Growing Prompt Length
**Problem**: As we generate more conformers, the prompt grows, potentially:
- Exceeding context window
- Slowing down generation
- Using more memory

**Potential Solutions**:
- Truncate prompt to keep only last N conformers
- Use sliding window approach
- Reset prompt after certain number of conformers

### Issue 2: No Parallelization
**Problem**: Current implementation is fully sequential:
- One SMILES at a time
- One conformer at a time
- Not leveraging batch processing

**Potential Solutions**:
1. **Batch multiple SMILES**: Generate conformers for multiple SMILES in parallel
2. **Parallel conformer generation**: Harder because each conformer depends on previous ones
3. **Multi-GPU**: Process different SMILES on different GPUs

### Issue 3: Conformer Quality Over Time
**Problem**: As prompt grows, quality of later conformers might degrade

**Potential Solutions**:
- Monitor quality metrics over conformer index
- Reset prompt periodically
- Use different generation strategies for later conformers

### Issue 4: Forcing Conformer Start Token
**Current**: We rely on logit processor to disallow SMILES/EOS, but don't force `[CONFORMER]` token

**Alternative**: After `[/CONFORMER]`, force next token to be `[CONFORMER]` using ForceTokenLogitsProcessor for first token only

### Issue 5: Error Recovery
**Problem**: If generation fails for one conformer, entire batch for that SMILES might be lost

**Potential Solutions**:
- Implement retry logic
- Continue with next SMILES on error
- Save partial results incrementally

## Parallelization Strategies

### Strategy A: Batch Different SMILES
```python
# Generate 8 conformers each for SMILES A, B, C in parallel
batch_prompts = [
    "[SMILES]A[/SMILES]",
    "[SMILES]B[/SMILES]",
    "[SMILES]C[/SMILES]",
]
# Generate in parallel, but each would still need sequential conformer generation
```

### Strategy B: Duplicate Same SMILES
```python
# Generate 8 conformers for same SMILES by creating 8 separate generation streams
batch_prompts = [
    "[SMILES]A[/SMILES]",
    "[SMILES]A[/SMILES]",
    # ... 8 times
]
# But this doesn't enforce sequential conformer building
```

### Strategy C: Multi-Stage Pipeline
1. Stage 1: Generate first conformer for batch of SMILES
2. Stage 2: Generate second conformer for each, using results from Stage 1
3. Repeat until N conformers per SMILES

## Performance Considerations

### Memory Usage
- Prompt length grows: O(n) per conformer
- Model KV cache grows with prompt
- Consider memory-efficient attention mechanisms

### Speed
- Sequential generation is slow
- Consider using smaller `max_new_tokens` after first few conformers
- Could compile model for specific prompt lengths

### Throughput Optimization
- Current: ~8 conformers / (N * gen_time) where N = num_SMILES
- Could achieve: ~8 * batch_size conformers / gen_time with proper batching

## Testing Strategy

### Unit Tests
1. Test logit processors independently
2. Test conformer extraction logic
3. Test prompt building

### Integration Tests
1. Run on small dataset (limit=5, conformers_per_batch=2)
2. Verify output format matches expected
3. Check statistics (mismatch rates, parse failures)

### Performance Tests
1. Measure generation speed vs traditional batch approach
2. Monitor memory usage over time
3. Profile bottlenecks

## Usage Examples

### Basic Usage - Generate 2x Ground Truths
```bash
python src/molgen3D/evaluation/inference_multiconf.py \
    --device local \
    --test_set distinct \
    --conformers_per_batch 8 \
    --conformer_multiplier 2
```

### Generate 3x Ground Truths
```bash
python src/molgen3D/evaluation/inference_multiconf.py \
    --device local \
    --test_set distinct \
    --conformers_per_batch 8 \
    --conformer_multiplier 3
```

### Debug Mode (Small Test)
```bash
python src/molgen3D/evaluation/inference_multiconf.py \
    --device local \
    --test_set distinct \
    --conformers_per_batch 4 \
    --conformer_multiplier 2 \
    --limit 3
```

### Example Scenario
If you have:
- SMILES_A: 10 ground truths → generates 20 conformers (10 × 2)
- SMILES_B: 5 ground truths → generates 10 conformers (5 × 2)  
- SMILES_C: 30 ground truths → generates 60 conformers (30 × 2)

With `conformers_per_batch=8`:
- SMILES_A: 3 batches (8 + 8 + 4) = 20 conformers, 3 model.generate() calls
- SMILES_B: 2 batches (8 + 2) = 10 conformers, 2 model.generate() calls
- SMILES_C: 8 batches (8×7 + 4) = 60 conformers, 8 model.generate() calls

### Batch Size Recommendations
- **Default (8)**: Good balance of speed and memory
- **Small (4)**: For memory-constrained GPUs or large molecules
- **Large (16)**: For high-memory GPUs (A100/H100) with small molecules

## Future Enhancements

1. **Adaptive conformer count**: Generate until diversity metric is satisfied
2. **Quality filtering**: Only keep high-quality conformers
3. **Incremental saving**: Save results after each SMILES to avoid data loss
4. **Distributed generation**: Use Ray or similar for multi-GPU/multi-node
5. **Smart prompt management**: Compress or summarize earlier conformers
6. **Hybrid approach**: Mix forced generation with normal sampling
