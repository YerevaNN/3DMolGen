# Batched Conformer Generation

## Overview

The multi-conformer inference now supports **batched generation**, where you can control how many conformers are generated per `model.generate()` call using the `--conformers_per_batch` parameter.

## Why Batching?

### Memory Management
- Generating many conformers at once (e.g., 50+) can exhaust GPU memory due to growing KV cache
- Batching allows splitting into manageable chunks

### Quality & Diversity
- Smaller batches may produce more diverse conformers
- Model sees fresh context at the start of each batch
- Can help avoid repetitive patterns in long sequences

### Flexibility
- Different molecules need different amounts - some need 10 conformers, others 50
- Batching allows same code to handle both efficiently

## How It Works

### Example: 20 Target Conformers with batch_size=8

```
SMILES needs 20 conformers (10 ground truths × 2 multiplier)
conformers_per_batch = 8

Batch 1: Generate 8 conformers
  Input:  [SMILES]CCO[/SMILES]
  Output: [SMILES]CCO[/SMILES][CONFORMER]...[/CONFORMER]...[CONFORMER]...[/CONFORMER] (8 times)

Batch 2: Generate 8 more conformers
  Input:  [SMILES]CCO[/SMILES][CONFORMER]...[/CONFORMER]...[CONFORMER]...[/CONFORMER] (previous 8)
  Output: [SMILES]CCO[/SMILES][CONFORMER]...[/CONFORMER]...[CONFORMER]...[/CONFORMER] (16 times)

Batch 3: Generate 4 remaining conformers
  Input:  [SMILES]CCO[/SMILES][CONFORMER]...[/CONFORMER]...[CONFORMER]...[/CONFORMER] (previous 16)
  Output: [SMILES]CCO[/SMILES][CONFORMER]...[/CONFORMER]...[CONFORMER]...[/CONFORMER] (20 times)
```

### Key Points

1. **Accumulative**: Each batch continues from the previous batch's output
2. **Context preserved**: Model sees all previously generated conformers
3. **Exact count**: Last batch generates only remaining conformers (4 in example above)
4. **Automatic**: System calculates batch sizes and handles partial batches

## Implementation Details

### Function Signature Change

```python
def generate_multiple_conformers(
    model,
    tokenizer,
    smiles_prompt: str,
    num_conformers: int,  # Now means "conformers in THIS batch"
    gen_config,
    binned: bool,
    stats: Counter,
    geom_smiles: str,
    current_output: str = None,  # NEW: previous output to continue from
) -> tuple[List, str]:  # NEW: returns (mol_objects, decoded_output)
```

**Changes:**
- Added `current_output` parameter - if provided, continues from this output
- Now returns tuple: `(mol_objects, decoded_output)` 
- `decoded_output` is used as `current_output` for next batch

### Main Loop Structure

```python
for geom_smiles, sub_smiles, target_count in smiles_to_process:
    conformers_generated = 0
    current_output = None
    
    # Generate in batches until target reached
    while conformers_generated < target_count:
        remaining = target_count - conformers_generated
        batch_size = min(conformers_per_batch, remaining)
        
        # Generate batch
        mol_objects, current_output = generate_multiple_conformers(
            ...,
            num_conformers=batch_size,
            current_output=current_output,  # Continue from previous
        )
        
        conformers_generated += len(mol_objects)
```

## Usage

### Basic Usage

```bash
# Generate 2x ground truths, 8 conformers per batch (default)
python src/molgen3D/evaluation/inference_multiconf.py \
    --device local \
    --test_set distinct \
    --conformers_per_batch 8 \
    --conformer_multiplier 2
```

### Small Batches (Conservative)

```bash
# Smaller batches for memory-constrained environments
python src/molgen3D/evaluation/inference_multiconf.py \
    --device local \
    --test_set distinct \
    --conformers_per_batch 4 \
    --conformer_multiplier 2
```

### Large Batches (Aggressive)

```bash
# Larger batches for faster generation (if memory allows)
python src/molgen3D/evaluation/inference_multiconf.py \
    --device local \
    --test_set distinct \
    --conformers_per_batch 16 \
    --conformer_multiplier 2
```

## Performance Considerations

### Speed vs Memory Trade-off

| Batch Size | Speed | Memory | Quality |
|------------|-------|--------|---------|
| 4 | Slower | Low | High diversity |
| 8 | Moderate | Moderate | Good balance |
| 16 | Faster | High | May repeat patterns |
| 32+ | Fastest | Very high | Risk of repetition |

### Recommended Settings

**For most use cases:**
```bash
--conformers_per_batch 8
```

**For large molecules or limited GPU memory:**
```bash
--conformers_per_batch 4
```

**For high-memory GPUs (A100, H100) with small molecules:**
```bash
--conformers_per_batch 16
```

## Examples

### Example 1: Molecule with 10 Ground Truths

```bash
python ... --conformers_per_batch 8 --conformer_multiplier 2
```

**Execution:**
- Target: 10 × 2 = 20 conformers
- Batch 1: Generate 8 conformers
- Batch 2: Generate 8 conformers
- Batch 3: Generate 4 conformers
- Total: 3 model.generate() calls

### Example 2: Molecule with 3 Ground Truths

```bash
python ... --conformers_per_batch 8 --conformer_multiplier 2
```

**Execution:**
- Target: 3 × 2 = 6 conformers
- Batch 1: Generate 6 conformers (less than batch size)
- Total: 1 model.generate() call

### Example 3: Molecule with 50 Ground Truths

```bash
python ... --conformers_per_batch 8 --conformer_multiplier 2
```

**Execution:**
- Target: 50 × 2 = 100 conformers
- Batch 1-12: Generate 8 conformers each (96 total)
- Batch 13: Generate 4 conformers
- Total: 13 model.generate() calls

## Technical Details

### ConformerControlLogitsProcessor Integration

The processor is created fresh for each batch with `target_k` set to the batch size:

```python
logits_processor = ConformerControlLogitsProcessor(
    conformer_start_ids=conformer_start_ids,
    conformer_end_ids=conformer_end_ids,
    banned_token_ids=banned_ids,
    target_k=batch_size,  # Not total target, just this batch!
    force_hard=True,
)
```

### Stopping Criteria

Similarly, stopping criteria is set for the batch:

```python
stopping_criteria = ConformerCountStoppingCriteria(
    conformer_end_ids=conformer_end_ids,
    target_k=batch_size,  # Stops after batch_size conformers
)
```

### Context Accumulation

The key insight is passing previous output as new input:

```python
# First batch
prompt = "[SMILES]CCO[/SMILES]"
output_1 = generate(prompt)
# -> "[SMILES]CCO[/SMILES][CONFORMER]...[/CONFORMER]...(8x)"

# Second batch
output_2 = generate(output_1)  # Continue from previous!
# -> "[SMILES]CCO[/SMILES][CONFORMER]...[/CONFORMER]...(16x)"
```

## Debugging

### Check Logs

The system logs each batch:

```
Processing CCO: target 20 conformers
  Batch: generating 8 conformers (0/20 so far)
  Batch: generating 8 conformers (8/20 so far)
  Batch: generating 4 conformers (16/20 so far)
Completed CCO: generated 20/20 valid conformers
```

### Common Issues

**Issue: Getting fewer conformers than expected**
- Check if generation is stopping early due to errors
- Look for "No valid conformers in batch" warnings
- Verify SMILES validation isn't rejecting too many

**Issue: Out of memory**
- Reduce `--conformers_per_batch` to 4 or lower
- Check that `max_new_tokens` isn't too large
- Monitor GPU memory with `nvidia-smi`

**Issue: Slow generation**
- Increase `--conformers_per_batch` to reduce overhead
- Ensure model is compiled with `torch.compile`
- Check that KV cache is being used

## Future Enhancements

Potential improvements:
1. **Adaptive batch size**: Automatically adjust based on available memory
2. **Parallel SMILES**: Process multiple SMILES in parallel on same GPU
3. **Multi-GPU**: Distribute different SMILES across GPUs
4. **Early stopping**: Stop if quality degrades across batches
5. **Checkpoint recovery**: Save intermediate results in case of failure
