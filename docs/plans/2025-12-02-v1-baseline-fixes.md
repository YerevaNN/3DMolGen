# v1 Baseline Fixes (2025-12-02)

## Issues Found in Initial v1

After running the first v1 baseline, we discovered critical bugs:

1. **Generation doesn't stop after `[/CONFORMER]`**
   - Found 6+ instances of `[/CONFORMER]` in decoded outputs
   - Model keeps generating more conformers

2. **Missing validation of conformer→SMILES mapping**
   - Need to verify embedded conformer decodes back to correct canonical SMILES

3. **`decoded_text` includes input prompt**
   - Expected behavior from `model.generate()` (returns full sequence)
   - Validation correctly handles this by extracting SMILES and CONFORMER blocks separately

## Fixes Applied

### 1. Logit Processor: Force EOS After Template Completion

**File**: `src/molgen3D/evaluation/constrained_logits_v1.py`

**Changes**:
- Added `eos_token_id` parameter to `__init__()`
- Modified `__call__()` to force EOS token when `state.done`:
  ```python
  if state.done or state.seg_idx >= len(template.segments):
      # Template fully consumed - force EOS to stop generation
      if self.eos_token_id is not None:
          scores[b, :] = -torch.inf
          scores[b, self.eos_token_id] = 0.0
      continue
  ```

**Before**: When done, processor would `continue` without masking → model free to generate
**After**: When done, processor forces EOS token → model must stop

### 2. Smoke Runner: Use Correct EOS Token

**File**: `scripts/run_constrained_smoke_v1.py`

**Changes**:
- Changed from using `[/CONFORMER]` first token as EOS
- Now use tokenizer's actual EOS token:
  ```python
  eos_token_id = tokenizer.eos_token_id
  if eos_token_id is None:
      eos_token_id = tokenizer.pad_token_id
  ```
- Pass `eos_token_id` to both processor and `model.generate()`

**Before**: Used `tokenizer.encode("[/CONFORMER]")[0]` as EOS (incorrect - multi-token sequence)
**After**: Processor forces all `[/CONFORMER]` tokens via template, THEN forces actual EOS

### 3. Analysis Script: Enhanced Diagnostics

**File**: `scripts/analyze_smoke_report.py`

**New Features**:
1. **Conformer→SMILES Mapping Validation**
   - Extracts first `[CONFORMER]...[/CONFORMER]` block
   - Uses `strip_smiles()` to decode to canonical
   - Compares to input SMILES
   - Reports mapping validity rate

2. **Generation Stopping Check**
   - Counts `[/CONFORMER]` instances per sample
   - Flags samples with >1 instance (didn't stop properly)

3. **Better Import Handling**
   - Gracefully falls back if `strip_smiles` not available
   - Uses simple regex fallback: `re.sub(r'<[^>]*>', '', s)`

## How to Use

### Run v1 Baseline

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
/bin/zsh -il -c "conda activate 3dmolgen && python scripts/run_constrained_smoke_v1.py \
  --dataset clean --sample-size 64 --batch-size 16 \
  --model-path /nfs/h100/raid/chem/checkpoints/hf/yerevann/Llama-3.2-380M_conformers/fea47a381e4046e4956dc44a/step-26100 \
  --tokenizer-name llama3_chem_v1 --sampling-config top_p_sampling1 \
  --device cuda --torch-dtype bfloat16 --attention sdpa_paged \
  --json-report outputs/smoke/constrained_clean_64_v1_fixed.json"
```

### Analyze Results

```bash
python scripts/analyze_smoke_report.py outputs/smoke/constrained_clean_64_v1_fixed.json
```

**New diagnostics shown**:
- Coordinate quality breakdown
- Conformer→SMILES mapping validity
- Generation stopping check
- Special token pollution

### Compare Versions

```bash
python scripts/analyze_smoke_report.py outputs/smoke/constrained_clean_64_v*.json --compare
```

## Expected Improvements

With these fixes, v1 should:
1. ✅ Stop immediately after first `[/CONFORMER]`
2. ✅ Show accurate conformer→SMILES mapping rate
3. ✅ Provide clear diagnostics on what's working vs broken

This gives us a true baseline to measure the model's **natural ability** to generate coordinates without constraints.

## Next Steps After v1

Based on v1 results, we can:
- If mapping fails → Debug template building or strip_smiles logic
- If coordinates are garbage → Add v2 with minimal coordinate constraints
- If stopping still fails → Debug EOS token handling further
- If model generates valid coords naturally → Minimal constraints needed!
