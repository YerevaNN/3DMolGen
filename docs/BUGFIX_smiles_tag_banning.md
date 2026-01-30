# Critical Bug Fix: Ban Both SMILES Tags

## Issue

The logits processor was only banning the **opening** `[SMILES]` tag but not the **closing** `[/SMILES]` tag during conformer generation. This caused the model to generate invalid sequences like:

```
[SMILES]CCO[/SMILES][CONFORMER]CCN(CC)CCNC(=O)Cn1ncc2c(=O)oc3ccc(C)cc3c21[/SMILES][CONFORMER]...
                                                                                    ^^^^^^^^^ Invalid!
```

The `[/SMILES]` tag appeared **inside** a conformer, breaking the expected format and potentially corrupting downstream parsing.

## Root Cause

In the original code:

```python
# logits_constraints.py
smiles_start_ids = tokenizer.encode(mol_tags[0], add_special_tokens=False)  # [SMILES]
banned_ids = set(smiles_start_ids)  # Only banned opening tag!
```

And in `inference_multiconf.py`:

```python
smiles_start_id = tokenizer.convert_tokens_to_ids("[SMILES]")
disallowed_tokens = []
if smiles_start_id is not None:
    disallowed_tokens.append(smiles_start_id)  # Only banned opening tag!
```

The closing `[/SMILES]` tag was never added to the banned list, so the model was free to generate it anywhere.

## Fix

### In `logits_constraints.py`:

```python
smiles_start_ids = tokenizer.encode(mol_tags[0], add_special_tokens=False)  # [SMILES]
smiles_end_ids = tokenizer.encode(mol_tags[1], add_special_tokens=False)    # [/SMILES]

banned_ids = set(smiles_start_ids)
banned_ids.update(smiles_end_ids)  # Ban closing tag too!
```

### In `inference_multiconf.py`:

```python
smiles_start_id = tokenizer.convert_tokens_to_ids("[SMILES]")
smiles_end_id = tokenizer.convert_tokens_to_ids("[/SMILES]")
eos_token_id = tokenizer.eos_token_id

disallowed_tokens = []
if smiles_start_id is not None:
    disallowed_tokens.append(smiles_start_id)
if smiles_end_id is not None:
    disallowed_tokens.append(smiles_end_id)  # Ban closing tag too!
if eos_token_id is not None:
    disallowed_tokens.append(eos_token_id)
```

### In `test_logits_processor.py`:

```python
smiles_start_ids = tokenizer.encode(mol_tags[0], add_special_tokens=False)
smiles_end_ids = tokenizer.encode(mol_tags[1], add_special_tokens=False)

banned_ids = set(smiles_start_ids)
banned_ids.update(smiles_end_ids)  # Ban closing tag too!
```

## Expected Behavior After Fix

The model should now generate only valid sequences:

```
[SMILES]CCO[/SMILES][CONFORMER][C]<x,y,z>[C]<x,y,z>[O]<x,y,z>[/CONFORMER][CONFORMER][C]<x,y,z>...
```

No `[SMILES]` or `[/SMILES]` tags should appear after the initial prompt.

## Testing

Run the test script to verify:

```bash
bash scripts/run_logits_test.sh
```

Check that:
1. ✅ No `[SMILES]` tokens in generated portion
2. ✅ No `[/SMILES]` tokens in generated portion
3. ✅ Only `[CONFORMER]` and `[/CONFORMER]` tags appear
4. ✅ Correct number of conformers are generated

## Impact

This fix affects:
- `src/molgen3D/training/grpo/logits_constraints.py` - Main logits processor
- `src/molgen3D/evaluation/inference_multiconf.py` - Multi-conformer inference
- `scripts/test_logits_processor.py` - Test script
- `docs/logits_processor_testing.md` - Documentation

## Files Changed

1. **logits_constraints.py**: Added `smiles_end_ids` and banned them
2. **inference_multiconf.py**: Added `smiles_end_id` and banned it
3. **test_logits_processor.py**: Added `smiles_end_ids` and banned them
4. **logits_processor_testing.md**: Updated docs to emphasize this requirement

## Prevention

When implementing token banning for tag-based systems:
- ⚠️ **Always ban BOTH opening and closing tags**
- ⚠️ **Test with actual generation to verify behavior**
- ⚠️ **Check for unexpected tokens in generated output**

## Related Issues

This type of bug can occur with any paired-tag system:
- `[SMILES]` / `[/SMILES]`
- `[CONFORMER]` / `[/CONFORMER]`
- Any custom XML-like or markdown-like tag pairs

When you want to prevent a semantic unit from appearing, you must ban ALL tokens associated with it.
