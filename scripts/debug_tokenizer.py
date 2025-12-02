#!/usr/bin/env python
"""Debug tokenizer behavior for constraint processor."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer
from molgen3D.config.paths import get_tokenizer_path

# Load the tokenizer
tokenizer_path = get_tokenizer_path("llama3_chem_v1")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Check how < and > are tokenized
print("Tokenizing angle brackets:")
lt_tokens = tokenizer.encode("<", add_special_tokens=False)
gt_tokens = tokenizer.encode(">", add_special_tokens=False)
print(f"  '<'  -> {lt_tokens} = {[tokenizer.decode([t]) for t in lt_tokens]}")
print(f"  '>'  -> {gt_tokens} = {[tokenizer.decode([t]) for t in gt_tokens]}")

# Check coordinate-like strings
test_strs = [
    '<1.234,2.345,3.456>',
    '<1O-]',
    '1O-][/SMILES]',
    '[/CONFORMER]',
    '[CONFORMER]',
    '[C]',
    '[c]',
    '=',
    '(',
    ')',
]
print()
print("Tokenizing test strings:")
for s in test_strs:
    tokens = tokenizer.encode(s, add_special_tokens=False)
    decoded = [tokenizer.decode([t]) for t in tokens]
    print(f"  {repr(s):30} -> {tokens} = {decoded}")

# Check EOS token
print()
print(f"EOS token: id={tokenizer.eos_token_id}, text={repr(tokenizer.eos_token)}")
print(f"Pad token: id={tokenizer.pad_token_id}, text={repr(tokenizer.pad_token)}")

# Also check if model can generate the sequence [/SMILES][CONFORMER]
seq = "[/SMILES][CONFORMER]"
tokens = tokenizer.encode(seq, add_special_tokens=False)
decoded = [tokenizer.decode([t]) for t in tokens]
print()
print(f"Sequence '{seq}':")
print(f"  Tokens: {tokens}")
print(f"  Decoded: {decoded}")

# Check digit tokens (0-9, including multi-char like "12", "345", etc.)
print()
print("Digit tokens:")
for d in "0123456789":
    tokens = tokenizer.encode(d, add_special_tokens=False)
    decoded = [tokenizer.decode([t]) for t in tokens]
    print(f"  '{d}' -> {tokens} = {decoded}")

# Check what tokens start with digits
print()
print("Finding all tokens that start with a digit...")
digit_start_tokens = []
for tok_id in range(tokenizer.vocab_size):
    try:
        text = tokenizer.decode([tok_id])
        if text and text[0].isdigit():
            digit_start_tokens.append((tok_id, text))
    except:
        pass

print(f"Found {len(digit_start_tokens)} tokens starting with digit")
print("Sample tokens:")
for tok_id, text in digit_start_tokens[:20]:
    print(f"  {tok_id}: {repr(text)}")

# Check what tokens end with a digit (important for > detection)
print()
print("Finding tokens that end with a digit...")
digit_end_tokens = []
for tok_id in range(tokenizer.vocab_size):
    try:
        text = tokenizer.decode([tok_id])
        if text and text[-1].isdigit():
            digit_end_tokens.append((tok_id, text))
    except:
        pass

print(f"Found {len(digit_end_tokens)} tokens ending with digit")
print("Sample tokens:")
for tok_id, text in digit_end_tokens[:20]:
    print(f"  {tok_id}: {repr(text)}")
