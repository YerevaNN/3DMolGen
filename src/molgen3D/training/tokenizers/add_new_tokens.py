#!/usr/bin/env python
import os
from transformers import PreTrainedTokenizerFast, AutoTokenizer

# ----- CONFIG -----

BASE_DIR = "/nfs/h100/raid/chem/checkpoints/yerevann/qwen3_06b/Qwen3-0.6B-Base"

# Path to the *tokenizer* files inside BASE_DIR
TOKENIZER_JSON = os.path.join(BASE_DIR, "tokenizer.json")

# Where to save the untouched original tokenizer snapshot
ORIG_TOK_DIR = os.path.join("./src/molgen3D/training/tokenizers", "Qwen3_tokenizer_original")

# Where to save the tokenizer with the 4 extra tokens
CUSTOM_TOK_DIR = os.path.join("./src/molgen3D/training/tokenizers", "Qwen3_tokenizer_custom")

# Your 4 new tokens as *normal* vocab items
NEW_TOKENS = ["[SMILES]", "[CONFORMERS]", "[/SMILES]", "[/CONFORMERS]"]

# ----- SCRIPT -----

def main():
    if not os.path.exists(TOKENIZER_JSON):
        raise FileNotFoundError(f"tokenizer.json not found at {TOKENIZER_JSON}")

    print(f"Loading fast tokenizer from: {TOKENIZER_JSON}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_JSON)
    # Optionally: load tokenizer_config.json to set special tokens if needed
    print(f"Tokenizer is_fast: {tokenizer.is_fast}")

    print(f"Original vocab size: {len(tokenizer)}")
    orig_vocab_size = len(tokenizer)

    # 1) Save a snapshot of the original tokenizer
    os.makedirs(ORIG_TOK_DIR, exist_ok=True)
    print(f"Saving original tokenizer to: {ORIG_TOK_DIR}")
    tokenizer.save_pretrained(ORIG_TOK_DIR)

    # 2) Add new tokens as *normal* tokens (not special tokens)
    print(f"Adding new tokens: {NEW_TOKENS}")
    num_added = tokenizer.add_tokens(NEW_TOKENS, special_tokens=False)
    print(f"Number of tokens actually added: {num_added}")

    new_vocab_size = len(tokenizer)
    print(f"New vocab size: {new_vocab_size}")

    # 3) Save the modified tokenizer
    os.makedirs(CUSTOM_TOK_DIR, exist_ok=True)
    print(f"Saving custom tokenizer with extra tokens to: {CUSTOM_TOK_DIR}")
    tokenizer.save_pretrained(CUSTOM_TOK_DIR)

    # ----- SANITY CHECKS -----
    print("\n=== Sanity checks ===")

    # 1) Fast tokenizer?
    print(f"Tokenizer is_fast before reload: {tokenizer.is_fast}")

    # 2) Vocab size increased by num_added
    expected_new_size = orig_vocab_size + num_added
    assert new_vocab_size == expected_new_size, (
        f"Vocab size mismatch: expected {expected_new_size}, got {new_vocab_size}"
    )

    # 3) New tokens have valid (non-UNK) IDs
    unk_id = tokenizer.unk_token_id
    for tok in NEW_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(tok)
        print(f"Token {tok!r} -> id {tid}")
        assert tid is not None and tid != unk_id, (
            f"Token {tok!r} did not get a valid id (got {tid}, unk={unk_id})"
        )

    # 4) Reload both original and custom tokenizers and re-check via AutoTokenizer
    print("\nReloading original tokenizer...")
    orig_tok = AutoTokenizer.from_pretrained(ORIG_TOK_DIR)
    print(f"Original reloaded vocab size: {len(orig_tok)} (is_fast={orig_tok.is_fast})")
    assert len(orig_tok) == orig_vocab_size, "Reloaded original vocab size mismatch"

    print("Reloading custom tokenizer...")
    custom_tok = AutoTokenizer.from_pretrained(CUSTOM_TOK_DIR)
    print(f"Custom reloaded vocab size: {len(custom_tok)} (is_fast={custom_tok.is_fast})")
    assert len(custom_tok) == new_vocab_size, "Reloaded custom vocab size mismatch"

    for tok in NEW_TOKENS:
        tid = custom_tok.convert_tokens_to_ids(tok)
        print(f"[reload] Token {tok!r} -> id {tid}")
        assert tid is not None and tid != custom_tok.unk_token_id, (
            f"[reload] Token {tok!r} did not get a valid id after reload"
        )

    print("\nAll sanity checks passed ✅")

if __name__ == "__main__":
    main()