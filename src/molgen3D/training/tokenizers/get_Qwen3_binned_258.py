#!/usr/bin/env python
import os
import json
from transformers import AutoTokenizer

# ----- CONFIG -----

# Base tokenizer: Qwen3 + 4 molecular tokens ([SMILES], [CONFORMER], etc.)
CUSTOM_TOK_DIR = os.path.join(
    os.path.dirname(__file__), "Qwen3_tokenizer_custom"
)

# Output directory for the new binned tokenizer
OUT_TOK_DIR = os.path.join(
    os.path.dirname(__file__), "Qwen3_tokenizer_binned_258"
)

# 258 bin tokens: "000" (BIN_L) + "001"-"256" (interior) + "257" (BIN_H)
N_INTERIOR_BINS = 256
BIN_TOKENS = [f"{i:03d}" for i in range(N_INTERIOR_BINS + 2)]  # "000" .. "257"


# ----- SCRIPT -----

def main():
    print(f"Loading base tokenizer from: {CUSTOM_TOK_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOK_DIR)
    base_vocab_size = len(tokenizer)
    print(f"  Base vocab size: {base_vocab_size}")

    # Verify the 4 molecular tokens are present
    for tok in ["[SMILES]", "[CONFORMER]", "[/SMILES]", "[/CONFORMER]"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        assert tid != tokenizer.unk_token_id, f"Missing token: {tok}"
        print(f"  {tok} -> {tid}")

    # Add bin tokens as normal (not special) tokens
    print(f"\nAdding {len(BIN_TOKENS)} bin tokens: {BIN_TOKENS[0]} .. {BIN_TOKENS[-1]}")
    num_added = tokenizer.add_tokens(BIN_TOKENS, special_tokens=False)
    print(f"  Tokens added: {num_added}")

    new_vocab_size = len(tokenizer)
    print(f"  New vocab size: {new_vocab_size}")
    assert num_added == len(BIN_TOKENS), (
        f"Expected {len(BIN_TOKENS)} new tokens, but only {num_added} were added. "
        f"Some tokens may already exist in the vocabulary."
    )

    # Verify all bin tokens have valid IDs
    unk_id = tokenizer.unk_token_id
    print("\nVerifying bin token IDs:")
    first_bin_id = tokenizer.convert_tokens_to_ids("000")
    last_bin_id = tokenizer.convert_tokens_to_ids(BIN_TOKENS[-1])
    print(f"  000 (BIN_L)   -> {first_bin_id}")
    print(f"  001 (bin 1)   -> {tokenizer.convert_tokens_to_ids('001')}")
    print(f"  128 (mid bin) -> {tokenizer.convert_tokens_to_ids('128')}")
    print(f"  256 (bin 256) -> {tokenizer.convert_tokens_to_ids('256')}")
    print(f"  257 (BIN_H)   -> {last_bin_id}")

    for tok in BIN_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(tok)
        assert tid is not None and tid != unk_id, f"Token {tok!r} has invalid id {tid}"

    # Check that bin token IDs are contiguous
    ids = [tokenizer.convert_tokens_to_ids(t) for t in BIN_TOKENS]
    assert ids == list(range(ids[0], ids[0] + len(BIN_TOKENS))), (
        "Bin token IDs are not contiguous!"
    )
    print(f"  All {len(BIN_TOKENS)} bin tokens contiguous: [{ids[0]}, {ids[-1]}]")

    # Verify ";" is already in the base vocabulary
    semi_ids = tokenizer.encode(";", add_special_tokens=False)
    print(f"\n  ';' encodes to token ids: {semi_ids}")

    # Save
    os.makedirs(OUT_TOK_DIR, exist_ok=True)
    print(f"\nSaving tokenizer to: {OUT_TOK_DIR}")
    tokenizer.save_pretrained(OUT_TOK_DIR)

    # Save metadata
    metadata = {
        "base_vocab_size": base_vocab_size,
        "num_bin_tokens_added": num_added,
        "new_vocab_size": new_vocab_size,
        "n_interior_bins": N_INTERIOR_BINS,
        "bin_l_token": "000",
        "bin_h_token": f"{N_INTERIOR_BINS + 1:03d}",
        "bin_token_id_range": [first_bin_id, last_bin_id],
    }
    meta_path = os.path.join(OUT_TOK_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata -> {meta_path}")

    # Sanity: reload and verify
    print("\n=== Reload sanity check ===")
    tok2 = AutoTokenizer.from_pretrained(OUT_TOK_DIR)
    assert len(tok2) == new_vocab_size, (
        f"Reloaded vocab size {len(tok2)} != {new_vocab_size}"
    )

    # Round-trip: encode a sample binned molecule string
    sample = "[SMILES]CC(=O)O[/SMILES][CONFORMER][C]128130125;[C]100102098;[/CONFORMER]"
    encoded = tok2.encode(sample, add_special_tokens=False)
    decoded = tok2.decode(encoded)
    print(f"\n  Sample: {sample}")
    print(f"  Encoded ({len(encoded)} tokens): {encoded}")
    print(f"  Decoded: {decoded}")

    # Show token-by-token breakdown for the bin portion
    print("\n  Token breakdown:")
    tokens = tok2.convert_ids_to_tokens(encoded)
    for tid, tok in zip(encoded, tokens):
        print(f"    {tid:>6d}  {tok!r}")

    print(f"\nDone. Tokenizer saved to {OUT_TOK_DIR}")
    print(f"  Total vocab size: {new_vocab_size}")
    print(f"  Bin tokens: {len(BIN_TOKENS)} (000=BIN_L, 001-256=interior, 257=BIN_H)")


if __name__ == "__main__":
    main()
