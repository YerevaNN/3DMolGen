"""Lightweight probes for the custom Qwen3 tokenizer.

This script inspects how the tokenizer handles:
- Standalone digits and numeric substrings.
- Angle bracket tokens used in conformer coordinates.
- Commas and dashes that appear inside coordinate tuples.

Run from repo root with the 3dmolgen conda env active:
    python scripts/logit_processor/qwen_tokenizer_probe.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_DIR = REPO_ROOT / "src/molgen3D/training/tokenizers/Qwen3_tokenizer_custom"


def load_tokenizer(tokenizer_dir: Path) -> PreTrainedTokenizer:
    """Load the tokenizer from disk."""
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        trust_remote_code=True,
    )
    return tokenizer


def tokenize_with_offsets(
    tokenizer: PreTrainedTokenizer,
    text: str,
) -> List[Tuple[str, int, int, int]]:
    """Return tokens with ids and character spans."""
    encoding: BatchEncoding = tokenizer.encode_plus(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    tokens: List[Tuple[str, int, int, int]] = []
    for token_id, (start, end) in zip(
        encoding["input_ids"], encoding["offset_mapping"]
    ):
        token_str: str = tokenizer.decode([token_id])
        tokens.append((token_str, token_id, start, end))
    return tokens


def collect_tokens_with_chars(
    tokenizer: PreTrainedTokenizer,
    chars: Iterable[str],
) -> List[Tuple[str, int]]:
    """Collect all vocab entries that contain any of the provided characters."""
    vocab = tokenizer.get_vocab()
    wanted: set[str] = set(chars)
    matches: List[Tuple[str, int]] = []
    for token_str, token_id in vocab.items():
        if any(char in token_str for char in wanted):
            matches.append((token_str, token_id))
    matches.sort(key=lambda pair: pair[1])
    return matches


def describe_numeric_tokenization(tokenizer: PreTrainedTokenizer) -> None:
    """Print how digits and simple numbers are tokenized."""
    samples: Sequence[str] = [
        "0123456789",
        "-1.2345",
        "10.001",
        "-0.0001",
    ]
    for text in samples:
        tokens = tokenize_with_offsets(tokenizer, text)
        print(f"\nText: {text!r}")
        print("Tokens (str, id, start, end):")
        for tok_str, tok_id, start, end in tokens:
            print(f"  {tok_str!r:6} id={tok_id:6} span=({start},{end})")


def describe_skeleton_example(tokenizer: PreTrainedTokenizer) -> None:
    """Print tokenization for an enriched skeleton example."""
    ref_str = "[C]<0.0000,0.0000,0.0000>[O]<1.2345,6.7890,-1.2345>"
    tokens = tokenize_with_offsets(tokenizer, ref_str)
    print(f"\nSkeleton example: {ref_str}")
    print("Tokens (str, id, start, end):")
    for tok_str, tok_id, start, end in tokens:
        print(f"  {tok_str!r:8} id={tok_id:6} span=({start},{end})")
    free_positions = [
        idx
        for idx, (tok_str, _, _, _) in enumerate(tokens)
        if all(c in "0123456789-., " for c in tok_str)
    ]
    print(f"Positions that look like coordinate content: {free_positions}")


def describe_block_lists(tokenizer: PreTrainedTokenizer) -> None:
    """Print counts and samples for bracket, comma, and dash tokens."""
    bracket_tokens = collect_tokens_with_chars(tokenizer, "<>")
    comma_dash_tokens = collect_tokens_with_chars(tokenizer, ",-")

    print(f"\nTokens containing '<' or '>': {len(bracket_tokens)}")
    print("  First 10:", bracket_tokens[:10])
    print("  Last 10:", bracket_tokens[-10:])

    print(f"\nTokens containing only comma/dash characters:")
    only_comma_dash: List[Tuple[str, int]] = []
    for tok_str, tok_id in comma_dash_tokens:
        if tok_str and all(ch in ",-" for ch in tok_str):
            only_comma_dash.append((tok_str, tok_id))
    only_comma_dash.sort(key=lambda pair: pair[1])
    print(f"  Count: {len(only_comma_dash)}")
    print("  Sample:", only_comma_dash[:20])


def main() -> None:
    """Run all probes."""
    tokenizer = load_tokenizer(TOKENIZER_DIR)
    print(f"Loaded tokenizer from {TOKENIZER_DIR}")

    describe_numeric_tokenization(tokenizer)
    describe_skeleton_example(tokenizer)
    describe_block_lists(tokenizer)


if __name__ == "__main__":
    main()
