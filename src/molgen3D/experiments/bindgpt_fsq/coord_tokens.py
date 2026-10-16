"""Original BindGPT scalar coordinate tokens.

The public BindGPT tokenizer (``insilicomedicine/bindgpt_finetuned``) carries a
1,286-entry added-token vocabulary in which a coordinate scalar is spelled as
exactly two tokens:

* an integer token -- ``"0"``..``"99"`` and ``"-0"``..``"-99"`` (200 tokens), the
  sign living on the integer part so that values in ``(-1, 0]`` use ``"-0"``; and
* a fraction token -- ``".000"``..``".999"`` (1,000 tokens), always three digits.

So one atom costs six tokens. Verified against the public tokenizer: the string
``"-1.234"`` encodes to ``['-1', '.234']``, while a short form such as ``"0.5"``
degrades to ``['0', '5']``, which is why :func:`format_scalar` always emits three
decimals.

Nothing here loads BindGPT model weights; only the tokenizer files are used, and
only to assert that our local spelling matches theirs.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

#: Largest representable magnitude of the integer part.
MAX_INT_PART = 99

#: Number of fractional digits in a BindGPT fraction token.
FRACTION_DIGITS = 3

#: Tokens per coordinate scalar, and per atom.
TOKENS_PER_SCALAR = 2
TOKENS_PER_ATOM = 3 * TOKENS_PER_SCALAR


def integer_tokens() -> List[str]:
    """Return the 200 integer-part tokens in public-vocabulary order."""
    return [str(i) for i in range(MAX_INT_PART + 1)] + [
        f"-{i}" for i in range(MAX_INT_PART + 1)
    ]


def fraction_tokens() -> List[str]:
    """Return the 1,000 fraction tokens in public-vocabulary order."""
    return [f".{i:03d}" for i in range(10 ** FRACTION_DIGITS)]


def coordinate_tokens() -> List[str]:
    """Return every coordinate token, integers first then fractions."""
    return integer_tokens() + fraction_tokens()


_INTEGER_SET = frozenset(integer_tokens())
_FRACTION_SET = frozenset(fraction_tokens())


class CoordinateRangeError(ValueError):
    """Raised when a coordinate cannot be spelled in the BindGPT vocabulary."""


def format_scalar(value: float) -> str:
    """Return ``value`` rounded to three decimals as ``"<int><frac>"``.

    The result is a single string such as ``"-1.234"``; the public tokenizer
    splits it into its two tokens. Rounding is half-away-from-zero so that the
    sign of a value in ``(-0.0005, 0)`` is preserved as ``"-0.000"``, matching
    the existence of a distinct ``"-0"`` integer token.
    """
    if not math.isfinite(value):
        raise CoordinateRangeError(f"non-finite coordinate: {value!r}")

    scale = 10 ** FRACTION_DIGITS
    magnitude = abs(float(value))
    # Half-away-from-zero, computed on the magnitude so the sign never rounds away.
    units = int(math.floor(magnitude * scale + 0.5))
    int_part, frac_part = divmod(units, scale)
    if int_part > MAX_INT_PART:
        raise CoordinateRangeError(
            f"coordinate {value!r} exceeds the BindGPT integer range "
            f"+/-{MAX_INT_PART}"
        )

    sign = "-" if value < 0 or (value == 0 and math.copysign(1.0, value) < 0) else ""
    return f"{sign}{int_part}.{frac_part:0{FRACTION_DIGITS}d}"


def scalar_to_tokens(value: float) -> Tuple[str, str]:
    """Return the ``(integer_token, fraction_token)`` pair for ``value``."""
    text = format_scalar(value)
    int_token, _, frac_digits = text.partition(".")
    return int_token, f".{frac_digits}"


def tokens_to_scalar(int_token: str, frac_token: str) -> float:
    """Invert :func:`scalar_to_tokens`."""
    if int_token not in _INTEGER_SET:
        raise ValueError(f"not a BindGPT integer token: {int_token!r}")
    if frac_token not in _FRACTION_SET:
        raise ValueError(f"not a BindGPT fraction token: {frac_token!r}")

    negative = int_token.startswith("-")
    magnitude = abs(int(int_token)) + int(frac_token[1:]) / 10 ** FRACTION_DIGITS
    return -magnitude if negative else magnitude


def triplet_to_tokens(xyz: Sequence[float]) -> List[str]:
    """Return the six tokens spelling one atom's coordinate triplet."""
    if len(xyz) != 3:
        raise ValueError(f"expected 3 coordinates, got {len(xyz)}")
    tokens: List[str] = []
    for value in xyz:
        tokens.extend(scalar_to_tokens(value))
    return tokens


def tokens_to_triplet(tokens: Sequence[str]) -> Tuple[float, float, float]:
    """Invert :func:`triplet_to_tokens`."""
    if len(tokens) != TOKENS_PER_ATOM:
        raise ValueError(f"expected {TOKENS_PER_ATOM} tokens, got {len(tokens)}")
    values = [
        tokens_to_scalar(tokens[i], tokens[i + 1])
        for i in range(0, TOKENS_PER_ATOM, TOKENS_PER_SCALAR)
    ]
    return values[0], values[1], values[2]


def quantization_error(values: Iterable[float]) -> List[float]:
    """Return signed encode/decode error for each scalar in ``values``."""
    errors: List[float] = []
    for value in values:
        int_token, frac_token = scalar_to_tokens(value)
        errors.append(tokens_to_scalar(int_token, frac_token) - float(value))
    return errors
