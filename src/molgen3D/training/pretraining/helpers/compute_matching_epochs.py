#!/usr/bin/env python3
"""Estimate compute-matched epochs for a coordinate-token representation.

The accounting model is intentionally narrow:

    D / x  = 6 - 4a
    D' / x = 3
    6 N D e = 6 N D' e' + 2 N' x

Each training example is the initial SMILES followed by the extended SMILES,
so the source tokens ``x`` are counted twice on both sides.  For the binned
representation every atom adds three coordinate-bin tokens plus one
delimiter, and the atom count is ``(1 - a) x``::

    D  = x + x + 4x(1 - a) = x(6 - 4a)
    D' = x + (x + x)       = 3x

Solving the supplied equation gives:

    e = (9 N e' + N') / (3 N (6 - 4a))

Here ``N`` is the Qwen parameter count, ``N'`` is the coordinate-tokenizer
parameter count, and ``e'`` is the number of epochs used by our FSQ run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, supported by this project
    import tomli as tomllib


DEFAULT_TRAINING_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "config"
    / "pretrain"
    / "qwen3_06b_fsq_bigdata_from_base.toml"
)
_FSQ_PAIR_RE = re.compile(r"(.*?)<(\d+)>")
_EPOCH_RE = re.compile(r"(?:^|_)(\d+(?:\.\d+)?)e(?:_|$)", re.IGNORECASE)


@dataclass(frozen=True)
class CharacterFraction:
    """Measured source-token composition of FSQ-serialized conformers."""

    character_tokens: int
    source_tokens: int
    sequences: int
    malformed_sequences: int

    @property
    def fraction(self) -> float:
        if self.source_tokens <= 0:
            raise ValueError("Cannot compute a: no source tokens were found.")
        return self.character_tokens / self.source_tokens


@dataclass(frozen=True)
class ComputeMatch:
    N: int
    N_prime: int
    e_prime: float
    a: float
    D_over_x: float
    D_prime_over_x: float
    e: float


def compute_matching_epochs(
    qwen_parameters: int,
    tokenizer_parameters: int,
    our_epochs: float,
    character_fraction: float,
) -> ComputeMatch:
    """Solve the compute-matching equation after validating its inputs.

    Parameter counts mean total unique model parameters, not only parameters
    whose ``requires_grad`` flag happens to be set at inference time.
    """

    if qwen_parameters <= 0:
        raise ValueError("N (Qwen parameters) must be > 0.")
    if tokenizer_parameters < 0:
        raise ValueError("N' (tokenizer parameters) must be >= 0.")
    if our_epochs <= 0:
        raise ValueError("e' (our epochs) must be > 0.")
    if not 0 <= character_fraction <= 1:
        raise ValueError("a (character-token fraction) must be in [0, 1].")

    d_over_x = 6.0 - 4.0 * character_fraction
    d_prime_over_x = 3.0
    epochs = (
        9.0 * qwen_parameters * our_epochs + tokenizer_parameters
    ) / (3.0 * qwen_parameters * d_over_x)
    return ComputeMatch(
        N=qwen_parameters,
        N_prime=tokenizer_parameters,
        e_prime=our_epochs,
        a=character_fraction,
        D_over_x=d_over_x,
        D_prime_over_x=d_prime_over_x,
        e=epochs,
    )


def _iter_embedded_smiles(obj: dict[str, Any]) -> Iterator[str]:
    embedded = obj.get("embedded_smiles")
    if isinstance(embedded, str) and embedded:
        yield embedded

    isomers = obj.get("isomers")
    if not isinstance(isomers, dict):
        return
    for conformers in isomers.values():
        if not isinstance(conformers, list):
            continue
        for conformer in conformers:
            if not isinstance(conformer, dict):
                continue
            embedded = conformer.get("embedded_smiles")
            if isinstance(embedded, str) and embedded:
                yield embedded


def _source_tokens(fsq_text: str) -> list[str] | None:
    """Return the source token paired with each ``<FSQ-code>``.

    A round-trip check rejects partial regex matches, which keeps malformed
    sequences from silently biasing ``a``.
    """

    pairs = _FSQ_PAIR_RE.findall(fsq_text)
    if not pairs:
        return None
    reconstructed = "".join(f"{token}<{code}>" for token, code in pairs)
    if reconstructed != fsq_text:
        return None
    return [token for token, _ in pairs]


def measure_character_fraction(
    jsonl_files: Iterable[Path], max_sequences: int | None = 100_000
) -> CharacterFraction:
    """Measure ``a`` from actual FSQ-tokenized conformer sequences.

    A *character token* is a coordinate-free SMILES topology token in the
    CoordToken/FSQ source vocabulary: bond, branch, ring, stereochemistry, or
    separator syntax such as ``(``, ``=``, ``1``, or ``/``. Atom tokens are
    bracketed (for example ``[C]`` or ``[C@@H]``), carry coordinates in the
    original enriched representation, and are therefore not character tokens.

    The denominator is all parsed source tokens, before appending FSQ codes.
    """

    if max_sequences is not None and max_sequences <= 0:
        raise ValueError("max_sequences must be > 0 or None.")

    character_tokens = 0
    source_tokens = 0
    sequences = 0
    malformed_sequences = 0

    for file_path in jsonl_files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if max_sequences is not None and sequences >= max_sequences:
                    return CharacterFraction(
                        character_tokens,
                        source_tokens,
                        sequences,
                        malformed_sequences,
                    )
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(obj, dict):
                    continue
                for embedded in _iter_embedded_smiles(obj):
                    if max_sequences is not None and sequences >= max_sequences:
                        return CharacterFraction(
                            character_tokens,
                            source_tokens,
                            sequences,
                            malformed_sequences,
                        )
                    tokens = _source_tokens(embedded)
                    if tokens is None:
                        malformed_sequences += 1
                        continue
                    # Preprocessing normalizes every atom to a bracketed token.
                    character_tokens += sum(
                        1 for token in tokens if not token.startswith("[")
                    )
                    source_tokens += len(tokens)
                    sequences += 1

    return CharacterFraction(
        character_tokens, source_tokens, sequences, malformed_sequences
    )


def _shape_numel(shape: Iterable[int]) -> int:
    return math.prod(int(dimension) for dimension in shape)


def count_hf_safetensors_parameters(model_path: Path) -> int:
    """Count unique stored tensors in an HF safetensors model."""

    from safetensors import safe_open

    if model_path.is_file() and model_path.suffix == ".safetensors":
        files = [model_path]
    else:
        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            files = sorted(
                {model_path / name for name in index.get("weight_map", {}).values()}
            )
        else:
            files = sorted(model_path.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No model safetensors found under {model_path}.")

    seen_names: set[str] = set()
    total = 0
    for file_path in files:
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in seen_names:
                    continue
                seen_names.add(name)
                total += _shape_numel(handle.get_slice(name).get_shape())
    return total


def count_fsq_parameters(checkpoint_path: Path) -> int:
    """Instantiate the checkpoint-compatible FSQ architecture and count it."""

    from molgen3D.data_processing.smiles_encoder_decoder_fsq import MolFSQModel

    model = MolFSQModel.load_from_checkpoint(
        checkpoint_path, map_location="cpu", device="cpu"
    )
    # load_from_checkpoint freezes the inference model, so requires_grad is not
    # a meaningful way to count the parameters used when it was trained.
    return sum(parameter.numel() for parameter in model.parameters())


def _load_training_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _resolve_config_value(value: str) -> Path:
    if value.startswith("data:"):
        from molgen3D.config.paths import get_data_path

        return get_data_path(value.split(":", 1)[1])
    return Path(value).expanduser()


def _epochs_from_description(config: dict[str, Any]) -> float:
    description = str(config.get("job", {}).get("description", ""))
    match = _EPOCH_RE.search(description)
    if match is None:
        raise ValueError(
            "Could not infer e' from job.description; pass --our-epochs explicitly."
        )
    return float(match.group(1))


def _jsonl_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file():
        return [dataset_path]
    files = sorted(dataset_path.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found under {dataset_path}.")
    return files


def _format_report(
    result: ComputeMatch,
    *,
    character_stats: CharacterFraction | None,
    qwen_source: str,
    tokenizer_source: str,
) -> str:
    lines = [
        "Compute-matched epoch estimate",
        f"N: {result.N:,}",
        f"N': {result.N_prime:,}",
        f"e': {result.e_prime:.12g}",
        f"a: {result.a:.12g}",
        f"D/x = 6 - 4a: {result.D_over_x:.12g}",
        f"D'/x: {result.D_prime_over_x:.12g}",
        f"computed e: {result.e:.12g}",
        f"N source: {qwen_source}",
        f"N' source: {tokenizer_source}",
    ]
    if character_stats is not None:
        lines.extend(
            [
                "character-token definition: coordinate-free source SMILES "
                "topology tokens; bracketed atom tokens are excluded",
                "a measurement: "
                f"{character_stats.character_tokens:,}/"
                f"{character_stats.source_tokens:,} tokens from "
                f"{character_stats.sequences:,} sequences",
                f"malformed sequences skipped: {character_stats.malformed_sequences:,}",
            ]
        )
    lines.append(
        "parameter-size assumption: total unique parameters (not trainable-only)"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute epochs that match Qwen + coordinate-tokenizer training compute."
    )
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--qwen-model-path", type=Path)
    parser.add_argument("--tokenizer-checkpoint", type=Path)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--qwen-parameters", type=int)
    parser.add_argument("--tokenizer-parameters", type=int)
    parser.add_argument("--our-epochs", type=float)
    parser.add_argument("--character-fraction", type=float)
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=100_000,
        help=(
            "Maximum actual tokenized conformers used to estimate a. Pass 0 to "
            "measure every sequence in the dataset. Note that the shards are "
            "not shuffled, so any cap samples their head and biases a."
        ),
    )
    parser.add_argument("--output", type=Path, help="Optional JSON result path.")
    args = parser.parse_args()

    config = _load_training_config(args.training_config)

    qwen_model_path = args.qwen_model_path
    if qwen_model_path is None:
        raw_model_path = str(config.get("molgen_run", {}).get("base_model_tag", ""))
        if not raw_model_path:
            raise ValueError("Training config has no molgen_run.base_model_tag.")
        qwen_model_path = _resolve_config_value(raw_model_path)

    if args.qwen_parameters is None:
        qwen_parameters = count_hf_safetensors_parameters(qwen_model_path)
        qwen_source = str(qwen_model_path)
    else:
        qwen_parameters = args.qwen_parameters
        qwen_source = "CLI override"

    tokenizer_checkpoint = args.tokenizer_checkpoint
    if tokenizer_checkpoint is None:
        from molgen3D.data_processing.smiles_encoder_decoder_fsq import (
            _resolve_fsq_ckpt_path,
        )

        tokenizer_checkpoint = _resolve_fsq_ckpt_path()
    if args.tokenizer_parameters is None:
        tokenizer_parameters = count_fsq_parameters(tokenizer_checkpoint)
        tokenizer_source = str(tokenizer_checkpoint)
    else:
        tokenizer_parameters = args.tokenizer_parameters
        tokenizer_source = "CLI override"

    our_epochs = (
        args.our_epochs
        if args.our_epochs is not None
        else _epochs_from_description(config)
    )

    character_stats: CharacterFraction | None = None
    if args.character_fraction is None:
        dataset_path = args.dataset_path
        if dataset_path is None:
            raw_dataset_path = str(
                config.get("molgen_data", {}).get("train_path")
                or config.get("training", {}).get("dataset_path", "")
            )
            if not raw_dataset_path:
                raise ValueError("Training config has no training dataset path.")
            dataset_path = _resolve_config_value(raw_dataset_path)
        character_stats = measure_character_fraction(
            _jsonl_files(dataset_path),
            max_sequences=args.max_sequences or None,
        )
        character_fraction = character_stats.fraction
    else:
        character_fraction = args.character_fraction

    result = compute_matching_epochs(
        qwen_parameters,
        tokenizer_parameters,
        our_epochs,
        character_fraction,
    )
    print(
        _format_report(
            result,
            character_stats=character_stats,
            qwen_source=qwen_source,
            tokenizer_source=tokenizer_source,
        )
    )

    if args.output is not None:
        payload = asdict(result)
        payload["parameter_size_assumption"] = "total unique parameters"
        payload["qwen_parameter_source"] = qwen_source
        payload["tokenizer_parameter_source"] = tokenizer_source
        if character_stats is not None:
            payload["character_fraction_measurement"] = asdict(character_stats)
            payload["character_token_definition"] = (
                "coordinate-free source SMILES topology token; bracketed atom "
                "tokens are excluded"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
