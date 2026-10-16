"""Unconditional and SMILES-conditioned generation for one checkpoint.

Both evaluations use one decoding configuration -- stochastic sampling,
temperature 1.0, top-p 1.0, top-k 50, no beams, no guidance, no reranking -- and
an arm mask that forbids the other arm's geometry tokens. Every raw attempt is
written out before any parsing or filtering.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from molgen3D.experiments.bindgpt_fsq import config as experiment_config
from molgen3D.experiments.bindgpt_fsq import public_tokenizer, vocabulary
from molgen3D.experiments.bindgpt_fsq.config import ExperimentConfig
from molgen3D.experiments.bindgpt_fsq.decode import decode_generation, record
from molgen3D.experiments.bindgpt_fsq.sequences import (
    EOS,
    LIGAND,
    PAD,
    XYZ,
    Arm,
)


@dataclass
class Sampler:
    """A loaded checkpoint plus its arm mask and decoding settings."""

    model: torch.nn.Module
    device: torch.device
    arm: Arm
    vocab: vocabulary.Vocabulary
    settings: object
    fsq_tokenizer: object = None
    layout: str = "suffix"

    @property
    def table(self) -> Dict[str, int]:
        return self.vocab.token_to_id

    def ids(self, tokens: Sequence[str]) -> List[int]:
        return self.vocab.ids(tokens)

    def tokens(self, ids: Iterable[int]) -> List[str]:
        lookup = self.vocab.tokens
        return [lookup[int(index)] for index in ids]


def load_sampler(
    checkpoint: str | Path,
    run_config: ExperimentConfig,
    device: str = "cuda",
    load_fsq: Optional[bool] = None,
) -> Sampler:
    """Load a trained checkpoint for sampling."""
    from transformers import AutoModelForCausalLM

    arm = Arm(run_config.arm)
    union = vocabulary.build_vocabulary()
    target = torch.device(device if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), local_files_only=True, dtype=torch.bfloat16
    ).eval().to(target)

    fsq = None
    if load_fsq if load_fsq is not None else arm is Arm.FSQ:
        from molgen3D.experiments.bindgpt_fsq.fsq_frozen import FrozenFSQTokenizer

        fsq = FrozenFSQTokenizer(run_config.fsq_checkpoint, device=str(target))
        fsq.assert_no_gradients()

    return Sampler(
        model=model,
        device=target,
        arm=arm,
        vocab=union,
        settings=run_config.decoding,
        fsq_tokenizer=fsq,
        layout=run_config.layout,
    )


def _bad_words(sampler: Sampler) -> List[List[int]]:
    """The other arm's geometry tokens, as single-token bad-word sequences."""
    return [[token_id] for token_id in sampler.vocab.forbidden_ids(sampler.arm)]


@torch.no_grad()
def sample_batch(
    sampler: Sampler,
    prompt_ids: Sequence[int],
    batch_size: int,
    seed: int,
    max_new_tokens: Optional[int] = None,
) -> List[List[int]]:
    """Return ``batch_size`` completions of one prompt, prompt tokens included."""
    from transformers import GenerationConfig, set_seed

    table = sampler.table
    # transformers.generate draws from the global RNG, so the per-batch seed has
    # to be set on it; a local torch.Generator would be ignored.
    set_seed(seed)

    inputs = torch.tensor([list(prompt_ids)] * batch_size, device=sampler.device)
    attention = torch.ones_like(inputs)
    generation = GenerationConfig(
        do_sample=sampler.settings.do_sample,
        temperature=sampler.settings.temperature,
        top_p=sampler.settings.top_p,
        top_k=sampler.settings.top_k,
        num_beams=sampler.settings.num_beams,
        max_new_tokens=max_new_tokens or sampler.settings.max_new_tokens,
        eos_token_id=table[EOS],
        pad_token_id=table[PAD],
        bad_words_ids=_bad_words(sampler) or None,
    )
    outputs = sampler.model.generate(
        input_ids=inputs,
        attention_mask=attention,
        generation_config=generation,
    )
    pad_id = table[PAD]
    return [
        [int(value) for value in row if int(value) != pad_id]
        for row in outputs.detach().cpu().numpy()
    ]


def generate_unconditional(
    sampler: Sampler,
    destination: str | Path,
    target_valid: int = 1_000,
    max_attempts: int = 20_000,
    batch_size: int = 256,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Sample from ``<LIGAND>`` until ``target_valid`` connected molecules exist."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    seed = seed if seed is not None else sampler.settings.sampling_seed

    prompt = sampler.ids([LIGAND])
    attempts = 0
    valid_connected = 0
    raw_valid = 0
    decoded_ok = 0
    failures: Dict[str, int] = {}
    started = time.perf_counter()

    with open(destination / "attempts.jsonl", "w") as handle:
        batch_index = 0
        while valid_connected < target_valid and attempts < max_attempts:
            size = min(batch_size, max_attempts - attempts)
            rows = sample_batch(sampler, prompt, size, seed + batch_index)
            batch_index += 1
            for row in rows:
                attempts += 1
                tokens = sampler.tokens(row)
                hit_limit = tokens[-1] != EOS
                result = decode_generation(
                    tokens, sampler.arm, sampler.fsq_tokenizer,
                    hit_length_limit=hit_limit, layout=sampler.layout,
                )
                if result.left_smiles and result.flags.get("left_canonical"):
                    raw_valid += 1
                if result.ok:
                    decoded_ok += 1
                is_primary = bool(result.ok and result.connected)
                if is_primary:
                    valid_connected += 1
                else:
                    reason = result.failure_reason or (
                        "disconnected" if result.ok else "other"
                    )
                    failures[reason] = failures.get(reason, 0) + 1
                handle.write(
                    json.dumps(
                        record(
                            result,
                            {
                                "attempt_index": attempts,
                                "sampling_seed": seed,
                                "in_primary_population": is_primary
                                and valid_connected <= target_valid,
                                "primary_rank": valid_connected if is_primary else None,
                                "generation_wall_time": time.perf_counter() - started,
                            },
                        ),
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                if valid_connected >= target_valid:
                    break

    elapsed = time.perf_counter() - started
    summary = {
        "arm": sampler.arm.value,
        "sampling_seed": seed,
        "attempts": attempts,
        "raw_smiles_valid": raw_valid,
        "coordinate_decode_success": decoded_ok,
        "valid_connected": valid_connected,
        "reached_target": valid_connected >= target_valid,
        "raw_validity": raw_valid / max(1, attempts),
        "valid_connected_yield": valid_connected / max(1, attempts),
        "attempts_per_valid_connected": attempts / max(1, valid_connected),
        "failure_reasons": failures,
        "wall_seconds": elapsed,
        "gpu_seconds": elapsed,
        "decoding": {
            "temperature": sampler.settings.temperature,
            "top_p": sampler.settings.top_p,
            "top_k": sampler.settings.top_k,
            "max_new_tokens": sampler.settings.max_new_tokens,
        },
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def generate_conditional(
    sampler: Sampler,
    molecules: Sequence[Dict[str, object]],
    destination: str | Path,
    generations_per_reference: int = 2,
    max_attempt_multiplier: int = 4,
    batch_size: int = 64,
    seed: Optional[int] = None,
    shard: int = 0,
) -> Dict[str, object]:
    """Sample ``2K`` conformers for each test molecule with ``K`` references."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    seed = seed if seed is not None else sampler.settings.sampling_seed
    smiles_tokens = vocabulary.load_smiles_tokens()

    per_molecule: List[Dict[str, object]] = []
    started = time.perf_counter()
    attempts_path = destination / f"attempts_{shard:03d}.jsonl"

    with open(attempts_path, "w") as handle:
        for position, entry in enumerate(molecules):
            key = str(entry["key"])
            prompt_smiles = str(entry["prompt_smiles"])
            references = int(entry["num_references"])
            wanted = generations_per_reference * references
            budget = max_attempt_multiplier * references

            try:
                prompt = sampler.ids(
                    [LIGAND]
                    + public_tokenizer.greedy_tokenize(prompt_smiles, smiles_tokens)
                    + [XYZ]
                )
            except ValueError as exc:
                per_molecule.append(
                    {
                        "key": key,
                        "references": references,
                        "wanted": wanted,
                        "obtained": 0,
                        "attempts": 0,
                        "failed": True,
                        "failure_reasons": {"smiles_out_of_vocabulary": 1},
                        "error": str(exc)[:200],
                    }
                )
                continue

            obtained = 0
            attempts = 0
            failures: Dict[str, int] = {}
            batch_index = 0
            while obtained < wanted and attempts < budget:
                size = min(batch_size, budget - attempts, (wanted - obtained) * 2 or 1)
                rows = sample_batch(
                    sampler, prompt, max(1, size), seed + 1_000_003 * position + batch_index
                )
                batch_index += 1
                for row in rows:
                    attempts += 1
                    tokens = sampler.tokens(row)
                    hit_limit = tokens[-1] != EOS
                    result = decode_generation(
                        tokens,
                        sampler.arm,
                        sampler.fsq_tokenizer,
                        prompt_smiles=prompt_smiles,
                        hit_length_limit=hit_limit,
                        layout=sampler.layout,
                    )
                    accepted = bool(result.ok and result.matches_prompt)
                    if accepted:
                        obtained += 1
                    else:
                        reason = result.failure_reason or "other"
                        failures[reason] = failures.get(reason, 0) + 1
                    handle.write(
                        json.dumps(
                            record(
                                result,
                                {
                                    "key": key,
                                    "attempt_index": attempts,
                                    "sampling_seed": seed,
                                    "accepted": accepted,
                                    "accepted_rank": obtained if accepted else None,
                                },
                            ),
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    if obtained >= wanted:
                        break

            per_molecule.append(
                {
                    "key": key,
                    "references": references,
                    "wanted": wanted,
                    "obtained": obtained,
                    "attempts": attempts,
                    "failed": obtained < wanted,
                    "failure_reasons": failures,
                }
            )

    failed = [entry for entry in per_molecule if entry["failed"]]
    summary = {
        "arm": sampler.arm.value,
        "sampling_seed": seed,
        "shard": shard,
        "molecules": len(per_molecule),
        "failed_molecules": len(failed),
        "failed_fraction": len(failed) / max(1, len(per_molecule)),
        "attempts": sum(int(entry["attempts"]) for entry in per_molecule),
        "obtained": sum(int(entry["obtained"]) for entry in per_molecule),
        "wanted": sum(int(entry["wanted"]) for entry in per_molecule),
        "decode_yield": sum(int(entry["obtained"]) for entry in per_molecule)
        / max(1, sum(int(entry["attempts"]) for entry in per_molecule)),
        "wall_seconds": time.perf_counter() - started,
        "per_molecule": per_molecule,
    }
    (destination / f"summary_{shard:03d}.json").write_text(json.dumps(summary, indent=1))
    return summary


def load_test_molecules(shard: int = 0, num_shards: int = 1) -> List[Dict[str, object]]:
    """Return the frozen test molecules with deterministic canonical prompts."""
    import cloudpickle
    from rdkit import Chem

    with open(experiment_config.BENCHMARK_TEST_PICKLE, "rb") as handle:
        payload = cloudpickle.load(handle)

    molecules: List[Dict[str, object]] = []
    for index, (key, entry) in enumerate(sorted(payload.items())):
        if index % num_shards != shard:
            continue
        confs = entry.get("confs") or []
        if not confs:
            continue
        reference = Chem.RemoveHs(confs[0])
        molecules.append(
            {
                "key": key,
                "prompt_smiles": Chem.MolToSmiles(
                    reference, isomericSmiles=True, canonical=True
                ),
                "num_references": int(entry.get("num_confs") or len(confs)),
            }
        )
    return molecules
