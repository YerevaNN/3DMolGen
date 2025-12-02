from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Sequence

import cloudpickle
import torch
from transformers import GenerationConfig, LogitsProcessorList, PreTrainedModel, PreTrainedTokenizer

from molgen3D.config.paths import get_data_path
from molgen3D.data_processing.smiles_encoder_decoder import strip_smiles
from molgen3D.evaluation.constrained_logits import (
    ConformerConstraintLogitsProcessor,
    build_templates_for_batch,
)
from molgen3D.evaluation.utils import extract_between, same_molecular_graph

SmokeDataset = Literal["clean", "distinct"]
_DATASET_TO_KEY = {
    "clean": "clean_smi",
    "distinct": "distinct_smi",
}


@dataclass
class SmokeRecord:
    """Holds the decoded text and validation issues for a single prompt."""

    prompt_smiles: str
    decoded_text: str
    canonical_smiles: str | None = None
    conformer_block: str | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues


@dataclass
class SmokeValidationResult:
    """Aggregated validation output for a smoke batch."""

    records: list[SmokeRecord]

    @property
    def total(self) -> int:
        return len(self.records)

    @property
    def failures(self) -> list[SmokeRecord]:
        return [rec for rec in self.records if rec.issues]

    @property
    def num_passed(self) -> int:
        return self.total - len(self.failures)

    def raise_for_failures(self) -> None:
        if not self.failures:
            return
        details = "\n".join(
            f"SMILES {rec.prompt_smiles}: {', '.join(rec.issues)}" for rec in self.failures
        )
        raise AssertionError(f"Constrained smoke validation failed for {len(self.failures)} items:\n{details}")


def load_ground_truth(dataset: SmokeDataset) -> Dict[str, dict]:
    """Load the GEOM pickle for the specified dataset into memory."""

    key = _DATASET_TO_KEY.get(dataset)
    if key is None:
        raise ValueError(f"Unknown dataset '{dataset}', expected one of {sorted(_DATASET_TO_KEY)}")
    path = get_data_path(key)
    with open(path, "rb") as handle:
        return cloudpickle.load(handle)


def sample_smiles(
    ground_truth: Dict[str, dict],
    sample_size: int,
    *,
    seed: int | None = None,
) -> list[str]:
    """Sample unique SMILES strings from the dataset."""

    smiles = list(ground_truth.keys())
    if sample_size > len(smiles):
        raise ValueError(f"Requested sample_size={sample_size} exceeds dataset of {len(smiles)}")
    rng = random.Random(seed)
    rng.shuffle(smiles)
    return smiles[:sample_size]


def build_prompts(smiles_list: Sequence[str]) -> list[str]:
    """Wrap raw SMILES strings in the conformer prompt tags."""

    return [f"[SMILES]{s}[/SMILES]" for s in smiles_list]


def _resolve_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device") and model.device is not None:
        return torch.device(model.device)
    try:
        return next(model.parameters()).device
    except StopIteration as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("Model has no parameters to infer device from") from exc


def _prepare_logits_processor(
    tokenizer: PreTrainedTokenizer,
    smiles_chunk: Sequence[str],
    attention_mask: torch.Tensor,
) -> LogitsProcessorList:
    templates = build_templates_for_batch(smiles_chunk, tokenizer)
    prompt_lengths = [int(mask.sum().item()) for mask in attention_mask]
    processor = ConformerConstraintLogitsProcessor(templates, prompt_lengths, tokenizer=tokenizer)
    return LogitsProcessorList([processor])


def generate_constrained_outputs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    smiles_list: Sequence[str],
    *,
    prompts: Sequence[str] | None = None,
    generation_config: GenerationConfig | None = None,
    batch_size: int = 64,
    max_new_tokens: int = 2000,
) -> list[str]:
    """Run constrained generation for the provided prompts and SMILES strings."""

    if prompts is None:
        prompts = build_prompts(smiles_list)
    if len(prompts) != len(smiles_list):
        raise ValueError("Number of prompts must match number of SMILES strings")

    eos_token = tokenizer.encode("[/CONFORMER]")[0]
    device = _resolve_device(model)
    gen_config = deepcopy(generation_config) if generation_config is not None else model.generation_config
    decoded: list[str] = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            end = start + batch_size
            prompt_chunk = prompts[start:end]
            smiles_chunk = smiles_list[start:end]
            tokenized = tokenizer(prompt_chunk, return_tensors="pt", padding=True)
            tokenized = {k: v.to(device, non_blocking=True) for k, v in tokenized.items()}
            logits_processor = _prepare_logits_processor(
                tokenizer, smiles_chunk, tokenized["attention_mask"]
            )
            outputs = model.generate(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token,
                generation_config=gen_config,
                logits_processor=logits_processor,
                use_cache=True,
                return_dict_in_generate=False,
            )
            # V11: Decode normally, then strip BOS/EOS tokens in post-processing
            batch_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            # Remove common special token pollution
            cleaned = []
            for text in batch_decoded:
                # Strip leading/trailing BOS/EOS tokens
                text = text.replace("<|begin_of_text|>", "")
                text = text.replace("<|end_of_text|>", "")
                cleaned.append(text)
            decoded.extend(cleaned)
    return decoded


def validate_smoke_outputs(
    smiles_list: Sequence[str],
    decoded_outputs: Sequence[str],
    *,
    require_conformer: bool = True,
) -> SmokeValidationResult:
    """Compare decoded outputs against prompts to ensure structural fidelity."""

    if len(smiles_list) != len(decoded_outputs):
        raise ValueError("Decoded outputs must match number of SMILES inputs")

    records: list[SmokeRecord] = []
    for prompt_smi, decoded in zip(smiles_list, decoded_outputs):
        record = SmokeRecord(prompt_smiles=prompt_smi, decoded_text=decoded)
        canonical_smiles = extract_between(decoded, "[SMILES]", "[/SMILES]")
        conformer_block = extract_between(decoded, "[CONFORMER]", "[/CONFORMER]")
        record.canonical_smiles = canonical_smiles
        record.conformer_block = conformer_block

        if not canonical_smiles:
            record.issues.append("missing [SMILES] block")
        elif canonical_smiles != prompt_smi:
            record.issues.append("SMILES block mismatch")

        if not conformer_block:
            if require_conformer:
                record.issues.append("missing [CONFORMER] block")
        else:
            stripped = strip_smiles(conformer_block)
            if stripped != prompt_smi:
                record.issues.append("conformer SMILES mismatch")
            elif not same_molecular_graph(prompt_smi, stripped):
                record.issues.append("different molecular graph")

            if ">" not in conformer_block:
                record.issues.append("coordinate block never closed")

        records.append(record)

    return SmokeValidationResult(records=records)


def run_smoke_check(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    smiles_list: Sequence[str],
    *,
    generation_config: GenerationConfig | None = None,
    batch_size: int = 64,
    max_new_tokens: int = 2000,
    require_conformer: bool = True,
) -> SmokeValidationResult:
    """Full pipeline: constrained generation followed by validation."""

    prompts = build_prompts(smiles_list)
    decoded = generate_constrained_outputs(
        model,
        tokenizer,
        smiles_list,
        prompts=prompts,
        generation_config=generation_config,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    return validate_smoke_outputs(smiles_list, decoded, require_conformer=require_conformer)


__all__ = [
    "SmokeRecord",
    "SmokeValidationResult",
    "build_prompts",
    "generate_constrained_outputs",
    "load_ground_truth",
    "run_smoke_check",
    "sample_smiles",
    "validate_smoke_outputs",
]
