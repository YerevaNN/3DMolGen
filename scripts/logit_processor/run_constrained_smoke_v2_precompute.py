#!/usr/bin/env python
"""
Run constrained conformer smoke test using v2 simple pre-computed mask.

This uses the simple position-based approach per the original spec.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import time
from loguru import logger
import cloudpickle
import random
from typing import Dict, Literal, Sequence
import torch
from transformers import LogitsProcessorList
from dataclasses import dataclass, field

from molgen3D.config.paths import get_ckpt, get_tokenizer_path
from molgen3D.config.sampling_config import sampling_configs
from molgen3D.evaluation.constraint_logit_processor import (
    ConformerConstraintLogitsProcessor,
    build_templates_for_batch,
)
from molgen3D.config.paths import get_data_path
from molgen3D.evaluation.inference import load_model_tokenizer
from molgen3D.evaluation.utils import extract_between, same_molecular_graph
from molgen3D.data_processing.smiles_encoder_decoder import strip_smiles


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
            # Use same_molecular_graph for comparison - handles implicit/explicit H differences
            # (e.g., "CCNC" vs "[NH]" notation)
            if not same_molecular_graph(prompt_smi, stripped):
                record.issues.append("generated conformer <> prompt SMILES mismatch")

            if ">" not in conformer_block:
                record.issues.append("coordinate block never closed")

        records.append(record)

    return SmokeValidationResult(records=records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v2 simple precompute mask smoke test")
    parser.add_argument("--dataset", choices=["clean", "distinct"], default="distinct")
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-alias", default="m380_conf_v2")
    parser.add_argument("--model-step", default="2e")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tokenizer-name", default="llama3_chem_v1")
    parser.add_argument("--sampling-config", default="greedy",
                        choices=list(sampling_configs.keys()))
    parser.add_argument("--max-new-tokens", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention", default="sdpa_paged")  # aim for sdpa_paged right now.
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-logit-processor", action="store_true",
                        help="Disable logit processor for timing comparison")
    return parser.parse_args()


def _load_model_and_tokenizer(args):
    model_path = args.model_path or get_ckpt(args.model_alias, args.model_step)
    tokenizer_path = get_tokenizer_path(args.tokenizer_name)
    return load_model_tokenizer(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        torch_dtype=args.torch_dtype,
        attention_imp=args.attention,
        device=args.device,
    )


def _extract_between(text: str, start_tag: str, end_tag: str) -> str | None:
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return None
    start_idx += len(start_tag)
    end_idx = text.find(end_tag, start_idx)
    if end_idx == -1:
        return None
    return text[start_idx:end_idx]


def _extract_generated_smiles(decoded_text: str) -> str | None:
    """Extract SMILES from conformer block in decoded text."""
    conformer = _extract_between(decoded_text, "[CONFORMER]", "[/CONFORMER]")
    if not conformer:
        return None
    try:
        return strip_smiles(conformer)
    except Exception:
        return None


def _has_real_coordinates(decoded_text: str) -> bool:
    """Check if conformer has real (non-placeholder) coordinates."""
    import re
    conformer = _extract_between(decoded_text, "[CONFORMER]", "[/CONFORMER]")
    if not conformer:
        return False
    # Find all coordinate blocks <x,y,z>
    coords = re.findall(r'<([^>]+)>', conformer)
    if not coords:
        return False
    # Check if any coordinate is NOT the placeholder
    placeholder = "0.0000,0.0000,0.0000"
    return any(c != placeholder for c in coords)


def _build_pass_record(rec) -> dict:
    """Build pass record with generated_smiles for proof of work."""
    gen_smiles = _extract_generated_smiles(rec.decoded_text)
    return {
        "prompt_smiles": rec.prompt_smiles,
        "generated_smiles": gen_smiles,
        "smiles_exact_match": gen_smiles == rec.prompt_smiles if gen_smiles else None,
        "has_real_coordinates": _has_real_coordinates(rec.decoded_text),
        "decoded_text": rec.decoded_text,
    }


def _build_failure_record(rec) -> dict:
    """Build failure record with generated_smiles_from_conformer field."""
    record = {
        "prompt_smiles": rec.prompt_smiles,
        "issues": rec.issues,
        "decoded_text": rec.decoded_text,
    }
    conformer = _extract_between(rec.decoded_text, "[CONFORMER]", "[/CONFORMER]")
    if not conformer:
        record["generated_smiles_from_conformer"] = "NO_CONFORMER_BLOCK"
        return record
    try:
        gen_smiles = strip_smiles(conformer)
        if same_molecular_graph(rec.prompt_smiles, gen_smiles):
            record["generated_smiles_from_conformer"] = True
        else:
            record["generated_smiles_from_conformer"] = gen_smiles
    except Exception as e:
        record["generated_smiles_from_conformer"] = f"PARSE_ERROR: {e}"
    return record


def _generate(model, tokenizer, smiles_list, gen_config, batch_size, max_new_tokens, use_logit_processor):
    prompts = [f"[SMILES]{s}[/SMILES]" for s in smiles_list]
    eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    device = next(model.parameters()).device
    decoded = []

    model.eval()
    start_time = time()

    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            end = start + batch_size
            prompt_chunk = prompts[start:end]
            smiles_chunk = smiles_list[start:end]

            tokenized = tokenizer(prompt_chunk, return_tensors="pt", padding=True)
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            logits_processor = None
            if use_logit_processor:
                templates = build_templates_for_batch(smiles_chunk, tokenizer)
                prompt_lengths = [int(m.sum().item()) for m in tokenized["attention_mask"]]
                processor = ConformerConstraintLogitsProcessor(
                    templates, prompt_lengths,
                    tokenizer=tokenizer,
                    eos_token_id=eos_token_id,
                )
                logits_processor = LogitsProcessorList([processor])

            outputs = model.generate(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                generation_config=gen_config,
                logits_processor=logits_processor,
                use_cache=True,
            )

            batch_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            for text in batch_decoded:
                text = text.replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "")
                decoded.append(text)

    elapsed = time() - start_time
    logger.info(f"Generation time: {elapsed:.2f}s")
    return decoded, elapsed


def main():
    args = _parse_args()
    logger.info(f"Loading dataset '{args.dataset}'")
    ground_truth = load_ground_truth(args.dataset)
    smiles_subset = sample_smiles(ground_truth, args.sample_size, seed=args.seed)
    logger.info(f"Selected {len(smiles_subset)} SMILES")

    logger.info("Loading model/tokenizer...")
    model, tokenizer = _load_model_and_tokenizer(args)
    gen_config = deepcopy(sampling_configs[args.sampling_config])

    use_lp = not args.no_logit_processor
    logger.info(f"Starting generation (logit_processor={use_lp})")

    decoded, time_taken = _generate(
        model, tokenizer, smiles_subset, gen_config,
        args.batch_size, args.max_new_tokens, use_lp
    )

    result = validate_smoke_outputs(smiles_subset, decoded, require_conformer=True)
    logger.info(f"Passed: {result.num_passed}/{result.total}")

    if result.failures:
        for rec in result.failures[:5]:
            logger.warning(f"FAIL {rec.prompt_smiles}: {rec.issues}")

    if args.json_report:
        pass_records = [_build_pass_record(r) for r in result.records if not r.issues]
        # Compute summary stats for verification
        exact_matches = sum(1 for r in pass_records if r.get("smiles_exact_match"))
        has_real_coords = sum(1 for r in pass_records if r.get("has_real_coordinates"))

        payload = {
            "metadata": {
                "version": ConformerConstraintLogitsProcessor.VERSION,
                "timestamp": datetime.now().isoformat(),
                "logit_processor_enabled": use_lp,
                "sampling": args.sampling_config,
                "batch_size": args.batch_size,
                "sample_size": len(smiles_subset),
            },
            "time_taken": time_taken,
            "num_passed": result.num_passed,
            "num_failed": len(result.failures),
            "summary": {
                "smiles_exact_matches": exact_matches,
                "has_real_coordinates": has_real_coords,
                "total_passes": len(pass_records),
            },
            "passes": pass_records,
            "failures": [_build_failure_record(r) for r in result.failures],
        }
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(payload, indent=2))
        logger.info(f"Wrote report to {args.json_report}")

    return 1 if args.fail_fast and result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
