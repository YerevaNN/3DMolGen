#!/usr/bin/env python
"""
Run constrained conformer smoke test using v2 pre-computed mask processor.

v2: Pre-computed position-based mask:
- Pre-compute reference skeleton with placeholder coordinates
- Derive COPY/FREE mask from token positions (outside/inside <...>)
- Pure position-based lookup: O(1) per token per sequence
- No state machine, no segment tracking

Includes validation using strip_smiles and same_molecular_graph.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import time
from loguru import logger

from molgen3D.config.paths import get_ckpt, get_tokenizer_path
from molgen3D.config.sampling_config import sampling_configs
from molgen3D.evaluation.constrained_logits_v2 import (
    ConformerConstraintLogitsProcessorV2,
    build_templates_for_batch,
)
from molgen3D.evaluation.constrained_smoke import (
    load_ground_truth,
    sample_smiles,
    validate_smoke_outputs,
)
from molgen3D.evaluation.inference import load_model_tokenizer
from molgen3D.evaluation.utils import same_molecular_graph
from molgen3D.data_processing.smiles_encoder_decoder import strip_smiles, decode_cartesian_v2
import torch
from transformers import LogitsProcessorList


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v2 pre-computed mask constrained conformer smoke test")
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
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--fail-fast", action="store_true",
                        help="Exit with status 1 if any mismatches occur")
    parser.add_argument("--no-logit-processor", action="store_true",
                        help="Disable logit processor for timing comparison")
    return parser.parse_args()


def _load_model_and_tokenizer(args: argparse.Namespace):
    model_path = (
        args.model_path
        if args.model_path is not None
        else get_ckpt(args.model_alias, args.model_step)
    )
    tokenizer_path = get_tokenizer_path(args.tokenizer_name)
    return load_model_tokenizer(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        torch_dtype=args.torch_dtype,
        attention_imp=args.attention,
        device=args.device,
    )


def _build_metadata(args: argparse.Namespace) -> dict:
    """Capture experiment configuration for reproducibility."""
    model_path = (
        args.model_path
        if args.model_path is not None
        else get_ckpt(args.model_alias, args.model_step)
    )

    logit_processor_enabled = not args.no_logit_processor

    return {
        "version": ConformerConstraintLogitsProcessorV2.VERSION,
        "timestamp": datetime.now().isoformat(),
        "model_checkpoint": str(model_path),
        "tokenizer": args.tokenizer_name,
        "logit_processor_enabled": logit_processor_enabled,
        "logit_processor_config": ConformerConstraintLogitsProcessorV2.CONFIG if logit_processor_enabled else None,
        "generation_config": {
            "sampling": args.sampling_config,
            "max_new_tokens": args.max_new_tokens,
        },
    }


def _compute_summary(result) -> dict:
    """Compute detailed summary statistics."""
    issue_counts = Counter()
    for fail in result.failures:
        for issue in fail.issues:
            issue_counts[issue] += 1

    return {
        "total": result.total,
        "passed": result.num_passed,
        "failed": len(result.failures),
        "failure_breakdown": dict(issue_counts),
    }


def _extract_between(text: str, start_tag: str, end_tag: str) -> str | None:
    """Extract content between start_tag and end_tag."""
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return None
    start_idx += len(start_tag)
    end_idx = text.find(end_tag, start_idx)
    if end_idx == -1:
        return None
    return text[start_idx:end_idx]


def _build_failure_record(rec) -> dict:
    """
    Build failure record with generated_smiles_from_conformer field.

    If the stripped SMILES from conformer matches prompt_smiles, set to True.
    If not, include the stripped SMILES string for comparison.
    If extraction/parsing fails, include error info.
    """
    record = {
        "prompt_smiles": rec.prompt_smiles,
        "issues": rec.issues,
        "decoded_text": rec.decoded_text,
    }

    # Try to extract and validate conformer -> SMILES mapping
    generated_conformer = _extract_between(rec.decoded_text, "[CONFORMER]", "[/CONFORMER]")

    if not generated_conformer:
        record["generated_smiles_from_conformer"] = "NO_CONFORMER_BLOCK"
        return record

    try:
        generated_smiles = strip_smiles(generated_conformer)

        # Check if it matches prompt_smiles
        if same_molecular_graph(rec.prompt_smiles, generated_smiles):
            record["generated_smiles_from_conformer"] = True
        else:
            # Include the mismatched SMILES for easy comparison
            record["generated_smiles_from_conformer"] = generated_smiles
    except Exception as e:
        record["generated_smiles_from_conformer"] = f"PARSE_ERROR: {str(e)}"

    return record


def _validate_conformer_smiles_mapping(
    smiles_list: list[str],
    decoded_list: list[str],
) -> dict:
    """
    Validate that generated conformers map back to correct SMILES.

    Uses strip_smiles and same_molecular_graph for validation.
    """
    stats = {
        "total": len(smiles_list),
        "valid_mapping": 0,
        "invalid_mapping": 0,
        "no_conformer": 0,
        "parse_error": 0,
        "details": [],
    }

    for i, (input_smiles, decoded_text) in enumerate(zip(smiles_list, decoded_list)):
        # Extract SMILES and CONFORMER blocks
        canonical_smiles = _extract_between(decoded_text, "[SMILES]", "[/SMILES]")
        generated_conformer = _extract_between(decoded_text, "[CONFORMER]", "[/CONFORMER]")

        detail = {
            "index": i,
            "input_smiles": input_smiles,
            "canonical_smiles": canonical_smiles,
            "status": None,
        }

        if not generated_conformer:
            stats["no_conformer"] += 1
            detail["status"] = "no_conformer"
            stats["details"].append(detail)
            continue

        try:
            # Strip coordinates to get SMILES structure
            generated_smiles = strip_smiles(generated_conformer)
            detail["generated_smiles"] = generated_smiles

            # Check if molecular graphs match
            if same_molecular_graph(canonical_smiles or input_smiles, generated_smiles):
                stats["valid_mapping"] += 1
                detail["status"] = "valid"
            else:
                stats["invalid_mapping"] += 1
                detail["status"] = "mismatch"
                detail["conformer_preview"] = generated_conformer[:200]
        except Exception as e:
            stats["parse_error"] += 1
            detail["status"] = "parse_error"
            detail["error"] = str(e)

        stats["details"].append(detail)

    return stats


def _generate_constrained(
    model,
    tokenizer,
    smiles_list: list[str],
    generation_config,
    batch_size: int,
    max_new_tokens: int,
    use_logit_processor: bool = True,
) -> tuple[list[str], float]:
    """Generate conformers with optional v2 pre-computed mask constraints."""
    prompts = [f"[SMILES]{s}[/SMILES]" for s in smiles_list]

    # Use tokenizer's actual EOS token
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.pad_token_id

    device = next(model.parameters()).device
    decoded: list[str] = []

    model.eval()
    start_time = time()

    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            end = start + batch_size
            prompt_chunk = prompts[start:end]
            smiles_chunk = smiles_list[start:end]

            tokenized = tokenizer(prompt_chunk, return_tensors="pt", padding=True)
            tokenized = {k: v.to(device, non_blocking=True) for k, v in tokenized.items()}

            # Build logits processor if enabled
            logits_processor = None
            if use_logit_processor:
                templates = build_templates_for_batch(smiles_chunk, tokenizer)
                prompt_lengths = [int(mask.sum().item()) for mask in tokenized["attention_mask"]]
                processor = ConformerConstraintLogitsProcessorV2(
                    templates,
                    prompt_lengths,
                    tokenizer=tokenizer,
                    eos_token_id=eos_token_id,
                )
                logits_processor = LogitsProcessorList([processor])

            outputs = model.generate(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                generation_config=generation_config,
                logits_processor=logits_processor,
                use_cache=True,
                return_dict_in_generate=False,
            )

            # Decode and clean special tokens
            batch_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            cleaned = []
            for text in batch_decoded:
                text = text.replace("<|begin_of_text|>", "")
                text = text.replace("<|end_of_text|>", "")
                cleaned.append(text)
            decoded.extend(cleaned)

    end_time = time()
    elapsed = end_time - start_time
    logger.info(f"Generation time: {elapsed:.2f} seconds")
    return decoded, elapsed


def main() -> int:
    args = _parse_args()
    logger.info("Loading ground-truth dataset '{}'.", args.dataset)
    ground_truth = load_ground_truth(args.dataset)
    smiles_subset = sample_smiles(ground_truth, args.sample_size, seed=args.seed)
    logger.info("Selected %d SMILES for smoke test.", len(smiles_subset))

    logger.info("Loading model/tokenizer...")
    model, tokenizer = _load_model_and_tokenizer(args)

    gen_config = deepcopy(sampling_configs[args.sampling_config])

    use_logit_processor = not args.no_logit_processor
    if use_logit_processor:
        logger.info("Starting v2 pre-computed mask generation.")
    else:
        logger.info("Starting UNCONSTRAINED generation (logit processor disabled).")

    decoded, time_taken = _generate_constrained(
        model,
        tokenizer,
        smiles_subset,
        gen_config,
        args.batch_size,
        args.max_new_tokens,
        use_logit_processor=use_logit_processor,
    )

    # Standard smoke test validation
    result = validate_smoke_outputs(smiles_subset, decoded, require_conformer=True)
    failure_count = len(result.failures)
    logger.info(
        "Smoke test finished: %d/%d passed.", result.num_passed, result.total
    )

    # Additional validation: conformer -> SMILES mapping
    mapping_stats = _validate_conformer_smiles_mapping(smiles_subset, decoded)
    logger.info(
        "Conformer->SMILES mapping: %d/%d valid, %d invalid, %d no_conformer, %d parse_error",
        mapping_stats["valid_mapping"],
        mapping_stats["total"],
        mapping_stats["invalid_mapping"],
        mapping_stats["no_conformer"],
        mapping_stats["parse_error"],
    )

    if failure_count:
        for rec in result.failures[:5]:
            logger.warning(
                "Mismatch for %s: %s", rec.prompt_smiles, "; ".join(rec.issues)
            )

    if args.json_report:
        payload = {
            "metadata": _build_metadata(args),
            "summary": _compute_summary(result),
            "time_taken": time_taken,
            "logit_processor_enabled": use_logit_processor,
            "conformer_mapping": {
                "valid": mapping_stats["valid_mapping"],
                "invalid": mapping_stats["invalid_mapping"],
                "no_conformer": mapping_stats["no_conformer"],
                "parse_error": mapping_stats["parse_error"],
            },
            "dataset": args.dataset,
            "sample_size": args.sample_size,
            "num_passed": result.num_passed,
            "num_failed": failure_count,
            "passes": [
                {
                    "prompt_smiles": rec.prompt_smiles,
                    "decoded_text": rec.decoded_text,
                }
                for rec in result.records
                if not rec.issues
            ],
            "failures": [
                _build_failure_record(rec)
                for rec in result.failures
            ],
            "mapping_details": mapping_stats["details"][:10],  # First 10 for debugging
        }
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(payload, indent=2))
        logger.info("Wrote JSON report to %s", args.json_report)

    if args.fail_fast and failure_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
