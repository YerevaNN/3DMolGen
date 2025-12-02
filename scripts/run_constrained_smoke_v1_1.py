#!/usr/bin/env python
"""
Run constrained conformer smoke test using v1.1 baseline processor.

v1.1: Structural constraints + minimal coord protection:
- Blocks special tags ([CONFORMER], [/CONFORMER], etc.) in coord blocks
- Max coord tokens safety limit (forces > after N tokens)
- Simple > detection (no smart digit-based detection)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from loguru import logger

from molgen3D.config.paths import get_ckpt, get_tokenizer_path
from molgen3D.config.sampling_config import sampling_configs
from molgen3D.evaluation.constrained_logits_v1_1 import (
    ConformerConstraintLogitsProcessorV1_1,
    build_templates_for_batch,
)
from molgen3D.evaluation.constrained_smoke import (
    load_ground_truth,
    sample_smiles,
    validate_smoke_outputs,
)
from molgen3D.evaluation.inference import load_model_tokenizer
import torch
from transformers import LogitsProcessorList


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v1.1 constrained conformer smoke test")
    parser.add_argument("--dataset", choices=["clean", "distinct"], default="distinct")
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-alias", default="m380_conf_v2")
    parser.add_argument("--model-step", default="2e")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tokenizer-name", default="llama3_chem_v1")
    parser.add_argument("--sampling-config", default="top_p_sampling1",
                        choices=list(sampling_configs.keys()))
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--max-coord-tokens", type=int, default=100,
                        help="Max tokens per coordinate block before forcing >")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--fail-fast", action="store_true",
                        help="Exit with status 1 if any mismatches occur")
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

    config = ConformerConstraintLogitsProcessorV1_1.CONFIG.copy()
    config["max_coord_tokens"] = args.max_coord_tokens

    return {
        "version": ConformerConstraintLogitsProcessorV1_1.VERSION,
        "timestamp": datetime.now().isoformat(),
        "model_checkpoint": str(model_path),
        "tokenizer": args.tokenizer_name,
        "logit_processor_config": config,
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


def _generate_constrained(
    model,
    tokenizer,
    smiles_list: list[str],
    generation_config,
    batch_size: int,
    max_new_tokens: int,
    max_coord_tokens: int,
) -> list[str]:
    """Generate conformers with v1.1 structural + protection constraints."""
    prompts = [f"[SMILES]{s}[/SMILES]" for s in smiles_list]

    # Use tokenizer's actual EOS token
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.pad_token_id

    device = next(model.parameters()).device
    decoded: list[str] = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            end = start + batch_size
            prompt_chunk = prompts[start:end]
            smiles_chunk = smiles_list[start:end]

            tokenized = tokenizer(prompt_chunk, return_tensors="pt", padding=True)
            tokenized = {k: v.to(device, non_blocking=True) for k, v in tokenized.items()}

            # Build v1.1 processor with EOS token and max_coord_tokens
            templates = build_templates_for_batch(smiles_chunk, tokenizer)
            prompt_lengths = [int(mask.sum().item()) for mask in tokenized["attention_mask"]]
            processor = ConformerConstraintLogitsProcessorV1_1(
                templates,
                prompt_lengths,
                tokenizer=tokenizer,
                eos_token_id=eos_token_id,
                max_coord_tokens=max_coord_tokens,
            )

            outputs = model.generate(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                generation_config=generation_config,
                logits_processor=LogitsProcessorList([processor]),
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

    return decoded


def main() -> int:
    args = _parse_args()
    logger.info("Loading ground-truth dataset '{}'.", args.dataset)
    ground_truth = load_ground_truth(args.dataset)
    smiles_subset = sample_smiles(ground_truth, args.sample_size, seed=args.seed)
    logger.info("Selected %d SMILES for smoke test.", len(smiles_subset))

    logger.info("Loading model/tokenizer...")
    model, tokenizer = _load_model_and_tokenizer(args)

    gen_config = deepcopy(sampling_configs[args.sampling_config])

    logger.info("Starting v1.1 constrained generation (structural + protection).")
    logger.info("max_coord_tokens=%d", args.max_coord_tokens)
    decoded = _generate_constrained(
        model,
        tokenizer,
        smiles_subset,
        gen_config,
        args.batch_size,
        args.max_new_tokens,
        args.max_coord_tokens,
    )

    result = validate_smoke_outputs(smiles_subset, decoded, require_conformer=True)
    failure_count = len(result.failures)
    logger.info(
        "Smoke test finished: %d/%d passed.", result.num_passed, result.total
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
                {
                    "prompt_smiles": rec.prompt_smiles,
                    "issues": rec.issues,
                    "decoded_text": rec.decoded_text,
                }
                for rec in result.failures
            ],
        }
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(payload, indent=2))
        logger.info("Wrote JSON report to %s", args.json_report)

    if args.fail_fast and failure_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
