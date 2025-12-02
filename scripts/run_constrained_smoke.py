#!/usr/bin/env python
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
from molgen3D.evaluation.constrained_logits import ConformerConstraintLogitsProcessor
from molgen3D.evaluation.constrained_smoke import (
    load_ground_truth,
    run_smoke_check,
    sample_smiles,
)
from molgen3D.evaluation.inference import load_model_tokenizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run constrained conformer smoke test")
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

    return {
        "version": ConformerConstraintLogitsProcessor.VERSION,
        "timestamp": datetime.now().isoformat(),
        "model_checkpoint": str(model_path),
        "tokenizer": args.tokenizer_name,
        "logit_processor_config": ConformerConstraintLogitsProcessor.CONFIG.copy(),
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


def main() -> int:
    args = _parse_args()
    logger.info("Loading ground-truth dataset '{}'.", args.dataset)
    ground_truth = load_ground_truth(args.dataset)
    smiles_subset = sample_smiles(ground_truth, args.sample_size, seed=args.seed)
    logger.info("Selected %d SMILES for smoke test.", len(smiles_subset))

    logger.info("Loading model/tokenizer...")
    model, tokenizer = _load_model_and_tokenizer(args)

    gen_config = deepcopy(sampling_configs[args.sampling_config])

    logger.info("Starting constrained generation.")
    result = run_smoke_check(
        model,
        tokenizer,
        smiles_subset,
        generation_config=gen_config,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

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
