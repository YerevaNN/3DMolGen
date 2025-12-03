#!/usr/bin/env python
"""
Run constrained conformer smoke test using v2 simple pre-computed mask.

This uses the simple position-based approach per the original spec.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import time
from loguru import logger

from molgen3D.config.paths import get_ckpt, get_tokenizer_path
from molgen3D.config.sampling_config import sampling_configs
from molgen3D.evaluation.constrained_logits_v2_precompute_mask import (
    ConformerConstraintLogitsProcessorV2PrecomputeMask,
    build_templates_for_batch,
)
from molgen3D.evaluation.constrained_smoke import (
    load_ground_truth,
    sample_smiles,
    validate_smoke_outputs,
)
from molgen3D.evaluation.inference import load_model_tokenizer
from molgen3D.evaluation.utils import same_molecular_graph
from molgen3D.data_processing.smiles_encoder_decoder import strip_smiles
import torch
from transformers import LogitsProcessorList


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
    parser.add_argument("--attention", default="sdpa_paged") # aim for sdpa_paged right now.
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
                processor = ConformerConstraintLogitsProcessorV2PrecomputeMask(
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
        payload = {
            "metadata": {
                "version": ConformerConstraintLogitsProcessorV2PrecomputeMask.VERSION,
                "timestamp": datetime.now().isoformat(),
                "logit_processor_enabled": use_lp,
                "sampling": args.sampling_config,
            },
            "time_taken": time_taken,
            "num_passed": result.num_passed,
            "num_failed": len(result.failures),
            "passes": [{"prompt_smiles": r.prompt_smiles, "decoded_text": r.decoded_text}
                       for r in result.records if not r.issues],
            "failures": [_build_failure_record(r) for r in result.failures],
        }
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(payload, indent=2))
        logger.info(f"Wrote report to {args.json_report}")

    return 1 if args.fail_fast and result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
