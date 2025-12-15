#!/usr/bin/env python
"""
Run constrained conformer smoke test using v2 simple pre-computed mask.

This uses the simple position-based approach per the original spec.

Optimization flags:
    --kv-cache static    Enable static KV cache for ~1.8x speedup
    --device h100        Submit job to h100 via slurm (uses submitit)

Usage:
    # Local run with static KV cache:
    python scripts/logit_processor/run_logit_processor_smoke.py --kv-cache static

    # Submit to h100 with static cache:
    python scripts/logit_processor/run_logit_processor_smoke.py --kv-cache static --device h100
"""
from __future__ import annotations

import argparse
import json
import os
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

from concurrent.futures import ThreadPoolExecutor
from functools import partial


def _get_optimal_workers() -> int:
    """Get optimal number of workers for parallel template building.

    Priority:
    1. SLURM_CPUS_PER_TASK (if running under slurm)
    2. os.cpu_count() with cap at 8 (more workers not always better)

    Returns:
        Number of workers to use for parallel template building
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)
    # Default: use half of available CPUs, capped at 8
    cpu_count = os.cpu_count() or 4
    return min(cpu_count // 2 + 1, 8)

from molgen3D.config.paths import get_ckpt, get_tokenizer_path
from molgen3D.config.sampling_config import sampling_configs
from molgen3D.evaluation.constraint_logit_processor import (
    ConformerConstraintLogitsProcessor,
    build_templates_for_batch,
    build_precomputed_template,
)
from molgen3D.evaluation.qwen_constraint_logit_processor import (
    QwenConformerConstraintLogitsProcessor,
    build_templates_for_batch as qwen_build_templates_for_batch,
    build_precomputed_template as qwen_build_precomputed_template,
)
from molgen3D.evaluation.qwen_vectorized_constraint_lp import (
    QwenVectorizedConstraintLogitsProcessor,
    build_templates_for_batch as vectorized_build_templates_for_batch,
    build_precomputed_template as vectorized_build_precomputed_template,
)
from molgen3D.evaluation.qwen_simple_vectorized_lp import (
    QwenSimpleVectorizedLogitsProcessor,
    build_precomputed_template as simple_vectorized_build_precomputed_template,
)
from molgen3D.config.paths import get_data_path
from molgen3D.evaluation.inference import load_model_tokenizer
from molgen3D.evaluation.utils import extract_between, same_molecular_graph
from molgen3D.data_processing.smiles_encoder_decoder import strip_smiles, decode_cartesian_v2


SmokeDataset = Literal["clean", "distinct"]
_DATASET_TO_KEY = {
    "clean": "clean_smi",
    "distinct": "distinct_smi",
}


def build_templates_parallel(
    smiles_list: list[str],
    tokenizer,
    processor_type: str = "generic",
    num_workers: int = 4,
    use_tokenizer_cache: bool = False,
) -> list:
    """
    Build templates in parallel using ThreadPoolExecutor.

    This is a CPU-side optimization that parallelizes the template construction
    which involves tokenization and SMILES parsing.

    Args:
        smiles_list: List of SMILES strings
        tokenizer: HuggingFace tokenizer
        processor_type: "generic", "qwen", "vectorized-qwen", or "vectorized-qwen-v2"
        num_workers: Number of threads to use
        use_tokenizer_cache: Enable tokenizer caching for repeated skeletons

    Returns:
        List of PrecomputedTemplate objects
    """
    # Map processor type to template builder function
    # Note: vectorized processors use the same template format as qwen
    build_fn_map = {
        "generic": build_precomputed_template,
        "qwen": qwen_build_precomputed_template,
        "vectorized-qwen": vectorized_build_precomputed_template,
        "vectorized-qwen-v2": vectorized_build_precomputed_template,
        "simple-vectorized": simple_vectorized_build_precomputed_template,
    }
    build_fn = build_fn_map.get(processor_type, build_precomputed_template)

    # Use partial to bind tokenizer and cache flag to build function
    build_with_tokenizer = partial(
        build_fn, tokenizer=tokenizer, use_tokenizer_cache=use_tokenizer_cache
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        templates = list(executor.map(build_with_tokenizer, smiles_list))

    return templates


@dataclass
class SmokeRecord:
    """Holds the decoded text and validation issues for a single prompt."""

    prompt_smiles: str
    decoded_text: str
    canonical_smiles: str | None = None
    conformer_block: str | None = None
    issues: list[str] = field(default_factory=list)
    mol_obj: object = None  # RDKit Mol object if parsing succeeded
    parse_error: str | None = None  # Error message if decode_cartesian_v2 failed

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

            # Try to parse with decode_cartesian_v2 (mimics inference.py behavior)
            # This catches malformed coordinate blocks that pass SMILES matching
            if not record.issues:  # Only try parsing if no other issues
                try:
                    record.mol_obj = decode_cartesian_v2(conformer_block)
                except Exception as e:
                    record.parse_error = str(e)
                    record.issues.append(f"decode_cartesian_v2 failed: {e}")

        records.append(record)

    return SmokeValidationResult(records=records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v2 simple precompute mask smoke test")
    parser.add_argument("--dataset", choices=["clean", "distinct"], default="distinct")
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--num-generations", type=int, default=1,
                        help="Number of conformer generations per SMILES (default: 1). "
                             "Use >1 to simulate inference.py behavior and test tokenizer caching.")
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
    parser.add_argument("--attention", default="sdpa")  # sdpa is optimal on H100 with static cache?
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-logit-processor", action="store_true",
                        help="Disable logit processor for timing comparison")
    parser.add_argument("--processor-type",
                        choices=["generic", "qwen", "vectorized-qwen", "vectorized-qwen-v2", "simple-vectorized"],
                        default="generic",
                        help="Select logit processor type: 'generic' (blocklist), 'qwen' (blocklist v3.7), "
                             "'vectorized-qwen' (v4.x torch.compile compatible), "
                             "'simple-vectorized' (v5.0 simple in-place, recommended)")
    # Optimization flags
    parser.add_argument("--kv-cache", choices=["dynamic", "static"], default="dynamic",
                        help="KV cache mode: 'dynamic' (default) or 'static' (~1.8x speedup on H100)")
    parser.add_argument("--max-cache-len", type=int, default=None,
                        help="Max cache length for static KV cache. If None, auto-calculated as "
                             "padded_prompt_len + max_new_tokens. Only used when --kv-cache=static")
    parser.add_argument("--parallel-templates", action="store_true",
                        help="Build templates in parallel using ThreadPoolExecutor (CPU optimization)")
    parser.add_argument("--template-workers", type=int, default=None,
                        help="Number of workers for parallel template building. "
                             "Default: auto-detect from SLURM_CPUS_PER_TASK or os.cpu_count()")
    parser.add_argument("--cache-tokenizer", action="store_true",
                        help="Cache tokenizer encodings for repeated skeletons (CPU optimization)")
    parser.add_argument("--submit", choices=["local", "h100", "a100"], default="local",
                        help="Where to run: 'local' (default) or submit to 'h100'/'a100' via slurm")
    # Profiling flags
    parser.add_argument("--profile", action="store_true",
                        help="Enable torch.profiler and generate Chrome trace JSON")
    parser.add_argument("--profile-output", type=Path, default=None,
                        help="Output path for profile trace (default: outputs/profiles/smoke_profile_<timestamp>.json)")
    return parser.parse_args()


def _load_model_and_tokenizer(args) -> tuple:
    """Load model and tokenizer, returning (model, tokenizer, model_path, tokenizer_path)."""
    model_path = args.model_path or get_ckpt(args.model_alias, args.model_step)
    tokenizer_path = get_tokenizer_path(args.tokenizer_name)

    # For static cache, recommend single GPU to enable CUDA graphs
    device = args.device
    if args.kv_cache == "static" and device == "auto":
        logger.warning("Static KV cache works best with single GPU. Consider using --device cuda:0")

    # Check GPU capability - static cache benefits mainly H100/A100
    if args.kv_cache == "static" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        if cap[0] < 8:  # Pre-Ampere
            logger.warning(f"Static KV cache may not help on {gpu_name} (sm{cap[0]}{cap[1]})")
        elif "3070" in gpu_name or "3080" in gpu_name or "3090" in gpu_name:
            logger.warning(f"Static KV cache is optimized for datacenter GPUs (A100/H100). "
                          f"{gpu_name} may see slower performance.")

    model, tokenizer = load_model_tokenizer(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        torch_dtype=args.torch_dtype,
        attention_imp=args.attention,
        device=device,
    )

    # Configure static KV cache if requested
    if args.kv_cache == "static":
        logger.info("Enabling static KV cache (cache_implementation='static')")
        model.generation_config.cache_implementation = "static"

        # Print useful sizing info
        config = model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        head_dim = config.hidden_size // config.num_attention_heads
        bytes_per_elem = 2  # bfloat16

        # Cache size per token per batch item: 2 (K,V) * num_layers * num_kv_heads * head_dim * bytes
        cache_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem
        max_cache_len = args.max_cache_len or (64 + args.max_new_tokens)  # rough estimate

        logger.info(f"  Model: {num_layers} layers, {num_kv_heads} KV heads, head_dim={head_dim}")
        logger.info(f"  Cache per token per batch item: {cache_per_token / 1024:.1f} KB")
        logger.info(f"  Est. cache for batch={args.batch_size}, len={max_cache_len}: "
                    f"{(cache_per_token * max_cache_len * args.batch_size) / (1024**3):.2f} GB")

    return model, tokenizer, model_path, tokenizer_path


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
        "parse_success": rec.mol_obj is not None,  # decode_cartesian_v2 succeeded
        "decoded_text": rec.decoded_text,
    }


def _build_failure_record(rec) -> dict:
    """Build failure record with generation_has_same_molecular_graph field."""
    record = {
        "prompt_smiles": rec.prompt_smiles,
        "issues": rec.issues,
        "decoded_text": rec.decoded_text,
    }
    # Include parse_error if present (from decode_cartesian_v2 failure)
    if rec.parse_error:
        record["parse_error"] = rec.parse_error
    conformer = _extract_between(rec.decoded_text, "[CONFORMER]", "[/CONFORMER]")
    if not conformer:
        record["generation_has_same_molecular_graph"] = "NO_CONFORMER_BLOCK"
        return record
    try:
        gen_smiles = strip_smiles(conformer)
        if same_molecular_graph(rec.prompt_smiles, gen_smiles):
            record["generation_has_same_molecular_graph"] = True
        else:
            # Include the mismatched SMILES for debugging
            record["generation_has_same_molecular_graph"] = False
            record["extracted_smiles"] = gen_smiles
    except Exception as e:
        record["generation_has_same_molecular_graph"] = f"PARSE_ERROR: {e}"
    return record


def _get_processor_class_and_template_fn(processor_type: str):
    """Return (ProcessorClass, template_build_fn) for the given processor type."""
    if processor_type == "generic":
        return ConformerConstraintLogitsProcessor, build_precomputed_template
    elif processor_type == "qwen":
        return QwenConformerConstraintLogitsProcessor, qwen_build_precomputed_template
    elif processor_type in ("vectorized-qwen", "vectorized-qwen-v2"):
        # Both map to the same v4.x implementation (v2 is now an alias)
        return QwenVectorizedConstraintLogitsProcessor, vectorized_build_precomputed_template
    elif processor_type == "simple-vectorized":
        # v5.0: Simple in-place modification with pre-stacked templates
        return QwenSimpleVectorizedLogitsProcessor, simple_vectorized_build_precomputed_template
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")


def _generate(
    model, tokenizer, smiles_list, gen_config, batch_size, max_new_tokens,
    use_logit_processor, processor_type: str = "generic", static_cache: bool = False,
    parallel_templates: bool = False, template_workers: int = 4,
    use_tokenizer_cache: bool = False,
    enable_profiling: bool = False,
    profile_output_path: Path | None = None,
):
    prompts = [f"[SMILES]{s}[/SMILES]" for s in smiles_list]
    eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    device = next(model.parameters()).device
    decoded = []

    # Tokenizer kwargs - pad_to_multiple_of=64 helps with static cache efficiency
    tokenize_kwargs = {"return_tensors": "pt", "padding": True}
    if static_cache:
        tokenize_kwargs["pad_to_multiple_of"] = 64

    model.eval()

    # Get processor class and template function
    ProcessorClass, template_fn = _get_processor_class_and_template_fn(processor_type)

    # Setup profiler if enabled
    profiler = None
    if enable_profiling:
        from torch.profiler import profile as torch_profile, ProfilerActivity
        if profile_output_path is None:
            profile_output_path = Path(f"outputs/profiles/smoke_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        elif isinstance(profile_output_path, str):
            profile_output_path = Path(profile_output_path)
        profile_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Profiling a long generate() can explode memory; keep it lightweight
        profiler = torch_profile(
            activities=[ProfilerActivity.CUDA],  # CUDA only to reduce event volume
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        logger.info(f"Profiling enabled, will save to: {profile_output_path}")

    start_time = time()

    # Wrap generation loop with profiler context if enabled
    ctx = profiler if profiler else torch.inference_mode()

    with ctx:
        if profiler:
            # Need inference_mode inside profiler context
            with torch.inference_mode():
                decoded = _run_generation_loop(
                    model, tokenizer, prompts, smiles_list, gen_config,
                    batch_size, max_new_tokens, use_logit_processor,
                    ProcessorClass, template_fn, processor_type,
                    parallel_templates, template_workers, use_tokenizer_cache,
                    eos_token_id, device, tokenize_kwargs,
                )
        else:
            decoded = _run_generation_loop(
                model, tokenizer, prompts, smiles_list, gen_config,
                batch_size, max_new_tokens, use_logit_processor,
                ProcessorClass, template_fn, processor_type,
                parallel_templates, template_workers, use_tokenizer_cache,
                eos_token_id, device, tokenize_kwargs,
            )

    elapsed = time() - start_time

    # Export profiler results
    if profiler:
        logger.info("Exporting profiler trace...")
        profiler.export_chrome_trace(str(profile_output_path))
        logger.info(f"Profile saved to: {profile_output_path}")

        # Also print summary to console
        print("\n" + "=" * 80)
        print("PROFILER SUMMARY (Top 20 CUDA operations by total time)")
        print("=" * 80)
        print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        print("=" * 80 + "\n")

    logger.info(f"Generation time: {elapsed:.2f}s")
    return decoded, elapsed


def _run_generation_loop(
    model, tokenizer, prompts, smiles_list, gen_config,
    batch_size, max_new_tokens, use_logit_processor,
    ProcessorClass, template_fn, processor_type,
    parallel_templates, template_workers, use_tokenizer_cache,
    eos_token_id, device, tokenize_kwargs,
):
    """Inner loop for generation - separated for profiler context management."""
    decoded = []

    for start in range(0, len(prompts), batch_size):
        end = start + batch_size
        prompt_chunk = prompts[start:end]
        smiles_chunk = smiles_list[start:end]

        tokenized = tokenizer(prompt_chunk, **tokenize_kwargs)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        logits_processor = None
        if use_logit_processor:
            prompt_lengths = [int(m.sum().item()) for m in tokenized["attention_mask"]]

            # Build templates (optionally in parallel, optionally with tokenizer caching)
            if parallel_templates:
                templates = build_templates_parallel(
                    smiles_chunk, tokenizer,
                    processor_type=processor_type,
                    num_workers=template_workers,
                    use_tokenizer_cache=use_tokenizer_cache,
                )
            else:
                templates = [
                    template_fn(smi, tokenizer, use_tokenizer_cache=use_tokenizer_cache)
                    for smi in smiles_chunk
                ]

            # Create the processor
            processor = ProcessorClass(
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
            # Strip any special tokens the tokenizer leaves in decoded text so JSONs are readable
            text = text.replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "")
            text = text.replace("<|endoftext|>", "")  # Qwen-style EOS/pad token
            decoded.append(text)

    return decoded


def run_smoke_test(config: dict) -> int:
    """
    Run the smoke test. This function can be called directly or submitted via submitit.

    Args:
        config: Dictionary with all configuration parameters

    Returns:
        Exit code (0 = success, 1 = failures with fail_fast)
    """
    # Recreate args-like object from config
    class Args:
        pass
    args = Args()
    for k, v in config.items():
        setattr(args, k, v)

    # Convert json_report back to Path if it was serialized as string
    if args.json_report and isinstance(args.json_report, str):
        args.json_report = Path(args.json_report)

    logger.info(f"Loading dataset '{args.dataset}'")
    ground_truth = load_ground_truth(args.dataset)
    smiles_subset = sample_smiles(ground_truth, args.sample_size, seed=args.seed)
    num_generations = getattr(args, 'num_generations', 1)
    logger.info(f"Selected {len(smiles_subset)} SMILES, {num_generations} generation(s) each")

    logger.info("Loading model/tokenizer...")
    model, tokenizer, model_path, tokenizer_path = _load_model_and_tokenizer(args)
    gen_config = deepcopy(sampling_configs[args.sampling_config])

    use_lp = not args.no_logit_processor
    static_cache = args.kv_cache == "static"
    parallel_templates = getattr(args, 'parallel_templates', False)
    cache_tokenizer = getattr(args, 'cache_tokenizer', False)

    # Auto-detect workers if not specified
    template_workers = getattr(args, 'template_workers', None)
    if template_workers is None:
        template_workers = _get_optimal_workers()
    if parallel_templates:
        logger.info(f"Using {template_workers} workers for parallel template building "
                    f"(SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK', 'N/A')})")

    logger.info(f"Starting generation (logit_processor={use_lp}, processor_type={args.processor_type}, "
                f"kv_cache={args.kv_cache}, parallel_templates={parallel_templates}, "
                f"cache_tokenizer={cache_tokenizer})")

    # Expand SMILES list for multiple generations per molecule
    # This simulates inference.py behavior where same SMILES is generated multiple times
    expanded_smiles = smiles_subset * num_generations  # [s1,s2,s3] * 2 = [s1,s2,s3,s1,s2,s3]

    # Get profiling settings
    enable_profiling = getattr(args, 'profile', False)
    profile_output = getattr(args, 'profile_output', None)

    decoded, time_taken = _generate(
        model, tokenizer, expanded_smiles, gen_config,
        args.batch_size, args.max_new_tokens, use_lp,
        processor_type=args.processor_type,
        static_cache=static_cache,
        parallel_templates=parallel_templates,
        template_workers=template_workers,
        use_tokenizer_cache=cache_tokenizer,
        enable_profiling=enable_profiling,
        profile_output_path=profile_output,
    )

    # Group decoded outputs back by original SMILES
    # decoded_grouped[i] = list of num_generations outputs for smiles_subset[i]
    n_smiles = len(smiles_subset)
    decoded_grouped = [[] for _ in range(n_smiles)]
    for gen_idx in range(num_generations):
        for smi_idx in range(n_smiles):
            decoded_grouped[smi_idx].append(decoded[gen_idx * n_smiles + smi_idx])

    # For validation, use first generation of each SMILES (matches original behavior)
    decoded_first = [group[0] for group in decoded_grouped]
    result = validate_smoke_outputs(smiles_subset, decoded_first, require_conformer=True)
    logger.info(f"Passed: {result.num_passed}/{result.total}")

    if result.failures:
        for rec in result.failures[:5]:
            logger.warning(f"FAIL {rec.prompt_smiles}: {rec.issues}")

    if args.json_report:
        pass_records = [_build_pass_record(r) for r in result.records if not r.issues]
        # Compute summary stats for verification
        exact_matches = sum(1 for r in pass_records if r.get("smiles_exact_match"))
        has_real_coords = sum(1 for r in pass_records if r.get("has_real_coordinates"))
        # Count parse failures (decode_cartesian_v2 errors) - mimics inference.py's "mol_parse_fail" stat
        parse_failures = sum(1 for r in result.failures if r.parse_error is not None)

        # Get the version from the processor actually used
        processor_version_map = {
            "generic": ConformerConstraintLogitsProcessor.VERSION,
            "qwen": QwenConformerConstraintLogitsProcessor.VERSION,
            "vectorized-qwen": QwenVectorizedConstraintLogitsProcessor.VERSION,
            "vectorized-qwen-v2": QwenVectorizedConstraintLogitsProcessor.VERSION,
            "simple-vectorized": QwenSimpleVectorizedLogitsProcessor.VERSION,
        }
        processor_version = processor_version_map.get(args.processor_type, "unknown")

        # Calculate throughput for comparison (strip special tokens so Qwen padding doesn't inflate)
        def _strip_special(text: str) -> str:
            return (
                text.replace("<|begin_of_text|>", "")
                    .replace("<|end_of_text|>", "")
                    .replace("<|endoftext|>", "")
            )

        total_chars_raw = sum(len(d) for d in decoded)
        decoded_clean = [_strip_special(d) for d in decoded]
        total_chars_clean = sum(len(d) for d in decoded_clean)
        throughput = total_chars_clean / time_taken if time_taken > 0 else 0

        # Build pass records with all generations if num_generations > 1
        if num_generations > 1:
            # Include all generations as array per SMILES
            pass_records_with_all_gens = []
            for smi_idx, rec in enumerate(result.records):
                if not rec.issues:
                    base_record = _build_pass_record(rec)
                    # Add all generations for this SMILES
                    base_record["all_generations"] = decoded_grouped[smi_idx]
                    pass_records_with_all_gens.append(base_record)
            pass_records = pass_records_with_all_gens

        payload = {
            "metadata": {
                "version": processor_version,
                "processor_type": args.processor_type,
                "timestamp": datetime.now().isoformat(),
                # Model info
                "model_alias": args.model_alias,
                "model_step": args.model_step,
                "model_path": str(model_path),
                # Tokenizer info
                "tokenizer_name": args.tokenizer_name,
                "tokenizer_path": str(tokenizer_path),
                # Generation config
                "logit_processor_enabled": use_lp,
                "sampling": args.sampling_config,
                "batch_size": args.batch_size,
                "sample_size": len(smiles_subset),
                "num_generations": num_generations,
                "total_generations": len(smiles_subset) * num_generations,
                "kv_cache": args.kv_cache,
                "attention": args.attention,
                "parallel_templates": parallel_templates,
                "template_workers": template_workers if parallel_templates else None,
                "cache_tokenizer": cache_tokenizer,
            },
            "time_taken": time_taken,
            "throughput_chars_per_sec": throughput,
            "total_chars_raw": total_chars_raw,
            "total_chars_clean": total_chars_clean,
            "num_passed": result.num_passed,
            "num_failed": len(result.failures),
            "summary": {
                "smiles_exact_matches": exact_matches,
                "has_real_coordinates": has_real_coords,
                "total_passes": len(pass_records),
                "parse_failures": parse_failures,  # decode_cartesian_v2 failures (like inference.py's mol_parse_fail)
            },
            "passes": pass_records,
            "failures": [_build_failure_record(r) for r in result.failures],
        }
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(payload, indent=2))
        logger.info(f"Wrote report to {args.json_report}")

    return 1 if args.fail_fast and result.failures else 0


def main():
    args = _parse_args()

    # Build config dict from args (for submitit serialization)
    config = {
        "dataset": args.dataset,
        "sample_size": args.sample_size,
        "num_generations": args.num_generations,
        "batch_size": args.batch_size,
        "model_alias": args.model_alias,
        "model_step": args.model_step,
        "model_path": str(args.model_path) if args.model_path else None,
        "tokenizer_name": args.tokenizer_name,
        "sampling_config": args.sampling_config,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "attention": args.attention,
        "json_report": str(args.json_report) if args.json_report else None,
        "fail_fast": args.fail_fast,
        "no_logit_processor": args.no_logit_processor,
        "processor_type": args.processor_type,
        "kv_cache": args.kv_cache,
        "max_cache_len": args.max_cache_len,
        "parallel_templates": args.parallel_templates,
        "template_workers": args.template_workers or _get_optimal_workers(),  # Resolve None to actual value
        "cache_tokenizer": args.cache_tokenizer,
        # Profiling
        "profile": args.profile,
        "profile_output": str(args.profile_output) if args.profile_output else None,
    }

    if args.submit == "local":
        # Run locally
        return run_smoke_test(config)
    else:
        # Submit to slurm via submitit
        import submitit

        partition = args.submit  # h100 or a100

        executor = submitit.AutoExecutor(
            folder="~/slurm_jobs/smoke/job_%j"
        )
        executor.update_parameters(
            name=f"smoke_{args.kv_cache}_{args.sample_size}",
            timeout_min=60,
            gpus_per_node=1,
            nodes=1,
            mem_gb=40,
            cpus_per_task=4,
            slurm_additional_parameters={"partition": partition},
        )

        job = executor.submit(run_smoke_test, config)
        print(f"Submitted job to {partition} partition")
        print(f"Job ID: {job.job_id}")
        print(f"Output will be in: ~/slurm_jobs/smoke/job_{job.job_id}/")
        print(f"\nTo check status: squeue -j {job.job_id}")
        print(f"To view output: cat ~/slurm_jobs/smoke/job_{job.job_id}/{job.job_id}_0_log.out")

        if config["json_report"]:
            print(f"\nJSON report will be saved to: {config['json_report']}")

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
