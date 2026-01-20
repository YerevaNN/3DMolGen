#!/usr/bin/env python3
"""
Quick A/B benchmark for inference optimizations.

This script tests different configurations and measures throughput/memory
to help identify optimal settings for inference. Supports both single-GPU
and multi-GPU benchmarking.

Usage:
    # Basic benchmark with expandable_segments A/B comparison
    python scripts/benchmark_inference.py --limit 100 --test_expandable_segments

    # Test different batch sizes
    python scripts/benchmark_inference.py --limit 50 --batch_sizes 128,256,384,512

    # Test compile modes
    python scripts/benchmark_inference.py --limit 50 --compile_modes none,reduce-overhead,max-autotune

    # Full benchmark suite (all combinations)
    python scripts/benchmark_inference.py --limit 100 --test_expandable_segments --batch_sizes 128,256,512

    # Multi-GPU benchmark via Slurm (submits jobs, waits for results)
    python scripts/benchmark_inference.py --limit 200 --num_gpus 4 --device h100

    # Stack optimizations: test all combinations of expandable_segments x batch_sizes x compile_modes
    python scripts/benchmark_inference.py --limit 50 --test_expandable_segments \\
        --batch_sizes 128,256 --compile_modes reduce-overhead,max-autotune

    # Profile prefill vs decode phases (to evaluate "prefill once" optimization potential)
    python scripts/benchmark_inference.py --limit 50 --profile_phases --batch_sizes 256

    # Slurm job with phase profiling
    python scripts/benchmark_inference.py --limit 100 --profile_phases --device a100

    # A/B test: current approach vs "prefill once, decode many" caching (local)
    python scripts/benchmark_inference.py --test_prefill_caching --limit 5 --gens_per_mol 50

    # A/B test submitted to A100 via Slurm (returns immediately)
    python scripts/benchmark_inference.py --test_prefill_caching --limit 5 --gens_per_mol 100 --device a100

    # Test with more molecules and higher generation count
    python scripts/benchmark_inference.py --test_prefill_caching --limit 10 --gens_per_mol 200 --device a100
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

import torch
import cloudpickle
import submitit
from loguru import logger
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from molgen3D.config.paths import get_ckpt, get_tokenizer_path, get_data_path, get_base_path
from molgen3D.config.sampling_config import sampling_configs


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    config_name: str
    batch_size: int
    num_molecules: int
    total_time_sec: float
    tokens_per_sec: float
    molecules_per_sec: float
    peak_memory_gb: float
    oom_occurred: bool
    expandable_segments: bool
    compile_mode: str | None
    attn_implementation: str = "flash_attention_2"
    warmup_time_sec: float = 0.0
    # Phase timing breakdown (when --profile_phases is enabled)
    prefill_time_sec: float = 0.0
    decode_time_sec: float = 0.0
    prefill_pct: float = 0.0  # Percentage of time spent in prefill
    avg_prompt_tokens: float = 0.0
    avg_generated_tokens: float = 0.0


@dataclass
class BenchmarkSummary:
    """Full benchmark session summary."""
    timestamp: str
    gpu_name: str
    model_alias: str
    model_step: str
    # Run configuration for reproducibility
    test_set: str = ""
    num_molecules: int = 0
    device: str = ""
    batch_sizes_tested: list[int] = field(default_factory=list)
    compile_modes_tested: list[str] = field(default_factory=list)
    expandable_segments_tested: list[bool] = field(default_factory=list)
    # Results
    results: list[dict] = field(default_factory=list)
    best_throughput_config: str = ""
    max_stable_batch_size: int = 0


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_model_tokenizer(
    model_path,
    tokenizer_path,
    torch_dtype="bfloat16",
    compile_mode="reduce-overhead",
    attn_implementation="flash_attention_2",
):
    """Load model and tokenizer with specified settings.

    Args:
        model_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer
        torch_dtype: Data type for model weights
        compile_mode: torch.compile mode (none, default, reduce-overhead, max-autotune)
        attn_implementation: Attention implementation (flash_attention_2, sdpa, eager)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), padding_side="left", local_files_only=True
    )
    dtype_obj = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype_obj,
        attn_implementation=attn_implementation,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    ).eval()

    # Apply torch.compile if requested
    if compile_mode and compile_mode != "none":
        try:
            model = torch.compile(model, mode=compile_mode)
            logger.info(f"torch.compile succeeded with mode={compile_mode}")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def run_warmup(model, tokenizer, gen_config, eos_token_id, num_warmup: int = 3) -> float:
    """Run warmup iterations to compile CUDA kernels."""
    warmup_prompt = "[SMILES]CC[/SMILES]"
    warmup_start = time.perf_counter()

    for _ in range(num_warmup):
        inputs = tokenizer(warmup_prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            _ = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                eos_token_id=eos_token_id,
                generation_config=gen_config,
                use_cache=True,
            )
        torch.cuda.synchronize()

    return time.perf_counter() - warmup_start


def measure_prefill_only(
    model,
    tokenizer,
    prompts: list[str],
    num_runs: int = 3,
) -> tuple[float, float]:
    """Measure ONLY the prefill phase (no token generation).

    Prefill = processing all prompt tokens in parallel to build the KV cache.
    This is the "understand the prompt" phase before generation starts.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompts: List of prompt strings
        num_runs: Number of runs to average

    Returns:
        (avg_prefill_time_ms, avg_prompt_tokens)
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, pad_to_multiple_of=8)
    inputs = {k: v.to(model.device, non_blocking=True) for k, v in inputs.items()}

    avg_prompt_tokens = inputs["attention_mask"].sum().item() / len(prompts)

    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            # Forward pass only (prefill) - no generation
            # This computes the KV cache for all prompt tokens
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=True,  # Build KV cache
            )
            # Force synchronization to get accurate timing
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)
        del outputs

    avg_time_sec = sum(times) / len(times)
    return avg_time_sec, avg_prompt_tokens


def expand_kv_cache(past_key_values, num_copies: int):
    """Expand KV cache from batch_size=1 to batch_size=num_copies.

    This is the key operation for "prefill once, decode many":
    - We prefill a single prompt to get KV cache for 1 sequence
    - We tile/repeat it to N copies so we can decode N different outputs

    Args:
        past_key_values: KV cache from model forward pass. Can be:
                        - Tuple of (key, value) tuples (older format)
                        - DynamicCache object (newer transformers)
        num_copies: Number of copies to expand to

    Returns:
        Expanded past_key_values with batch_size=num_copies
    """
    from transformers.cache_utils import DynamicCache

    # Handle DynamicCache (newer transformers format)
    if isinstance(past_key_values, DynamicCache):
        expanded_cache = DynamicCache()
        for layer_idx in range(len(past_key_values)):
            # New API: access via .layers[idx].keys/.values
            layer = past_key_values.layers[layer_idx]
            key_states = layer.keys
            value_states = layer.values

            # Expand batch dimension: (1, H, S, D) -> (N, H, S, D)
            # IMPORTANT: Use .clone() to break CUDA graph dependencies when using torch.compile
            # The repeat() operation creates new storage, but we clone first to ensure
            # we're not referencing tensors that CUDA graphs might overwrite
            expanded_key = key_states.clone().repeat(num_copies, 1, 1, 1)
            expanded_value = value_states.clone().repeat(num_copies, 1, 1, 1)

            expanded_cache.update(expanded_key, expanded_value, layer_idx)

        return expanded_cache

    # Handle tuple format (older transformers)
    expanded = []
    for layer_kv in past_key_values:
        # layer_kv is a tuple (key_states, value_states)
        # Each has shape (batch, num_heads, seq_len, head_dim)
        key_states, value_states = layer_kv

        # Expand batch dimension: (1, H, S, D) -> (N, H, S, D)
        # Clone first to break CUDA graph dependencies
        expanded_key = key_states.clone().repeat(num_copies, 1, 1, 1)
        expanded_value = value_states.clone().repeat(num_copies, 1, 1, 1)

        expanded.append((expanded_key, expanded_value))

    return tuple(expanded)


def generate_with_cached_prefill(
    model,
    tokenizer,
    prompt: str,
    num_generations: int,
    gen_config,
    eos_token_id,
    max_new_tokens: int = 2500,
) -> tuple[list[str], float, float]:
    """Generate multiple outputs from one prompt using cached prefill.

    Uses the official HuggingFace approach:
    1. Initialize StaticCache and prefill with prompt
    2. Clone the cache with copy.deepcopy()
    3. Use model.generate() with the cloned cache

    This approach is compatible with torch.compile and CUDA graphs.

    Args:
        model: The model
        tokenizer: The tokenizer
        prompt: Single prompt string
        num_generations: Number of different outputs to generate
        gen_config: Generation config (sampling parameters)
        eos_token_id: EOS token ID
        max_new_tokens: Max tokens to generate

    Returns:
        (decoded_outputs, prefill_time_sec, decode_time_sec)
    """
    import copy
    from transformers import StaticCache

    # Tokenize single prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    # === PHASE 1: Prefill once using StaticCache ===
    # Estimate max cache length: prompt + max_new_tokens
    max_cache_len = prompt_len + max_new_tokens

    torch.cuda.synchronize()
    prefill_start = time.perf_counter()

    with torch.no_grad():
        # Initialize StaticCache - this is the official HuggingFace approach
        prompt_cache = StaticCache(config=model.config, max_cache_len=max_cache_len, batch_size=1)

        # Prefill: run forward pass to populate the cache
        _ = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            past_key_values=prompt_cache,
            use_cache=True,
        )
        torch.cuda.synchronize()

    prefill_time = time.perf_counter() - prefill_start

    # === PHASE 2: Generate multiple sequences using cloned caches ===
    torch.cuda.synchronize()
    decode_start = time.perf_counter()

    decoded_outputs = []

    with torch.no_grad():
        for _ in range(num_generations):
            # Clone the prefilled cache - official HuggingFace pattern
            cloned_cache = copy.deepcopy(prompt_cache)

            # Generate using model.generate() with the cloned cache
            # Important: pass full input_ids for position tracking
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                past_key_values=cloned_cache,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                generation_config=gen_config,
                use_cache=True,
            )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            decoded_outputs.append(decoded)

        torch.cuda.synchronize()

    decode_time = time.perf_counter() - decode_start

    return decoded_outputs, prefill_time, decode_time


def benchmark_prefill_caching(
    model,
    tokenizer,
    unique_molecules: dict[str, int],
    gen_config,
    eos_token_id,
    max_generations_per_mol: int = 200,
) -> dict:
    """Compare current approach vs cached prefill approach.

    Args:
        model: The model
        tokenizer: The tokenizer
        unique_molecules: Dict of {smiles: num_conformers_needed}
        gen_config: Generation config
        eos_token_id: EOS token ID
        max_generations_per_mol: Cap on generations per molecule for testing

    Returns:
        Dict with timing comparison results
    """
    results = {
        "current_approach": {"total_time": 0, "prefill_time": 0, "decode_time": 0, "num_generated": 0},
        "cached_prefill": {"total_time": 0, "prefill_time": 0, "decode_time": 0, "num_generated": 0},
    }

    # Test on a few molecules
    test_mols = list(unique_molecules.items())[:5]  # Test 5 molecules

    for smiles, num_gens in tqdm(test_mols, desc="Comparing approaches"):
        num_gens = min(num_gens, max_generations_per_mol)
        prompt = f"[SMILES]{smiles}[/SMILES]"

        # === Current approach: duplicate prompts, batch together ===
        logger.info(f"Testing CURRENT approach: {smiles[:30]}... x{num_gens}")
        prompts = [prompt] * num_gens
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        torch.cuda.synchronize()
        current_start = time.perf_counter()

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=800,  # Shorter for testing
                eos_token_id=eos_token_id,
                generation_config=gen_config,
                use_cache=True,
            )
        torch.cuda.synchronize()

        current_time = time.perf_counter() - current_start
        results["current_approach"]["total_time"] += current_time
        results["current_approach"]["num_generated"] += num_gens
        del outputs
        torch.cuda.empty_cache()

        # === Cached prefill approach: prefill once, decode many ===
        logger.info(f"Testing CACHED PREFILL approach: {smiles[:30]}... x{num_gens}")

        try:
            _, prefill_time, decode_time = generate_with_cached_prefill(
                model, tokenizer, prompt, num_gens,
                gen_config, eos_token_id, max_new_tokens=800,
            )
            cached_total = prefill_time + decode_time
            results["cached_prefill"]["total_time"] += cached_total
            results["cached_prefill"]["prefill_time"] += prefill_time
            results["cached_prefill"]["decode_time"] += decode_time
            results["cached_prefill"]["num_generated"] += num_gens
        except Exception as e:
            import traceback
            logger.warning(f"Cached prefill failed: {e}")
            logger.warning(f"Full traceback:\n{traceback.format_exc()}")
            # Fall back to current approach timing
            results["cached_prefill"]["total_time"] += current_time
            results["cached_prefill"]["num_generated"] += num_gens

        torch.cuda.empty_cache()

    # Calculate speedup
    if results["cached_prefill"]["total_time"] > 0:
        results["speedup"] = results["current_approach"]["total_time"] / results["cached_prefill"]["total_time"]
    else:
        results["speedup"] = 1.0

    return results


def run_single_benchmark_with_phases(
    model,
    tokenizer,
    mols_list: list,
    batch_size: int,
    gen_config,
    eos_token_id,
    config_name: str,
    expandable_segments: bool,
    compile_mode: str | None,
    warmup_time: float,
    attn_implementation: str = "flash_attention_2",
) -> BenchmarkResult:
    """Run benchmark WITH detailed prefill vs decode phase timing.

    This runs the benchmark in two passes:
    1. Prefill-only pass: Measure time to process prompts (no generation)
    2. Full generation pass: Measure total time

    Decode time = Total time - Prefill time
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    total_tokens = 0
    total_prompt_tokens = 0
    total_generated_tokens = 0
    oom_occurred = False

    # Accumulators for phase timing
    total_prefill_time = 0.0
    total_generation_time = 0.0
    num_batches = 0

    try:
        for start_idx in tqdm(range(0, len(mols_list), batch_size), desc=f"{config_name} (profiling)", leave=False):
            batch = mols_list[start_idx:start_idx + batch_size]
            prompts = [item[1] for item in batch]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, pad_to_multiple_of=8)
            inputs = {k: v.to(model.device, non_blocking=True) for k, v in inputs.items()}

            prompt_lens = inputs["attention_mask"].sum(dim=1)
            total_prompt_tokens += int(prompt_lens.sum().item())

            # === PHASE 1: Prefill only ===
            torch.cuda.synchronize()
            prefill_start = time.perf_counter()

            with torch.inference_mode():
                prefill_outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=True,
                )
                torch.cuda.synchronize()

            prefill_time = time.perf_counter() - prefill_start
            total_prefill_time += prefill_time
            del prefill_outputs

            # === PHASE 2: Full generation ===
            torch.cuda.synchronize()
            gen_start = time.perf_counter()

            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=2500,
                    eos_token_id=eos_token_id,
                    generation_config=gen_config,
                    use_cache=True,
                )
                torch.cuda.synchronize()

            gen_time = time.perf_counter() - gen_start
            total_generation_time += gen_time

            # Count tokens
            gen_lens = (outputs != tokenizer.pad_token_id).sum(dim=1) - prompt_lens.to(outputs.device)
            batch_generated = int(gen_lens.sum().item())
            total_generated_tokens += batch_generated
            total_tokens += batch_generated

            num_batches += 1
            del outputs

    except torch.cuda.OutOfMemoryError:
        oom_occurred = True
        torch.cuda.empty_cache()
        logger.warning(f"OOM at batch_size={batch_size}")

    torch.cuda.synchronize()

    # Calculate metrics
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    # Decode time = Generation time - Prefill time
    # (because generate() includes prefill internally)
    total_decode_time = max(0, total_generation_time - total_prefill_time)

    # But wait - generate() does its OWN prefill, so we measured prefill twice.
    # The actual breakdown is:
    # - total_prefill_time: Our separate prefill measurement
    # - total_generation_time: Includes HF's internal prefill + decode
    #
    # So the TRUE prefill percentage based on our measurement:
    prefill_pct = (total_prefill_time / total_generation_time * 100) if total_generation_time > 0 else 0

    tps = total_tokens / total_generation_time if total_generation_time > 0 and not oom_occurred else 0
    mps = len(mols_list) / total_generation_time if total_generation_time > 0 and not oom_occurred else 0

    avg_prompt = total_prompt_tokens / len(mols_list) if mols_list else 0
    avg_gen = total_generated_tokens / len(mols_list) if mols_list else 0

    return BenchmarkResult(
        config_name=config_name,
        batch_size=batch_size,
        num_molecules=len(mols_list),
        total_time_sec=total_generation_time,
        tokens_per_sec=tps,
        molecules_per_sec=mps,
        peak_memory_gb=peak_mem,
        oom_occurred=oom_occurred,
        expandable_segments=expandable_segments,
        compile_mode=compile_mode,
        attn_implementation=attn_implementation,
        warmup_time_sec=warmup_time,
        # Phase breakdown
        prefill_time_sec=total_prefill_time,
        decode_time_sec=total_decode_time,
        prefill_pct=prefill_pct,
        avg_prompt_tokens=avg_prompt,
        avg_generated_tokens=avg_gen,
    )


def run_single_benchmark(
    model,
    tokenizer,
    mols_list: list,
    batch_size: int,
    gen_config,
    eos_token_id,
    config_name: str,
    expandable_segments: bool,
    compile_mode: str | None,
    warmup_time: float,
    attn_implementation: str = "flash_attention_2",
) -> BenchmarkResult:
    """Run inference on mols_list with given batch_size, return metrics."""

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    total_tokens = 0
    oom_occurred = False
    start_time = time.perf_counter()

    try:
        for start in tqdm(range(0, len(mols_list), batch_size), desc=config_name, leave=False):
            batch = mols_list[start:start + batch_size]
            prompts = [item[1] for item in batch]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, pad_to_multiple_of=8)
            inputs = {k: v.to(model.device, non_blocking=True) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=2500,
                    eos_token_id=eos_token_id,
                    generation_config=gen_config,
                    use_cache=True,
                )
                # Count generated tokens
                prompt_lens = inputs["attention_mask"].sum(dim=1)
                gen_lens = (outputs != tokenizer.pad_token_id).sum(dim=1) - prompt_lens
                total_tokens += int(gen_lens.sum().item())
                del outputs

            torch.cuda.synchronize()

    except torch.cuda.OutOfMemoryError:
        oom_occurred = True
        torch.cuda.empty_cache()
        logger.warning(f"OOM at batch_size={batch_size}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    tps = total_tokens / elapsed if elapsed > 0 and not oom_occurred else 0
    mps = len(mols_list) / elapsed if elapsed > 0 and not oom_occurred else 0

    return BenchmarkResult(
        config_name=config_name,
        batch_size=batch_size,
        num_molecules=len(mols_list),
        total_time_sec=elapsed,
        tokens_per_sec=tps,
        molecules_per_sec=mps,
        peak_memory_gb=peak_mem,
        oom_occurred=oom_occurred,
        expandable_segments=expandable_segments,
        compile_mode=compile_mode,
        attn_implementation=attn_implementation,
        warmup_time_sec=warmup_time,
    )


def find_max_stable_batch_size(results: list[BenchmarkResult]) -> int:
    """Find largest batch_size that didn't OOM."""
    stable = [r.batch_size for r in results if not r.oom_occurred]
    return max(stable) if stable else 0


def run_single_config_job(
    config: dict,
    mols_list: list,
    output_file: Path,
) -> BenchmarkResult:
    """Run a single benchmark configuration. Can be submitted as a Slurm job.

    Args:
        config: Dict with keys: batch_size, expandable_segments, compile_mode, model_alias, model_step
        mols_list: List of (geom_smiles, prompt) tuples
        output_file: Path to write result JSON

    Returns:
        BenchmarkResult
    """
    set_seed(42)

    # Set expandable_segments
    if config["expandable_segments"]:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    # Load model
    attn_impl = config.get("attn_implementation", "flash_attention_2")
    model, tokenizer = load_model_tokenizer(
        model_path=get_ckpt(config["model_alias"], config["model_step"]),
        tokenizer_path=get_tokenizer_path("qwen3_0.6b_custom"),
        torch_dtype="bfloat16",
        compile_mode=config["compile_mode"],
        attn_implementation=attn_impl,
    )

    eos_token_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
    gen_config = sampling_configs["top_p_sampling1"]

    # Warmup
    warmup_time = run_warmup(model, tokenizer, gen_config, eos_token_id)

    # Build config name
    expand_str = "expand" if config["expandable_segments"] else "no_expand"
    compile_str = config["compile_mode"] if config["compile_mode"] else "none"
    attn_str = attn_impl.replace("_", "")  # flash_attention_2 -> flashattention2
    config_name = f"{expand_str}_{attn_str}_compile={compile_str}_bs={config['batch_size']}"

    # Run benchmark (use phase profiling if requested)
    profile_phases = config.get("profile_phases", False)
    if profile_phases:
        result = run_single_benchmark_with_phases(
            model, tokenizer, mols_list, config["batch_size"],
            gen_config, eos_token_id, config_name,
            config["expandable_segments"], config["compile_mode"],
            warmup_time, attn_impl,
        )
    else:
        result = run_single_benchmark(
            model, tokenizer, mols_list, config["batch_size"],
            gen_config, eos_token_id, config_name,
            config["expandable_segments"], config["compile_mode"],
            warmup_time, attn_impl,
        )

    # Save result
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    return result


def load_test_data(test_set: str, limit: int) -> list[tuple[str, str]]:
    """Load test molecules and build prompts."""
    with open(get_data_path(f"{test_set}_smi"), 'rb') as f:
        test_data = cloudpickle.load(f)

    mols_list = []
    for geom_smiles, data in test_data.items():
        if test_set == "clean":
            mols_list.append((geom_smiles, f"[SMILES]{data['corrected_smi']}[/SMILES]"))
        else:
            # distinct, xl, qm9
            for sub_smiles in data.get("sub_smiles_counts", {}).keys():
                mols_list.append((geom_smiles, f"[SMILES]{sub_smiles}[/SMILES]"))

    # Sort by length (like inference.py) and limit
    mols_list.sort(key=lambda x: len(x[0])) 
    return mols_list[:limit] 


def load_unique_molecules_with_counts(test_set: str, limit: int) -> dict[str, int]:
    """Load unique molecules with their conformer counts for prefill caching test.

    This is different from load_test_data which duplicates molecules.
    Here we return {smiles: num_conformers_needed} for the caching comparison.

    Args:
        test_set: Test set name (clean, distinct, xl, qm9)
        limit: Max number of unique molecules to return

    Returns:
        Dict mapping SMILES to number of conformers needed
    """
    with open(get_data_path(f"{test_set}_smi"), 'rb') as f:
        test_data = cloudpickle.load(f)

    unique_mols = {}
    for geom_smiles, data in test_data.items():
        if test_set == "clean":
            smiles = data['corrected_smi']
            # In clean mode, conformer count is num_confs * 2 (like inference.py)
            num_confs = data.get("num_confs", 1) * 2
            unique_mols[smiles] = num_confs
        else:
            # distinct, xl, qm9 - each sub_smiles has a count
            for sub_smiles, count in data.get("sub_smiles_counts", {}).items():
                num_confs = count * 2  # Match inference.py duplication
                unique_mols[sub_smiles] = num_confs

    # Sort by SMILES length and limit
    sorted_mols = sorted(unique_mols.items(), key=lambda x: len(x[0]))
    return dict(sorted_mols[:limit])


def run_benchmark(args: argparse.Namespace) -> None:
    """Main benchmark routine."""
    set_seed(42)

    # Create descriptive run directory: {device}_{test_set}_{limit}mols_{timestamp}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.device}_{args.test_set}_{args.limit}mols_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting benchmark: {run_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Full args: {args}")

    # Load test data
    mols_list = load_test_data(args.test_set, args.limit)
    logger.info(f"Loaded {len(mols_list)} molecules for benchmarking")

    # Parse configurations
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    compile_modes = args.compile_modes.split(",") if args.compile_modes else ["reduce-overhead"]

    # Determine expandable_segments configurations to test
    if args.test_expandable_segments:
        expandable_opts = [True, False]
    else:
        expandable_opts = [True]  # Default: only test with expandable_segments enabled

    # Attention implementation (single value for now, could be extended to test multiple)
    attn_impl = args.attn_impl

    # Generate all configuration combinations: (expand, compile, batch_size, attn_impl)
    all_configs = [
        (expand, compile, bs, attn_impl)
        for expand, compile, bs in itertools.product(expandable_opts, compile_modes, batch_sizes)
    ]
    logger.info(f"Testing {len(all_configs)} configurations (attn_impl={attn_impl})")

    # Slurm mode: submit to cluster when device is specified
    if args.device in ["a100", "h100"]:
        results = run_multi_gpu_benchmark(args, mols_list, all_configs, output_dir)
        # Slurm mode returns None (non-blocking), collect results later with --collect
        if results is None:
            return
    else:
        # Local mode: add file logging and run sequentially
        logger.add(output_dir / "benchmark.log")
        results = run_single_gpu_benchmark(args, mols_list, all_configs)

    # Build summary with full configuration for reproducibility
    valid_results = [r for r in results if not r.oom_occurred]
    best_result = max(valid_results, key=lambda r: r.tokens_per_sec) if valid_results else None

    summary = BenchmarkSummary(
        timestamp=datetime.now().isoformat(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        model_alias=args.model_alias,
        model_step=args.model_step,
        # Full run configuration
        test_set=args.test_set,
        num_molecules=len(mols_list),
        device=args.device,
        batch_sizes_tested=batch_sizes,
        compile_modes_tested=compile_modes,
        expandable_segments_tested=expandable_opts,
        # Results
        results=[asdict(r) for r in results],
        best_throughput_config=best_result.config_name if best_result else "N/A",
        max_stable_batch_size=find_max_stable_batch_size(results),
    )

    # Save results (filename is simple since folder is already descriptive)
    output_file = output_dir / "results.json"
    with open(output_file, 'w') as f:
        json.dump(asdict(summary), f, indent=2)

    logger.info(f"Results saved to: {output_file}")

    # Print summary
    print_benchmark_summary(args, summary, results, output_file)


def run_single_gpu_benchmark(
    args: argparse.Namespace,
    mols_list: list,
    all_configs: list,
) -> list[BenchmarkResult]:
    """Run benchmark configurations sequentially on single GPU."""
    results = []

    # Sort configs by (expandable, compile_mode, attn_impl) to minimize model reloads
    sorted_configs = sorted(all_configs, key=lambda x: (x[0], x[1], x[3]))

    current_expand = None
    current_compile = None
    current_attn = None
    model = None
    tokenizer = None

    for expand, compile_mode, bs, attn_impl in sorted_configs:
        # Reload model if expandable_segments, compile_mode, or attn_impl changed
        if expand != current_expand or compile_mode != current_compile or attn_impl != current_attn:
            if model is not None:
                del model
                torch.cuda.empty_cache()

            # Set expandable_segments
            if expand:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                logger.info("expandable_segments: ENABLED")
            else:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
                logger.info("expandable_segments: DISABLED")

            logger.info(f"Loading model with compile_mode={compile_mode}, attn_impl={attn_impl}")
            model, tokenizer = load_model_tokenizer(
                model_path=get_ckpt(args.model_alias, args.model_step),
                tokenizer_path=get_tokenizer_path("qwen3_0.6b_custom"),
                torch_dtype="bfloat16",
                compile_mode=compile_mode,
                attn_implementation=attn_impl,
            )

            eos_token_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
            gen_config = sampling_configs["top_p_sampling1"]

            # Warmup
            logger.info("Running warmup iterations...")
            warmup_time = run_warmup(model, tokenizer, gen_config, eos_token_id)
            logger.info(f"Warmup completed in {warmup_time:.2f}s")

            current_expand = expand
            current_compile = compile_mode
            current_attn = attn_impl

        # Run benchmark for this batch size
        expand_str = "expand" if expand else "no_expand"
        compile_str = compile_mode if compile_mode else "none"
        attn_str = attn_impl.replace("_", "")
        config_name = f"{expand_str}_{attn_str}_compile={compile_str}_bs={bs}"
        logger.info(f"Benchmarking: {config_name}")

        # Choose benchmark function based on --profile_phases flag
        if args.profile_phases:
            result = run_single_benchmark_with_phases(
                model, tokenizer, mols_list, bs,
                gen_config, eos_token_id, config_name,
                expand, compile_mode,
                warmup_time, attn_impl,
            )
        else:
            result = run_single_benchmark(
                model, tokenizer, mols_list, bs,
                gen_config, eos_token_id, config_name,
                expand, compile_mode,
                warmup_time, attn_impl,
            )
        results.append(result)

        # Log results - include phase timing if profiling
        if args.profile_phases and not result.oom_occurred:
            logger.info(
                f"  -> {result.tokens_per_sec:.1f} tok/s, "
                f"peak={result.peak_memory_gb:.2f} GB | "
                f"PREFILL: {result.prefill_time_sec:.2f}s ({result.prefill_pct:.1f}%), "
                f"DECODE: {result.decode_time_sec:.2f}s ({100-result.prefill_pct:.1f}%)"
            )
        else:
            logger.info(
                f"  -> {result.tokens_per_sec:.1f} tok/s, "
                f"{result.molecules_per_sec:.2f} mol/s, "
                f"peak={result.peak_memory_gb:.2f} GB, "
                f"OOM={result.oom_occurred}"
            )

    # Cleanup
    if model is not None:
        del model
        torch.cuda.empty_cache()

    return results


def run_multi_gpu_benchmark(
    args: argparse.Namespace,
    mols_list: list,
    all_configs: list,
    output_dir: Path,
) -> list[BenchmarkResult] | None:
    """Submit benchmark jobs to Slurm (non-blocking).

    Returns None immediately after submission. Use --collect to gather results later.
    """
    logger.info(f"Slurm mode: submitting {len(all_configs)} jobs to partition '{args.device}'")

    # Create executor
    executor = submitit.AutoExecutor(folder=str(output_dir / "slurm_jobs"))
    executor.update_parameters(
        name="benchmark",
        timeout_min=60,
        gpus_per_node=1,
        nodes=1,
        mem_gb=80,
        cpus_per_task=12,
        slurm_additional_parameters={"partition": args.device},
    )

    # Submit jobs and save metadata
    job_metadata = []
    for i, (expand, compile_mode, bs, attn_impl) in enumerate(all_configs):
        config = {
            "batch_size": bs,
            "expandable_segments": expand,
            "compile_mode": compile_mode,
            "attn_implementation": attn_impl,
            "model_alias": args.model_alias,
            "model_step": args.model_step,
            "profile_phases": args.profile_phases,
        }
        result_file = output_dir / f"result_{i}.json"

        job = executor.submit(run_single_config_job, config, mols_list, result_file)
        job_metadata.append({
            "job_id": job.job_id,
            "config": config,
            "result_file": str(result_file),
        })
        logger.info(f"Submitted job {job.job_id} for config: expand={expand}, attn={attn_impl}, compile={compile_mode}, bs={bs}")

    # Save job metadata for later collection
    metadata_file = output_dir / "jobs.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "num_jobs": len(job_metadata),
            "num_molecules": len(mols_list),
            "jobs": job_metadata,
        }, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Jobs submitted successfully! Terminal returned.")
    logger.info(f"Monitor with: squeue -u $USER")
    logger.info(f"Collect results when done:")
    logger.info(f"  python scripts/benchmark_inference.py --collect {output_dir}")
    logger.info("=" * 60)

    return None  # Signal that results will be collected later


def print_benchmark_summary(
    args: argparse.Namespace,
    summary: BenchmarkSummary,
    results: list[BenchmarkResult],
    output_file: Path,
) -> None:
    """Print formatted benchmark summary."""
    logger.info("=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)
    logger.info(f"GPU: {summary.gpu_name}")
    logger.info(f"Model: {summary.model_alias} @ {summary.model_step}")
    logger.info(f"Test molecules: {args.limit}")
    logger.info(f"Best config: {summary.best_throughput_config}")
    logger.info(f"Max stable batch size: {summary.max_stable_batch_size}")
    logger.info(f"Results saved to: {output_file}")

    # Print comparison table if testing expandable_segments
    if args.test_expandable_segments:
        logger.info("\nexpandable_segments A/B Comparison:")
        logger.info("-" * 60)
        for r in results:
            if not r.oom_occurred:
                expand_str = "ON " if r.expandable_segments else "OFF"
                logger.info(
                    f"  expand={expand_str} bs={r.batch_size:3d}: "
                    f"{r.tokens_per_sec:7.1f} tok/s, "
                    f"peak={r.peak_memory_gb:.2f} GB"
                )

    # Print phase timing breakdown if profiling was enabled
    if args.profile_phases:
        logger.info("\n" + "=" * 60)
        logger.info("PREFILL vs DECODE PHASE BREAKDOWN")
        logger.info("=" * 60)
        logger.info("")
        logger.info("What these numbers mean:")
        logger.info("  PREFILL: Processing the prompt (SMILES) to build KV cache")
        logger.info("  DECODE:  Generating coordinates token-by-token")
        logger.info("")
        logger.info("-" * 60)
        for r in results:
            if not r.oom_occurred and r.prefill_time_sec > 0:
                logger.info(f"Config: {r.config_name}")
                logger.info(f"  Avg prompt tokens:  {r.avg_prompt_tokens:.0f}")
                logger.info(f"  Avg generated tokens: {r.avg_generated_tokens:.0f}")
                logger.info(f"  Total time:    {r.total_time_sec:.2f}s")
                logger.info(f"  Prefill time:  {r.prefill_time_sec:.2f}s ({r.prefill_pct:.1f}%)")
                logger.info(f"  Decode time:   {r.decode_time_sec:.2f}s ({100-r.prefill_pct:.1f}%)")
                logger.info("")

        # Calculate aggregate stats
        valid = [r for r in results if not r.oom_occurred and r.prefill_time_sec > 0]
        if valid:
            avg_prefill_pct = sum(r.prefill_pct for r in valid) / len(valid)
            logger.info("-" * 60)
            logger.info(f"AVERAGE PREFILL PERCENTAGE: {avg_prefill_pct:.1f}%")
            logger.info("")
            if avg_prefill_pct < 5:
                logger.info("CONCLUSION: Prefill is <5% of time - 'prefill once' optimization")
                logger.info("            would save minimal time. Focus on other optimizations.")
            elif avg_prefill_pct < 15:
                logger.info("CONCLUSION: Prefill is 5-15% of time - moderate savings possible")
                logger.info("            from 'prefill once' but may not be worth implementation effort.")
            else:
                logger.info("CONCLUSION: Prefill is >15% of time - 'prefill once' optimization")
                logger.info("            could provide meaningful speedup. Consider vLLM migration.")


def collect_results(run_dir: Path) -> None:
    """Collect results from a completed Slurm benchmark run."""
    jobs_file = run_dir / "jobs.json"
    if not jobs_file.exists():
        logger.error(f"No jobs.json found in {run_dir}")
        logger.error("This directory may not be a Slurm benchmark run.")
        return

    with open(jobs_file) as f:
        job_data = json.load(f)

    logger.info(f"Collecting results from {job_data['num_jobs']} jobs...")

    results = []
    for job_info in job_data["jobs"]:
        result_file = Path(job_info["result_file"])
        config = job_info["config"]

        if result_file.exists():
            with open(result_file) as f:
                result_dict = json.load(f)
            result = BenchmarkResult(**result_dict)
            results.append(result)
            logger.info(f"  ✓ Job {job_info['job_id']}: {result.tokens_per_sec:.1f} tok/s")
        else:
            # Check if job is still running
            logger.warning(f"  ✗ Job {job_info['job_id']}: result not found (still running or failed?)")
            expand_str = "expand" if config["expandable_segments"] else "no_expand"
            compile_str = config["compile_mode"] if config["compile_mode"] else "none"
            config_name = f"{expand_str}_compile={compile_str}_bs={config['batch_size']}"
            results.append(BenchmarkResult(
                config_name=config_name,
                batch_size=config["batch_size"],
                num_molecules=job_data["num_molecules"],
                total_time_sec=0,
                tokens_per_sec=0,
                molecules_per_sec=0,
                peak_memory_gb=0,
                oom_occurred=True,
                expandable_segments=config["expandable_segments"],
                compile_mode=config["compile_mode"],
            ))

    if not results:
        logger.error("No results collected!")
        return

    # Build summary
    valid_results = [r for r in results if not r.oom_occurred]
    best_result = max(valid_results, key=lambda r: r.tokens_per_sec) if valid_results else None

    # Infer config from job data
    all_bs = list(set(j["config"]["batch_size"] for j in job_data["jobs"]))
    all_compile = list(set(j["config"]["compile_mode"] for j in job_data["jobs"]))
    all_expand = list(set(j["config"]["expandable_segments"] for j in job_data["jobs"]))

    summary = BenchmarkSummary(
        timestamp=datetime.now().isoformat(),
        gpu_name=results[0].config_name.split("_")[0] if results else "N/A",
        model_alias=job_data["jobs"][0]["config"]["model_alias"],
        model_step=job_data["jobs"][0]["config"]["model_step"],
        test_set="(from slurm run)",
        num_molecules=job_data["num_molecules"],
        device="slurm",
        batch_sizes_tested=all_bs,
        compile_modes_tested=all_compile,
        expandable_segments_tested=all_expand,
        results=[asdict(r) for r in results],
        best_throughput_config=best_result.config_name if best_result else "N/A",
        max_stable_batch_size=find_max_stable_batch_size(results),
    )

    # Save results
    output_file = run_dir / "results.json"
    with open(output_file, 'w') as f:
        json.dump(asdict(summary), f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS COLLECTED")
    logger.info("=" * 60)
    logger.info(f"Best config: {summary.best_throughput_config}")
    logger.info(f"Max stable batch size: {summary.max_stable_batch_size}")
    logger.info(f"Results saved to: {output_file}")

    # Print comparison if expandable_segments was tested
    expand_on = [r for r in valid_results if r.expandable_segments]
    expand_off = [r for r in valid_results if not r.expandable_segments]
    if expand_on and expand_off:
        logger.info("")
        logger.info("expandable_segments A/B Comparison:")
        for r in valid_results:
            expand_str = "ON " if r.expandable_segments else "OFF"
            logger.info(f"  expand={expand_str} bs={r.batch_size}: {r.tokens_per_sec:.1f} tok/s, peak={r.peak_memory_gb:.2f} GB")

    # Print phase timing breakdown if results include phase data
    phase_results = [r for r in valid_results if r.prefill_time_sec > 0]
    if phase_results:
        logger.info("")
        logger.info("=" * 60)
        logger.info("PREFILL vs DECODE PHASE BREAKDOWN")
        logger.info("=" * 60)
        logger.info("")
        logger.info("What these numbers mean:")
        logger.info("  PREFILL: Processing the prompt (SMILES) to build KV cache")
        logger.info("  DECODE:  Generating coordinates token-by-token")
        logger.info("")
        logger.info("-" * 60)
        for r in phase_results:
            logger.info(f"Config: {r.config_name}")
            logger.info(f"  Avg prompt tokens:  {r.avg_prompt_tokens:.0f}")
            logger.info(f"  Avg generated tokens: {r.avg_generated_tokens:.0f}")
            logger.info(f"  Total time:    {r.total_time_sec:.2f}s")
            logger.info(f"  Prefill time:  {r.prefill_time_sec:.2f}s ({r.prefill_pct:.1f}%)")
            logger.info(f"  Decode time:   {r.decode_time_sec:.2f}s ({100-r.prefill_pct:.1f}%)")
            logger.info("")

        avg_prefill_pct = sum(r.prefill_pct for r in phase_results) / len(phase_results)
        logger.info("-" * 60)
        logger.info(f"AVERAGE PREFILL PERCENTAGE: {avg_prefill_pct:.1f}%")
        logger.info("")
        if avg_prefill_pct < 5:
            logger.info("CONCLUSION: Prefill is <5% of time - 'prefill once' optimization")
            logger.info("            would save minimal time. Focus on other optimizations.")
        elif avg_prefill_pct < 15:
            logger.info("CONCLUSION: Prefill is 5-15% of time - moderate savings possible")
            logger.info("            from 'prefill once' but may not be worth implementation effort.")
        else:
            logger.info("CONCLUSION: Prefill is >15% of time - 'prefill once' optimization")
            logger.info("            could provide meaningful speedup. Consider vLLM migration.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--collect", type=str, metavar="RUN_DIR",
                        help="Collect results from a previous Slurm run directory")
    parser.add_argument("--limit", type=int, default=100,
                        help="Number of molecules to benchmark (default: 100)")
    parser.add_argument("--batch_sizes", type=str, default="64,128,256",
                        help="Comma-separated batch sizes to test (default: 64,128,256)")
    parser.add_argument("--model_alias", type=str, default="qwen3_grpo_251226_1635",
                        help="Model alias from paths.yaml")
    parser.add_argument("--model_step", type=str, default="4000",
                        help="Model checkpoint step")
    parser.add_argument("--test_set", type=str, default="distinct",
                        choices=["clean", "distinct", "xl", "qm9"],
                        help="Test set to use (default: distinct)")
    parser.add_argument("--compile_modes", type=str, default="reduce-overhead",
                        help="Comma-separated compile modes: none,default,reduce-overhead,max-autotune")
    parser.add_argument("--test_expandable_segments", action="store_true",
                        help="Run A/B comparison with/without expandable_segments")
    parser.add_argument("--profile_phases", action="store_true",
                        help="Measure prefill vs decode time breakdown (adds ~50%% overhead)")
    parser.add_argument("--test_prefill_caching", action="store_true",
                        help="A/B test: current approach vs 'prefill once, decode many' (runs locally)")
    parser.add_argument("--gens_per_mol", type=int, default=50,
                        help="Number of generations per molecule for prefill caching test (default: 50)")
    parser.add_argument("--output_dir", type=str, default="outputs/gen_benchmarking",
                        help="Base directory for benchmark results (default: outputs/gen_benchmarking)")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs for parallel benchmarking (default: 1)")
    parser.add_argument("--device", type=str, default="local",
                        choices=["local", "a100", "h100"],
                        help="Device/partition for Slurm jobs (default: local)")
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"],
                        help="Attention implementation (default: flash_attention_2)")

    args = parser.parse_args()

    # Handle --collect mode
    if args.collect:
        collect_results(Path(args.collect))
        return

    # Handle --test_prefill_caching mode (A/B test, runs locally)
    if args.test_prefill_caching:
        run_prefill_caching_test(args)
        return

    run_benchmark(args)


def run_prefill_caching_job(config: dict, output_dir: Path) -> dict:
    """Run the prefill caching A/B test. Can be submitted as a Slurm job.

    Args:
        config: Dict with keys: test_set, limit, gens_per_mol, model_alias, model_step
        output_dir: Directory to save results

    Returns:
        Results dict with timing comparison
    """
    set_seed(42)

    # Set expandable_segments
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Setup logging
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.add(output_dir / "benchmark.log")

    logger.info("=" * 60)
    logger.info("PREFILL CACHING A/B TEST (Slurm Job)")
    logger.info("=" * 60)
    logger.info(f"Config: {config}")

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model_tokenizer(
        model_path=get_ckpt(config["model_alias"], config["model_step"]),
        tokenizer_path=get_tokenizer_path("qwen3_0.6b_custom"),
        torch_dtype="bfloat16",
        compile_mode="reduce-overhead",
    )
    logger.info(f"Model loaded: {model.dtype}, {model.device}")

    eos_token_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
    gen_config = sampling_configs["top_p_sampling1"]

    # Warmup
    logger.info("Running warmup...")
    run_warmup(model, tokenizer, gen_config, eos_token_id)

    # Load unique molecules
    logger.info("Loading test molecules...")
    unique_mols = load_unique_molecules_with_counts(config["test_set"], config["limit"])
    logger.info(f"Loaded {len(unique_mols)} unique molecules")

    # Run comparison
    logger.info("Starting A/B comparison...")
    results = benchmark_prefill_caching(
        model, tokenizer, unique_mols,
        gen_config, eos_token_id,
        max_generations_per_mol=config["gens_per_mol"],
    )

    # Save results
    results_file = output_dir / "prefill_caching_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {results_file}")
    return results


def run_prefill_caching_test(args: argparse.Namespace) -> None:
    """Run A/B comparison: current approach vs 'prefill once, decode many'.

    This test compares:
    - CURRENT: Duplicate prompts N times, batch them, each computes its own KV cache
    - CACHED:  Prefill prompt once, expand KV cache to N copies, decode N sequences

    Supports both local execution and Slurm submission via --device flag.
    """
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"prefill_caching_test_{args.test_set}_{args.limit}mols_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "test_set": args.test_set,
        "limit": args.limit,
        "gens_per_mol": args.gens_per_mol,
        "model_alias": args.model_alias,
        "model_step": args.model_step,
    }

    # Submit to Slurm if device is specified
    if args.device in ["a100", "h100"]:
        logger.info(f"Submitting prefill caching test to Slurm partition '{args.device}'...")

        executor = submitit.AutoExecutor(folder=str(output_dir / "slurm_jobs"))
        executor.update_parameters(
            name="prefill_cache_test",
            timeout_min=120,
            gpus_per_node=1,
            nodes=1,
            mem_gb=80,
            cpus_per_task=12,
            slurm_additional_parameters={"partition": args.device},
        )

        job = executor.submit(run_prefill_caching_job, config, output_dir)

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Job submitted! Job ID: {job.job_id}")
        logger.info(f"Monitor with: squeue -u $USER")
        logger.info(f"View logs: tail -f {output_dir}/slurm_jobs/{job.job_id}_log.out")
        logger.info(f"Results will be at: {output_dir}/prefill_caching_results.json")
        logger.info("=" * 60)
        return

    # Local execution
    set_seed(42)
    logger.add(output_dir / "benchmark.log")

    logger.info("=" * 60)
    logger.info("PREFILL CACHING A/B TEST")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Comparing two approaches:")
    logger.info("  CURRENT: Duplicate prompts, each computes its own KV cache")
    logger.info("  CACHED:  Prefill once, expand KV cache, decode many")
    logger.info("")
    logger.info(f"Test set: {args.test_set}")
    logger.info(f"Unique molecules: {args.limit}")
    logger.info(f"Generations per molecule: {args.gens_per_mol}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Set expandable_segments (use it for both to be fair)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model_tokenizer(
        model_path=get_ckpt(args.model_alias, args.model_step),
        tokenizer_path=get_tokenizer_path("qwen3_0.6b_custom"),
        torch_dtype="bfloat16",
        compile_mode="reduce-overhead",
    )
    logger.info(f"Model loaded: {model.dtype}, {model.device}")

    eos_token_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
    gen_config = sampling_configs["top_p_sampling1"]

    # Warmup
    logger.info("Running warmup...")
    run_warmup(model, tokenizer, gen_config, eos_token_id)

    # Load unique molecules
    logger.info("Loading test molecules...")
    unique_mols = load_unique_molecules_with_counts(args.test_set, args.limit)
    logger.info(f"Loaded {len(unique_mols)} unique molecules")

    # Get conformer counts
    total_gens = sum(min(count, args.gens_per_mol) for count in unique_mols.values())
    avg_gens = total_gens / len(unique_mols) if unique_mols else 0
    logger.info(f"Total generations to compare: {total_gens} (avg {avg_gens:.1f} per mol)")
    logger.info("")

    # Run comparison
    logger.info("Starting A/B comparison...")
    results = benchmark_prefill_caching(
        model, tokenizer, unique_mols,
        gen_config, eos_token_id,
        max_generations_per_mol=args.gens_per_mol,
    )

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("PREFILL CACHING TEST RESULTS")
    logger.info("=" * 60)
    logger.info("")

    current = results["current_approach"]
    cached = results["cached_prefill"]

    logger.info(f"CURRENT APPROACH (duplicate prompts):")
    logger.info(f"  Total time:     {current['total_time']:.2f}s")
    logger.info(f"  Generations:    {current['num_generated']}")
    if current['total_time'] > 0:
        logger.info(f"  Throughput:     {current['num_generated'] / current['total_time']:.2f} gen/s")
    logger.info("")

    logger.info(f"CACHED PREFILL (prefill once, decode many):")
    logger.info(f"  Total time:     {cached['total_time']:.2f}s")
    logger.info(f"  Prefill time:   {cached['prefill_time']:.3f}s")
    logger.info(f"  Decode time:    {cached['decode_time']:.2f}s")
    logger.info(f"  Generations:    {cached['num_generated']}")
    if cached['total_time'] > 0:
        logger.info(f"  Throughput:     {cached['num_generated'] / cached['total_time']:.2f} gen/s")
    logger.info("")

    speedup = results["speedup"]
    logger.info("-" * 60)
    if speedup > 1.05:
        logger.info(f"SPEEDUP: {speedup:.2f}x FASTER with cached prefill!")
        logger.info("")
        logger.info("CONCLUSION: 'Prefill once' optimization provides measurable benefit.")
        logger.info("            Consider implementing this in production inference.")
    elif speedup > 0.95:
        logger.info(f"SPEEDUP: {speedup:.2f}x (roughly equal)")
        logger.info("")
        logger.info("CONCLUSION: No significant difference between approaches.")
        logger.info("            Current approach is fine, optimization not needed.")
    else:
        logger.info(f"SPEEDUP: {speedup:.2f}x (cached approach is SLOWER)")
        logger.info("")
        logger.info("CONCLUSION: Cached prefill is slower (unexpected).")
        logger.info("            This might indicate overhead from KV cache expansion.")

    # Save results
    results_file = output_dir / "prefill_caching_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("")
    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
