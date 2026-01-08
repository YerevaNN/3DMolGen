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
    warmup_time_sec: float = 0.0


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


def load_model_tokenizer(model_path, tokenizer_path, torch_dtype="bfloat16", compile_mode="reduce-overhead"):
    """Load model and tokenizer with specified settings."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), padding_side="left", local_files_only=True
    )
    dtype_obj = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype_obj,
        attn_implementation="flash_attention_2",
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
    model, tokenizer = load_model_tokenizer(
        model_path=get_ckpt(config["model_alias"], config["model_step"]),
        tokenizer_path=get_tokenizer_path("qwen3_0.6b_custom"),
        torch_dtype="bfloat16",
        compile_mode=config["compile_mode"],
    )

    eos_token_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
    gen_config = sampling_configs["top_p_sampling1"]

    # Warmup
    warmup_time = run_warmup(model, tokenizer, gen_config, eos_token_id)

    # Build config name
    expand_str = "expand" if config["expandable_segments"] else "no_expand"
    compile_str = config["compile_mode"] if config["compile_mode"] else "none"
    config_name = f"{expand_str}_compile={compile_str}_bs={config['batch_size']}"

    # Run benchmark
    result = run_single_benchmark(
        model, tokenizer, mols_list, config["batch_size"],
        gen_config, eos_token_id, config_name,
        config["expandable_segments"], config["compile_mode"],
        warmup_time,
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

    # Generate all configuration combinations
    all_configs = list(itertools.product(expandable_opts, compile_modes, batch_sizes))
    logger.info(f"Testing {len(all_configs)} configurations")

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

    # Sort configs by (expandable, compile_mode) to minimize model reloads
    sorted_configs = sorted(all_configs, key=lambda x: (x[0], x[1]))

    current_expand = None
    current_compile = None
    model = None
    tokenizer = None

    for expand, compile_mode, bs in sorted_configs:
        # Reload model if expandable_segments or compile_mode changed
        if expand != current_expand or compile_mode != current_compile:
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

            logger.info(f"Loading model with compile_mode={compile_mode}")
            model, tokenizer = load_model_tokenizer(
                model_path=get_ckpt(args.model_alias, args.model_step),
                tokenizer_path=get_tokenizer_path("qwen3_0.6b_custom"),
                torch_dtype="bfloat16",
                compile_mode=compile_mode,
            )

            eos_token_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
            gen_config = sampling_configs["top_p_sampling1"]

            # Warmup
            logger.info("Running warmup iterations...")
            warmup_time = run_warmup(model, tokenizer, gen_config, eos_token_id)
            logger.info(f"Warmup completed in {warmup_time:.2f}s")

            current_expand = expand
            current_compile = compile_mode

        # Run benchmark for this batch size
        expand_str = "expand" if expand else "no_expand"
        compile_str = compile_mode if compile_mode else "none"
        config_name = f"{expand_str}_compile={compile_str}_bs={bs}"
        logger.info(f"Benchmarking: {config_name}")

        result = run_single_benchmark(
            model, tokenizer, mols_list, bs,
            gen_config, eos_token_id, config_name,
            expand, compile_mode,
            warmup_time,
        )
        results.append(result)

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
    for i, (expand, compile_mode, bs) in enumerate(all_configs):
        config = {
            "batch_size": bs,
            "expandable_segments": expand,
            "compile_mode": compile_mode,
            "model_alias": args.model_alias,
            "model_step": args.model_step,
        }
        result_file = output_dir / f"result_{i}.json"

        job = executor.submit(run_single_config_job, config, mols_list, result_file)
        job_metadata.append({
            "job_id": job.job_id,
            "config": config,
            "result_file": str(result_file),
        })
        logger.info(f"Submitted job {job.job_id} for config: expand={expand}, compile={compile_mode}, bs={bs}")

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
    parser.add_argument("--output_dir", type=str, default="outputs/gen_benchmarking",
                        help="Base directory for benchmark results (default: outputs/gen_benchmarking)")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs for parallel benchmarking (default: 1)")
    parser.add_argument("--device", type=str, default="local",
                        choices=["local", "a100", "h100"],
                        help="Device/partition for Slurm jobs (default: local)")

    args = parser.parse_args()

    # Handle --collect mode
    if args.collect:
        collect_results(Path(args.collect))
        return

    run_benchmark(args)


if __name__ == "__main__":
    main()
