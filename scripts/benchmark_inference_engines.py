#!/usr/bin/env python3
"""
A/B comparison of inference engines: HuggingFace vs vLLM vs SGLang.

Tests prefix caching benefits for the 3DMolGen workload pattern:
- Same SMILES prompt repeated N times per molecule
- Measures tok/s, memory, and latency

Workload context:
    - GEOM-DRUGS test set: ~1,000 molecules, ~213k total conformers
    - Average ~200 conformers per molecule (variable per molecule)
    - Same prompt repeated N times = ideal for prefix caching

Usage:
    # Quick vLLM test (simplest to install)
    python scripts/benchmark_inference_engines.py --backend vllm --limit 20 --gens_per_mol 50

    # SGLang test
    python scripts/benchmark_inference_engines.py --backend sglang --limit 20 --gens_per_mol 50

    # Compare all backends (quick)
    python scripts/benchmark_inference_engines.py --backend all --limit 10 --gens_per_mol 50

    # Realistic workload simulation (~200 gens/mol)
    python scripts/benchmark_inference_engines.py --backend vllm --limit 10 --gens_per_mol 200

    # Submit to A100 via Slurm (non-blocking, returns immediately)
    python scripts/benchmark_inference_engines.py --backend vllm --limit 20 --gens_per_mol 50 --device a100

    # Submit to H100
    python scripts/benchmark_inference_engines.py --backend all --limit 10 --gens_per_mol 100 --device h100

    # Test expandable_segments A/B comparison (stack with prefix caching)
    python scripts/benchmark_inference_engines.py --backend vllm --test_expandable_segments --limit 20 --gens_per_mol 50

    # Collect results after jobs complete
    python scripts/benchmark_inference_engines.py --collect outputs/engine_benchmarks/a100_vllm_20260120_123456

Installation:
    # vLLM (try this first - most stable)
    pip install vllm

    # SGLang
    pip install "sglang[all]"

    # If CUDA conflicts, create a separate env:
    conda create -n inference-test python=3.11
    conda activate inference-test
    pip install torch vllm  # or sglang
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import torch
import submitit

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from molgen3D.config.paths import get_ckpt, get_tokenizer_path, get_data_path


@dataclass
class EngineResult:
    """Single engine benchmark result."""
    backend: str
    num_prompts: int
    total_tokens: int
    total_time_sec: float
    tokens_per_sec: float
    peak_memory_gb: float
    expandable_segments: bool = True  # CUDA allocator config
    first_token_latency_ms: float | None = None
    error: str | None = None


def load_test_prompts(test_set: str, limit: int, gens_per_mol: int) -> list[str]:
    """Load test molecules and create prompts with repetition for prefix caching test.

    Args:
        test_set: Test set name (distinct, clean, xl, qm9)
        limit: Max unique molecules
        gens_per_mol: Number of generations per molecule (tests prefix reuse)

    Returns:
        List of prompt strings, with each molecule repeated gens_per_mol times
    """
    import cloudpickle

    with open(get_data_path(f"{test_set}_smi"), 'rb') as f:
        test_data = cloudpickle.load(f)

    unique_smiles = []
    for geom_smiles, data in test_data.items():
        if test_set == "clean":
            unique_smiles.append(data['corrected_smi'])
        else:
            for sub_smiles in data.get("sub_smiles_counts", {}).keys():
                unique_smiles.append(sub_smiles)

    # Sort by length (shorter molecules first, like benchmark_inference.py)
    unique_smiles.sort(key=len)
    unique_smiles = unique_smiles[:limit]

    # Create prompts with repetition
    prompts = []
    for smiles in unique_smiles:
        prompt = f"[SMILES]{smiles}[/SMILES]"
        prompts.extend([prompt] * gens_per_mol)

    return prompts


def get_model_paths(model_alias: str, model_step: str) -> tuple[str, str]:
    """Get model and tokenizer paths."""
    model_path = get_ckpt(model_alias, model_step)
    tokenizer_path = get_tokenizer_path("qwen3_0.6b_custom")
    return str(model_path), str(tokenizer_path)


def benchmark_huggingface(
    prompts: list[str],
    model_path: str,
    tokenizer_path: str,
    max_new_tokens: int,
    batch_size: int,
) -> EngineResult:
    """Benchmark HuggingFace transformers (baseline)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[HuggingFace] Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model.eval()

    # Warmup
    print("[HuggingFace] Warmup...")
    _ = model.generate(
        tokenizer(prompts[0], return_tensors="pt").input_ids.cuda(),
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    print(f"[HuggingFace] Running {len(prompts)} prompts in batches of {batch_size}...")
    total_tokens = 0
    start = time.perf_counter()

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Count generated tokens (exclude prompt)
        for j, output in enumerate(outputs):
            prompt_len = inputs.input_ids[j].ne(tokenizer.pad_token_id).sum()
            total_tokens += len(output) - prompt_len

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return EngineResult(
        backend="huggingface",
        num_prompts=len(prompts),
        total_tokens=total_tokens,
        total_time_sec=elapsed,
        tokens_per_sec=total_tokens / elapsed,
        peak_memory_gb=peak_mem,
    )


def benchmark_vllm(
    prompts: list[str],
    model_path: str,
    tokenizer_path: str,
    max_new_tokens: int,
) -> EngineResult:
    """Benchmark vLLM with prefix caching enabled."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        return EngineResult(
            backend="vllm",
            num_prompts=len(prompts),
            total_tokens=0,
            total_time_sec=0,
            tokens_per_sec=0,
            peak_memory_gb=0,
            error="vllm not installed. Run: pip install vllm",
        )

    print(f"[vLLM] Loading model from {model_path}")
    print(f"[vLLM] Using tokenizer from {tokenizer_path}")

    try:
        llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            dtype="bfloat16",
            enable_prefix_caching=True,  # Key feature for our workload
            gpu_memory_utilization=0.9,
            max_model_len=2048,
        )
    except Exception as e:
        return EngineResult(
            backend="vllm",
            num_prompts=len(prompts),
            total_tokens=0,
            total_time_sec=0,
            tokens_per_sec=0,
            peak_memory_gb=0,
            error=f"Failed to load model: {e}",
        )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_new_tokens,
    )

    # Warmup
    print("[vLLM] Warmup...")
    _ = llm.generate([prompts[0]], sampling_params)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    print(f"[vLLM] Running {len(prompts)} prompts...")
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return EngineResult(
        backend="vllm",
        num_prompts=len(prompts),
        total_tokens=total_tokens,
        total_time_sec=elapsed,
        tokens_per_sec=total_tokens / elapsed,
        peak_memory_gb=peak_mem,
    )


def benchmark_sglang(
    prompts: list[str],
    model_path: str,
    tokenizer_path: str,
    max_new_tokens: int,
) -> EngineResult:
    """Benchmark SGLang with RadixAttention (automatic prefix caching)."""
    try:
        import sglang as sgl
    except ImportError:
        return EngineResult(
            backend="sglang",
            num_prompts=len(prompts),
            total_tokens=0,
            total_time_sec=0,
            tokens_per_sec=0,
            peak_memory_gb=0,
            error="sglang not installed. Run: pip install 'sglang[all]'",
        )

    print(f"[SGLang] Loading model from {model_path}")

    try:
        # SGLang Runtime with RadixAttention enabled by default
        runtime = sgl.Runtime(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            tp_size=1,
            dtype="bfloat16",
        )
        sgl.set_default_backend(runtime)
    except Exception as e:
        return EngineResult(
            backend="sglang",
            num_prompts=len(prompts),
            total_tokens=0,
            total_time_sec=0,
            tokens_per_sec=0,
            peak_memory_gb=0,
            error=f"Failed to load model: {e}",
        )

    # Warmup
    print("[SGLang] Warmup...")
    _ = runtime.generate(
        prompts[0],
        sampling_params={"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 50},
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    print(f"[SGLang] Running {len(prompts)} prompts...")
    start = time.perf_counter()

    outputs = runtime.generate(
        prompts,
        sampling_params={
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": max_new_tokens,
        },
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Count tokens
    total_tokens = sum(len(o["meta_info"]["completion_tokens"]) for o in outputs)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    runtime.shutdown()
    gc.collect()
    torch.cuda.empty_cache()

    return EngineResult(
        backend="sglang",
        num_prompts=len(prompts),
        total_tokens=total_tokens,
        total_time_sec=elapsed,
        tokens_per_sec=total_tokens / elapsed,
        peak_memory_gb=peak_mem,
    )


def print_results(results: list[EngineResult]) -> None:
    """Print comparison table."""
    print("\n" + "=" * 95)
    print("RESULTS COMPARISON")
    print("=" * 95)

    # Header
    print(f"{'Backend':<15} {'Expand':>7} {'Prompts':>8} {'Tokens':>10} {'Time (s)':>10} "
          f"{'Tok/s':>10} {'Memory (GB)':>12} {'Error':<15}")
    print("-" * 95)

    for r in results:
        error_str = r.error[:12] + "..." if r.error and len(r.error) > 15 else (r.error or "")
        expand_str = "ON" if r.expandable_segments else "OFF"
        print(f"{r.backend:<15} {expand_str:>7} {r.num_prompts:>8} {r.total_tokens:>10} "
              f"{r.total_time_sec:>10.2f} {r.tokens_per_sec:>10.1f} "
              f"{r.peak_memory_gb:>12.2f} {error_str:<15}")

    print("=" * 95)

    # Speedup comparison (use HF with expand=ON as baseline)
    baseline = next(
        (r for r in results if r.backend == "huggingface" and r.expandable_segments and not r.error),
        None,
    )
    if baseline:
        print("\nSpeedup vs HuggingFace baseline (expand=ON):")
        for r in results:
            if not r.error and r.tokens_per_sec > 0 and r != baseline:
                speedup = r.tokens_per_sec / baseline.tokens_per_sec
                expand_str = "expand=ON" if r.expandable_segments else "expand=OFF"
                print(f"  {r.backend} ({expand_str}): {speedup:.2f}x")

    # expandable_segments A/B comparison if both tested
    expand_on = [r for r in results if r.expandable_segments and not r.error]
    expand_off = [r for r in results if not r.expandable_segments and not r.error]
    if expand_on and expand_off:
        print("\nexpandable_segments A/B comparison (same backend):")
        backends_tested = set(r.backend for r in results)
        for backend in backends_tested:
            on_result = next((r for r in expand_on if r.backend == backend), None)
            off_result = next((r for r in expand_off if r.backend == backend), None)
            if on_result and off_result and on_result.tokens_per_sec > 0 and off_result.tokens_per_sec > 0:
                diff_pct = (on_result.tokens_per_sec - off_result.tokens_per_sec) / off_result.tokens_per_sec * 100
                print(f"  {backend}: ON={on_result.tokens_per_sec:.1f} vs OFF={off_result.tokens_per_sec:.1f} tok/s "
                      f"({diff_pct:+.1f}%)")


def run_single_backend_job(
    backend: str,
    prompts: list[str],
    model_path: str,
    tokenizer_path: str,
    max_new_tokens: int,
    batch_size: int,
    expandable_segments: bool,
    result_file: Path,
) -> EngineResult:
    """Run a single backend benchmark (for Slurm job submission)."""
    # Set CUDA allocator config
    if expandable_segments:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    if backend == "huggingface":
        result = benchmark_huggingface(
            prompts, model_path, tokenizer_path,
            max_new_tokens, batch_size,
        )
    elif backend == "vllm":
        result = benchmark_vllm(
            prompts, model_path, tokenizer_path,
            max_new_tokens,
        )
    elif backend == "sglang":
        result = benchmark_sglang(
            prompts, model_path, tokenizer_path,
            max_new_tokens,
        )
    else:
        result = EngineResult(
            backend=backend,
            num_prompts=len(prompts),
            total_tokens=0,
            total_time_sec=0,
            tokens_per_sec=0,
            peak_memory_gb=0,
            error=f"Unknown backend: {backend}",
        )

    # Set expandable_segments flag in result
    result.expandable_segments = expandable_segments

    # Save result to file
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    return result


def submit_slurm_jobs(
    args: argparse.Namespace,
    backends: list[str],
    expandable_opts: list[bool],
    prompts: list[str],
    model_path: str,
    tokenizer_path: str,
    output_dir: Path,
) -> None:
    """Submit benchmark jobs to Slurm (non-blocking)."""
    total_jobs = len(backends) * len(expandable_opts)
    print(f"Slurm mode: submitting {total_jobs} job(s) to partition '{args.device}'")
    if len(expandable_opts) > 1:
        print("  (Testing both expandable_segments ON and OFF)")

    executor = submitit.AutoExecutor(folder=str(output_dir / "slurm_jobs"))
    executor.update_parameters(
        name="bench_engine",
        timeout_min=120,  # 2 hours for longer benchmarks
        gpus_per_node=1,
        nodes=1,
        mem_gb=80,
        cpus_per_task=12,
        slurm_additional_parameters={"partition": args.device},
    )

    job_metadata = []
    for expand in expandable_opts:
        expand_suffix = "expand" if expand else "no_expand"
        for backend in backends:
            result_file = output_dir / f"result_{backend}_{expand_suffix}.json"

            job = executor.submit(
                run_single_backend_job,
                backend,
                prompts,
                model_path,
                tokenizer_path,
                args.max_new_tokens,
                args.batch_size,
                expand,  # expandable_segments
                result_file,
            )

            job_metadata.append({
                "job_id": job.job_id,
                "backend": backend,
                "expandable_segments": expand,
                "result_file": str(result_file),
            })
            expand_str = "expand=ON" if expand else "expand=OFF"
            print(f"Submitted job {job.job_id} for {backend} ({expand_str})")

    # Save metadata for later collection
    metadata_file = output_dir / "jobs_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "submitted_at": datetime.now().isoformat(),
            "args": vars(args),
            "jobs": job_metadata,
        }, f, indent=2)

    print(f"\nJobs submitted. Metadata saved to: {metadata_file}")
    print(f"Results will be saved to: {output_dir}/result_*.json")
    print(f"\nTo collect results after jobs complete:")
    print(f"  python scripts/benchmark_inference_engines.py --collect {output_dir}")


def collect_results(output_dir: Path) -> None:
    """Collect and display results from completed Slurm jobs."""
    metadata_file = output_dir / "jobs_metadata.json"
    if not metadata_file.exists():
        print(f"Error: No jobs_metadata.json found in {output_dir}")
        return

    with open(metadata_file) as f:
        metadata = json.load(f)

    results = []
    for job_info in metadata["jobs"]:
        result_file = Path(job_info["result_file"])
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
                results.append(EngineResult(**data))
        else:
            print(f"Warning: Result file not found: {result_file}")
            results.append(EngineResult(
                backend=job_info["backend"],
                num_prompts=0,
                total_tokens=0,
                total_time_sec=0,
                tokens_per_sec=0,
                peak_memory_gb=0,
                error="Job not completed or failed",
            ))

    print_results(results)


def main():
    parser = argparse.ArgumentParser(description="Compare inference engines")
    parser.add_argument("--backend", default="vllm",
                        choices=["huggingface", "vllm", "sglang", "all"],
                        help="Which backend(s) to test")
    parser.add_argument("--limit", type=int, default=50,
                        help="Number of unique molecules")
    parser.add_argument("--gens_per_mol", type=int, default=50,
                        help="Generations per molecule (tests prefix caching). "
                             "Real workload: ~200/mol. Quick test: 50.")
    parser.add_argument("--max_new_tokens", type=int, default=500,
                        help="Max tokens to generate per prompt")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for HuggingFace (vLLM/SGLang batch automatically)")
    parser.add_argument("--test_set", default="distinct",
                        choices=["distinct", "clean", "xl", "qm9"],
                        help="Test set to use")
    parser.add_argument("--model_alias", default="qw600_conf",
                        help="Model alias from paths.yaml (e.g., qw600_conf, qw600_pre)")
    parser.add_argument("--model_step", default="1e",
                        help="Model checkpoint step key (e.g., 1e, 2e, 3e, 4e)")
    parser.add_argument("--device", default="local",
                        choices=["local", "a100", "h100"],
                        help="Run locally or submit to Slurm partition")
    parser.add_argument("--test_expandable_segments", action="store_true",
                        help="Run A/B comparison with/without expandable_segments. "
                             "Tests whether it stacks with prefix caching or is redundant.")
    parser.add_argument("--output_dir", default="outputs/engine_benchmarks",
                        help="Output directory for results")
    parser.add_argument("--collect", type=str, default=None,
                        help="Collect results from a previous Slurm run (provide output dir)")
    args = parser.parse_args()

    # Collect mode: just gather and display results
    if args.collect:
        collect_results(Path(args.collect))
        return

    # Get paths
    model_path, tokenizer_path = get_model_paths(args.model_alias, args.model_step)
    print(f"Model: {model_path}")
    print(f"Tokenizer: {tokenizer_path}")

    # Load prompts
    prompts = load_test_prompts(args.test_set, args.limit, args.gens_per_mol)
    print(f"Total prompts: {len(prompts)} ({args.limit} molecules × {args.gens_per_mol} gens)")

    # Determine backends to test
    if args.backend == "all":
        backends = ["huggingface", "vllm", "sglang"]
    else:
        backends = [args.backend]

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{args.device}_{args.backend}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine expandable_segments configurations to test
    if args.test_expandable_segments:
        expandable_opts = [True, False]
        print("Testing expandable_segments A/B comparison")
    else:
        expandable_opts = [True]  # Default: only test with expandable_segments enabled

    # Slurm mode: submit jobs and return
    if args.device in ["a100", "h100"]:
        submit_slurm_jobs(
            args, backends, expandable_opts, prompts,
            model_path, tokenizer_path, output_dir,
        )
        return

    # Local mode: run directly
    results = []

    for expand in expandable_opts:
        # Set CUDA allocator config
        if expand:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            expand_str = "expand=ON"
        else:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            expand_str = "expand=OFF"

        for backend in backends:
            print(f"\n{'=' * 60}")
            print(f"Testing: {backend.upper()} | {expand_str}")
            print(f"{'=' * 60}")

            if backend == "huggingface":
                result = benchmark_huggingface(
                    prompts, model_path, tokenizer_path,
                    args.max_new_tokens, args.batch_size,
                )
            elif backend == "vllm":
                result = benchmark_vllm(
                    prompts, model_path, tokenizer_path,
                    args.max_new_tokens,
                )
            elif backend == "sglang":
                result = benchmark_sglang(
                    prompts, model_path, tokenizer_path,
                    args.max_new_tokens,
                )

            # Set expandable_segments flag in result
            result.expandable_segments = expand
            results.append(result)

            # Save individual result
            expand_suffix = "expand" if expand else "no_expand"
            result_file = output_dir / f"result_{backend}_{expand_suffix}.json"
            with open(result_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)

            if result.error:
                print(f"[{backend}] Error: {result.error}")
            else:
                print(f"[{backend}] Completed: {result.tokens_per_sec:.1f} tok/s, "
                      f"{result.peak_memory_gb:.2f} GB peak memory")

            # Clean up between runs
            gc.collect()
            torch.cuda.empty_cache()

    print_results(results)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
