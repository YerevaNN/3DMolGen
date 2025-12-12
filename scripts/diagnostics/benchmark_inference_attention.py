#!/usr/bin/env python3
"""
Benchmark real inference with different attention implementations.
Runs on a sample of real molecules to compare actual throughput.

Uses submitit to automatically submit jobs to slurm (h100/a100).

Available attention implementations:
    - sdpa: PyTorch SDPA (auto-selects best backend: cuDNN on H100)
    - eager: Naive attention (baseline)
    - flash_attention_2: Requires flash_attn package (has dependency issues)

NOTE on SDPA backend forcing:
    Forcing specific SDPA backends (sdpa_backend_*) can hang or fail due to
    unsupported configurations. PyTorch's SDPA auto-selection is recommended.
    The auto-selected backend on H100 is typically cuDNN (best for Hopper).

Static KV Cache:
    Use --static-cache to enable static KV cache. This pre-allocates the KV cache
    to a fixed size, which enables CUDA graph capture and may fix issues with
    specific SDPA backends that fail with dynamic cache shapes.

Usage:
    # Recommended comparison (safe defaults):
    python scripts/diagnostics/benchmark_inference_attention.py \
        --model m600_qwen --tokenizer qwen3_0.6b_custom

    # Llama model:
    python scripts/diagnostics/benchmark_inference_attention.py \
        --model m380_conf_v2 --tokenizer llama3_chem_v1

    # Test with static KV cache:
    python scripts/diagnostics/benchmark_inference_attention.py \
        --model m380_conf_v2 --tokenizer llama3_chem_v1 --static-cache

    # Run locally (if already on GPU node):
    python scripts/diagnostics/benchmark_inference_attention.py \
        --model m600_qwen --tokenizer qwen3_0.6b_custom --device local
"""

import argparse
import json
import random
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Project root (absolute path for NFS access across nodes)
PROJECT_ROOT = "/auto/home/aram.dovlatyan/3DMolGen-new/3DMolGen"

# Default path to smoke test results with real SMILES (absolute for NFS)
DEFAULT_SMILES_JSON = f"{PROJECT_ROOT}/outputs/smoke/experiments/logit_processing/strip_smiles_distinct/v2_1_new_strip_smiles_distinct_top_p.json"

# Fallback SMILES if JSON file not available
FALLBACK_SMILES = [
    "CCO",  # Ethanol
    "CC(=O)O",  # Acetic acid
    "c1ccccc1",  # Benzene
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
]


def resolve_nfs_path(path: str) -> str:
    """
    Resolve path for NFS access across nodes.
    Paths starting with /auto/home/ are accessible via NFS on all nodes.
    """
    # If already absolute path starting with /auto/, it should work via NFS
    if path.startswith("/auto/"):
        return path
    # If relative, make it absolute relative to project root
    if not path.startswith("/"):
        return f"{PROJECT_ROOT}/{path}"
    return path


def load_smiles_from_json(json_path: str, num_samples: int, seed: int = 42) -> list[str]:
    """Load SMILES from a smoke test JSON file."""
    resolved_path = resolve_nfs_path(json_path)
    path = Path(resolved_path)
    if not path.exists():
        print(f"  Warning: {resolved_path} not found, using fallback SMILES")
        return (FALLBACK_SMILES * ((num_samples // len(FALLBACK_SMILES)) + 1))[:num_samples]

    with open(path) as f:
        data = json.load(f)

    passes = data.get("passes", [])
    if not passes:
        print(f"  Warning: No passes in {json_path}, using fallback SMILES")
        return (FALLBACK_SMILES * ((num_samples // len(FALLBACK_SMILES)) + 1))[:num_samples]

    # Extract prompt_smiles from passes
    all_smiles = [p["prompt_smiles"] for p in passes if "prompt_smiles" in p]

    # Sample randomly (with seed for reproducibility)
    random.seed(seed)
    if len(all_smiles) >= num_samples:
        sampled = random.sample(all_smiles, num_samples)
    else:
        # If not enough, repeat with replacement
        sampled = random.choices(all_smiles, k=num_samples)

    print(f"  Loaded {len(sampled)} SMILES from {json_path}")
    return sampled


def configure_sdpa_backend(attention_impl: str):
    """
    Configure SDPA backend via torch.backends settings.

    For sdpa_* variants, we enable only the specified backend and disable others.
    This is more reliable than using sdpa_kernel() context manager during generate().
    """
    # Reset all backends to enabled first
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(True)

    backend_config = {
        "sdpa_flash": {
            "flash": True, "mem_efficient": False, "math": False, "cudnn": False,
            "name": "Flash only"
        },
        "sdpa_cudnn": {
            "flash": False, "mem_efficient": False, "math": False, "cudnn": True,
            "name": "cuDNN only"
        },
        "sdpa_efficient": {
            "flash": False, "mem_efficient": True, "math": False, "cudnn": False,
            "name": "Memory-efficient only"
        },
        "sdpa_math": {
            "flash": False, "mem_efficient": False, "math": True, "cudnn": False,
            "name": "Math only (slowest)"
        },
    }

    if attention_impl in backend_config:
        cfg = backend_config[attention_impl]
        torch.backends.cuda.enable_flash_sdp(cfg["flash"])
        torch.backends.cuda.enable_mem_efficient_sdp(cfg["mem_efficient"])
        torch.backends.cuda.enable_math_sdp(cfg["math"])
        torch.backends.cuda.enable_cudnn_sdp(cfg["cudnn"])
        print(f"  SDPA backend: {cfg['name']}")
        return cfg["name"]
    elif attention_impl == "sdpa":
        print(f"  SDPA backend: auto-select (cuDNN > Flash > Efficient > Math)")
        return "auto"

    return None


def load_model(
    model_path: str,
    tokenizer_path: str,
    attention_impl: str,
    static_cache: bool = False,
    max_cache_len: int = 2048,
):
    """Load model with specified attention implementation.

    Args:
        model_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer
        attention_impl: Attention implementation (sdpa, eager, flash_attention_2, etc.)
        static_cache: If True, enable static KV cache (pre-allocated to max_cache_len)
        max_cache_len: Maximum cache length for static KV cache (prompt + max_new_tokens)
    """
    print(f"\nLoading model with attention={attention_impl}...")

    # Configure SDPA backend if using sdpa variant
    if attention_impl.startswith("sdpa"):
        configure_sdpa_backend(attention_impl)

    # Map extended attention names to base implementation for HuggingFace
    base_impl = attention_impl
    if attention_impl.startswith("sdpa_"):
        base_impl = "sdpa"

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        padding_side='left',
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        attn_implementation=base_impl,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()

    print(f"  Model loaded: {model.dtype}, {model.device}")
    print(f"  Attention: {getattr(model.config, '_attn_implementation', 'unknown')}")

    # Configure static KV cache if requested
    if static_cache:
        print(f"  Static KV cache: enabled (max_len={max_cache_len})")
        # Set cache implementation to static - this pre-allocates the KV cache
        model.generation_config.cache_implementation = "static"
        # Note: The actual cache size is determined by max_new_tokens + prompt_len
        # during generate(), but we configure the model to use static cache
    else:
        print(f"  KV cache: dynamic (default)")

    return model, tokenizer


def check_sdpa_backends():
    """Check which SDPA backends are available and print status."""
    print("\n  Checking SDPA backend availability...")

    try:
        from torch.nn.functional import scaled_dot_product_attention
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backends = {
            "Flash": SDPBackend.FLASH_ATTENTION,
            "cuDNN": SDPBackend.CUDNN_ATTENTION,
            "Efficient": SDPBackend.EFFICIENT_ATTENTION,
            "Math": SDPBackend.MATH,
        }

        # Test 1: Prefill scenario (all same length)
        print("    Prefill scenario (q=k=v=32 tokens):")
        q = torch.randn(1, 8, 32, 64, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1, 8, 32, 64, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(1, 8, 32, 64, device='cuda', dtype=torch.bfloat16)

        for name, backend in backends.items():
            try:
                with sdpa_kernel([backend]):
                    _ = scaled_dot_product_attention(q, k, v, is_causal=True)
                print(f"      {name}: ✓")
            except RuntimeError as e:
                print(f"      {name}: ✗ ({str(e)[:40]}...)")

        # Test 2: Decode scenario (q=1, k/v=N - autoregressive generation)
        print("    Decode scenario (q=1, k=v=100 tokens):")
        q = torch.randn(1, 8, 1, 64, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1, 8, 100, 64, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(1, 8, 100, 64, device='cuda', dtype=torch.bfloat16)

        for name, backend in backends.items():
            try:
                with sdpa_kernel([backend]):
                    _ = scaled_dot_product_attention(q, k, v, is_causal=False)
                print(f"      {name}: ✓")
            except RuntimeError as e:
                print(f"      {name}: ✗ ({str(e)[:40]}...)")

    except Exception as e:
        print(f"    Could not check backends: {e}")


def benchmark_flash_attn_kernels():
    """
    Benchmark flash-attn library kernels directly (FA3 on H100).
    Tests FP16 and BF16.
    """
    print("\n" + "=" * 70)
    print("Flash-Attn Kernel Benchmarks (FA3 on H100)")
    print("=" * 70)

    try:
        from flash_attn import flash_attn_func
        print(f"\n  flash-attn library: ✓ available")
    except ImportError:
        print(f"\n  flash-attn library: ✗ not installed")
        return {}

    # Test configuration matching our model
    batch_size = 16
    seq_len = 2048  # Typical generation length
    num_heads = 32
    head_dim = 64

    results = {}

    # Test FP16 and BF16
    dtypes_to_test = [
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
    ]

    print(f"\n  Config: batch={batch_size}, seq={seq_len}, heads={num_heads}, dim={head_dim}")

    for dtype_name, dtype in dtypes_to_test:
        print(f"\n  Testing {dtype_name}...")
        try:
            # flash_attn_func expects [batch, seq, heads, dim]
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype)

            # Warmup
            for _ in range(5):
                _ = flash_attn_func(q, k, v, causal=True)
            torch.cuda.synchronize()

            # Benchmark
            num_iters = 50
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_iters):
                _ = flash_attn_func(q, k, v, causal=True)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            avg_time_ms = (elapsed / num_iters) * 1000
            # Calculate TFLOPS: 4 * batch * seq^2 * heads * dim (for causal, roughly half)
            flops = 2 * batch_size * seq_len * seq_len * num_heads * head_dim
            tflops = (flops / (avg_time_ms / 1000)) / 1e12

            results[dtype_name] = {
                'time_ms': avg_time_ms,
                'tflops': tflops,
            }
            print(f"    {dtype_name}: {avg_time_ms:.3f} ms, {tflops:.2f} TFLOPS")

        except Exception as e:
            print(f"    {dtype_name}: FAILED - {e}")
            results[dtype_name] = None

    return results


def run_inference_benchmark(
    model, tokenizer, smiles_list: list, batch_size: int,
    max_new_tokens: int = 4000, attention_impl: str = "sdpa",
    static_cache: bool = False,
):
    """Run inference and return timing stats.

    Note: PyTorch SDPA auto-selects the best backend based on hardware and config.
    Forcing specific backends can hang or fail - we let SDPA auto-select.

    Args:
        model: The model to run inference with
        tokenizer: Tokenizer for the model
        smiles_list: List of SMILES strings to generate conformers for
        batch_size: Batch size for inference
        max_new_tokens: Maximum new tokens to generate per sample
        attention_impl: Attention implementation name (for logging)
        static_cache: If True, use static KV cache with padded inputs
    """

    # Prepare prompts
    prompts = [f"[SMILES]{smi}[/SMILES]" for smi in smiles_list]

    # Get EOS token
    eos_ids = tokenizer.encode("[/CONFORMER]", add_special_tokens=False)
    eos_token_id = eos_ids if eos_ids else [tokenizer.eos_token_id]

    # Build generation config - note: cache_implementation is set on model.generation_config
    # if static_cache was enabled, so we don't override it here
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
    )

    # For static cache, we inherit cache_implementation from model.generation_config
    if static_cache and hasattr(model.generation_config, 'cache_implementation'):
        gen_config.cache_implementation = model.generation_config.cache_implementation
        print(f"  Using cache_implementation: {gen_config.cache_implementation}")

    total_tokens_generated = 0
    batch_times = []

    # Process in batches
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    # Tokenizer padding settings for static cache
    # pad_to_multiple_of helps reduce recompilations with static cache
    tokenize_kwargs = {
        "return_tensors": "pt",
        "padding": True,
    }
    if static_cache:
        # Pad to multiple of 64 for better tensor core utilization and fewer cache reallocations
        tokenize_kwargs["pad_to_multiple_of"] = 64

    # Warmup (1 batch)
    warmup_batch = prompts[:min(batch_size, len(prompts))]
    warmup_inputs = tokenizer(warmup_batch, **tokenize_kwargs)
    warmup_inputs = {k: v.to(model.device) for k, v in warmup_inputs.items()}
    with torch.inference_mode():
        _ = model.generate(
            **warmup_inputs,
            generation_config=gen_config,
            eos_token_id=eos_token_id,
        )
    torch.cuda.synchronize()

    # Actual benchmark
    print(f"  Running {num_batches} batches of size {batch_size}...")

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, **tokenize_kwargs)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]

        torch.cuda.synchronize()
        start = time.time()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                generation_config=gen_config,
                eos_token_id=eos_token_id,
            )

        torch.cuda.synchronize()
        elapsed = time.time() - start

        output_len = outputs.shape[1]
        tokens_generated = (output_len - input_len) * len(batch)
        total_tokens_generated += tokens_generated
        batch_times.append(elapsed)

        print(f"    Batch {i//batch_size + 1}/{num_batches}: {elapsed:.2f}s, "
              f"{tokens_generated} tokens ({tokens_generated/elapsed:.1f} tok/s)")

    total_time = sum(batch_times)

    return {
        'total_time': total_time,
        'total_tokens': total_tokens_generated,
        'tokens_per_sec': total_tokens_generated / total_time,
        'samples_per_sec': len(prompts) / total_time,
        'batch_times': batch_times,
        'num_samples': len(prompts),
    }


def run_benchmark(config: dict):
    """
    Run the actual benchmark. This function is called either directly or via submitit.

    Args:
        config: Dictionary with benchmark configuration
    """
    from molgen3D.config.paths import get_ckpt, get_tokenizer_path, get_model_tokenizer

    model_path = get_ckpt(config['model'], config['model_step'])

    # Auto-detect tokenizer from model config if not explicitly provided
    tokenizer_name = config.get('tokenizer')
    if not tokenizer_name:
        tokenizer_name = get_model_tokenizer(config['model'])
        if not tokenizer_name:
            raise ValueError(
                f"No tokenizer specified and model '{config['model']}' has no default tokenizer. "
                "Use --tokenizer to specify one."
            )
    tokenizer_path = get_tokenizer_path(tokenizer_name)

    # Get static cache setting
    static_cache = config.get('static_cache', False)
    max_cache_len = config.get('max_cache_len', 2048)

    print("=" * 70)
    print("3DMolGen Inference Attention Benchmark")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Model: {config['model']} ({config['model_step']})")
    print(f"  Model path: {model_path}")
    print(f"  Tokenizer: {tokenizer_name} (auto-detected)" if not config.get('tokenizer') else f"  Tokenizer: {tokenizer_name}")
    print(f"  SMILES source: {config.get('smiles_json', DEFAULT_SMILES_JSON)}")
    print(f"  Num samples: {config['num_samples']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Max new tokens: {config['max_new_tokens']}")
    print(f"  Static KV cache: {static_cache}")
    if static_cache:
        print(f"  Max cache length: {max_cache_len}")
    print(f"  Attention impls to test: {config['attention']}")

    # GPU info
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute capability: {cap[0]}.{cap[1]}")

        # Check SDPA backend availability
        check_sdpa_backends()

        # Run flash-attn kernel benchmarks (FA3 on H100)
        kernel_results = benchmark_flash_attn_kernels()
    else:
        print("\nWARNING: No CUDA GPU available!")
        return None

    # Load SMILES from JSON file or use fallback
    smiles_json = config.get('smiles_json', DEFAULT_SMILES_JSON)
    resolved_smiles_path = resolve_nfs_path(smiles_json)
    print(f"\nLoading SMILES from: {resolved_smiles_path}")
    smiles_list = load_smiles_from_json(smiles_json, config['num_samples'])
    print(f"Test molecules: {len(smiles_list)} samples")

    # Benchmark each attention implementation
    results = {}

    for attn_impl in config['attention']:
        print(f"\n{'=' * 70}")
        print(f"Testing: {attn_impl}")
        print("=" * 70)

        try:
            model, tokenizer = load_model(
                model_path, tokenizer_path, attn_impl,
                static_cache=static_cache,
                max_cache_len=max_cache_len,
            )

            result = run_inference_benchmark(
                model, tokenizer, smiles_list,
                batch_size=config['batch_size'],
                max_new_tokens=config['max_new_tokens'],
                attention_impl=attn_impl,
                static_cache=static_cache,
            )
            results[attn_impl] = result

            print(f"\n  Results for {attn_impl}:")
            print(f"    Total time: {result['total_time']:.2f}s")
            print(f"    Tokens generated: {result['total_tokens']}")
            print(f"    Throughput: {result['tokens_per_sec']:.1f} tokens/sec")
            print(f"    Samples/sec: {result['samples_per_sec']:.2f}")

            # Cleanup
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[attn_impl] = None

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    valid_results = {k: v for k, v in results.items() if v is not None}

    if not valid_results:
        print("No successful benchmarks!")
        return results

    # Find baseline (eager if available, otherwise first result)
    baseline_key = "eager" if "eager" in valid_results else list(valid_results.keys())[0]
    baseline_tps = valid_results[baseline_key]['tokens_per_sec']

    header = f"{'Implementation':<20} {'Time (s)':<12} {'Tokens/sec':<15} {'Speedup':<10}"
    print(f"\n{header}")
    print("-" * 60)

    max_speedup = max(r['tokens_per_sec'] / baseline_tps for r in valid_results.values())
    for impl, result in sorted(valid_results.items(), key=lambda x: -x[1]['tokens_per_sec']):
        speedup = result['tokens_per_sec'] / baseline_tps
        marker = " 🏆" if speedup == max_speedup else ""
        print(f"{impl:<20} {result['total_time']:<12.2f} "
              f"{result['tokens_per_sec']:<15.1f} {speedup:<10.2f}x{marker}")

    # Recommendation
    best_impl = max(valid_results.keys(), key=lambda k: valid_results[k]['tokens_per_sec'])
    print(f"\n✅ Recommendation: Use '{best_impl}' for best performance")

    # Memory usage
    if torch.cuda.is_available():
        print("\nGPU Memory (after cleanup):")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    return results


def main():
    import submitit

    parser = argparse.ArgumentParser(
        description="Benchmark inference with different attention implementations"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model alias (e.g., m600_qwen, m380_conf_v2)"
    )
    parser.add_argument(
        "--model-step", type=str, default="2e",
        help="Model step (default: 2e)"
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="Tokenizer name (e.g., qwen3_0.6b_custom, llama3_chem_v1). "
             "If not specified, auto-detected from model config."
    )
    parser.add_argument(
        "--num-samples", type=int, default=64,
        help="Number of molecules to generate (default: 64)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=1000,
        help="Max new tokens per generation (default: 1000)"
    )
    # Available attention implementations
    ATTENTION_CHOICES = [
        # HuggingFace transformers attn_implementation options
        "sdpa",               # PyTorch SDPA (auto-selects best backend: cuDNN on H100)
        "eager",              # Naive attention (baseline)
        "flash_attention_2",  # FlashAttention-2/3 via flash-attn package
        # SDPA with specific backend forced (via torch.backends settings)
        "sdpa_flash",         # SDPA with only Flash backend enabled
        "sdpa_cudnn",         # SDPA with only cuDNN backend enabled
        "sdpa_efficient",     # SDPA with only Memory-efficient backend enabled
        "sdpa_math",          # SDPA with only Math backend enabled (slowest)
    ]
    parser.add_argument(
        "--attention", type=str, nargs='+',
        default=["sdpa", "sdpa_flash", "sdpa_cudnn", "sdpa_efficient", "eager"],
        choices=ATTENTION_CHOICES,
        help="Attention implementations to test. sdpa_* variants force specific SDPA backends."
    )
    parser.add_argument(
        "--smiles-json", type=str, default=DEFAULT_SMILES_JSON,
        help=f"Path to JSON file with SMILES (default: {DEFAULT_SMILES_JSON})"
    )
    parser.add_argument(
        "--static-cache", action="store_true",
        help="Enable static KV cache (pre-allocated). May fix SDPA backend issues with dynamic shapes."
    )
    parser.add_argument(
        "--max-cache-len", type=int, default=2048,
        help="Maximum cache length for static KV cache (default: 2048)"
    )
    parser.add_argument(
        "--device", type=str, default="h100",
        choices=["h100", "a100", "local"],
        help="Device to run on: h100/a100 submit via slurm, local runs directly (default: h100)"
    )
    args = parser.parse_args()

    # Build config dict for the benchmark
    config = {
        'model': args.model,
        'model_step': args.model_step,
        'tokenizer': args.tokenizer,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'max_new_tokens': args.max_new_tokens,
        'attention': args.attention,
        'smiles_json': args.smiles_json,
        'static_cache': args.static_cache,
        'max_cache_len': args.max_cache_len,
    }

    if args.device == "local":
        # Run directly on current node
        print("Running locally...")
        run_benchmark(config)
    else:
        # Submit via slurm
        partition = args.device  # h100 or a100

        executor = submitit.AutoExecutor(
            folder="~/slurm_jobs/diagnostics/job_%j"
        )
        executor.update_parameters(
            name=f"attn_bench_{args.model}",
            timeout_min=60,
            gpus_per_node=1,
            nodes=1,
            mem_gb=80,
            cpus_per_task=4,
            slurm_additional_parameters={"partition": partition},
        )

        job = executor.submit(run_benchmark, config)
        print(f"Submitted job to {partition} partition")
        print(f"Job ID: {job.job_id}")
        print(f"Output will be in: ~/slurm_jobs/diagnostics/job_{job.job_id}/")
        print(f"\nTo check status: squeue -j {job.job_id}")
        print(f"To view output: cat ~/slurm_jobs/diagnostics/job_{job.job_id}/{job.job_id}_0_log.out")


if __name__ == "__main__":
    main()
