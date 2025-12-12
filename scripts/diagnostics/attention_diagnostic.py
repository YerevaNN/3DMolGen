#!/usr/bin/env python3
"""
Diagnostic script for 3DMolGen attention backends.
Run on H100 in 3dmolgen environment to determine optimal attention implementation.

Usage:
    python scripts/diagnostics/attention_diagnostic.py

    # Or submit to H100:
    sbatch --partition=h100 --gpus=1 --mem=40G --time=00:30:00 \
           --wrap="python scripts/diagnostics/attention_diagnostic.py"
"""

import torch
import time
import sys

def print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def get_environment_info():
    """Gather environment information."""
    print_section("Environment Information")

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            cap = torch.cuda.get_device_capability(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Compute capability: {cap[0]}.{cap[1]}")
            print(f"  Total memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  SM count: {props.multi_processor_count}")

    # cuDNN info
    print(f"\ncuDNN available: {torch.backends.cudnn.is_available()}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")


def get_sdpa_backend_info():
    """Check SDPA backend availability."""
    print_section("SDPA Backend Availability")

    # Check flash attention availability (PyTorch built-in)
    if hasattr(torch.backends.cuda, 'is_flash_attention_available'):
        print(f"PyTorch Flash Attention available: {torch.backends.cuda.is_flash_attention_available()}")
    else:
        print("PyTorch Flash Attention check: Not available in this PyTorch version")

    # cuDNN SDPA
    print(f"cuDNN SDPA enabled: {torch.backends.cuda.cudnn_sdp_enabled()}")

    # Check flash_attn package
    try:
        import flash_attn as fa
        print(f"flash_attn package version: {fa.__version__}")
        return True
    except ImportError as e:
        print(f"flash_attn package: Not installed or import error ({e})")
        return False
    except Exception as e:
        print(f"flash_attn package: Error ({e})")
        return False


def benchmark_sdpa_backends(batch_size: int = 4, num_heads: int = 16,
                            seq_len: int = 2048, head_dim: int = 64,
                            num_iterations: int = 100):
    """Benchmark different SDPA backends."""
    print_section(f"SDPA Backend Benchmarks (batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim})")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks")
        return {}

    device = "cuda"
    dtype = torch.bfloat16

    # Create test tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention

    results = {}

    # Enable cuDNN explicitly
    torch.backends.cuda.enable_cudnn_sdp(True)
    print(f"\ncuDNN SDPA enabled: {torch.backends.cuda.cudnn_sdp_enabled()}")

    backend_configs = [
        ("All backends (auto)", [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH,
                                  SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]),
        ("cuDNN only", [SDPBackend.CUDNN_ATTENTION]),
        ("Flash only (built-in)", [SDPBackend.FLASH_ATTENTION]),
        ("Memory-efficient only", [SDPBackend.EFFICIENT_ATTENTION]),
        ("Math only (baseline)", [SDPBackend.MATH]),
    ]

    for name, backends in backend_configs:
        try:
            # Warmup
            for _ in range(10):
                with sdpa_kernel(backends):
                    _ = scaled_dot_product_attention(q, k, v, is_causal=True)

            torch.cuda.synchronize()
            start = time.time()

            for _ in range(num_iterations):
                with sdpa_kernel(backends):
                    _ = scaled_dot_product_attention(q, k, v, is_causal=True)

            torch.cuda.synchronize()
            elapsed = time.time() - start
            avg_ms = (elapsed / num_iterations) * 1000
            results[name] = avg_ms
            print(f"{name:30s}: {avg_ms:.3f} ms/iter")

        except Exception as e:
            print(f"{name:30s}: FAILED ({e})")
            results[name] = None

    return results


def benchmark_model_inference(model_alias: str = "m600_qwen",
                              model_step: str = "2e",
                              tokenizer_name: str | None = None,
                              test_prompt: str = "[SMILES]CCO[/SMILES]"):
    """Benchmark actual model inference with different attention implementations.

    Tokenizer is auto-detected from model config if not specified.
    """
    print_section("Model Inference Benchmarks")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping model benchmarks")
        return {}

    try:
        from molgen3D.config.paths import get_ckpt, get_tokenizer_path as get_tok_path, get_model_tokenizer
        model_path = get_ckpt(model_alias, model_step)

        # Auto-detect tokenizer from model config if not specified
        if tokenizer_name is None:
            tokenizer_name = get_model_tokenizer(model_alias)
            if not tokenizer_name:
                raise ValueError(
                    f"No tokenizer specified and model '{model_alias}' has no default tokenizer."
                )
            print(f"Model: {model_alias} ({model_step}) -> {model_path}")
            print(f"Tokenizer: {tokenizer_name} (auto-detected)")
        else:
            print(f"Model: {model_alias} ({model_step}) -> {model_path}")
            print(f"Tokenizer: {tokenizer_name}")

        tokenizer_path = get_tok_path(tokenizer_name)
    except ImportError:
        print("Could not import molgen3D config, skipping model benchmarks")
        return {}

    from transformers import AutoTokenizer, AutoModelForCausalLM

    results = {}
    attention_impls = ["sdpa", "eager"]

    # Check if flash_attn is available
    try:
        import flash_attn
        attention_impls.insert(0, "flash_attention_2")
    except ImportError:
        print("flash_attn not available, skipping flash_attention_2")

    for attn_impl in attention_impls:
        print(f"\nTesting {attn_impl}...")

        try:
            # Enable cuDNN for SDPA
            if attn_impl == "sdpa":
                torch.backends.cuda.enable_cudnn_sdp(True)

            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                padding_side='left',
            )
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                device_map="cuda",
                trust_remote_code=True,
            ).eval()

            inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
            # Use tokenizer's eos_token_id as fallback if [/CONFORMER] not in vocab
            eos_ids = tokenizer.encode("[/CONFORMER]", add_special_tokens=False)
            eos_token_id = eos_ids if eos_ids else [tokenizer.eos_token_id]
            print(f"  EOS token IDs: {eos_token_id}")

            # Warmup
            with torch.inference_mode():
                for _ in range(3):
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=500,
                        eos_token_id=eos_token_id,
                        do_sample=False,
                        use_cache=True,
                    )

            torch.cuda.synchronize()

            # Benchmark
            num_runs = 10
            times = []
            tokens_generated = []

            with torch.inference_mode():
                for _ in range(num_runs):
                    torch.cuda.synchronize()
                    start = time.time()

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=500,
                        eos_token_id=eos_token_id,
                        do_sample=False,
                        use_cache=True,
                    )

                    torch.cuda.synchronize()
                    elapsed = time.time() - start
                    times.append(elapsed)
                    tokens_generated.append(outputs.shape[1] - inputs['input_ids'].shape[1])

            avg_time = sum(times) / len(times)
            avg_tokens = sum(tokens_generated) / len(tokens_generated)
            tokens_per_sec = avg_tokens / avg_time

            results[attn_impl] = {
                'avg_time': avg_time,
                'tokens_per_sec': tokens_per_sec,
                'avg_tokens': avg_tokens,
            }

            print(f"  Average time: {avg_time:.3f}s")
            print(f"  Tokens generated: {avg_tokens:.0f}")
            print(f"  Tokens/sec: {tokens_per_sec:.1f}")
            print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[attn_impl] = None

    return results


def print_recommendations(sdpa_results: dict, model_results: dict):
    """Print recommendations based on benchmark results."""
    print_section("Recommendations")

    # Analyze SDPA results
    if sdpa_results:
        valid_results = {k: v for k, v in sdpa_results.items() if v is not None}
        if valid_results:
            best_sdpa = min(valid_results, key=valid_results.get)
            print(f"\nBest SDPA backend: {best_sdpa} ({valid_results[best_sdpa]:.3f} ms)")

            if "Math only (baseline)" in valid_results:
                baseline = valid_results["Math only (baseline)"]
                print("\nSpeedups vs Math baseline:")
                for name, time_ms in valid_results.items():
                    if time_ms and time_ms > 0:
                        speedup = baseline / time_ms
                        print(f"  {name}: {speedup:.2f}x")

    # Analyze model results
    if model_results:
        valid_model = {k: v for k, v in model_results.items() if v is not None}
        if valid_model:
            best_impl = max(valid_model, key=lambda x: valid_model[x]['tokens_per_sec'])
            print(f"\nBest attention implementation: {best_impl}")
            print(f"  Tokens/sec: {valid_model[best_impl]['tokens_per_sec']:.1f}")

            if "eager" in valid_model:
                baseline_tps = valid_model["eager"]['tokens_per_sec']
                print("\nSpeedups vs eager baseline:")
                for name, data in valid_model.items():
                    speedup = data['tokens_per_sec'] / baseline_tps
                    print(f"  {name}: {speedup:.2f}x ({data['tokens_per_sec']:.1f} tok/s)")

    # Final recommendation
    print("\n" + "-" * 60)
    print("RECOMMENDATION:")
    print("-" * 60)

    # Check compute capability
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 9:  # Hopper (H100)
            print("""
For H100 (Hopper, compute capability 9.0+):

1. BEST OPTION: Use attn_implementation='sdpa' with cuDNN backend
   - No external dependencies needed
   - PyTorch 2.5+ includes cuDNN SDPA optimized for H100
   - Should match or exceed FlashAttention-2 performance

2. Enable cuDNN SDPA explicitly:
   torch.backends.cuda.enable_cudnn_sdp(True)

3. In inference.py, change default from 'flash_attention_2' to 'sdpa'

4. FlashAttention-3 (if needed later):
   - Install from hopper/ directory of flash-attention repo
   - Requires CUDA 12.3+, recommend CUDA 12.8
   - Currently beta, may have stability issues
""")
        elif cap[0] >= 8:  # Ampere (A100, RTX 30xx)
            print("""
For Ampere GPUs (A100, RTX 30xx, compute capability 8.x):

1. BEST OPTION: Use attn_implementation='flash_attention_2'
   - FlashAttention-2 achieves ~70% of theoretical max FLOPS on Ampere
   - Requires flash_attn package

2. FALLBACK: Use attn_implementation='sdpa'
   - No external dependencies
   - Will use PyTorch's built-in optimizations
""")
        else:
            print("Older GPU detected. Use 'sdpa' or 'eager' attention.")
    else:
        print("No GPU detected. Cannot make hardware-specific recommendations.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="3DMolGen Attention Diagnostic")
    parser.add_argument("--model", type=str, default="qwen3_06b_pre",
                        help="Model alias (default: qwen3_06b_pre for latest Qwen, use m380_conf_v2 for Llama)")
    parser.add_argument("--model-step", type=str, default=None,
                        help="Model step (default: auto-select last available step)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer name. If not specified, auto-detected from model config.")
    parser.add_argument("--skip-model-bench", action="store_true",
                        help="Skip model inference benchmarks (useful for quick SDPA kernel tests)")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Sequence length for SDPA benchmarks (default: 2048)")
    args = parser.parse_args()

    # Display tokenizer info
    tokenizer_display = args.tokenizer if args.tokenizer else "(auto-detect from model)"

    print("3DMolGen Attention Diagnostic")
    print("Run this on H100 for accurate recommendations")
    print(f"\nConfiguration:")
    print(f"  Model: {args.model} ({args.model_step or 'auto'})")
    print(f"  Tokenizer: {tokenizer_display}")
    print(f"  Seq length: {args.seq_len}")

    # Get environment info
    get_environment_info()

    # Check SDPA backends
    get_sdpa_backend_info()

    # Benchmark SDPA backends
    sdpa_results = benchmark_sdpa_backends(
        batch_size=4,
        num_heads=16,  # Qwen3 0.6B has 16 heads
        seq_len=args.seq_len,
        head_dim=64,   # Qwen3 0.6B head dim
    )

    # Benchmark actual model inference
    model_results = {}
    if not args.skip_model_bench:
        model_results = benchmark_model_inference(
            model_alias=args.model,
            model_step=args.model_step,  # None means auto-select
            tokenizer_name=args.tokenizer,  # None means auto-detect
        )
    else:
        print("\n[Skipping model inference benchmarks]")

    # Print recommendations
    print_recommendations(sdpa_results, model_results)


if __name__ == "__main__":
    main()
