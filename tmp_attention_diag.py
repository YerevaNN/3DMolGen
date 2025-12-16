#!/usr/bin/env python3
"""
Diagnostic script for 3DMolGen attention backends
Run on H100 in 3dmolgen environment
"""
import torch, time
print("=" * 60)
print("3DMolGen Attention Diagnostic")
print("=" * 60)
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Capability: {torch.cuda.get_device_capability(0)}")
print(f"\ncuDNN available: {torch.backends.cudnn.is_available()}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
print("\n--- SDPA Backend Availability ---")
print(f"Flash Attention available: {torch.backends.cuda.is_flash_attention_available()}")
print(f"cuDNN SDPA enabled: {torch.backends.cuda.cudnn_sdp_enabled()}")
print("\n--- cuDNN Attention Check ---")
batch_size = 4
num_heads = 16
seq_len = 512
head_dim = 64
device = "cuda"
dtype = torch.bfloat16
if torch.cuda.is_available():
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    from torch.nn.attention import SDPBackend
    from torch.nn.functional import scaled_dot_product_attention
    torch.backends.cuda.enable_cudnn_sdp(True)
    print(f"After enabling - cuDNN SDPA enabled: {torch.backends.cuda.cudnn_sdp_enabled()}")
    def benchmark_sdpa(name, enable_flash=True, enable_math=True, enable_mem_efficient=True, enable_cudnn=True):
        for _ in range(5):
            with torch.nn.attention.sdpa_kernel([
                b for b, e in [
                    (SDPBackend.FLASH_ATTENTION, enable_flash),
                    (SDPBackend.MATH, enable_math),
                    (SDPBackend.EFFICIENT_ATTENTION, enable_mem_efficient),
                    (SDPBackend.CUDNN_ATTENTION, enable_cudnn)
                ] if e]):
                _ = scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            with torch.nn.attention.sdpa_kernel([
                b for b, e in [
                    (SDPBackend.FLASH_ATTENTION, enable_flash),
                    (SDPBackend.MATH, enable_math),
                    (SDPBackend.EFFICIENT_ATTENTION, enable_mem_efficient),
                    (SDPBackend.CUDNN_ATTENTION, enable_cudnn)
                ] if e]):
                _ = scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"{name}: {elapsed*20:.2f} ms per call")
    try:
        benchmark_sdpa("All backends (auto)", True, True, True, True)
    except Exception as e:
        print("All backends failed:", e)
    try:
        benchmark_sdpa("cuDNN only", False, False, False, True)
    except Exception as e:
        print("cuDNN only failed:", e)
    try:
        benchmark_sdpa("Flash only", True, False, False, False)
    except Exception as e:
        print("Flash only failed:", e)
    try:
        benchmark_sdpa("Memory-efficient only", False, False, True, False)
    except Exception as e:
        print("Memory-efficient only failed:", e)
