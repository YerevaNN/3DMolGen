# Inference Attention Investigation

**Status**: Complete | **Last Updated**: 2025-12-17

## TL;DR - Recommendations

| GPU | Batch Size | Best Attention | Notes |
|-----|------------|----------------|-------|
| **A100** | Large (≥128) | `flash_attention_2` | 5% faster than SDPA |
| **A100** | Small (≤16) | `sdpa` | 20% faster than FA2 |
| **H100** | Large (≥128) | TBD... | TBD... |
| **Consumer (GTX 30xx)** | Any | `sdpa` | FA2 not recommended |

**With logit processor**: Always use `--kv-cache dynamic` (static hurts 2-3x) (_Cant figure this one out on how to get static cache to work with logit processor_)

---

## Key Findings

### 1. Batch Size Determines Winner (A100)

**New finding** (2025-12-17): flash_attention_2 performance depends heavily on batch size.

#### A100 - Small Batch (samples=64, batch=16)
*Job 422535: `~/slurm_jobs/diagnostics/flash_attn_build_422535.out`*

| Implementation | Tokens/sec | Speedup |
|----------------|------------|---------|
| **sdpa** | 416.8 | **1.03x** |
| sdpa_efficient | 412.2 | 1.02x |
| eager | 404.7 | 1.00x |
| sdpa_cudnn | 388.2 | 0.96x |
| flash_attention_2 | 327.7 | 0.81x |

#### A100 - Large Batch (samples=1024, batch=128)
*Job 422667: `~/slurm_jobs/diagnostics/flash_attn_build_422667.out`*

| Implementation | Tokens/sec | Speedup |
|----------------|------------|---------|
| **flash_attention_2** | 1572.4 | **1.05x** |
| sdpa | 1516.2 | 1.01x |
| sdpa_efficient | 1509.8 | 1.01x |
| eager | 1500.0 | 1.00x |
| sdpa_cudnn | 1463.1 | 0.98x |

**Conclusion**: At large batches, flash_attention_2 wins. At small batches, SDPA wins.

**Memory limit**: Batch size 192 causes OOM on A100 40GB.

### 3. Static KV Cache

- **Without logit processor**: 1.8x speedup (enables CUDA graphs)
- **With logit processor**: 2-3x **slowdown** (graphs broken by per-token Python)

**Rule**: If using constrained generation (logit processor), always use dynamic cache.

### 4. CPU-Side Optimizations

| Optimization | Effect | When to Use |
|--------------|--------|-------------|
| `--parallel-templates` | Not measured | Template building is fast; unlikely to be bottleneck |
| `--cache-tokenizer` | No benefit in smoke tests | Only for multi-gen per molecule |

---

## Quick Reference

### Production Config (A100, Large Batch)
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
```

### Production Config (H100 or Small Batch)
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
```

### With Logit Processor
```bash
python scripts/logit_processor/run_logit_processor_smoke.py \
    --kv-cache dynamic
```

---

## Environment

```
PyTorch: 2.9.1+cu128
CUDA: 12.8
flash-attn: 2.8.3 (from prebuilt wheel)
```

**Wheel location**: `/auto/home/aram.dovlatyan/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl`

---

## Detailed Experiment Log



<details>
<summary>A100 Experiments (2025-12-17)</summary>

### Small Batch (Job 422535)
- samples=64, batch=16
- SDPA wins (416.8 tok/s vs FA2 327.7 tok/s)

### Large Batch (Job 422667)
- samples=1024, batch=128
- FA2 wins (1572.4 tok/s vs SDPA 1516.2 tok/s)

### Memory Limits
- Batch 192: OOM on A100 40GB

</details>

---

## Why These Results?

1. **Batch size effect**: flash-attn has higher per-call overhead but better throughput at scale. Small batches = overhead dominates. Large batches = throughput wins.

2. **H100 vs A100**: PyTorch SDPA has Hopper-specific optimizations (TMA, WGMMA) that flash-attn 2.8.3 lacks. On A100 (Ampere), both are more comparable.

3. **Static cache + LP conflict**: Static cache enables CUDA graphs, but logit processor forces Python execution every token, breaking graphs and leaving only pre-allocation overhead.

---

## References

1. [FlashAttention-2 Paper (ICLR 2024)](https://arxiv.org/abs/2307.08691)
2. [FlashAttention-3 Blog](https://tridao.me/blog/2024/flash3/)
3. [PyTorch SDPA Docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
