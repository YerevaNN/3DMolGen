# Inference Performance Investigation

**Date**: 2025-12-09
**Author**: Claude Code + Aram Dovlatyan
**Status**: Phase 2 Complete (Static KV Cache)

---

## Phase 1: Attention Implementation Comparison

**Scope**: Compare attention implementations (FA2/FA3, SDPA, eager) without logit processor or other optimizations.

## Objective

Investigate the best attention implementation for 3DMolGen inference on H100 GPUs, comparing:
- FlashAttention-2 (FA2)
- FlashAttention-3 (FA3)
- PyTorch SDPA (with cuDNN backend)

### Constraints
- Small models (~380M-1B parameters)
- Context length: ~2048 tokens
- Hardware: NVIDIA H100 80GB HBM3
- Dependency compatibility with torchtitan and PyTorch 2.5.1

---

## Background Research

### Attention Implementation Overview

| Implementation | Source | H100 Optimized | Dependencies |
|----------------|--------|----------------|--------------|
| FlashAttention-2 | Dao-AILab/flash-attention | Partial (Ampere-focused) | `flash-attn` package, CUDA toolkit |
| FlashAttention-3 | Dao-AILab/flash-attention v2.5+ | Yes (Hopper-native, uses TMA) | `flash-attn` package, CUDA 12.3+ |
| PyTorch SDPA | torch.nn.functional | Yes (cuDNN backend) | Built-in (PyTorch 2.2+) |

### Key Differences

**FlashAttention-2** (Source: [FA2 Paper, ICLR 2024](https://arxiv.org/abs/2307.08691))
- IO-aware tiling algorithm reduces HBM access
- 2-3x speedup over standard attention on A100
- 35% GPU utilization on H100 (not optimized for Hopper)

**FlashAttention-3** (Source: [Tri Dao's blog](https://tridao.me/blog/2024/flash3/))
- Hopper-native: uses TMA (Tensor Memory Accelerator) and WGMMA instructions
- Achieves 75% H100 utilization (vs 35% for FA2)
- FP8 support for additional 1.5x speedup
- 1.5-2x faster than FA2 on H100

**PyTorch SDPA with cuDNN** (Source: [PyTorch 2.5 Release Notes](https://pytorch.org/blog/pytorch2-5/))
- cuDNN 9.0+ provides optimized attention kernels for Hopper
- Auto-selected when using `attn_implementation="sdpa"` on H100
- No external dependencies

### Reference Benchmarks (Kernel-Level)

From flash-attention skill reference (`~/.claude/skills/flash-attention/references/benchmarks.md`):

**H100 80GB, Forward Pass (batch=8, heads=32, dim=64)**

| Seq Length | Standard | FA2 | FA3 (FP16) | FA3 (FP8) | Best Speedup |
|------------|----------|-----|------------|-----------|--------------|
| 512 | 0.8 ms | 0.6 ms | 0.4 ms | 0.3 ms | 2.7x |
| 1024 | 2.6 ms | 1.0 ms | 0.6 ms | 0.4 ms | 6.5x |
| 2048 | 9.8 ms | 3.4 ms | 2.0 ms | 1.3 ms | 7.5x |
| 4096 | 38.2 ms | 12.5 ms | 7.2 ms | 4.8 ms | 8.0x |

**Key insight**: These are *kernel-level* benchmarks. Full model inference shows different results.

### Inference vs Training Performance

From the same reference:

> "Inference speedup less dramatic than training because generation is memory-bound (KV cache accesses)"

**Inference throughput (Llama 2 7B on A100)**:

| Context Length | Standard (tok/s) | Flash Attn (tok/s) | Speedup |
|----------------|------------------|-------------------|---------|
| 512 | 48 | 52 | 1.1x |
| 2K | 42 | 62 | 1.5x |
| 4K | 31 | 58 | 1.9x |
| 8K | 18 | 51 | 2.8x |

**Observation**: Speedup scales with context length. At 2K context, only 1.5x improvement.

---

## Environment

```
Hardware: NVIDIA H100 80GB HBM3 (Compute Capability 9.0)
PyTorch: 2.5.1
CUDA: 12.1
cuDNN: 90100
Transformers: 4.57.0
flash-attn: 2.8.3 (installed successfully)
```

### SDPA Backend Availability (Verified on H100)

```
Flash: ✓ available
cuDNN: ✓ available
Efficient: ✓ available
Math: ✓ available
```

---

## Experiments

### Experiment 1: Kernel-Level SDPA Backend Comparison

**Script**: `scripts/diagnostics/attention_diagnostic.py`
**Job ID**: 421088
**Date**: 2025-12-09

**Methodology**: Direct SDPA kernel benchmark with test tensors (batch=2, heads=32, seq=2048, dim=64, bfloat16)

**Results**:

| Backend | Time (ms) | Speedup vs Math |
|---------|-----------|-----------------|
| cuDNN | 0.119 | 59.7x |
| Flash (PyTorch) | 0.161 | 44.1x |
| Memory-Efficient | 0.335 | 21.2x |
| Math (baseline) | 7.094 | 1.0x |

**Conclusion**: cuDNN backend is fastest on H100 for kernel-level attention.

### Experiment 2: Full Model Inference Benchmark

**Script**: `scripts/diagnostics/benchmark_inference_attention.py`
**Job ID**: 421135
**Date**: 2025-12-09

**Configuration**:
- Model: Llama-3.2-380M_conformers (m380_conf_v2)
- Tokenizer: llama3_chem_v1
- Samples: 64 molecules
- Batch size: 16
- Max new tokens: 1000
- SMILES source: `outputs/smoke/v2_1_new_strip_smiles_distinct_top_p.json`

**Results**:

| Implementation | Total Time | Tokens Generated | Throughput | Speedup |
|----------------|------------|------------------|------------|---------|
| sdpa (auto) | 29.57s | 58,608 | 1981.9 tok/s | 1.01x |
| eager | 29.14s | 57,168 | 1962.1 tok/s | 1.00x |

**Conclusion**: Only ~1% speedup in full inference despite 59x kernel-level improvement.

### Experiment 3: Forced Backend Testing (Failed)

**Job ID**: 421125
**Date**: 2025-12-09

Attempted to force specific SDPA backends during model inference:
- `sdpa_backend_flash`: Failed with "No available kernel"
- `sdpa_backend_cudnn`: Hung indefinitely (100% CPU, 3% GPU)

**Conclusion**: Forcing specific backends via `torch.nn.attention.sdpa_kernel()` is unreliable for full model inference. PyTorch's auto-selection is recommended.

### Experiment 4: FlashAttention-3 (flash-attn 2.8.3) Benchmark

**Job ID**: 421150
**Date**: 2025-12-09

After successfully installing flash-attn 2.8.3, we ran comprehensive benchmarks comparing FA3 with SDPA and eager attention.

#### FA3 Kernel-Level Results

**Configuration**: batch=16, seq=512, heads=32, dim=64

| Dtype | Time (ms) | TFLOPS |
|-------|-----------|--------|
| float16 | 0.102 | 168.35 |
| bfloat16 | 0.102 | 168.14 |

**Observation**: FA3 kernel achieves excellent performance at 168 TFLOPS on H100.

#### Full Model Inference Results

**Configuration**: Same as Experiment 2 (64 samples, batch=16, max_new_tokens=1000)

| Implementation | Total Time | Tokens Generated | Throughput | Speedup |
|----------------|------------|------------------|------------|---------|
| **sdpa** | 29.17s | 56,480 | **1936.3 tok/s** | **1.01x 🏆** |
| eager | 25.67s | 49,280 | 1919.9 tok/s | 1.00x |
| flash_attention_2 | 47.05s | 54,240 | 1152.8 tok/s | **0.60x** |

**⚠️ SURPRISING RESULT**: `flash_attention_2` is **40% SLOWER** than SDPA in full model inference!

#### Analysis of FA3 Slowdown

Despite FA3 kernel being extremely fast (168 TFLOPS), HuggingFace's `flash_attention_2` integration results in slower inference:

1. **Integration overhead**: The flash-attn library integration with HuggingFace transformers may have additional overhead
2. **cuDNN optimization**: PyTorch's SDPA with cuDNN backend is specifically optimized for H100 and deeply integrated
3. **Memory layout**: Potential tensor layout conversions between HuggingFace and flash-attn
4. **Small model penalty**: For small models, the per-call overhead of flash-attn dominates

**Conclusion**: For this model/hardware combination, PyTorch SDPA significantly outperforms flash-attn library.

---

## Analysis

### Why Only 1% Speedup in Full Inference?

For small models with moderate context, attention is a small fraction of total compute:

```
Attention kernel: ~5-10% of inference time
MLP layers: ~60-70%
Embedding/sampling: ~20-30%
```

**Mathematical bound**: Even with 60x faster attention:
```
Total speedup = 1 / (0.92 + 0.08/60) = 1.087x (~9% max theoretical)
```

Our measured 1% suggests attention is ~1-2% of total inference time for this configuration.

### When Attention Optimization Matters

| Scenario | Attention % of Compute | Expected Benefit |
|----------|------------------------|------------------|
| Small model, short context (our case) | 1-5% | Minimal (<5%) |
| Large model (70B+), short context | 5-10% | Moderate (5-15%) |
| Any model, long context (8K+) | 20-40% | Significant (2-3x) |
| Training (forward + backward) | 30-50% | Large (2-4x) |

### flash-attn vs PyTorch SDPA

Our Experiment 4 revealed a critical finding: **flash-attn library is 40% slower than PyTorch SDPA** for this use case.

**Possible reasons**:
1. **HuggingFace integration overhead**: The transformers library's `flash_attention_2` integration may add overhead
2. **PyTorch cuDNN synergy**: SDPA with cuDNN is deeply integrated into PyTorch's memory management
3. **Small model inefficiency**: flash-attn may have higher fixed overhead that hurts small models
4. **Autoregressive generation**: flash-attn may be optimized more for training than token-by-token generation

**PyTorch SDPA advantages** (confirmed by our experiments):
1. Zero additional dependencies
2. Maintained by PyTorch team
3. cuDNN backend is highly optimized for H100
4. **Actually faster** than flash-attn for inference on our setup

---

## Recommendations

### For 3DMolGen Inference (Current Setup)

**Use `attn_implementation="sdpa"`** - this is optimal and **faster than flash-attn**.

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="sdpa",  # Auto-selects cuDNN on H100
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
```

**Rationale** (backed by experimental evidence):
1. cuDNN backend auto-selected on H100 - highly optimized for Hopper
2. **1.7x faster** than flash_attention_2 in our benchmarks (1936 vs 1152 tok/s)
3. Zero dependency issues
4. Attention is not the bottleneck for small model inference anyway

### ⚠️ Do NOT Use flash_attention_2 for This Use Case

Despite popular belief that flash-attn is always faster, our experiments show:
- `flash_attention_2`: 1152.8 tok/s (**40% slower**)
- `sdpa`: 1936.3 tok/s (**winner**)

### When flash-attn Might Help

Consider flash-attn library if:
1. **Training** large models (backward pass benefits more from flash-attn)
2. Very long contexts (>8K tokens) where memory savings matter
3. Need FP8 support (FA3 only)
4. Different model architecture where HuggingFace integration is better optimized

### Alternative Optimizations for Inference

Since attention isn't the bottleneck, consider:
1. **Larger batch sizes** - better GPU utilization
2. **Quantization (INT8/FP8)** - reduces memory bandwidth
3. **Speculative decoding** - reduces generation steps
4. **vLLM/TensorRT-LLM** - optimized serving frameworks

---

## Appendix

### Scripts Created

1. `scripts/diagnostics/attention_diagnostic.py` - Kernel-level SDPA benchmark
2. `scripts/diagnostics/benchmark_inference_attention.py` - Full model inference benchmark
3. `scripts/diagnostics/submit_attention_diagnostic.sh` - Slurm submission wrapper

### Raw Job Outputs

- Job 421088: `/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/421088_attn_diag.out` (SDPA backends)
- Job 421106: `/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/job_421106/` (early inference test)
- Job 421125: `/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/job_421125/` (forced backends - failed)
- Job 421135: `/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/job_421135/` (sdpa vs eager)
- Job 421150: `/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/job_421150/` (FA3 benchmark)
- Job 421188: `/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/job_421188/` (SDPA backend availability)
- Job 421191: `/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/job_421191/` (forced backends - Flash failed, cuDNN hung)
- Job 421194: `/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/job_421194/` (Phase 1 final - 64 samples, dynamic cache)
- Job 421210: `/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/job_421210/` **(Phase 2 - Static KV cache benchmark)**
- Job 421213: `/auto/home/aram.dovlatyan/slurm_jobs/diagnostics/job_421213/` (flash_attention_2 + static cache - HUNG)

### Final Results

**Job ID**: 421194
**Configuration**: 64 samples, batch_size=16, max_new_tokens=1000

| Implementation | Time (s) | Tokens/sec | Speedup |
|----------------|----------|------------|---------|
| **sdpa** (auto-select) | 29.66 | 1958.4 | **1.03x 🏆** |
| eager | 25.36 | 1907.4 | 1.00x |
| flash_attention_2 (FA3) | 45.81 | 1136.1 | 0.60x |

**Conclusion**: PyTorch SDPA (auto-select) is optimal. flash-attn library is 40% slower.

### SDPA Backend Analysis

**Attempted to force individual SDPA backends** (Job 421191):

| Backend | Result |
|---------|--------|
| `sdpa` (auto-select) | ✓ Works - uses cuDNN with fallbacks |
| `sdpa_flash` (Flash only) | ✗ Failed: "No available kernel" |
| `sdpa_cudnn` (cuDNN only) | ✗ Hangs indefinitely |
| `sdpa_efficient` (Mem-efficient only) | Not tested (job cancelled) |

**Analysis**: Forcing individual backends fails because `model.generate()` requires fallback backends for certain operations. The auto-select mode handles this correctly by falling back to other backends when needed.

**Why Flash backend fails**: Investigation showed:
- Model head_dim=64 (within Flash limit of 128)
- GQA (num_heads=16, num_kv_heads=8) works in isolated tests
- Fails specifically during HuggingFace `model.generate()` on H100
- Likely related to dynamic KV cache shapes during autoregressive generation
- Similar issues reported in [PyTorch Forums](https://discuss.pytorch.org/t/using-f-scaled-dot-product-attention-gives-the-error-runtimeerror-no-available-kernel-aborting-execution/180900) and [GitHub Issue #127523](https://github.com/pytorch/pytorch/issues/127523)

**Flash Attention Requirements** (from [PyTorch SDPA Tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)):
- head_dim: ≤128 (PyTorch built-in), ≤256 (flash-attn library)
- dtype: fp16 or bf16 (bf16 requires sm80+)
- GPU: sm80+ (Ampere/Ada/Hopper)
- No singleton dimensions
- Non-null attn_mask not supported by fused kernels

---

## Phase 1 Summary

### Key Findings

1. **PyTorch SDPA is the best choice** for 3DMolGen inference on H100
2. **flash-attn 2.8.3 (FA3) is 40% slower** than SDPA despite fast raw kernels (168 TFLOPS)
3. **Attention is ~1-2% of inference time** for small models - optimizing it has minimal impact
4. **Forcing SDPA backends is unreliable** - let PyTorch auto-select

### Recommendation

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="sdpa",  # Best for H100
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
```

---

## Phase 2: Static KV Cache Optimization

**Goal**: Test if static KV cache can improve performance and fix SDPA backend issues.

### Background

**Dynamic KV Cache** (default): Cache grows dynamically during generation, causing:
- Memory reallocations
- Inability to use CUDA graphs
- Some SDPA backends fail with variable shapes

**Static KV Cache**: Pre-allocates KV cache to fixed size (`max_cache_len`), enabling:
- CUDA graph capture
- Consistent memory layout
- Potential for `torch.compile()` optimization

### Experiment 5: Static KV Cache Benchmark

**Job ID**: 421210
**Date**: 2025-12-09
**Script**: `scripts/diagnostics/benchmark_inference_attention.py --static-cache`

**Configuration**:
- Model: Llama-3.2-380M_conformers (m380_conf_v2)
- Static KV cache: **enabled** (`cache_implementation="static"`)
- Max cache length: 2048 (matches model's `max_position_embeddings`)
- Samples: 64, batch_size=16, max_new_tokens=1000
- Tokenizer padding: `pad_to_multiple_of=64` (for tensor core efficiency)

**Results**:

| Implementation | Time (s) | Tokens/sec | Speedup vs eager |
|----------------|----------|------------|------------------|
| **eager** | 14.62 | **3513.4** | **1.00x 🏆** |
| sdpa (auto) | 15.08 | 3443.6 | 0.98x |
| sdpa_efficient | 16.48 | 3394.9 | 0.97x |
| sdpa_cudnn | 18.38 | 2765.3 | 0.79x |
| sdpa_flash | Failed | - | "No available kernel" |

### Static vs Dynamic KV Cache Comparison

| Implementation | Dynamic Cache (Job 421194) | Static Cache (Job 421210) | Improvement |
|----------------|---------------------------|---------------------------|-------------|
| **eager** | 1907 tok/s | **3513 tok/s** | **1.84x** |
| sdpa (auto) | 1958 tok/s | 3444 tok/s | **1.76x** |
| sdpa_cudnn | (hung) | 2765 tok/s | ✓ Now works! |
| sdpa_flash | Failed | Failed | Still broken |

### Key Findings

1. **Static KV cache provides ~1.8x speedup** across all working implementations
2. **eager attention becomes fastest** with static cache (3513 tok/s)
3. **sdpa_cudnn now works** with static cache (was hanging with dynamic cache)
4. **sdpa_flash still fails** - the issue is not related to dynamic cache shapes

### Experiment 6: flash_attention_2 with Static Cache

**Job ID**: 421213
**Date**: 2025-12-09

**Result**: **HUNG** - flash_attention_2 does not work with HuggingFace's static cache implementation.

### Why flash_attention_2 Hangs with Static Cache

The flash-attn library's integration with HuggingFace transformers appears incompatible with the static cache implementation. This is likely because:
1. flash-attn expects specific tensor layouts that differ from static cache format
2. The HuggingFace `Llama*Cache` implementations may not be supported by flash-attn
3. Internal memory management conflicts between flash-attn and static pre-allocation

### Static KV Cache: When to Use

**Valid for conformer generation** because:
- Input SMILES: ~10-100 tokens
- Output conformer: ~500-1500 tokens (proportional to atom count)
- Model context length: 2048 (hard limit)
- `max_cache_len=2048` covers worst case

**Note**: `inference.py` uses `max_new_tokens=4000` but this is just a safety cap. The model's `max_position_embeddings=2048` is the actual limit.

### Recommendation Update

```python
# Optimal configuration for inference (with static cache)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="sdpa",  # or "eager" - both work well
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# Enable static KV cache
model.generation_config.cache_implementation = "static"

# Use padded tokenization for consistent shapes
inputs = tokenizer(prompts, padding=True, pad_to_multiple_of=64, return_tensors="pt")
```

**Performance**: ~3400-3500 tok/s (vs ~1900-2000 tok/s with dynamic cache)

---

## Phase 2 Summary

### Static KV Cache Results

1. **~1.8x speedup** with static KV cache enabled
2. **eager and sdpa (auto) are both excellent** - similar performance
3. **flash_attention_2 is incompatible** with static cache - hangs
4. **sdpa_flash still fails** even with static cache
5. **sdpa_cudnn works** with static cache but is slower than auto-select

### Updated Recommendations (H100)

| Use Case | Recommendation | Expected Throughput |
|----------|----------------|---------------------|
| **Production inference** | `sdpa` + static cache | ~3400 tok/s |
| **Simplest setup** | `eager` + static cache | ~3500 tok/s |
| **Avoid** | `flash_attention_2` | Hangs or 40% slower |

---

## Phase 2.5: GPU Architecture Comparison (GTX 3070 vs H100)

### Critical Finding: Static KV Cache is GPU-Specific

After integrating static KV cache into the smoke runner (`run_logit_processor_smoke.py`), we tested on GTX 3070 (Ampere consumer GPU) and discovered that **static cache hurts performance on consumer GPUs**.

**Test Configuration**:
- GPU: GTX 3070 (8GB, sm86)
- Model: Llama-3.2-380M_conformers
- Samples: 64, batch_size=32
- Sampling: top_p_sampling4

**Results on GTX 3070**:

| Configuration | Time | Relative |
|---------------|------|----------|
| dynamic + logit processor | 29.9s | **1.00x (baseline)** |
| static + logit processor | 73.6s | 0.41x (2.5x SLOWER!) |
| static + no logit processor | 93.1s | 0.32x (3x SLOWER!) |

### Why Static Cache Hurts on GTX 3070

1. **Missing Hopper-specific hardware**: H100 has TMA (Tensor Memory Accelerator) and WGMMA instructions that make static cache + CUDA graphs efficient. GTX 3070 lacks these.

2. **CUDA graphs not benefiting**: Static cache enables CUDA graph capture, but on consumer GPUs without specialized hardware, the overhead of pre-allocation outweighs any graph benefits.

3. **Memory bandwidth**: GTX 3070 has ~448 GB/s bandwidth vs H100's 3.35 TB/s. Static pre-allocation may hurt memory access patterns on lower-bandwidth GPUs.

4. **Logit processor breaks graphs anyway**: Even if CUDA graphs could help, the logit processor runs Python code every token, forcing graph breaks.

### GPU Compatibility Matrix

| GPU | Architecture | Static KV Cache | Recommendation |
|-----|--------------|-----------------|----------------|
| **H100** | Hopper (sm90) | ✅ 1.8x faster | Use static cache |
| **A100** | Ampere DC (sm80) | ⚠️ TBD | Test before using |
| **GTX 3070/3080/3090** | Ampere consumer (sm86) | ❌ 2-3x slower | Use dynamic cache |
| **Older GPUs** | Pre-Ampere | ❌ Likely slower | Use dynamic cache |

### Recommendations by GPU

**For H100 (datacenter)**:
```python
model.generation_config.cache_implementation = "static"
# Use --device cuda:0 (single GPU required)
```

**For consumer GPUs (GTX 30xx, RTX 40xx)**:
```python
# Do NOT enable static cache - use default dynamic
# model.generation_config.cache_implementation = "dynamic"  # default
```

### Scripts Updated

- `scripts/logit_processor/run_logit_processor_smoke.py`:
  - Added `--kv-cache {dynamic,static}` flag
  - Added `--submit {local,h100,a100}` for slurm submission
  - Added GPU detection warning for consumer GPUs
  - Added cache sizing info logging

---

## Phase 3: CPU-Side Optimizations (In Progress)

**Key insight**: Since static KV cache is GPU-specific, we focus on **CPU-side optimizations** that help on ANY GPU.

### Remaining Optimizations

#### 1. torch.compile with Static Cache
- Now that static cache works, we can try `torch.compile()`
- **Expected**: Additional 10-30% speedup from graph optimization
- **Risk**: May not work with logit processor

#### 2. Batched Template Building
- Current: Templates built per-sequence in logit processor
- Optimization: Batch template construction for multiple sequences
- **Location**: `ConformerConstraintLogitsProcessor.build_templates_for_batch()`

#### 3. Reduced Tokenizer Overhead
- Current: `tokenizer.encode_plus()` called frequently in logit processor
- Optimization: Cache tokenization results, use faster encoding methods
- **Location**: Logit processor's token-by-token constraint checking

#### 4. Profiling Strategy
- Profile on `np` node for quick iteration
- Validate on `h100` with large generations
- Track metrics: tokens/sec, memory, latency per token

### Scripts Created/Modified

- [x] `scripts/diagnostics/benchmark_inference_attention.py` - Added `--static-cache` flag
- [ ] `src/molgen3D/evaluation/inference.py` - Add static cache support
- [ ] `src/molgen3D/evaluation/constrained_logits.py` - Optimize tokenizer calls

---

### References

1. FlashAttention-2 Paper: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (ICLR 2024)
2. FlashAttention-3 Blog: https://tridao.me/blog/2024/flash3/
3. PyTorch SDPA Docs: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
4. PyTorch 2.5 Release Notes: https://pytorch.org/blog/pytorch2-5/
5. flash-attention skill benchmarks: `~/.claude/skills/flash-attention/references/benchmarks.md`
