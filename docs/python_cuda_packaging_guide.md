# Python CUDA Packaging: Wheels, Building, and flash-attn

A practical guide for understanding how Python packages with CUDA code work on clusters.

---

## Part 0: What IS Flash Attention? (The Conceptual Foundation)

Before diving into packaging, let's understand what we're actually building.

### What is Attention in a Transformer?

When your Qwen 0.6B model generates text, at each step it asks: "Given all the tokens I've seen so far, what should come next?"

**Attention** is the mechanism that lets the model look at ALL previous tokens and decide which ones are relevant.

```
Input: "The cat sat on the"
                          ↓
        ┌─────────────────────────────────────┐
        │  Attention: "What's relevant here?" │
        │                                     │
        │  "The" ──► 0.05 (low relevance)     │
        │  "cat" ──► 0.40 (high - subject!)   │
        │  "sat" ──► 0.15                     │
        │  "on"  ──► 0.10                     │
        │  "the" ──► 0.30 (high - needs noun) │
        └─────────────────────────────────────┘
                          ↓
                   Output: "mat"
```

Mathematically, attention computes:
```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

Where:
- **Q** (Query): "What am I looking for?"
- **K** (Key): "What does each token offer?"
- **V** (Value): "What information does each token contain?"

### Why is Standard Attention Slow?

The problem: **Q × K^T** compares EVERY token to EVERY other token.

```
Sequence length: 1000 tokens
Comparisons: 1000 × 1000 = 1,000,000 operations
Memory for attention matrix: 1000 × 1000 × 4 bytes = 4 MB

Sequence length: 10,000 tokens
Comparisons: 100,000,000 operations
Memory: 400 MB  ← Grows quadratically! (O(n²))
```

But the real bottleneck isn't computation - it's **memory bandwidth**.

```
┌─────────────────────────────────────────────────────────────────┐
│                         GPU Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐                    ┌───────────────────────┐ │
│   │   SRAM       │ ◄── Very fast     │    HBM (VRAM)         │ │
│   │   (20 MB)    │     but tiny      │    (40-80 GB)         │ │
│   │              │                    │                       │ │
│   │ Where actual │                    │  Where tensors live   │ │
│   │ compute      │ ◄─────────────────►│  (Q, K, V matrices)   │ │
│   │ happens      │   Memory transfer  │                       │ │
│   └──────────────┘   is the bottleneck └───────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Standard Attention:
1. Load Q, K from HBM → SRAM
2. Compute Q × K^T
3. Write attention scores to HBM  ← SLOW!
4. Load attention scores from HBM
5. Compute softmax
6. Write softmax result to HBM    ← SLOW!
7. Load softmax, V from HBM
8. Compute output
9. Write output to HBM            ← SLOW!

Total HBM reads/writes: O(n²) - this is the killer!
```

### What Flash Attention Does Differently

Flash Attention is **the same math** but with a smarter memory access pattern.

```
Flash Attention: "Tiled" Computation
──────────────────────────────────────

Instead of computing the full attention matrix:

Standard:                      Flash:
┌───────────────────┐         ┌────┬────┬────┬────┐
│                   │         │ T1 │    │    │    │ Process tile T1,
│  Full n×n matrix  │   →     ├────┼────┼────┼────┤ keep in SRAM,
│  (won't fit in    │         │    │ T2 │    │    │ write only final
│   SRAM!)          │         ├────┼────┼────┼────┤ output to HBM
│                   │         │    │    │ T3 │    │
│                   │         ├────┼────┼────┼────┤
└───────────────────┘         │    │    │    │ T4 │
                              └────┴────┴────┴────┘
```

Key insight: We never need the full attention matrix! We only need the OUTPUT.

```
Flash Attention Algorithm:
1. Load a TILE of Q, K, V into SRAM (fits!)
2. Compute attention for that tile
3. Accumulate result (clever math to combine tiles)
4. Repeat for all tiles
5. Write ONLY the final output to HBM

HBM reads/writes: O(n) instead of O(n²)!
```

**Result:** Same mathematical output, 2-4x faster, uses 10-20x less memory.

### How Does a `.so` File Fit In?

The flash attention **algorithm** is implemented as **CUDA kernels** - low-level code that runs on GPU cores.

```
Source code (flash_attn/csrc/*.cu):
┌─────────────────────────────────────────────────────┐
│ __global__ void flash_fwd_kernel(...) {            │
│     // Load tiles into shared memory (SRAM)        │
│     // Compute attention in tiles                  │
│     // Accumulate with online softmax              │
│ }                                                   │
└─────────────────────────────────────────────────────┘
            │
            │ nvcc compiler
            ▼
Compiled binary (flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so):
┌─────────────────────────────────────────────────────┐
│ Binary machine code that runs on NVIDIA GPU        │
│ (PTX → SASS assembly for your specific GPU arch)   │
└─────────────────────────────────────────────────────┘
            │
            │ Python import
            ▼
Python can call it:
┌─────────────────────────────────────────────────────┐
│ from flash_attn import flash_attn_func             │
│ output = flash_attn_func(q, k, v, causal=True)     │
└─────────────────────────────────────────────────────┘
```

The `.so` file is a **shared library** - compiled code that Python loads at runtime.

### How HuggingFace Swaps Attention Implementations

Your Qwen model has attention layers. HuggingFace lets you choose HOW those layers compute attention:

```python
# In your inference code:
model = AutoModelForCausalLM.from_pretrained(
    "path/to/qwen",
    attn_implementation="flash_attention_2",  # ← This chooses the implementation
)
```

Inside the model, this happens:

```python
# Simplified view of what HuggingFace does internally:

class QwenAttention(nn.Module):
    def forward(self, hidden_states, ...):
        q, k, v = self.project_qkv(hidden_states)

        if self.config._attn_implementation == "eager":
            # Standard PyTorch attention (slow but always works)
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
            attn_weights = F.softmax(attn_weights, dim=-1)
            output = torch.matmul(attn_weights, v)

        elif self.config._attn_implementation == "sdpa":
            # PyTorch's built-in optimized attention
            output = F.scaled_dot_product_attention(q, k, v)

        elif self.config._attn_implementation == "flash_attention_2":
            # Flash Attention library (fastest, but needs installation)
            from flash_attn import flash_attn_func
            output = flash_attn_func(q, k, v, causal=True)

        return output
```

**All three produce the same mathematical result!** They're just different implementations with different speed/memory tradeoffs.

### During Inference: What Actually Happens?

When you call `model.generate()`:

```
model.generate(input_ids, max_new_tokens=100)
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│ For each new token:                                              │
│                                                                  │
│  1. Forward pass through all layers                             │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │ Layer 1: Attention → FFN                                │ │
│     │ Layer 2: Attention → FFN                                │ │
│     │ ...                                                      │ │
│     │ Layer N: Attention → FFN                                │ │
│     └─────────────────────────────────────────────────────────┘ │
│                          │                                       │
│                          ▼                                       │
│  2. Get logits (probability for each token in vocabulary)       │
│                          │                                       │
│                          ▼                                       │
│  3. Sample next token based on logits                           │
│                          │                                       │
│                          ▼                                       │
│  4. Add new token to sequence, repeat                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

The ATTENTION step in each layer is where flash_attention_2 speeds things up.
With 28 layers (Qwen 0.6B), that's 28 attention computations per token!
```

### The KV Cache: Why Attention Gets Faster During Generation

There's another optimization you've seen in logs: **KV Cache**.

```
Generating: "The cat sat on the mat and then"
                                          ↑
                                    generating this

Without KV Cache:
  Every new token → Recompute K, V for ALL previous tokens

With KV Cache:
  Store K, V for previous tokens → Only compute K, V for NEW token

┌─────────────────────────────────────────────────────────────────┐
│ Token 1: Compute K₁, V₁, store in cache                        │
│ Token 2: Load K₁,V₁ from cache, compute K₂,V₂, store           │
│ Token 3: Load K₁,K₂,V₁,V₂ from cache, compute K₃,V₃, store     │
│ ...                                                             │
└─────────────────────────────────────────────────────────────────┘
```

**Static KV Cache** (even faster): Pre-allocate the full cache size upfront instead of growing dynamically.

### Summary: The Full Picture

```
Your Qwen Model
      │
      ├── Embedding Layer
      │
      ├── Transformer Layers (×28)
      │       │
      │       ├── Attention ◄─── THIS is what flash_attention_2 optimizes
      │       │      │
      │       │      └── Can use: eager (slow), sdpa (fast), flash_attention_2 (fastest)
      │       │
      │       └── Feed-Forward Network
      │
      └── Output Layer (logits)

flash-attn package provides:
  └── flash_attn_func() → Optimized CUDA kernel for attention computation
                          Same math, better memory access pattern
                          Compiled to .so file that Python calls
```

---

## The Two Ways to Install Python Packages

### 1. Wheels (Pre-compiled Binaries)

A **wheel** is a pre-compiled package. Someone else already compiled the code, and you just download and use it.

```
flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl
         │       │        │              │     │
         │       │        │              │     └── Platform (Linux x86_64)
         │       │        │              └── Python version (CPython 3.13)
         │       │        └── C++ ABI flag (must match PyTorch's)
         │       └── CUDA version + PyTorch version
         └── Package version
```

**Advantages:**
- Fast install (just download and unzip)
- No compiler needed
- Works immediately

**Disadvantages:**
- Must find a wheel that matches YOUR exact environment
- If no matching wheel exists, you're stuck

### 2. Building from Source

When no matching wheel exists, you compile the code yourself.

**What you need:**
- Source code (downloaded by pip)
- Compiler (nvcc for CUDA, gcc/clang for C++)
- Matching libraries (PyTorch headers, CUDA headers)
- Time and memory (compilation is resource-intensive)

**Advantages:**
- Works for any environment
- Can optimize for your specific GPU

**Disadvantages:**
- Slow (flash-attn takes 20-60 minutes)
- Needs compiler toolchain
- Can fail in many ways

## Why flash-attn is Particularly Tricky

flash-attn has **CUDA kernel code** - code that runs directly on the GPU. This code must be compiled with `nvcc` (NVIDIA's CUDA compiler).

### The Version Matching Problem

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR ENVIRONMENT                              │
├─────────────────────────────────────────────────────────────────┤
│  PyTorch 2.9.1+cu128                                            │
│       │                                                          │
│       └── Built with CUDA 12.8 runtime                          │
│           (bundled inside PyTorch, no nvcc)                     │
├─────────────────────────────────────────────────────────────────┤
│  CUDA Driver (nvidia-smi shows 12.8)                            │
│       │                                                          │
│       └── Supports running CUDA 12.8 code                       │
│           (but has no compiler)                                 │
├─────────────────────────────────────────────────────────────────┤
│  CUDA Toolkit (/usr/local/cuda-12.8)                            │
│       │                                                          │
│       └── Has nvcc compiler                                     │
│           (needed to BUILD cuda code)                           │
└─────────────────────────────────────────────────────────────────┘
```

**The confusion:** PyTorch comes with CUDA *runtime* libraries (to run GPU code), but NOT the *compiler* (to build GPU code). These are different things!

### ABI Compatibility

ABI = Application Binary Interface. Think of it as a "dialect" of compiled code.

```
PyTorch compiled with:     CXX11_ABI=TRUE
flash-attn compiled with:  CXX11_ABI=FALSE  ← MISMATCH!

Result: ImportError: undefined symbol: _ZN3c105ErrorC2E...
```

The cryptic error means: "I'm looking for a function, but it was compiled with different settings."

**Fix:** Set `FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE` to match PyTorch's ABI.

## How This Works on the Cluster

### Build Once, Use Everywhere

```
┌──────────────────┐         ┌──────────────────┐
│   BUILD NODE     │         │    RUN NODE      │
│   (dgx/a100)     │         │   (dgx/a100)     │
├──────────────────┤         ├──────────────────┤
│ Has nvcc 12.8    │         │ No nvcc needed   │
│ Compiles code    │  ────►  │ Just runs code   │
│ Creates .so file │         │ Uses .so file    │
└──────────────────┘         └──────────────────┘
        │                            │
        └────────────────────────────┘
                    │
            NFS shared storage
         (~/miniconda3/envs/3dmolgen/)
```

Because your conda environment is on NFS (`/auto/home/...`), once you build flash-attn on ANY node, it's available on ALL nodes.

### Why We Submit a Slurm Job to Build

Building flash-attn needs:
1. **GPU** - To detect GPU architecture (sm_80 for A100)
2. **nvcc** - The CUDA compiler
3. **Lots of RAM** - Compilation is memory-hungry (we use 120G)
4. **Time** - 20-60 minutes of CPU time

Your laptop/workstation doesn't have these. The cluster nodes do.

## The Build Process Step-by-Step

```bash
pip install flash-attn --no-build-isolation
```

What happens:

1. **pip downloads source** → `/tmp/pip-install-xxx/flash-attn_xxx/`

2. **pip runs setup.py** → Detects environment:
   - CUDA version from nvcc
   - PyTorch version
   - GPU architecture

3. **nvcc compiles CUDA kernels** → Creates `.cu` → `.o` files
   - This is the slow, memory-intensive part
   - Multiple files compiled in parallel (MAX_JOBS)

4. **Linker creates shared library** → `flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so`

5. **pip installs to site-packages** → `~/miniconda3/envs/3dmolgen/lib/python3.10/site-packages/flash_attn/`

### The --no-build-isolation Flag

```
WITH isolation (default):
┌─────────────────────────────────┐
│  Temporary virtual environment  │
│  - Fresh pip, setuptools        │
│  - NO access to your packages   │  ← Can't find torch!
└─────────────────────────────────┘

WITHOUT isolation (--no-build-isolation):
┌─────────────────────────────────┐
│  Your actual environment        │
│  - Has torch installed          │  ← Can find torch headers
│  - Has all your packages        │
└─────────────────────────────────┘
```

flash-attn needs PyTorch headers during compilation. With isolation, it can't find them.

## Environment Variables Explained

| Variable | Purpose | Our Value |
|----------|---------|-----------|
| `CUDA_HOME` | Where to find nvcc and CUDA headers | `/usr/local/cuda-12.8` |
| `FLASH_ATTENTION_FORCE_BUILD` | Build even if wheel exists | `TRUE` |
| `FLASH_ATTENTION_FORCE_CXX11_ABI` | Match PyTorch's ABI setting | `FALSE` |
| `MAX_JOBS` | Parallel compilation jobs | `4` (lower = less RAM) |
| `NVCC_THREADS` | Threads per nvcc process | `2` |

## Common Errors and What They Mean

### "CUDA 11.7 and above required"
```
RuntimeError: FlashAttention is only supported on CUDA 11.7 and above.
torch.__version__ = 2.9.1+cu128
```
**Meaning:** nvcc is too old. The script found `/usr/bin/nvcc` (CUDA 11.5) instead of `/usr/local/cuda-12.8/bin/nvcc`.

**Fix:** Set `CUDA_HOME=/usr/local/cuda-12.8` and add to PATH.

### "undefined symbol" errors
```
ImportError: flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so:
undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationESs
```
**Meaning:** ABI mismatch between flash-attn and PyTorch.

**Fix:** Rebuild with matching `FLASH_ATTENTION_FORCE_CXX11_ABI` setting.

### "Killed" during compilation
```
FAILED: flash_bwd_hdim128_bf16_sm80.o
Killed
ninja: build stopped: subcommand failed.
```
**Meaning:** Out of memory. The OOM killer terminated the process.

**Fix:** Reduce `MAX_JOBS` and `NVCC_THREADS`, increase `--mem` in Slurm.

### "ModuleNotFoundError: packaging"
```
ModuleNotFoundError: No module named 'packaging'
```
**Meaning:** pip's subprocess can't see conda packages.

**Fix:** Add `PYTHONPATH=$CONDA_PREFIX/lib/python3.10/site-packages:$PYTHONPATH`

## After Building: Using flash-attn

Once built, flash-attn is just another package in your conda env:

```python
# In your code (e.g., loading a HuggingFace model)
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    attn_implementation="flash_attention_2",  # Uses flash-attn
    torch_dtype=torch.bfloat16,
)
```

HuggingFace Transformers automatically uses flash-attn if:
1. `attn_implementation="flash_attention_2"` is set
2. flash-attn is installed
3. GPU supports it (Ampere+ = A100, H100, etc.)

## Summary

| Concept | Simple Explanation |
|---------|-------------------|
| **Wheel** | Pre-compiled package, just download and use |
| **Build from source** | Compile code yourself, needs compiler + time |
| **nvcc** | NVIDIA's CUDA compiler (builds GPU code) |
| **CUDA runtime** | Libraries to RUN GPU code (bundled in PyTorch) |
| **CUDA toolkit** | Full package with compiler + headers + runtime |
| **ABI** | Binary compatibility setting, must match |
| **NFS** | Network storage, build once → use everywhere |

## Our Cluster Setup

```
dgx node:
├── CUDA Driver: 12.8 (nvidia-smi)
├── CUDA Toolkit: /usr/local/cuda-12.8 (has nvcc)
├── PyTorch: 2.9.1+cu128
└── conda env: /auto/home/.../3dmolgen (NFS shared)

Build on dgx → Install to NFS → Use from any node
```

---

## Frequently Asked Questions

### Does it matter which node I build on (dgx vs h100)?

**No, it doesn't matter** - as long as both nodes have the same CUDA toolkit version.

The flash-attn build compiles for **multiple GPU architectures** simultaneously:

```bash
# These flags are added automatically by the build system:
-gencode arch=compute_80,code=sm_80   # A100 (dgx)
-gencode arch=compute_90,code=sm_90   # H100
-gencode arch=compute_100,code=sm_100 # Blackwell
-gencode arch=compute_120,code=sm_120 # Future GPUs
```

The compiled binary contains code for ALL these architectures. At runtime, the GPU driver selects the right code path for the current GPU.

**What matters:**
- CUDA toolkit version (nvcc) - determines which architectures can be compiled
- CUDA driver version - must be >= toolkit version
- PyTorch CUDA version - should match toolkit for ABI compatibility

**What doesn't matter:**
- Which GPU is physically present during build
- Whether you build on A100 or H100

### Where is the built package stored?

The package is installed into your conda environment's site-packages directory:

```
/auto/home/aram.dovlatyan/miniconda3/envs/3dmolgen/lib/python3.10/site-packages/flash_attn/
├── __init__.py
├── flash_attn_interface.py
├── flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so  ← The compiled binary
└── ... other Python files
```

**Key point:** This path is on **NFS** (`/auto/home/...`), which means:
- Build once on ANY node with nvcc
- Use from ALL cluster nodes immediately
- No need to rebuild on each node

### How much storage space does flash-attn take?

flash-attn is relatively small:

| Component | Size |
|-----------|------|
| Compiled `.so` files | ~200-400 MB |
| Python source files | ~5 MB |
| **Total** | **~300-500 MB** |

The size varies based on how many GPU architectures are compiled. With 4 architectures (sm_80, sm_90, sm_100, sm_120), expect ~400-500 MB.

**For comparison:**

| Package | Typical Size |
|---------|--------------|
| flash-attn | 300-500 MB |
| PyTorch | 2-3 GB |
| Full ML conda env | 10-15 GB |

flash-attn is <5% of a typical ML environment - not a storage concern.

### Why does building take so long?

CUDA kernel compilation is inherently slow because:

1. **Multiple architectures**: Each `-gencode` flag means compiling the same kernels again for a different GPU
2. **Complex templates**: flash-attn uses C++ templates that expand to many specialized functions
3. **Optimization**: nvcc performs heavy optimization passes
4. **Many kernels**: Different kernels for forward/backward, different head dimensions, different data types

Typical build times:
- `MAX_JOBS=2`: 30-45 minutes
- `MAX_JOBS=4`: 20-30 minutes
- `MAX_JOBS=8`: 15-20 minutes (but needs lots of RAM)

### Can I restrict which architectures are built?

Yes, use `FLASH_ATTN_CUDA_ARCHS` to build only for specific GPUs:

```bash
# Only build for A100 (faster build, smaller binary)
export FLASH_ATTN_CUDA_ARCHS="80"

# Only build for H100
export FLASH_ATTN_CUDA_ARCHS="90"

# Build for both A100 and H100
export FLASH_ATTN_CUDA_ARCHS="80;90"
```

**Trade-off:** Faster build and smaller binary, but won't work on GPUs not included.
