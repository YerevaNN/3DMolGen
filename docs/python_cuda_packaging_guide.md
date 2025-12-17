# Installing flash-attn for 3DMolGen

## Quick Install (Recommended)

A pre-built wheel for **PyTorch 2.9.1 + CUDA 12.8 + Python 3.10** is available on the cluster:

```
/auto/home/aram.dovlatyan/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl
```

### Install Steps

```bash
# 1. Activate the 3dmolgen environment
conda activate 3dmolgen

# 2. Copy the wheel to your home (or any folder)
cp /auto/home/aram.dovlatyan/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl ~/

# 3. Install locally
cd ~
pip install ./flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl
```

The wheel location doesn't matter - you can copy it to `~/`, `~/wheels/`, or anywhere else.

### Verify Installation

```python
python -c "import flash_attn; print(flash_attn.__version__)"
# Should print: 2.8.3
```

## Different Environment?

If your environment differs (different PyTorch/CUDA/Python version), download a matching wheel from:

**https://github.com/mjun0812/flash-attention-prebuild-wheels**

Wheel naming convention:
```
flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl
           │      │       │     │
           │      │       │     └── Python 3.10
           │      │       └── PyTorch 2.9
           │      └── CUDA 12.8
           └── flash-attn version
```

## What is Flash Attention?

Flash Attention computes the same attention math as standard attention but with **optimized memory access patterns**:

- **2-4x faster** inference
- **10-20x less GPU memory**
- Same mathematical output

In HuggingFace, enable it with:
```python
model = AutoModelForCausalLM.from_pretrained(
    "path/to/model",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
```

## Why Pre-built Wheels?

Building flash-attn from source requires:
- CUDA compiler (nvcc)
- 20-60 minutes compile time
- Lots of RAM (~120GB for Slurm job)
- Correct ABI matching with PyTorch

Pre-built wheels skip all of this - just download and install in seconds.

---

**For detailed explanations** (how attention works, why flash-attn is tricky to build, common errors, cluster architecture), see the original comprehensive guide in git history or ask for the expanded version.
