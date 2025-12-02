# Constrained Conformer Smoke Harness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Provide a reusable harness that samples SMILES from the GEOM pickle, runs constrained generation, and verifies the custom logits processor preserves SMILES structure while leaving coordinate blocks free.

**Architecture:** Central helper module orchestrates sampling, prompt construction, constrained generation, and validation; both a CLI and pytest wrapper call into it. Real inference uses HF checkpoints, while tests rely on lightweight stubs. Validation focuses on SMILES/tag integrity rather than coordinate accuracy.

**Tech Stack:** Python 3.10, PyTorch, Hugging Face transformers, RDKit, pytest.

### Task 1: Create shared constrained smoke helper

**Files:**
- Create: `src/molgen3D/evaluation/constrained_smoke.py`
- Modify: `src/molgen3D/evaluation/__init__.py`

**Step 1: Scaffold module**
```python
"""Utilities to run constrained conformer smoke checks."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
```

**Step 2: Implement data-loading helpers**
- Function `load_ground_truth(dataset: str) -> dict` using `get_data_path` + `cloudpickle.load`.
- Function `sample_smiles(gt: dict, sample_size: int, seed: int, stratify: bool) -> list[str]` that draws unique keys.

**Step 3: Implement prompt/conformer assembly**
- Function `build_prompts(smiles_list: Sequence[str]) -> list[str]` returning `[SMILES]{s}[/SMILES]` strings.

**Step 4: Implement runner dataclass**
```python
@dataclass
class ConstrainedSmokeConfig:
    model: AutoModelForCausalLM
    tokenizer: PreTrainedTokenizer
    generation_config: GenerationConfig
    batch_size: int = 64
    max_geom_len: int = 80
```
- Method `run_smoke(smiles_subset)` that reuses `process_batch` internals (tokenization, template build, logits processor) to produce stats and decoded outputs.

**Step 5: Implement validation utilities**
- Function `validate_outputs(result, check_coords=True)` ensuring each decoded `[SMILES]` matches prompt, `strip_smiles` agrees with canonical form, coordinate sequences exist, etc. Raise `AssertionError` with descriptive payload when mismatch occurs.

**Step 6: Export helpers**
- Update `src/molgen3D/evaluation/__init__.py` to expose the helper for reuse.

### Task 2: Add CLI smoke runner

**Files:**
- Create: `scripts/run_constrained_smoke.py`

**Step 1: Parse CLI args**
```python
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["clean", "distinct"], default="distinct")
parser.add_argument("--sample-size", type=int, default=64)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--model", default="m380_conf_v2")nparser.add_argument("--ckpt", default="2e")
parser.add_argument("--device", default="auto")
```

**Step 2: Load model/tokenizer**
- Reuse `load_model_tokenizer` from `inference.py` but allow overriding dtype/attention via flags.

**Step 3: Wire helper**
- Load dataset, sample smiles, run smoke, and print a concise report (counts, example mismatches, stats).

**Step 4: Exit codes**
- Non-zero exit if validation fails.

### Task 3: Pytest wrapper using stubs

**Files:**
- Create: `tests/inference/test_constrained_smoke.py`

**Step 1: Implement stub tokenizer/model**
- Extend dummy tokenizer to cover tokens used in helper.
- Stub model with `generate` that emits controlled sequences consistent with templates.

**Step 2: Positive test**
```python
def test_smoke_passes_with_stub(tmp_path):
    runner = make_stub_runner()
    result = runner.run_smoke(smiles_subset)
    validate_outputs(result)
```

**Step 3: Negative test**
- Modify stub to emit wrong fixed token and assert `validate_outputs` raises.

**Step 4: Run pytest**
- `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python -m pytest tests/inference/test_constrained_smoke.py -q`

### Task 4: Update constrained logits plan document

**Files:**
- Modify: `docs/constrained_logits_plan.md`

**Steps:**
1. Add “Smoke harness + CLI + pytest guard” progress log entry.
2. Document next actions discovered during work.

### Task 5: Verification

1. Run targeted pytest: `LD_PRELOAD=... python -m pytest tests/inference/test_constrained_logits.py tests/inference/test_constrained_smoke.py -q`
2. Optionally run CLI smoke with a tiny sample to confirm end-to-end behavior.
