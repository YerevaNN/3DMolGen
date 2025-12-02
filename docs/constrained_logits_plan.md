# Constrained Conformer Generation — Working Plan

Last updated: 2025-12-01 (v11 implementation + analysis tools)

## Objective
Guarantee that every non-coordinate token in the conformer output matches the input SMILES while allowing the model to freely generate the numeric coordinates. The output format stays:

```
[SMILES]{prompt_smiles}[/SMILES][CONFORMER]{enriched_conformer}[/CONFORMER]
```

Where `enriched_conformer` interleaves `[AtomDescriptor]<x,y,z>` blocks with SMILES structural tokens.

## Clarifying questions/dimensions to think about for this broad task that was asked at beginning

### Design Questions from v11 Brainstorming (2025-12-01)

1. **Success Metric**: What should we prioritize?
   - Option A: Pass rate (maximize 64/64 passing, even if coords somewhat malformed)
   - Option B: Coordinate quality (ensure passes have valid `<x,y,z>` structure)
   - Option C: Both together (valid structure AND high pass rate)
   - **Decision**: Both together - need structurally valid coordinates with high pass rate

2. **Constraint Approach**: How should we enforce coordinate structure?
   - Option A: Character-level masking (force each character to match `<float,float,float>` pattern)
   - Option B: Strict token pattern (state machine tracking position in pattern, pre-compute valid token sequences)
   - Option C: Post-generation repair (let model generate freely, parse/validate/fix after)
   - **Decision**: Strict token pattern (v11 pivoted to minimal blocklist as simpler first step)

3. **Special Token Handling**: Where to remove `<|begin_of_text|>` and `<|end_of_text|>` pollution?
   - Option A: Decode-time (`skip_special_tokens=True` in `tokenizer.batch_decode()`)
   - Option B: Logits processor (add special token IDs to forbidden set during generation)
   - Option C: Post-processing (trim leading/trailing special tokens from decoded strings)
   - **Decision**: Post-processing (simpler, doesn't interfere with generation)

4. **Iteration Scope**: How much to tackle per version?
   - Option A: Comprehensive (fix all issues: token pattern, special tokens, validation in v11)
   - Option B: Incremental (v11 = token pattern only, v12 = special tokens, etc.)
   - Option C: Hybrid (critical fixes in v11, defer nice-to-haves)
   - **Decision**: Incremental - one major change per version for easier debugging

5. **Float Format Strictness**: How strict should coordinate float parsing be?
   - Option A: Strict (enforce training format exactly - truncate-style, no leading zeros)
   - Option B: Lenient (accept any valid float format: `01.23`, `.5`, `1.0000`, etc.)
   - Option C: Regex-based (match `_NUMERIC_TOKEN_RE` pattern: `[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?`)
   - **Decision**: Regex-based (use existing pattern from codebase)

### Key Insight from Questions

The core tension identified: **token-level constraints vs model capability**
- Tokenizers produce arbitrary multi-token sequences (BPE learned `"120157094632576653"` as single token)
- Simple character-set allowlists permit garbage because tokens aren't characters
- Models *should* learn `[C]<` → coordinates if trained properly, but 380M model may need help
- Constraints should **help** the model stay on track, not **fight** its natural generation

This led to the v11 minimal blocklist philosophy: block known-bad tokens, trust model for the rest.


## Scope and assumptions
- Inputs to `model.generate` remain `[SMILES]...[/SMILES]` (no extra priming token yet, but design should tolerate adding one later).
- One conformer is generated per prompt today; the design should not preclude multiple conformers per prompt in the future.
- All SMILES-derived tokens (atoms, bonds, ring digits, parentheses, tags) are frozen. Coordinate substrings inside `<x,y,z>` are unconstrained; we also leave the `<` and `>` delimiters free for now (can tighten later).
- Batching is required: each prompt in a batch gets its own constraint template and per-sequence state.

## Plan
1) **Template builder**
   - Canonicalize the prompt SMILES via RDKit in the same way as `encode_cartesian_v2` to mirror training-time formatting.
   - Reuse `tokenize_smiles`, `_format_atom_descriptor`, `_normalize_atom_descriptor` to derive the exact `[AtomDescriptor]` strings and SMILES non-atom tokens.
   - Produce an ordered list of segments:
     - `fixed`: literal text fragments (e.g., `[C]`, `=`, `1`, `)`, `[CONFORMER]`, `[/CONFORMER]`).
     - `coord_block`: placeholder representing `<x,y,z>` where logits are not constrained; the block ends when the tokenizer emits the `>` token.
   - Tokenize each segment with the actual tokenizer to store expected token-id sequences (handles multi-token atoms or BPE splits).

2) **Logits processor**
   - Implement `ConformerConstraintLogitsProcessor(LogitsProcessor)` that tracks, per batch item:
     - current segment index and offset within the segment,
     - whether we are inside a coordinate block.
   - On each call, advance state based on newly generated tokens (difference in length since previous call).
   - If the next expected fragment is `fixed`, mask logits to `-inf` except for the exact expected token id at that position.
   - If in a `coord_block`, leave logits untouched until the closing `>` id is observed, then resume masking for the next fixed fragment.
   - Stop constraining after the final fixed segment (allow EOS/free continuation).

3) **Integration**
   - Wire the processor into `src/molgen3D/evaluation/inference.py` inside `process_batch` by building per-prompt templates and passing a `LogitsProcessorList` to `model.generate`.
   - Keep existing generation config untouched aside from adding the processor; ensure compatibility with padding/attention masks.

4) **Testing**
   - Unit tests under `tests/inference/`:
     - Template-building sanity: short SMILES (e.g., `CCO`, ring closures, branches) produce expected segments and token IDs.
     - Processor masking: with a toy logits tensor, verify that non-matching tokens are suppressed and coordinates remain free.
     - Batch behavior: two different SMILES in one batch advance independently.
   - Optional smoke script/notebook to run against `outputs/geom_drugs_data/test_smiles.csv` and confirm decoded SMILES matches the prompt while coordinates vary.

## Progress log
- Implemented `src/molgen3D/evaluation/constrained_logits.py` with template builder + `ConformerConstraintLogitsProcessor`.
- Integrated processor into `process_batch` in `src/molgen3D/evaluation/inference.py`.
- Added baseline unit tests in `tests/inference/test_constrained_logits.py`.
- Drafted constrained conformer smoke harness plan (2025-12-01) covering helper module, CLI, and pytest entry.
- Implemented reusable harness (`src/molgen3D/evaluation/constrained_smoke.py`), CLI runner (`scripts/run_constrained_smoke.py`), and pytest coverage (`tests/inference/test_constrained_smoke.py`); added plan + documentation hooks.
- Hardened coordinate handling by forcing `<` tokens at the start of every coordinate block inside `ConformerConstraintLogitsProcessor`, updating tests and smoke CLI accordingly after real-data smoke runs exposed prefix drift.

## Open choices to revisit
- Whether to later freeze the `<`, `,`, `>` delimiters inside coordinate blocks for stricter formatting.
- Support for multiple conformers per prompt (repeat `[CONFORMER]...[/CONFORMER]` blocks) once needed.
- How to handle prompts that deliberately include a priming token after `[/SMILES]`; template builder should accept an optional trailing fragment.
