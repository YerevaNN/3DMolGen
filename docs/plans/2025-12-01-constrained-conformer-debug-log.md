# Constrained Conformer Smoke Debug Log (through v11)

## Context
- Goal: ensure constrained generation keeps SMILES structure fixed while allowing numeric coordinates.
- Smoke harness: `scripts/run_constrained_smoke.py` sampling SMILES from GEOM pickle, using logits processor and real checkpoint. Reports passes/failures with decoded text.
- Current checkpoint used in smokes: `/nfs/h100/raid/chem/checkpoints/hf/yerevann/Llama-3.2-380M_conformers/fea47a381e4046e4956dc44a/step-26100` with tokenizer `llama3_chem_v1` and attention `sdpa_paged`.

## Constraint Processor Evolution
1. **Baseline**: froze non-coord tokens, coords unconstrained -> model injected tags and garbage before `>`.
2. Forced leading `<` per coord block; added forbidden tag IDs. Still leaked tag fragments due to BPE splits.
3. Added vocab-derived forbidden set (tokens containing `[`, `]`, `SMILES`, `CONFORMER`, BOS/EOS). Still leaked SMILES fragments.
4. Tried allowlist-only coords -> produced unreadable numeric blobs; removed/adjusted.
5. Added max coord length guard (currently 6 tokens). If no `>` by then, logits are forced to `>`.
6. Current allowlist: tokens whose text (minus SentencePiece boundary) is only `0123456789+-.,eE` plus `>`; everything else masked inside coords. Forbidden set retained for safety.

## Latest Smoke (v10)
- Command: see consolidation instructions above (clean set, sample 64, batch 16, bfloat16, sdpa_paged).
- Results: **19 passed / 45 failed**.
- Failure reason: all `conformer SMILES mismatch`.
- Pass example prompt: `COc1ccc(N2C(=O)c3ccc(C(=O)OC(C)C)cc3C2=O)cc1`.
- Fail example prompt: `Cc1ccc(CSCCNC(=O)CN(c2cccc([N+](=O)[O-])c2)S(C)(=O)=O)cc1`.
- Typical decoded head (fail):
  - `…<|begin_of_text|>[SMILES]...[/SMILES][CONFORMER][C]<120157094632576653>[c]<384196805974122015>1[c]<4192895506,,,,,,,,984>…` → coord tokens numeric but contain multi-digit junk and occasionally commas; structural tokens like `[c]` appear inside coords because they’re allowed via labels in the template (atom segments) followed immediately by numeric blocks.
- Typical decoded head (pass): still prefixed by multiple `<|end_of_text|>` markers before the prompt; conformer block is extremely numeric (`[C]<970103540803469640--,037843...`), but passed the graph check.

## Known Issues / Hypotheses
- **Special tokens leaking**: `<|end_of_text|>` prefixes remain in decoded text; we are not masking BOS/EOS in coord blocks because they are in the forbidden set, but they appear before the prompt. Might need `skip_special_tokens=True` on decode or to trim leading specials post-generate.
- **Coord allowlist too loose for delimiter punctuation**: commas are allowed; multi-digit concatenations like `120157094632576653` cause `strip_smiles` to leave numerals in SMILES before `>`; consider restricting coord charset further (e.g., allow only digits, sign, dot, maybe limited length per number, enforce pattern `<number,number,number>`).
- **Max coord tokens may still allow malformed sequences**: six tokens could be huge integers; need regex-style enforcement per coord triplet rather than token count.
- **Prompt padding effects**: multiple `<|end_of_text|>` suggest the model’s BOS/EOS specials are in the input; we aren’t stripping them before validation.

## Artifacts to Inspect
- Latest report: `outputs/smoke/constrained_clean_64_v10.json` (includes passes array and failures with decoded_text).
- Logs from earlier runs: v5–v9 in `outputs/smoke/` for regression comparisons.

## Ideas for Next Tool / Next Steps
1. Add post-processing: trim leading special tokens from decoded text before validation; consider `skip_special_tokens=True` in `tokenizer.batch_decode`.
2. Enforce coordinate pattern at token level: require `<` then (num [, num [, num]) then `>`; mask commas and very long integer tokens; or detect if token string contains non-digit/punct and force `>`.
3. Lower `max_coord_tokens` further or split into per-dimension counts.
4. Add debug logging inside logits processor (optional flag) to record first divergence token and step length for failing examples during smoke.
5. Check tokenizer vocab for large integer string tokens; optionally exclude multi-digit plain tokens from coord allowlist.

## Current Code State
- Logits processor: `src/molgen3D/evaluation/constrained_logits.py` (allowlist + max 6 coord tokens + forbidden set).
- Smoke harness: `scripts/run_constrained_smoke.py` writes passes/failures.
- Tests: `tests/inference/test_constrained_logits.py`, `test_constrained_smoke.py` passing.

## Quick Repro Command (current state)
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate 3dmolgen
python scripts/run_constrained_smoke.py \
  --dataset clean \
  --sample-size 64 \
  --batch-size 16 \
  --model-path /nfs/h100/raid/chem/checkpoints/hf/yerevann/Llama-3.2-380M_conformers/fea47a381e4046e4956dc44a/step-26100 \
  --tokenizer-name llama3_chem_v1 \
  --sampling-config top_p_sampling1 \
  --device cuda \
  --torch-dtype bfloat16 \
  --attention sdpa_paged \
  --json-report outputs/smoke/constrained_clean_64_v10.json \
  --fail-fast
```

## v11: Minimal Blocklist Approach (2025-12-01)

### Changes
- **Philosophy shift**: Remove tight allowlist, trust model to generate coordinates
- **Approach**: Simple blocklist - only block known-bad tokens
- **Blocklist criteria**:
  1. Tokens containing `[`, `]`, `SMILES`, `CONFORMER`, BOS/EOS tags
  2. Very long tokens (>8 chars) - prevents `"120157094632576653"` style garbage
  3. Repeated punctuation (e.g., `",,,,,"`, `"-----"`)
- **Decode fix**: Post-process to strip `<|begin_of_text|>` and `<|end_of_text|>` without removing structural tags
- **Max coord tokens**: Increased from 6 to 20 to give model more freedom
- **Analysis tools**: Added `scripts/analyze_smoke_report.py` for rapid analysis, enhanced JSON reports with metadata

### Results (v11 vs v10)
| Metric | v10 (Tight Allowlist) | v11 (Minimal Blocklist) |
|--------|----------------------|------------------------|
| Pass rate | 19/64 (29.7%) | 2/64 (3.1%) ❌ |
| Coord quality (valid format) | 0.8% | 28.9% ✅ |
| Special token pollution | All samples | None ✅ |
| Main failure | conformer SMILES mismatch (45) | conformer SMILES mismatch (62) |

### Key Findings
1. **Special token pollution SOLVED** ✅: Post-processing successfully removes BOS/EOS without breaking structural tags
2. **Coordinate quality MUCH better** ✅: 28.9% of coords now match `<float,float,float>` pattern (vs 0.8% in v10)
3. **Pass rate WORSE** ❌: 2/64 vs 19/64 - model diverges more often from SMILES structure
4. **Root cause of failures**: Coord blocks contain SMILES structural characters!
   - Example: `[C]<1O-P@@+)(OC)C(C)(C)C)cc(OC)c>`
   - Coord block has `)`, `(`, `c`, `C`, `O` - these are valid SMILES tokens
   - Model is hallucinating SMILES fragments inside coordinate blocks

### Analysis
**Why pass rate decreased:**
- v10's tight allowlist (only `0123456789+-.,eE>`) prevented SMILES char leakage
- v11's minimal blocklist only blocks brackets/tags, but allows `(`, `)`, `c`, `C`, `N`, `O`, `S`, digits, etc.
- These are valid SMILES structural characters that slip through
- Model generates them in coord blocks, causing `strip_smiles` to extract wrong structure

**Trade-off discovered:**
- **Tight constraints** → Better SMILES fidelity but garbage coordinates
- **Loose constraints** → Better coordinate format but SMILES drift

### Ideas for v12
Need a middle ground that blocks SMILES structural characters while allowing numeric tokens:

**Option A: Enhanced Blocklist**
- Block parentheses: `(`, `)`
- Block aromatic/element symbols when standalone: `c`, `C`, `N`, `O`, `S`, `P`, `F`, `Cl`, `Br`, etc.
- Block equals/hash (bonds): `=`, `#`
- Block slash (stereochemistry): `/`, `\`
- Challenge: Need to distinguish ring digit `"1"` from numeric `"1"` in context

**Option B: Allowlist with Better Validation**
- Revert to allowlist but tighten validation
- Accept only tokens that:
  1. Are pure digits: `"123"`, `"0"`, `"45"`
  2. Are float punctuation: `","`, `"."`, `"-"`, `"+"`
  3. Are scientific notation: `"e"`, `"E"` (only after seeing digits)
  4. Are close bracket: `">"`
- Block everything else

**Option C: Hybrid State Machine** (from brainstorming session)
- Track position in `<x,y,z>` pattern
- Use allowlist when building floats, stricter when expecting delimiters
- More complex but most precise

**Recommendation: Option A (Enhanced Blocklist) for v12**
- Simpler than state machine
- Should combine benefits of v10 (SMILES fidelity) and v11 (coord quality)
- Incremental fix that's easy to test

### Artifacts
- Report: `outputs/smoke/constrained_clean_64_v11.json`
- Analysis script: `scripts/analyze_smoke_report.py`
- Code: `src/molgen3D/evaluation/constrained_logits.py` (VERSION="v11")

### Quick Comparison Command
```bash
python scripts/analyze_smoke_report.py outputs/smoke/constrained_clean_64_v*.json --compare
```

## Hand-off
v11 successfully eliminated special token pollution and improved coordinate format quality (28.9% valid vs 0.8%), but pass rate dropped (2/64 vs 19/64) due to SMILES structural characters leaking into coord blocks. Need v12 with enhanced blocklist to block SMILES chars while preserving numeric freedom.
