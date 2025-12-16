# Outputs & Smoke Experiments

This directory collects smoke-test results and quick experiments around three CPU/GPU-side optimizations, measured with and without logit processing:

- Static KV cache vs Dynamic KV cache
- Batched template building (parallelizing CPU prep)
- `cache_tokenizer` / precomputed tokenizer state
- Logit processing (constrained logits on/off, plus SMILES stripping/distinct sampling)

For experiment context and methodology, see `docs/inference_attention_investigation.md` and `docs/constrained_logits.md`.

## Quick navigation
- `smoke/experiments/baseline/` — reference runs (no special opts); `3070_baseline.json`.
- `smoke/experiments/kv_cache/static/` — static KV cache only (H100 MG4); `h100_mg4_static_only.json`.
- `smoke/experiments/batched_template_building/` — batched/parallel CPU template prep; 3070 runs.
- `smoke/experiments/old_v2_logitprocessor_precompute/` — `cache_tokenizer` / precompute variants (v2_precompute* files).
- `smoke/experiments/logit_processing/` — effect of constrained logits:
  - `lp_vs_no_lp/` — LP on/off comparisons (static & dynamic KV, H100).
  - `strip_smiles_distinct/` — SMILES stripping / distinct-sampling tweaks.
- `smoke/experiments/combinations/` — stacks of the three optimizations:
  - `kv_static_parallel_cache/` — static KV + parallel template building (+ precompute where applicable) on H100.
  - `kv_dynamic_parallel_cache/` — dynamic KV + parallel template building (+ precompute where applicable) on H100, incl. newer torch run.
- `smoke/experiments/cache_reuse/` — cache reuse across generations (`cache_only`, `multigen_cache`, `static_cache_test`).
- `smoke/experiments/performance_scaling/` — scaling batch parallelism (e.g., `h100_parallel_256.json`).
- `smoke/diagnostics/` — failure/debug summaries (e.g., decode errors).
- `smoke/notebooks/` — exploratory notebooks for the above experiments.

## File naming hints
- Hardware: `h100_*` vs `3070_*` in filenames indicate GPU used.
- `static` / `dynamic` — KV cache mode; `parallel` indicates batched template building; `all` means a bundle of CPU optimizations (parallel + tokenizer caching).
- `_lp` / `_no_lp` — logit processor on/off; other logit experiments live under `logit_processing/strip_smiles_distinct/`.

## How to read
1) Identify the optimization combo you care about from the navigation above.
2) Drop into the matching folder; filenames encode hardware and toggles.
3) Cross-reference metrics/observations with `docs/inference_attention_investigation.md` (attention + CPU opt notes) and `docs/constrained_logits.md` (logit processor behavior).

If a new experiment mixes these knobs differently, add it under `smoke/experiments/combinations/` (with a short README or filename note) and update this map so an LLM can route to it quickly.
