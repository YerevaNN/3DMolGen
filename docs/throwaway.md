  python scripts/logit_processor/run_logit_processor_smoke.py \
      --model-alias qwen3_06b_pre \
      --model-step 40k \
      --tokenizer-name qwen3_0.6b_custom \
      --processor-type generic \
      --attention sdpa \
      --sample-size 1000 \
      --batch-size 128 \
      --parallel-templates \
      --json-report outputs/smoke/qwen_generic_lp_1000samples_h100.json \
      --submit h100




python scripts/logit_processor/run_logit_processor_smoke.py \
    --model-alias qwen3_06b_pre \
    --model-step 40k \
    --tokenizer-name qwen3_0.6b_custom \
    --processor-type qwen \
    --attention sdpa \
    --sample-size 256 \
    --batch-size 32 \
    --parallel-templates \
    --json-report outputs/smoke/qwen_lp_256samples_h100_batch128.json \
    --submit h100

python scripts/logit_processor/run_logit_processor_smoke.py \
    --model-alias qwen3_06b_pre \
    --model-step 40k \
    --tokenizer-name qwen3_0.6b_custom \
    --processor-type qwen \
    --attention sdpa \
    --sample-size 256 \
    --batch-size 128 \
    --parallel-templates \
    --json-report outputs/smoke/qwen_lp_256samples_h100_batch128.json \
    --submit h100


# v39

## no lp but with qwen3 sampling
python scripts/logit_processor/run_logit_processor_smoke.py \
    --model-alias qwen3_06b_pre --model-step 40k \
    --tokenizer-name qwen3_0.6b_custom \
    --no-logit-processor \
    --sampling-config qwen3 \
    --sample-size 500 --batch-size 64 \
    --json-report outputs/smoke/qwen_no_lp_v39_qwen3sampling_500.json \
    --submit a100

## lp but with qwen3 sampling
python scripts/logit_processor/run_logit_processor_smoke.py \
    --model-alias qwen3_06b_pre --model-step 40k \
    --tokenizer-name qwen3_0.6b_custom --processor-type qwen \
    --sampling-config qwen3 \
    --sample-size 500 --batch-size 64 \
    --json-report outputs/smoke/qwen_lp_v39_qwen3sampling_500.json \
    --submit a100

## greedy but with qwen3 sampling
  python scripts/logit_processor/run_logit_processor_smoke.py \
  --model-alias qwen3_06b_pre --model-step 40k \
  --tokenizer-name qwen3_0.6b_custom --processor-type qwen \
  --sampling-config qwen3 \
  --sample-size 500 --batch-size 64 \
  --json-report outputs/smoke/qwen_lp_v39_qwen3sampling_500.json \
  --submit a100


## greedy but with no lp
python scripts/logit_processor/run_logit_processor_smoke.py \
    --model-alias qwen3_06b_pre --model-step 40k \
    --tokenizer-name qwen3_0.6b_custom --no-logit-processor \
    --sampling-config greedy \
    --sample-size 500 --batch-size 64 \
    --json-report outputs/smoke/qwen_no_lp_greedy_v39_qwen3sampling_500.json \
    --submit a100

## top_p_sampling4 but with qwen lp
python scripts/logit_processor/run_logit_processor_smoke.py \
    --model-alias qwen3_06b_pre --model-step 40k \
    --tokenizer-name qwen3_0.6b_custom --processor-type qwen \
    --sampling-config top_p_sampling4 \
    --sample-size 500 --batch-size 64 \
    --json-report outputs/smoke/qwen_lp_top_p_sampling4_v39_qwen3_sampling_500.json \
    --submit a100


## top_p_sampling1 but with qwen lp and all 1000 samples
python scripts/logit_processor/run_logit_processor_smoke.py \
    --model-alias qwen3_06b_pre --model-step 40k \
    --tokenizer-name qwen3_0.6b_custom --processor-type qwen \
    --sampling-config top_p_sampling1 \
    --sample-size 1000 --batch-size 96 \
    --json-report outputs/smoke/qwen_lp_top_p_sampling1_v39_qwen3_sampling_1000.json \
    --submit a100

## top_p_sampling1 but with qwen lp and all 1000 samples and parallel templates and cache tokenizer
python scripts/logit_processor/run_logit_processor_smoke.py \
    --model-alias qwen3_06b_pre --model-step 40k \
    --tokenizer-name qwen3_0.6b_custom --processor-type qwen \
    --sampling-config top_p_sampling1 \
    --sample-size 1000 --batch-size 96 \
    --parallel-templates \
    --cache-tokenizer \
    --json-report outputs/smoke/qwen_lp_top_p_sampling1__pt_ct_v39_qwen3_sampling_1000.json \
    --submit a100


## top_p_sampling1 but with qwen lp and all 128 samples and parallel templates and cache tokenizer and 4 generations per sample (to test cache tokenizer)
python scripts/logit_processor/run_logit_processor_smoke.py \
    --model-alias qwen3_06b_pre --model-step 40k \
    --tokenizer-name qwen3_0.6b_custom --processor-type qwen \
    --sampling-config top_p_sampling1 \
    --sample-size 128 --batch-size 96 \
    --num-generations 4 \
    --parallel-templates \
    --cache-tokenizer \
    --json-report outputs/smoke/qwen_lp_top_p_sampling1__pt_ct_4gens_v39_qwen3_sampling_128.json \
    --submit a100

## top_p_sampling1 but with qwen lp and all 128 samples (no cache tokenzier or parllel templates) and 4 generations per sample (to test cache tokenizer)
python scripts/logit_processor/run_logit_processor_smoke.py \
    --model-alias qwen3_06b_pre --model-step 40k \
    --tokenizer-name qwen3_0.6b_custom --processor-type qwen \
    --sampling-config top_p_sampling1 \
    --sample-size 128 --batch-size 96 \
    --num-generations 4 \
    --json-report outputs/smoke/qwen_lp_top_p_sampling1_no_pt_no_ct_4gens_v39_qwen3_sampling_128.json \
    --submit a100

## top_p_sampling1 but without lp and all 1000 samples
python scripts/logit_processor/run_logit_processor_smoke.py \
    --model-alias qwen3_06b_pre --model-step 40k \
    --tokenizer-name qwen3_0.6b_custom --no-logit-processor \
    --sampling-config top_p_sampling1 \
    --sample-size 1000 --batch-size 96 \
    --json-report outputs/smoke/qwen_no_lp_top_p_sampling1_v39_qwen3_sampling_1000.json \
    --submit a100


## greedy but with qwen lp and all 1000 samples
python scripts/logit_processor/run_logit_processor_smoke.py \
    --model-alias qwen3_06b_pre --model-step 40k \
    --tokenizer-name qwen3_0.6b_custom --processor-type qwen \
    --sampling-config greedy \
    --sample-size 1000 --batch-size 96 \
    --json-report outputs/smoke/qwen_lp_greedy_v39_qwen3_sampling_1000.json \
    --submit a100


## top_p_sampling1 but with qwen lp and all 1000 samples and 4 generations per sample
python scripts/logit_processor/run_logit_processor_smoke.py \
    --model-alias qwen3_06b_pre --model-step 40k \
    --tokenizer-name qwen3_0.6b_custom --processor-type qwen \
    --sampling-config top_p_sampling1 \
    --sample-size 1000 --batch-size 96 \
    --num-generations 4 \
    --json-report outputs/smoke/qwen_lp_top_p_sampling1_v39_qwen3_sampling_1000_4generations.json \
    --submit a100




===============================

# v40 (extended blocklist)

## Blocklist v4.0 top_p_sampling1 but with qwen lp and all 1000 samples on a100
python scripts/logit_processor/run_logit_processor_smoke.py \
  --model-alias qwen3_06b_pre --model-step 40k \
  --tokenizer-name qwen3_0.6b_custom --processor-type qwen \
  --sampling-config top_p_sampling1 \
  --sample-size 1000 --batch-size 96 \
  --json-report outputs/smoke/qwen_lp_v40_top_p_sampling1_1000.json \
  --submit a100

## Blocklist v4.0 top_p_sampling1 but with qwen lp and all 1000 samples on h100
python scripts/logit_processor/run_logit_processor_smoke.py \
  --model-alias qwen3_06b_pre --model-step 40k \
  --tokenizer-name qwen3_0.6b_custom --processor-type qwen \
  --sampling-config top_p_sampling1 \
  --sample-size 1000 --batch-size 96 \
  --json-report outputs/smoke/qwen_lp_v40_top_p_sampling1_1000_h100.json \
  --submit h100

## Allowlist v4.1 (strict - only 66 tokens) (on a100)
python scripts/logit_processor/run_logit_processor_smoke.py \
  --model-alias qwen3_06b_pre --model-step 40k \
  --tokenizer-name qwen3_0.6b_custom --processor-type allowlist-qwen \
  --sampling-config top_p_sampling1 \
  --sample-size 1000 --batch-size 96 \
  --json-report outputs/smoke/qwen_lp_v41_allowlist_1000.json \
  --submit a100

## Allowlist v4.1 (strict - only 66 tokens) (on h100)
python scripts/logit_processor/run_logit_processor_smoke.py \
  --model-alias qwen3_06b_pre --model-step 40k \
  --tokenizer-name qwen3_0.6b_custom --processor-type allowlist-qwen \
  --sampling-config top_p_sampling1 \
  --sample-size 1000 --batch-size 96 \
  --json-report outputs/smoke/qwen_lp_v41_allowlist_1000_h100.json \
  --submit h100

## no lp with 1000 samples on h100 and top_p_sampling1
python scripts/logit_processor/run_logit_processor_smoke.py \
  --model-alias qwen3_06b_pre --model-step 40k \
  --tokenizer-name qwen3_0.6b_custom --no-logit-processor \
  --sampling-config top_p_sampling1 \
  --sample-size 1000 --batch-size 96 \
  --json-report outputs/smoke/qwen_no_lp_top_p_sampling1_1000_h100.json \
  --submit h100

## Test v4.2 (extended allowlist) (on h100)

  python scripts/logit_processor/run_logit_processor_smoke.py \
    --model-alias qwen3_06b_pre --model-step 40k \
    --tokenizer-name qwen3_0.6b_custom --processor-type allowlist-qwen \
    --sampling-config top_p_sampling1 \
    --sample-size 1000 --batch-size 96 \
    --json-report outputs/smoke/qwen_lp_v42_allowlist_1000.json \
    --submit h100


  ## Summary of v4.3 Changes

  | Position             | Blocks         | Reason                                                        |
  |----------------------|----------------|---------------------------------------------------------------|
  | First FREE (after <) | Comma only     | Coords don't start with , (but can start with - for negative) |
  | Last FREE (before >) | Comma AND dash | No trailing punctuation like ,> or ->                         |

  Reverted: LOOKAHEAD_RANGE back to 2 (v4.2's 3 caused 1000 failures)

  Test Command

python scripts/logit_processor/run_logit_processor_smoke.py \
  --model-alias qwen3_06b_pre --model-step 40k \
  --tokenizer-name qwen3_0.6b_custom --processor-type allowlist-qwen \
  --sampling-config top_p_sampling1 \
  --sample-size 1000 --batch-size 96 \
  --json-report outputs/smoke/qwen_lp_v43_allowlist_smart_1000.json \
  --submit h100

===============================

# `Inference.py` full runs

## h100

##  batch size 128 lead to ~10 it/s. h100


## oom w/out cuda mem mgmt changes in inference.py after 5 hours (h100)
python -m molgen3D.evaluation.inference \
  --device h100 \
  --test_set distinct \
  --logit-processor \
  --attention sdpa \
  --batch-size 512


python -m molgen3D.evaluation.inference \
  --device h100 \
  --test_set distinct \
  --logit-processor \
  --attention sdpa \
  --batch-size 1024


## oom with cuda memory management changes in inference.py after 7 hours (h100)
python -m molgen3D.evaluation.inference \
  --device h100 \
  --test_set distinct \
  --logit-processor \
  --attention sdpa \
  --batch-size 768

## a100

## safe batch size
 python -m molgen3D.evaluation.inference \
    --device a100 \
    --test_set distinct \
    --logit-processor \
    --attention sdpa \
    --batch-size 192

## aggressive batch size
 python -m molgen3D.evaluation.inference \
    --device a100 \
    --test_set distinct \
    --logit-processor \
    --attention sdpa \
    --batch-size 256
















===============================

# profiling
  python scripts/logit_processor/run_logit_processor_smoke.py \
  --model-alias qwen3_06b_pre \
  --model-step 40k \
  --tokenizer-name qwen3_0.6b_custom \
  --processor-type qwen \
  --attention sdpa \
  --sample-size 64 \
  --batch-size 32 \
  --parallel-templates \
  --profile \
  --profile-output outputs/profiles/qwen_lp_profile_h100.json \
  --json-report outputs/smoke/qwen_lp_64samples_h100_profiled.json \
  --submit h100



===============================

# vectorized allowlist logit processor attempts


## v1

 ## v4.3 (allowlist-qwen):
python scripts/logit_processor/run_logit_processor_smoke.py \
  --sample-size 64 \
  --num-generations 1 \
  --batch-size 32 \
  --model-alias qwen3_06b_pre \
  --model-step 40k \
  --tokenizer-name qwen3_0.6b_custom \
  --sampling-config top_p_sampling1 \
  --processor-type allowlist-qwen \
  --attention sdpa \
  --json-report outputs/smoke/vectorized/v43_allowlist_64_topps1_h100.json \
  --submit h100

  ## v5.0 (vectorized-allowlist-qwen):
  python scripts/logit_processor/run_logit_processor_smoke.py \
    --sample-size 64 \
    --num-generations 1 \
    --batch-size 32 \
    --model-alias qwen3_06b_pre \
    --model-step 40k \
    --tokenizer-name qwen3_0.6b_custom \
    --sampling-config top_p_sampling1 \
    --processor-type vectorized-allowlist-qwen \
    --attention sdpa \
    --json-report outputs/smoke/vectorized/v50_vectorized_64_topps1_h100.json \
    --submit h100

## v5.0 (vectorized-allowlist-qwen): v2 larger sample
python scripts/logit_processor/run_logit_processor_smoke.py \
  --sample-size 256 \
  --num-generations 1 \
  --batch-size 64 \
  --model-alias qwen3_06b_pre \
  --model-step 40k \
  --tokenizer-name qwen3_0.6b_custom \
  --sampling-config top_p_sampling1 \
  --processor-type vectorized-allowlist-qwen \
  --attention sdpa \
  --json-report outputs/smoke/vectorized/v50_vectorized_64_topps1_h100_v2_larger_sample.json \
  --submit h100

## baseline v4.3 (allowlist-qwen) with larger sample size
python scripts/logit_processor/run_logit_processor_smoke.py \
  --sample-size 256 \
  --num-generations 1 \
  --batch-size 64 \
  --model-alias qwen3_06b_pre \
  --model-step 40k \
  --tokenizer-name qwen3_0.6b_custom \
  --sampling-config top_p_sampling1 \
  --processor-type allowlist-qwen \
  --attention sdpa \
  --json-report outputs/smoke/vectorized/v43_allowlist_64_topps1_h100_larger_sample.json \
  --submit h100

## v5.0 (vectorized-allowlist-qwen): v2 1000 sample
python scripts/logit_processor/run_logit_processor_smoke.py \
  --sample-size 1000 \
  --num-generations 1 \
  --batch-size 96 \
  --model-alias qwen3_06b_pre \
  --model-step 40k \
  --tokenizer-name qwen3_0.6b_custom \
  --sampling-config top_p_sampling1 \
  --processor-type vectorized-allowlist-qwen \
  --attention sdpa \
  --json-report outputs/smoke/vectorized/v50_vectorized_1000_topps1_h100_v2.json \
  --submit h100


===============================

# benchmark_inference_attention.py

python scripts/diagnostics/benchmark_inference_attention.py \
  --model m600_qwen \
  --num-samples 1024 \
  --batch-size 192 \
  --attention sdpa sdpa_cudnn sdpa_efficient eager flash_attention_2 \
  --device a100