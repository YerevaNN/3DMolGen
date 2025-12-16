#!/bin/bash
# Compare Qwen vs LLaMA profiling traces
# Run this after collecting both traces

set -e

QWEN_TRACE="outputs/profiles/qwen_lp_profile_h100.json.gz"
LLAMA_TRACE="outputs/profiles/llama_lp_profile_h100.json.gz"

echo "=========================================="
echo "Qwen vs LLaMA LP Profiling Comparison"
echo "=========================================="
echo ""

# Check both traces exist
if [ ! -f "$QWEN_TRACE" ]; then
    echo "ERROR: Qwen trace not found at $QWEN_TRACE"
    exit 1
fi

if [ ! -f "$LLAMA_TRACE" ]; then
    echo "ERROR: LLaMA trace not found at $LLAMA_TRACE"
    echo "Run this command first:"
    echo ""
    echo "python scripts/logit_processor/run_logit_processor_smoke.py \\"
    echo "  --model-alias m380_conf_v2 --model-step 2e \\"
    echo "  --tokenizer-name llama3_chem_v1 --processor-type generic \\"
    echo "  --attention sdpa --sample-size 64 --batch-size 32 \\"
    echo "  --parallel-templates \\"
    echo "  --profile --profile-output $LLAMA_TRACE \\"
    echo "  --json-report outputs/smoke/llama_lp_64samples_h100_profiled.json \\"
    echo "  --submit h100"
    echo ""
    exit 1
fi

echo "Both traces found. Running comparison queries..."
echo ""

# Function to run query on both traces
run_comparison() {
    local query="$1"
    local description="$2"

    echo "=========================================="
    echo "$description"
    echo "=========================================="
    echo ""

    echo "QWEN:"
    echo "$query" | trace_processor_shell "$QWEN_TRACE" 2>&1 | grep -A 20 "name\|count\|ms\|sec" | head -25
    echo ""

    echo "LLAMA:"
    echo "$query" | trace_processor_shell "$LLAMA_TRACE" 2>&1 | grep -A 20 "name\|count\|ms\|sec" | head -25
    echo ""
    echo ""
}

# 1. Total trace duration
run_comparison \
    "SELECT dur/1e9 as total_seconds FROM slice WHERE name LIKE 'PyTorch Profiler%' LIMIT 1;" \
    "1. Total Trace Duration"

# 2. Total kernel launches
run_comparison \
    "SELECT name, COUNT(*) as launches, SUM(dur)/1e6 as total_ms FROM slice WHERE name LIKE '%LaunchKernel%' GROUP BY name ORDER BY total_ms DESC;" \
    "2. Kernel Launch Overhead"

# 3. Attention kernel performance
run_comparison \
    "SELECT name, COUNT(*) as count, SUM(dur)/1e6 as total_ms, AVG(dur)/1e3 as avg_us FROM slice WHERE name LIKE '%fmha%' OR name LIKE '%attention%' GROUP BY name ORDER BY total_ms DESC LIMIT 5;" \
    "3. Flash Attention Performance"

# 4. Memory operations
run_comparison \
    "SELECT name, COUNT(*) as count, SUM(dur)/1e6 as total_ms FROM slice WHERE name LIKE '%copy%' OR name LIKE '%Copy%' GROUP BY name ORDER BY total_ms DESC LIMIT 10;" \
    "4. Memory Copy Operations"

# 5. Elementwise kernels (includes masking)
run_comparison \
    "SELECT COUNT(*) as total_elementwise, SUM(dur)/1e6 as total_ms FROM slice WHERE name LIKE '%elementwise%';" \
    "5. Elementwise Operations (includes LP masking)"

# 6. Small frequent operations (potential overhead)
run_comparison \
    "SELECT name, COUNT(*) as launches FROM slice WHERE dur < 10000 AND name NOT LIKE 'Iteration%' GROUP BY name HAVING launches > 10000 ORDER BY launches DESC LIMIT 10;" \
    "6. Small Frequent Operations (<10us, >10K launches)"

echo "=========================================="
echo "Comparison Complete"
echo "=========================================="
echo ""
echo "Key metrics to check:"
echo "1. Total duration: Qwen should be ~3x slower than LLaMA"
echo "2. Kernel launches: Qwen should have ~2x more launches"
echo "3. Elementwise ops: Qwen should have ~2-3x more (extra masking)"
echo "4. Memory copies: Qwen should have more if creating intermediate tensors"
echo ""
echo "See outputs/profiles/LLAMA_VS_QWEN_HYPOTHESIS.md for detailed analysis"
