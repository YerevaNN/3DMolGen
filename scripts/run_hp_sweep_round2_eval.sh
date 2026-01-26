#!/bin/bash
# HP Sweep Round 2 Evaluation Script
# Submits RMSD evaluation jobs for round 2 HP sweep generation results

# === CONFIGURABLE PARAMETERS ===
NUM_WORKERS=12          # Workers per job (108 CPUs / 9 jobs ≈ 12)
DEVICE="a100"           # Slurm partition: a100, h100, all
POSEBUSTERS="None"      # None = skip PoseBusters, "mol" or "redock" to enable
TEST_SET="distinct"     # Test set: clean, distinct, xl, qm9
MEMORY_GB=40            # Memory per job
BATCH_SIZE=64           # Batch size for RMSD computation
# ===============================

# HP sweep round 2 directories from 2026-01-22 run
DIRS=(
    "20260122_131702_qw600_pre_binned_filtered_4e_min_p_r2_1_distinct"
    "20260122_131702_qw600_pre_binned_filtered_4e_min_p_r2_2_distinct"
    "20260122_131702_qw600_pre_binned_filtered_4e_min_p_r2_3_distinct"
    "20260122_131702_qw600_pre_binned_filtered_4e_top_p_r2_1_distinct"
    "20260122_131702_qw600_pre_binned_filtered_4e_top_p_r2_2_distinct"
    "20260122_131702_qw600_pre_binned_filtered_4e_top_p_r2_3_distinct"
    "20260122_141250_qw600_pre_binned_filtered_4e_top_k_r2_1_distinct"
    "20260122_165820_qw600_pre_binned_filtered_4e_top_k_r2_2_distinct"
    "20260122_165950_qw600_pre_binned_filtered_4e_top_k_r2_3_distinct"
)

echo "Submitting ${#DIRS[@]} evaluation jobs..."
echo "  Workers per job: $NUM_WORKERS"
echo "  Device: $DEVICE"
echo "  PoseBusters: $POSEBUSTERS"
echo "  Test set: $TEST_SET"
echo ""

for dir in "${DIRS[@]}"; do
    echo "Submitting: $dir"
    python -m molgen3D.evaluation.run_eval_optimized \
        --specific-dir "$dir" \
        --posebusters "$POSEBUSTERS" \
        --num-workers "$NUM_WORKERS" \
        --memory-gb "$MEMORY_GB" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --test_set "$TEST_SET"
done

echo ""
echo "All jobs submitted. Check queue with: sqr"
