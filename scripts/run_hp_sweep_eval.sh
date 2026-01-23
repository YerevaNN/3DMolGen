#!/bin/bash
# HP Sweep Evaluation Script
# Submits RMSD evaluation jobs for all 9 HP sweep generation results

# === CONFIGURABLE PARAMETERS ===
NUM_WORKERS=12          # Workers per job (108 CPUs / 9 jobs ≈ 12)
DEVICE="a100"           # Slurm partition: a100, h100, all
POSEBUSTERS="None"      # None = skip PoseBusters, "mol" or "redock" to enable
TEST_SET="distinct"     # Test set: clean, distinct, xl, qm9
MEMORY_GB=40            # Memory per job
BATCH_SIZE=64           # Batch size for RMSD computation
# ===============================

# HP sweep directories from 2026-01-21 run
DIRS=(
    "20260121_215350_qw600_pre_binned_filtered_4e_top_k_sweep3_distinct"
    "20260121_212950_qw600_pre_binned_filtered_4e_top_k_sweep2_distinct"
    "20260121_181015_qw600_pre_binned_filtered_4e_min_p_sweep3_distinct"
    "20260121_181010_qw600_pre_binned_filtered_4e_min_p_sweep1_distinct"
    "20260121_181010_qw600_pre_binned_filtered_4e_min_p_sweep2_distinct"
    "20260121_181010_qw600_pre_binned_filtered_4e_top_p_sweep1_distinct"
    "20260121_181010_qw600_pre_binned_filtered_4e_top_p_sweep3_distinct"
    "20260121_181010_qw600_pre_binned_filtered_4e_top_p_sweep2_distinct"
    "20260121_181014_qw600_pre_binned_filtered_4e_top_k_sweep1_distinct"
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
