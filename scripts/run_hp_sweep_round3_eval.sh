#!/bin/bash
# HP Sweep Round 3 Evaluation Script
# Submits RMSD evaluation jobs for round 3 HP sweep generation results
# Extended temperature range [0.6, 0.7, 1.3, 1.5] with top_k=[30, 50, 70]

# === CONFIGURABLE PARAMETERS ===
NUM_WORKERS=12          # Workers per job (108 CPUs / 12 jobs = 9, using 12 for faster eval)
DEVICE="a100"           # Slurm partition: a100, h100, all
POSEBUSTERS="None"      # None = skip PoseBusters, "mol" or "redock" to enable
TEST_SET="distinct"     # Test set: clean, distinct, xl, qm9
MEMORY_GB=40            # Memory per job
BATCH_SIZE=64           # Batch size for RMSD computation
# ===============================

# HP sweep round 3 directories from 2026-01-23/24 run
DIRS=(
    "20260123_194520_qw600_pre_binned_filtered_4e_top_k_r3_30_t06_distinct"
    "20260123_230422_qw600_pre_binned_filtered_4e_top_k_r3_30_t07_distinct"
    "20260124_022321_qw600_pre_binned_filtered_4e_top_k_r3_30_t13_distinct"
    "20260124_054922_qw600_pre_binned_filtered_4e_top_k_r3_30_t15_distinct"
    "20260124_092625_qw600_pre_binned_filtered_4e_top_k_r3_50_t06_distinct"
    "20260124_093724_qw600_pre_binned_filtered_4e_top_k_r3_50_t07_distinct"
    "20260124_101327_qw600_pre_binned_filtered_4e_top_k_r3_50_t13_distinct"
    "20260124_102026_qw600_pre_binned_filtered_4e_top_k_r3_50_t15_distinct"
    "20260124_115824_qw600_pre_binned_filtered_4e_top_k_r3_70_t06_distinct"
    "20260124_122553_qw600_pre_binned_filtered_4e_top_k_r3_70_t07_distinct"
    "20260124_124554_qw600_pre_binned_filtered_4e_top_k_r3_70_t13_distinct"
    "20260124_125724_qw600_pre_binned_filtered_4e_top_k_r3_70_t15_distinct"
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
