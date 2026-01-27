#!/bin/bash
# HP Sweep Round 4 Evaluation Script
# Submits RMSD evaluation jobs for round 4 HP sweep generation results
# Fill gaps for top_p and min_p temperature coverage

# === CONFIGURABLE PARAMETERS ===
NUM_WORKERS=12          # Workers per job
DEVICE="a100"           # Slurm partition: a100, h100, all
POSEBUSTERS="None"      # None = skip PoseBusters, "mol" or "redock" to enable
TEST_SET="distinct"     # Test set: clean, distinct, xl, qm9
MEMORY_GB=40            # Memory per job
BATCH_SIZE=64           # Batch size for RMSD computation
# ===============================

# HP sweep round 4 directories from 2026-01-26/27 run
DIRS=(
    "20260126_233340_qw600_pre_binned_filtered_4e_top_p_r4_08_t08_distinct"
    "20260126_233340_qw600_pre_binned_filtered_4e_top_p_r4_08_t12_distinct"
    "20260126_233340_qw600_pre_binned_filtered_4e_top_p_r4_95_t08_distinct"
    "20260126_233340_qw600_pre_binned_filtered_4e_top_p_r4_95_t12_distinct"
    "20260127_031556_qw600_pre_binned_filtered_4e_min_p_r4_05_t08_distinct"
    "20260127_031655_qw600_pre_binned_filtered_4e_min_p_r4_05_t12_distinct"
    "20260127_031755_qw600_pre_binned_filtered_4e_min_p_r4_15_t08_distinct"
    "20260127_032056_qw600_pre_binned_filtered_4e_min_p_r4_15_t12_distinct"
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
