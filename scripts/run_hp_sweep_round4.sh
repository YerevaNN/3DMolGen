#!/bin/bash
# HP Sweep Round 4 - Inference Script
# Fill gaps for top_p and min_p temperature coverage

# === CONFIGURABLE PARAMETERS ===
DEVICE="a100"           # Slurm partition: a100, h100, all
TEST_SET="distinct"     # Test set: clean, distinct, xl, qm9
BINNED="--binned"       # Use binned model
# ===============================

echo "HP Sweep Round 4 - Inference"
echo "  Device: $DEVICE"
echo "  Test set: $TEST_SET"
echo "  Binned: $BINNED"
echo ""
echo "Configs:"
echo "  top_p: p=[0.8, 0.95] @ temps=[0.8, 1.2]"
echo "  min_p: p=[0.05, 0.15] @ temps=[0.8, 1.2]"
echo "Total jobs: 8"
echo ""

python -m molgen3D.evaluation.inference \
    --device "$DEVICE" \
    --grid_run_inference \
    $BINNED \
    --test_set "$TEST_SET"

echo ""
echo "Jobs submitted. Check queue with: sqr"
