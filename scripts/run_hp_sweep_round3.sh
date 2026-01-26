#!/bin/bash
# HP Sweep Round 3 - Inference Script
# Extended temperature range [0.6, 0.7, 1.3, 1.5] with top_k=[30, 50, 70]

# === CONFIGURABLE PARAMETERS ===
DEVICE="a100"           # Slurm partition: a100, h100, all
TEST_SET="distinct"     # Test set: clean, distinct, xl, qm9
BINNED="--binned"       # Use binned model
# ===============================

echo "HP Sweep Round 3 - Inference"
echo "  Device: $DEVICE"
echo "  Test set: $TEST_SET"
echo "  Binned: $BINNED"
echo ""
echo "Configs: top_k=[30, 50, 70] @ temps=[0.6, 0.7, 1.3, 1.5]"
echo "Total jobs: 12"
echo ""

python -m molgen3D.evaluation.inference \
    --device "$DEVICE" \
    --grid_run_inference \
    $BINNED \
    --test_set "$TEST_SET"

echo ""
echo "Jobs submitted. Check queue with: sqr"
