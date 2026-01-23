#!/bin/bash
# HP Sweep Round 2 - Inference Script
# Runs generation with fixed temperature (1.0), varying parameters

# === CONFIGURABLE PARAMETERS ===
DEVICE="a100"           # Slurm partition: a100, h100, all
TEST_SET="distinct"     # Test set: clean, distinct, xl, qm9
BINNED="--binned"       # Use binned model
# ===============================

echo "HP Sweep Round 2 - Inference"
echo "  Device: $DEVICE"
echo "  Test set: $TEST_SET"
echo "  Binned: $BINNED"
echo ""
echo "Configs: top_p=[0.8,0.9,0.95], min_p=[0.05,0.1,0.15], top_k=[20,50,100] @ temp=1.0"
echo ""

python -m molgen3D.evaluation.inference \
    --device "$DEVICE" \
    --grid_run_inference \
    $BINNED \
    --test_set "$TEST_SET"

echo ""
echo "Jobs submitted. Check queue with: sqr"
