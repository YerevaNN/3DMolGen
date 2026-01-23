#!/usr/bin/env python3
"""
Quick runner script for LOQI energy evaluation.

Usage:
    # Single directory (local):
    python run_loqi_eval.py --gen-dir 20260122_093145_m600_qwen_pre_4seq_3e_loqi

    # Batch mode (evaluate all missing, locally):
    python run_loqi_eval.py --device local --max-recent 5

    # Submit to slurm (A100):
    python run_loqi_eval.py --device a100 --max-recent 5

    # Submit to slurm (H100):
    python run_loqi_eval.py --device h100 --specific-dir YOUR_DIR
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Just use the main function from the module
from molgen3D.evaluation.loqi_energy_eval import main

if __name__ == "__main__":
    main()
