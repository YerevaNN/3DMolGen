# Run tests from root of repo with 3dmolgen env activated
# The conda environment is configured to disable user site-packages
# and has pytest installed, so you can simply run:

pytest tests/evaluation/posebusters_check_tests/test_posebusters.py

# Or use:
python -m pytest tests/evaluation/posebusters_check_tests/test_posebusters.py

# Both commands work without needing PYTHONPATH since:
# 1. molgen3D is installed in editable mode (pip install -e .)
# 2. User site-packages is disabled (prevents conflicts)
# 3. pytest is installed in the conda environment