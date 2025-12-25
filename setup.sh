#!/bin/bash
# =============================================================================
# 3DMolGen Environment Setup Script
# =============================================================================
# One-command installation using conda + uv hybrid approach.
# - Conda: Manages Python 3.10 environment and system deps (rdkit)
# - uv: Fast pip replacement (10-100x faster) for Python packages
#
# Usage:
#   ./setup.sh              # Install all dependencies
#   ./setup.sh --dev        # Include development tools (pytest, black)
#   ./setup.sh --verify     # Only run verification (skip install)
#   ./setup.sh --clean      # Remove environment and start fresh
#
# Requirements:
#   - Linux x86_64
#   - Conda/Miniconda installed
#   - CUDA 12.8 drivers (system-level)
#   - Internet access for downloading packages
# =============================================================================

set -euo pipefail

# Configuration
ENV_NAME="3dmolgen"
PYTHON_VERSION="3.10"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"

# Flash Attention wheel locations (try local first, then GitHub)
FA_WHEEL_LOCAL="/nfs/ap/mnt/sxtn2/chem/wheels/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl"
FA_WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl"

# Script directory (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
INSTALL_EXTRAS=""
VERIFY_ONLY=false
CLEAN_INSTALL=false
for arg in "$@"; do
    case $arg in
        --dev|--all)
            INSTALL_EXTRAS="dev"
            ;;
        --verify)
            VERIFY_ONLY=true
            ;;
        --clean)
            CLEAN_INSTALL=true
            ;;
        --help|-h)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev       Install dev dependencies (pytest, black, ipython, einops)"
            echo "  --verify    Only run environment verification, skip installation"
            echo "  --clean     Remove existing environment and start fresh"
            echo "  --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./setup.sh              # Core (training + inference + eval)"
            echo "  ./setup.sh --dev        # Core + local dev tools"
            echo "  ./setup.sh --clean      # Fresh install"
            exit 0
            ;;
    esac
done

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Step 1: Check conda is available
# =============================================================================
check_conda() {
    if ! command -v conda &> /dev/null; then
        log_error "Conda not found. Please install Miniconda first:"
        echo "  curl -LsSf https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh | bash"
        exit 1
    fi
    log_success "Conda found: $(conda --version)"
}

# =============================================================================
# Step 2: Install uv (fast pip replacement)
# =============================================================================
install_uv() {
    if command -v uv &> /dev/null; then
        log_success "uv already installed: $(uv --version)"
    else
        log_info "Installing uv (fast pip replacement)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add to PATH for this session
        export PATH="$HOME/.local/bin:$PATH"

        if command -v uv &> /dev/null; then
            log_success "uv installed: $(uv --version)"
        else
            log_error "Failed to install uv"
            exit 1
        fi
    fi
}

# =============================================================================
# Step 3: Create/update conda environment
# =============================================================================
setup_conda_env() {
    # Clean install requested
    if [ "$CLEAN_INSTALL" = true ]; then
        log_info "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
    fi

    # Check if environment exists
    if conda env list | grep -q "^${ENV_NAME} "; then
        log_success "Conda environment '$ENV_NAME' exists"
    else
        log_info "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
        log_success "Conda environment created"
    fi

    # Activate environment
    log_info "Activating conda environment..."

    # Source conda for this script
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"

    log_success "Environment activated: $(python --version)"
}

# =============================================================================
# Step 4: Install rdkit via conda (better than pip for this package)
# =============================================================================
install_rdkit() {
    log_info "Checking rdkit..."

    if python -c "import rdkit" 2>/dev/null; then
        log_success "rdkit already installed"
    else
        log_info "Installing rdkit from conda-forge..."
        conda install -c conda-forge rdkit -y
        log_success "rdkit installed"
    fi
}

# =============================================================================
# Step 5: Install PyTorch with CUDA 12.8
# =============================================================================
install_pytorch() {
    log_info "Installing PyTorch 2.9.1+cu128..."

    # Note: We only install torch, not torchvision/torchaudio
    # They add ~2GB and are not needed for molecular generation
    uv pip install torch==2.9.1 --index-url "$PYTORCH_INDEX"

    log_success "PyTorch installed"
}

# =============================================================================
# Step 6: Install Flash Attention from prebuilt wheel
# =============================================================================
install_flash_attention() {
    log_info "Installing Flash Attention 2.8.3..."

    # Try local wheel first (faster)
    if [ -f "$FA_WHEEL_LOCAL" ]; then
        log_info "Using local wheel: $FA_WHEEL_LOCAL"
        uv pip install "$FA_WHEEL_LOCAL"
    else
        log_info "Downloading from GitHub..."
        uv pip install "$FA_WHEEL_URL"
    fi

    log_success "Flash Attention installed"
}

# =============================================================================
# Step 7: Install remaining dependencies
# =============================================================================
install_dependencies() {
    log_info "Installing dependencies from pyproject.toml..."

    cd "$SCRIPT_DIR"

    if [ -n "$INSTALL_EXTRAS" ]; then
        log_info "Including optional dependencies: [$INSTALL_EXTRAS]"
        uv pip install -e ".[$INSTALL_EXTRAS]"
    else
        uv pip install -e .
    fi

    log_success "Dependencies installed"
}

# =============================================================================
# Step 8: Verify environment
# =============================================================================
verify_environment() {
    log_info "Verifying environment..."

    cd "$SCRIPT_DIR"

    if [ -f "verify_env.py" ]; then
        python verify_env.py
    else
        # Inline verification
        python << 'EOF'
import sys

print("=" * 60)
print("3DMolGen Environment Verification")
print("=" * 60)

errors = []

# PyTorch
try:
    import torch
    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version:   {torch.version.cuda}")
        print(f"GPU:            {torch.cuda.get_device_name(0)}")
except ImportError as e:
    errors.append(f"PyTorch: {e}")

# Flash Attention
try:
    import flash_attn
    print(f"Flash Attention: {flash_attn.__version__}")
except ImportError as e:
    errors.append(f"Flash Attention: {e}")

# Core libraries
for lib in ["transformers", "trl", "datasets", "accelerate", "torchtitan"]:
    try:
        mod = __import__(lib)
        version = getattr(mod, "__version__", "installed")
        print(f"{lib}: {version}")
    except ImportError as e:
        errors.append(f"{lib}: {e}")

# RDKit
try:
    from rdkit import Chem
    print(f"RDKit: {Chem.rdBase.rdkitVersion}")
except ImportError as e:
    errors.append(f"RDKit: {e}")

print("=" * 60)

if errors:
    print("ERRORS:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("All checks passed!")
    sys.exit(0)
EOF
    fi
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo ""
    echo "=============================================="
    echo "  3DMolGen Environment Setup"
    echo "  Python 3.10 | PyTorch 2.9.1+cu128 | FA2"
    echo "=============================================="
    echo ""

    check_conda

    if [ "$VERIFY_ONLY" = true ]; then
        # Just verify, need to activate env first
        CONDA_BASE=$(conda info --base)
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
        verify_environment
        exit 0
    fi

    install_uv
    setup_conda_env
    install_rdkit
    install_pytorch
    install_flash_attention
    install_dependencies
    verify_environment

    echo ""
    log_success "=============================================="
    log_success "  Setup Complete!"
    log_success "=============================================="
    echo ""
    echo "To use the environment in a new shell:"
    echo "  conda activate $ENV_NAME"
    echo ""
    echo "To run training:"
    echo "  torchrun --nproc_per_node=8 \\"
    echo "    -m molgen3D.training.pretraining.torchtitan_runner \\"
    echo "    --train-toml src/molgen3D/config/pretrain/qwen3_06b.toml"
    echo ""
}

main "$@"
