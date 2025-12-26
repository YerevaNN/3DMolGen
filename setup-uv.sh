#!/bin/bash
# =============================================================================
# 3DMolGen Environment Setup - Pure uv (No Conda)
# =============================================================================
# Fast, reproducible environment using only uv.
# Designed for ephemeral cluster environments (Slurm jobs).
#
# Usage:
#   ./setup-uv.sh                    # Install to /scratch/$USER/3dmolgen
#   ./setup-uv.sh --dir /path/to/env # Install to custom location
#   ./setup-uv.sh --dev              # Include dev dependencies
#   ./setup-uv.sh --verify           # Only verify existing install
#
# Requirements:
#   - Linux x86_64
#   - CUDA 12.8 drivers (system-level)
#   - Internet access (first run) or uv cache populated
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
PYTHON_VERSION="3.10"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"
FA_WHEEL="/nfs/ap/mnt/sxtn2/chem/wheels/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl"
FA_WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl"

# Default install location (scratch is local, fast, ephemeral)
DEFAULT_ENV_DIR="/scratch/${USER}/3dmolgen"

# Script directory (where this script and pyproject.toml live)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# Parse Arguments
# =============================================================================
ENV_DIR="$DEFAULT_ENV_DIR"
INSTALL_EXTRAS=""
VERIFY_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dir)
            ENV_DIR="$2"
            shift 2
            ;;
        --dev|--all)
            INSTALL_EXTRAS="dev"
            shift
            ;;
        --verify)
            VERIFY_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./setup-uv.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dir PATH  Install to custom directory (default: /scratch/\$USER/3dmolgen)"
            echo "  --dev       Include dev dependencies (pytest, black, etc.)"
            echo "  --verify    Only verify existing installation"
            echo "  --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./setup-uv.sh                        # Standard install"
            echo "  ./setup-uv.sh --dev                  # With dev tools"
            echo "  ./setup-uv.sh --dir ~/envs/molgen    # Custom location"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Helper Functions
# =============================================================================
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Step 1: Install uv
# =============================================================================
install_uv() {
    if command -v uv &> /dev/null; then
        log_success "uv already installed: $(uv --version)"
    else
        log_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        log_success "uv installed: $(uv --version)"
    fi
}

# =============================================================================
# Step 2: Set up cache directory (avoid polluting NFS home)
# =============================================================================
setup_cache() {
    # Use scratch for cache if available, otherwise use env dir
    if [[ -d "/scratch" ]]; then
        export UV_CACHE_DIR="/scratch/${USER}/.cache/uv"
    else
        export UV_CACHE_DIR="${ENV_DIR}/.cache/uv"
    fi
    mkdir -p "$UV_CACHE_DIR"
    log_info "uv cache: $UV_CACHE_DIR"
}

# =============================================================================
# Step 3: Create virtual environment
# =============================================================================
create_venv() {
    if [[ -d "${ENV_DIR}/.venv" ]]; then
        log_success "venv exists: ${ENV_DIR}/.venv"
    else
        log_info "Creating venv with Python ${PYTHON_VERSION}..."
        mkdir -p "$ENV_DIR"
        uv venv --python "$PYTHON_VERSION" "${ENV_DIR}/.venv"
        log_success "venv created"
    fi

    # Activate
    source "${ENV_DIR}/.venv/bin/activate"
    log_success "Activated: $(python --version) @ $(which python)"
}

# =============================================================================
# Step 4: Install PyTorch
# =============================================================================
install_pytorch() {
    log_info "Installing PyTorch 2.9.1+cu128..."
    uv pip install torch==2.9.1 --index-url "$PYTORCH_INDEX"
    log_success "PyTorch installed"
}

# =============================================================================
# Step 5: Install Flash Attention
# =============================================================================
install_flash_attention() {
    log_info "Installing Flash Attention..."
    if [[ -f "$FA_WHEEL" ]]; then
        log_info "Using local wheel: $FA_WHEEL"
        uv pip install "$FA_WHEEL"
    else
        log_info "Downloading from GitHub..."
        uv pip install "$FA_WHEEL_URL"
    fi
    log_success "Flash Attention installed"
}

# =============================================================================
# Step 6: Install rdkit (pip wheel)
# =============================================================================
install_rdkit() {
    log_info "Installing rdkit..."
    uv pip install rdkit
    log_success "rdkit installed"
}

# =============================================================================
# Step 7: Install project dependencies
# =============================================================================
install_dependencies() {
    log_info "Installing molgen3D from ${SCRIPT_DIR}..."

    if [[ -n "$INSTALL_EXTRAS" ]]; then
        log_info "Including optional dependencies: [$INSTALL_EXTRAS]"
        uv pip install -e "${SCRIPT_DIR}[${INSTALL_EXTRAS}]"
    else
        uv pip install -e "${SCRIPT_DIR}"
    fi
    log_success "Dependencies installed"
}

# =============================================================================
# Step 8: Verify
# =============================================================================
verify_environment() {
    log_info "Verifying environment..."

    if [[ -f "${SCRIPT_DIR}/verify_env.py" ]]; then
        python "${SCRIPT_DIR}/verify_env.py"
    else
        python << 'EOF'
import sys
print("=" * 60)
print("3DMolGen Environment Verification")
print("=" * 60)

checks = []

# PyTorch
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"PyTorch:         {torch.__version__}")
    print(f"CUDA available:  {cuda_ok}")
    if cuda_ok:
        print(f"CUDA version:    {torch.version.cuda}")
    checks.append(("PyTorch", True))
except ImportError as e:
    checks.append(("PyTorch", False))
    print(f"PyTorch: FAILED - {e}")

# Flash Attention
try:
    import flash_attn
    print(f"Flash Attention: {flash_attn.__version__}")
    checks.append(("Flash Attention", True))
except ImportError as e:
    checks.append(("Flash Attention", False))
    print(f"Flash Attention: FAILED - {e}")

# Core libs
for lib in ["transformers", "trl", "torchtitan", "accelerate", "datasets"]:
    try:
        mod = __import__(lib)
        ver = getattr(mod, "__version__", "ok")
        print(f"{lib}: {ver}")
        checks.append((lib, True))
    except ImportError as e:
        checks.append((lib, False))
        print(f"{lib}: FAILED - {e}")

# RDKit
try:
    from rdkit import Chem
    print(f"rdkit: {Chem.rdBase.rdkitVersion}")
    checks.append(("rdkit", True))
except ImportError as e:
    checks.append(("rdkit", False))
    print(f"rdkit: FAILED - {e}")

print("=" * 60)
failed = [name for name, ok in checks if not ok]
if failed:
    print(f"FAILED: {', '.join(failed)}")
    sys.exit(1)
else:
    print("All checks passed!")
EOF
    fi
}

# =============================================================================
# Step 9: Print activation instructions
# =============================================================================
print_instructions() {
    echo ""
    log_success "=============================================="
    log_success "  Setup Complete!"
    log_success "=============================================="
    echo ""
    echo "To activate this environment:"
    echo ""
    echo "  source ${ENV_DIR}/.venv/bin/activate"
    echo ""
    echo "Or add to your Slurm job script:"
    echo ""
    echo "  export UV_CACHE_DIR=/scratch/\${USER}/.cache/uv"
    echo "  source ${ENV_DIR}/.venv/bin/activate"
    echo ""
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo ""
    echo "=============================================="
    echo "  3DMolGen - Pure uv Setup"
    echo "  Python ${PYTHON_VERSION} | PyTorch 2.9.1+cu128 | FA2"
    echo "=============================================="
    echo ""
    echo "Environment directory: ${ENV_DIR}"
    echo ""

    install_uv
    setup_cache

    if [[ "$VERIFY_ONLY" == true ]]; then
        source "${ENV_DIR}/.venv/bin/activate"
        verify_environment
        exit 0
    fi

    create_venv
    install_pytorch
    install_flash_attention
    install_rdkit
    install_dependencies
    verify_environment
    print_instructions
}

main "$@"
