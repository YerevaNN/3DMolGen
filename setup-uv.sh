#!/bin/bash
# =============================================================================
# 3DMolGen Environment Setup - Pure uv (No Conda)
# =============================================================================
# Fast, reproducible environment using only uv.
# Portable across clusters: YNN (with Slurm), new H100 cluster (SSH-only), etc.
#
# Usage:
#   ./setup-uv.sh                              # Defaults for YNN cluster
#   ./setup-uv.sh --dir /path/to/env           # Custom env location
#   ./setup-uv.sh --project /path/to/project   # Custom project (pyproject.toml)
#   ./setup-uv.sh --fa-wheel /path/to/wheel    # Custom Flash Attention wheel
#   ./setup-uv.sh --dev                        # Include dev dependencies
#
# Requirements:
#   - Linux x86_64
#   - CUDA 12.8 drivers (system-level)
#   - Internet access (or pre-downloaded wheels)
# =============================================================================

set -euo pipefail

# =============================================================================
# Defaults (YNN cluster)
# =============================================================================
PYTHON_VERSION="3.10"
PYTORCH_VERSION="2.9.1"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"

# Flash Attention: try local wheel first, fall back to GitHub
FA_WHEEL_DEFAULT="/nfs/ap/mnt/sxtn2/chem/wheels/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl"
FA_WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl"

# Script directory (default project location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# Helper Functions
# =============================================================================
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# Determine default env directory based on what exists
get_default_env_dir() {
    if [[ -d "/scratch" && -w "/scratch" ]]; then
        echo "/scratch/${USER}/3dmolgen"
    elif [[ -d "/tmp" ]]; then
        echo "/tmp/${USER}/3dmolgen"
    else
        echo "${HOME}/envs/3dmolgen"
    fi
}

# =============================================================================
# Parse Arguments
# =============================================================================
ENV_DIR=""
PROJECT_DIR="$SCRIPT_DIR"
FA_WHEEL="$FA_WHEEL_DEFAULT"
INSTALL_EXTRAS=""
VERIFY_ONLY=false
SKIP_FLASH_ATTN=false

show_help() {
    cat << EOF
Usage: ./setup-uv.sh [OPTIONS]

Creates a Python environment with PyTorch, Flash Attention, and project dependencies.

Options:
  --dir PATH        Environment directory (default: auto-detect /scratch or /tmp)
  --project PATH    Project directory containing pyproject.toml (default: script dir)
  --fa-wheel PATH   Flash Attention wheel path or URL (default: YNN cluster path)
  --skip-fa         Skip Flash Attention installation
  --dev             Include dev dependencies (pytest, black, etc.)
  --verify          Only verify existing installation
  --help            Show this help message

Environment Variables:
  UV_CACHE_DIR      Override uv cache location (default: auto-detect)
  PYTORCH_INDEX     Override PyTorch index URL (default: cu128)

Examples:
  # YNN cluster (defaults)
  ./setup-uv.sh --dev

  # New cluster with custom wheel location
  ./setup-uv.sh --dir /data/envs/molgen --fa-wheel ~/wheels/flash_attn.whl --dev

  # Download Flash Attention from GitHub (no local wheel)
  ./setup-uv.sh --fa-wheel https://github.com/.../flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl

  # Install without Flash Attention (CPU-only or incompatible GPU)
  ./setup-uv.sh --skip-fa

  # Different project
  ./setup-uv.sh --project /path/to/other/project --dir /tmp/other-env
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dir)
            ENV_DIR="$2"
            shift 2
            ;;
        --project)
            PROJECT_DIR="$2"
            shift 2
            ;;
        --fa-wheel)
            FA_WHEEL="$2"
            shift 2
            ;;
        --skip-fa)
            SKIP_FLASH_ATTN=true
            shift
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
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set defaults after parsing (so --dir can override)
if [[ -z "$ENV_DIR" ]]; then
    ENV_DIR="$(get_default_env_dir)"
fi

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
# Step 2: Set up cache directory
# =============================================================================
setup_cache() {
    # Allow override via environment variable
    if [[ -n "${UV_CACHE_DIR:-}" ]]; then
        log_info "Using UV_CACHE_DIR from environment: $UV_CACHE_DIR"
    elif [[ -d "/scratch" && -w "/scratch" ]]; then
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
    log_info "Installing PyTorch ${PYTORCH_VERSION}+cu128..."
    uv pip install "torch==${PYTORCH_VERSION}" --index-url "$PYTORCH_INDEX"
    log_success "PyTorch installed"
}

# =============================================================================
# Step 5: Install Flash Attention
# =============================================================================
install_flash_attention() {
    if [[ "$SKIP_FLASH_ATTN" == true ]]; then
        log_warn "Skipping Flash Attention (--skip-fa)"
        return
    fi

    log_info "Installing Flash Attention..."

    # Check if it's a URL or local path
    if [[ "$FA_WHEEL" == http* ]]; then
        log_info "Downloading from URL: $FA_WHEEL"
        uv pip install "$FA_WHEEL"
    elif [[ -f "$FA_WHEEL" ]]; then
        log_info "Using local wheel: $FA_WHEEL"
        uv pip install "$FA_WHEEL"
    elif [[ -f "$FA_WHEEL_DEFAULT" ]]; then
        log_info "Using default wheel: $FA_WHEEL_DEFAULT"
        uv pip install "$FA_WHEEL_DEFAULT"
    else
        log_info "Local wheel not found, downloading from GitHub..."
        uv pip install "$FA_WHEEL_URL"
    fi
    log_success "Flash Attention installed"
}

# =============================================================================
# Step 6: Install rdkit
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
    if [[ ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
        log_error "pyproject.toml not found at: ${PROJECT_DIR}/pyproject.toml"
        exit 1
    fi

    log_info "Installing project from ${PROJECT_DIR}..."

    if [[ -n "$INSTALL_EXTRAS" ]]; then
        log_info "Including optional dependencies: [$INSTALL_EXTRAS]"
        uv pip install -e "${PROJECT_DIR}[${INSTALL_EXTRAS}]"
    else
        uv pip install -e "${PROJECT_DIR}"
    fi
    log_success "Dependencies installed"
}

# =============================================================================
# Step 8: Verify
# =============================================================================
verify_environment() {
    log_info "Verifying environment..."

    local verify_script="${PROJECT_DIR}/verify_env.py"
    if [[ -f "$verify_script" ]]; then
        python "$verify_script"
    else
        # Inline verification
        python << 'EOF'
import sys
print("=" * 60)
print("Environment Verification")
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

# Flash Attention (optional)
try:
    import flash_attn
    print(f"Flash Attention: {flash_attn.__version__}")
    checks.append(("Flash Attention", True))
except ImportError:
    print("Flash Attention: not installed (optional)")

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
    if [[ -d "/scratch" ]]; then
        echo "For Slurm jobs, add to your script:"
        echo ""
        echo "  export UV_CACHE_DIR=/scratch/\${USER}/.cache/uv"
        echo "  source ${ENV_DIR}/.venv/bin/activate"
        echo ""
    fi
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo ""
    echo "=============================================="
    echo "  Environment Setup (uv)"
    echo "  Python ${PYTHON_VERSION} | PyTorch ${PYTORCH_VERSION}+cu128"
    echo "=============================================="
    echo ""
    echo "Environment:  ${ENV_DIR}"
    echo "Project:      ${PROJECT_DIR}"
    echo "FA wheel:     ${FA_WHEEL}"
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
