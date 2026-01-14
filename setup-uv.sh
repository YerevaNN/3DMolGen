#!/bin/bash
# =============================================================================
# 3DMolGen Environment Setup - Pure uv (No Conda)
# =============================================================================
# Fast, reproducible environment using only uv.
# Auto-detects CUDA version and cluster for optimal configuration.
#
# Clusters:
#   YNN (CUDA 12.8):     ./setup-uv.sh --dev --install-project
#   Superpod (CUDA 13.0): ./setup-uv.sh --nightly --dev --install-project
#
# The --nightly flag is required for Superpod because torchtitan 0.2.x
# requires PyTorch nightly (for torch.nn.attentigon.varlen). Flash Attention
# is automatically skipped with --nightly (no compatible wheels exist).
#
# Requirements:
#   - Linux x86_64
#   - CUDA 12.8+ drivers (system-level)
#   - Internet access (or pre-downloaded wheels)
# =============================================================================

set -euo pipefail

# =============================================================================
# Defaults (auto-detected at runtime)
# =============================================================================
PYTHON_VERSION="3.10"
PYTORCH_VERSION="2.9.1"
# CUDA_VERSION and PYTORCH_INDEX are set after argument parsing (auto-detected)
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

# Default env directory: project .venv (simpler, persistent)
get_default_env_dir() {
    echo "${SCRIPT_DIR}"
}

# Auto-detect CUDA version from nvidia-smi
detect_cuda_version() {
    local cuda_ver
    cuda_ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
    case "$cuda_ver" in
        13.*) echo "cu130" ;;
        12.*) echo "cu128" ;;
        *)    echo "cu128" ;;  # Default fallback
    esac
}

# Auto-detect cluster based on available paths
detect_cluster() {
    if [[ -d "/nfs/ap/mnt/sxtn2/chem" ]]; then
        echo "ynn"
    elif [[ -d "/home/chem-project" ]]; then
        echo "superpod"
    else
        echo "generic"
    fi
}

# Get flash attention wheel path based on python, cuda, and cluster
get_fa_wheel() {
    local py_ver="$1"   # "3.10" or "3.12"
    local cuda="$2"     # "cu128" or "cu130"
    local cluster="$3"  # "ynn", "superpod", or "generic"

    # Convert python version to cpython tag
    local cp_tag
    case "$py_ver" in
        3.10) cp_tag="cp310" ;;
        3.12) cp_tag="cp312" ;;
        *)    cp_tag="cp310" ;;
    esac

    local wheel_name="flash_attn-2.8.3+${cuda}torch2.9-${cp_tag}-${cp_tag}-linux_x86_64.whl"

    # Cluster-specific paths
    local ynn_path="/nfs/ap/mnt/sxtn2/chem/wheels/${wheel_name}"
    local superpod_path="/home/chem-project/flash-attention-wheels/${wheel_name}"
    local github_url="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/${wheel_name}"

    # Try cluster-specific path first
    case "$cluster" in
        ynn)
            if [[ -f "$ynn_path" ]]; then
                echo "$ynn_path"
                return
            fi
            ;;
        superpod)
            if [[ -f "$superpod_path" ]]; then
                echo "$superpod_path"
                return
            fi
            ;;
    esac

    # Fallback: try both paths
    if [[ -f "$ynn_path" ]]; then
        echo "$ynn_path"
    elif [[ -f "$superpod_path" ]]; then
        echo "$superpod_path"
    else
        # Last resort: GitHub URL
        echo "$github_url"
    fi
}

# =============================================================================
# Parse Arguments
# =============================================================================
ENV_DIR=""
PROJECT_DIR="$SCRIPT_DIR"
FA_WHEEL=""              # Set after parsing (auto-detected)
CUDA_VERSION=""          # Set after parsing (auto-detected)
INSTALL_EXTRAS=""
VERIFY_ONLY=false
SKIP_FLASH_ATTN=false
INSTALL_PROJECT=false
USE_NIGHTLY=false        # Use PyTorch nightly (required for torchtitan 0.2.x)

show_help() {
    cat << EOF
Usage: ./setup-uv.sh [OPTIONS]

Creates a Python environment with PyTorch, Flash Attention, and project dependencies.
Auto-detects CUDA version and cluster for optimal configuration.

Options:
  --nightly         Use PyTorch nightly (required for Superpod/torchtitan 0.2.x)
                    Automatically skips Flash Attention (no compatible wheels)
  --python VER      Python version: 3.10 or 3.12 (default: 3.10)
  --cuda VER        CUDA version: cu128 or cu130 (default: auto-detect from nvidia-smi)
  --dir PATH        Environment directory (default: .venv in project dir)
  --project PATH    Project directory containing pyproject.toml (default: script dir)
  --fa-wheel PATH   Flash Attention wheel path or URL (default: auto-detect)
  --skip-fa         Skip Flash Attention installation
  --dev             Include dev dependencies (pytest, black, etc.)
  --install-project Editable install of molgen3D package (default: deps only)
  --verify          Only verify existing installation
  --help            Show this help message

Environment Variables:
  UV_CACHE_DIR      Override uv cache location (default: auto-detect)

Clusters:
  YNN (YerevaNN):   CUDA 12.8, PyTorch stable, Flash Attention works
  Superpod:         CUDA 13.0, PyTorch nifrom torch.nn.attention.varlen import varlen_attnghtly required, uses SDPA attention

Examples:
  # YNN cluster (stable PyTorch + Flash Attention)
  ./setup-uv.sh --dev --install-project

  # Superpod cluster (nightly PyTorch + SDPA, no Flash Attention)
  ./setup-uv.sh --nightly --dev --install-project

  # Superpod with Python 3.12
  ./setup-uv.sh --nightly --python 3.12 --dev --install-project

  # Custom wheel location (YNN only)
  ./setup-uv.sh --fa-wheel ~/wheels/flash_attn.whl --dev

  # Skip Flash Attention explicitly
  ./setup-uv.sh --skip-fa --dev --install-project
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --nightly)
            USE_NIGHTLY=true
            SKIP_FLASH_ATTN=true  # No compatible flash_attn wheels for nightly
            shift
            ;;
        --python)
            PYTHON_VERSION="$2"
            if [[ "$PYTHON_VERSION" != "3.10" && "$PYTHON_VERSION" != "3.12" ]]; then
                log_error "Invalid Python version: $PYTHON_VERSION (use 3.10 or 3.12)"
                exit 1
            fi
            shift 2
            ;;
        --cuda)
            CUDA_VERSION="$2"
            if [[ "$CUDA_VERSION" != "cu128" && "$CUDA_VERSION" != "cu130" ]]; then
                log_error "Invalid CUDA version: $CUDA_VERSION (use cu128 or cu130)"
                exit 1
            fi
            shift 2
            ;;
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
        --install-project)
            INSTALL_PROJECT=true
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

# Set defaults after parsing
if [[ -z "$ENV_DIR" ]]; then
    ENV_DIR="$(get_default_env_dir)"
fi

# Auto-detect CUDA version if not specified
if [[ -z "$CUDA_VERSION" ]]; then
    CUDA_VERSION="$(detect_cuda_version)"
    log_info "Auto-detected CUDA version: $CUDA_VERSION"
fi

# Set PyTorch index URL based on CUDA version and nightly flag
if [[ "$USE_NIGHTLY" == true ]]; then
    case "$CUDA_VERSION" in
        cu128) PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128" ;;
        cu130) PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/cu130" ;;
        *)     PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128" ;;
    esac
    log_info "Using PyTorch NIGHTLY (required for torchtitan 0.2.x)"
else
    case "$CUDA_VERSION" in
        cu128) PYTORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
        cu130) PYTORCH_INDEX="https://download.pytorch.org/whl/cu130" ;;
        *)     PYTORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
    esac
fi

# Auto-detect cluster and flash attention wheel if not specified
if [[ -z "$FA_WHEEL" ]]; then
    DETECTED_CLUSTER="$(detect_cluster)"
    FA_WHEEL="$(get_fa_wheel "$PYTHON_VERSION" "$CUDA_VERSION" "$DETECTED_CLUSTER")"
    log_info "Auto-detected cluster: $DETECTED_CLUSTER"
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
    if [[ "$USE_NIGHTLY" == true ]]; then
        log_info "Installing PyTorch nightly+${CUDA_VERSION}..."
        uv pip install --pre torch --index-url "$PYTORCH_INDEX"
    else
        log_info "Installing PyTorch ${PYTORCH_VERSION}+${CUDA_VERSION}..."
        uv pip install "torch==${PYTORCH_VERSION}" --index-url "$PYTORCH_INDEX"
    fi
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
    else
        log_warn "Local wheel not found at: $FA_WHEEL"
        log_info "Attempting download (may be a URL)..."
        uv pip install "$FA_WHEEL"
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

    log_info "Installing dependencies from ${PROJECT_DIR}/pyproject.toml..."

    # Install tomli first (Python 3.10 doesn't have tomllib)
    uv pip install tomli --quiet

    # Extract and install dependencies from pyproject.toml (without installing the package)
    python3 << DEPS_EOF
try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Python 3.10 backport
import subprocess
import sys

with open("${PROJECT_DIR}/pyproject.toml", "rb") as f:
    config = tomllib.load(f)

deps = list(config["project"]["dependencies"])

# Add optional dependencies if requested
extras = "${INSTALL_EXTRAS}"
if extras:
    for extra in extras.split(","):
        extra = extra.strip()
        if extra in config["project"].get("optional-dependencies", {}):
            deps.extend(config["project"]["optional-dependencies"][extra])

print(f"Installing {len(deps)} dependencies...")
result = subprocess.run(["uv", "pip", "install"] + deps)
sys.exit(result.returncode)
DEPS_EOF

    if [[ $? -ne 0 ]]; then
        log_error "Failed to install dependencies"
        exit 1
    fi
    log_success "Dependencies installed"

    # Optionally install molgen3D package (editable)
    if [[ "$INSTALL_PROJECT" == true ]]; then
        log_info "Installing molgen3D package (editable)..."
        uv pip install --no-deps -e "${PROJECT_DIR}"
        log_success "molgen3D package installed"
    fi
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
    echo "For Slurm jobs, add to your script:"
    echo ""
    echo "  export UV_CACHE_DIR=${UV_CACHE_DIR}"
    echo "  source ${ENV_DIR}/.venv/bin/activate"
    echo ""
}

# =============================================================================
# Main
# =============================================================================
main() {
    local pytorch_label
    if [[ "$USE_NIGHTLY" == true ]]; then
        pytorch_label="nightly+${CUDA_VERSION}"
    else
        pytorch_label="${PYTORCH_VERSION}+${CUDA_VERSION}"
    fi

    echo ""
    echo "=============================================="
    echo "  Environment Setup (uv)"
    echo "  Python ${PYTHON_VERSION} | PyTorch ${pytorch_label}"
    echo "=============================================="
    echo ""
    echo "Environment:    ${ENV_DIR}/.venv"
    echo "Project:        ${PROJECT_DIR}"
    echo "CUDA version:   ${CUDA_VERSION}"
    echo "Nightly:        ${USE_NIGHTLY}"
    echo "PyTorch index:  ${PYTORCH_INDEX}"
    if [[ "$SKIP_FLASH_ATTN" == true ]]; then
        echo "Flash Attn:     SKIPPED (using SDPA)"
    else
        echo "Flash Attn:     ${FA_WHEEL}"
    fi
    echo "Install pkg:    ${INSTALL_PROJECT}"
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
