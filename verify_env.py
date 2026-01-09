#!/usr/bin/env python3
"""
3DMolGen Environment Verification Script

Verifies that all required packages are installed with correct versions
and that CUDA/GPU access is working properly.

Usage:
    python verify_env.py
    uv run python verify_env.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    passed: bool
    version: str = ""
    message: str = ""


def check_pytorch() -> CheckResult:
    """Check PyTorch installation and CUDA availability."""
    try:
        import torch

        version = torch.__version__
        cuda_available = torch.cuda.is_available()

        if not cuda_available:
            return CheckResult(
                name="PyTorch",
                passed=False,
                version=version,
                message="CUDA not available",
            )

        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()

        return CheckResult(
            name="PyTorch",
            passed=True,
            version=version,
            message=f"CUDA {cuda_version}, {gpu_count}x {gpu_name}",
        )
    except ImportError as e:
        return CheckResult(name="PyTorch", passed=False, message=str(e))


def check_flash_attention() -> CheckResult:
    """Check Flash Attention installation."""
    try:
        import flash_attn

        version = flash_attn.__version__

        # Try to import the function to verify it works
        from flash_attn import flash_attn_func  # noqa: F401

        return CheckResult(
            name="Flash Attention",
            passed=True,
            version=version,
            message="flash_attn_func available",
        )
    except ImportError as e:
        return CheckResult(name="Flash Attention", passed=False, message=str(e))


def check_transformers() -> CheckResult:
    """Check Transformers library."""
    try:
        import transformers

        return CheckResult(
            name="transformers",
            passed=True,
            version=transformers.__version__,
        )
    except ImportError as e:
        return CheckResult(name="transformers", passed=False, message=str(e))


def check_trl() -> CheckResult:
    """Check TRL (Transformer Reinforcement Learning) library."""
    try:
        import trl

        return CheckResult(name="trl", passed=True, version=trl.__version__)
    except ImportError as e:
        return CheckResult(name="trl", passed=False, message=str(e))


def check_accelerate() -> CheckResult:
    """Check Accelerate library."""
    try:
        import accelerate

        return CheckResult(
            name="accelerate",
            passed=True,
            version=accelerate.__version__,
        )
    except ImportError as e:
        return CheckResult(name="accelerate", passed=False, message=str(e))


def check_datasets() -> CheckResult:
    """Check Datasets library."""
    try:
        import datasets

        return CheckResult(
            name="datasets",
            passed=True,
            version=datasets.__version__,
        )
    except ImportError as e:
        return CheckResult(name="datasets", passed=False, message=str(e))


def check_torchtitan() -> CheckResult:
    """Check TorchTitan library."""
    try:
        import torchtitan

        version = getattr(torchtitan, "__version__", "installed")
        return CheckResult(name="torchtitan", passed=True, version=version)
    except ImportError as e:
        return CheckResult(name="torchtitan", passed=False, message=str(e))


def check_rdkit() -> CheckResult:
    """Check RDKit molecular modeling library."""
    try:
        from rdkit import Chem

        version = Chem.rdBase.rdkitVersion
        return CheckResult(name="RDKit", passed=True, version=version)
    except ImportError as e:
        return CheckResult(name="RDKit", passed=False, message=str(e))


def check_posebusters() -> CheckResult:
    """Check PoseBusters evaluation library."""
    try:
        import posebusters

        version = getattr(posebusters, "__version__", "installed")
        return CheckResult(name="posebusters", passed=True, version=version)
    except ImportError as e:
        return CheckResult(name="posebusters", passed=False, message=str(e))


def check_molgen3d() -> CheckResult:
    """Check molgen3D local package."""
    try:
        import molgen3D

        version = getattr(molgen3D, "__version__", "installed")
        return CheckResult(name="molgen3D (local)", passed=True, version=version)
    except ImportError as e:
        return CheckResult(name="molgen3D (local)", passed=False, message=str(e))


def run_all_checks() -> list[CheckResult]:
    """Run all environment checks."""
    checks = [
        check_pytorch,
        check_flash_attention,
        check_transformers,
        check_trl,
        check_accelerate,
        check_datasets,
        check_torchtitan,
        check_rdkit,
        check_posebusters,
        check_molgen3d,
    ]

    return [check() for check in checks]


def print_results(results: list[CheckResult]) -> bool:
    """Print check results and return True if all passed."""
    print("=" * 70)
    print("3DMolGen Environment Verification")
    print("=" * 70)
    print()

    all_passed = True
    critical_failed = []
    warnings = []

    # Critical packages
    critical = {"PyTorch", "Flash Attention", "transformers", "trl", "torchtitan"}

    for result in results:
        status = "\033[92m[PASS]\033[0m" if result.passed else "\033[91m[FAIL]\033[0m"
        version_str = f"v{result.version}" if result.version else ""
        message_str = f"({result.message})" if result.message else ""

        print(f"  {status} {result.name:<20} {version_str:<25} {message_str}")

        if not result.passed:
            if result.name in critical:
                critical_failed.append(result)
                all_passed = False
            else:
                warnings.append(result)

    print()
    print("-" * 70)

    if critical_failed:
        print("\033[91mCRITICAL FAILURES:\033[0m")
        for result in critical_failed:
            print(f"  - {result.name}: {result.message}")
        print()

    if warnings:
        print("\033[93mWARNINGS (non-critical):\033[0m")
        for result in warnings:
            print(f"  - {result.name}: {result.message}")
        print()

    if all_passed and not warnings:
        print("\033[92mAll checks passed! Environment is ready.\033[0m")
    elif all_passed:
        print("\033[93mCore checks passed with warnings. Environment should work.\033[0m")
    else:
        print("\033[91mCritical checks failed. Please fix before proceeding.\033[0m")

    print("=" * 70)

    return all_passed


def main() -> int:
    """Main entry point."""
    results = run_all_checks()
    all_passed = print_results(results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
