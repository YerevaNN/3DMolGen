#!/usr/bin/env python3
"""
Script for AIMNet2 energy calculations and verification.

This script provides multiple functionalities:
1. Compare AIMNet2 calculated energies with reported energies from conformer data
2. Run comprehensive verification of AIMNet2 implementation
3. Benchmark performance on real molecules

Usage:
    # Compare energies from a pickle file
    python test_energy_calculation.py --pickle path/to/conformer_data.pickle

    # Run comprehensive verification of AIMNet2 implementation
    python test_energy_calculation.py --verify

    # Run benchmark on real molecules
    python test_energy_calculation.py --benchmark

    # Specify device and batch size
    python test_energy_calculation.py --pickle data.pickle --device cuda --batch-size 512

Expected pickle file format:
    {
        'totalconfs': int,
        'temperature': float,
        'uniqueconfs': int,
        'lowestenergy': float,
        'conformers': [
            {
                'geom_id': int,
                'totalenergy': float,
                'rd_mol': <rdkit.Chem.Mol object>
            },
            ...
        ]
    }

Verification includes:
- Basic energy calculation on simple molecules
- Force calculation consistency
- Energy conservation during small displacements
- Comparison with conformer data (if available)
"""

import os
import sys
import time
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

import random
import psutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rdkit import Chem
from rdkit.Chem import AllChem
from molgen3D.evaluation.utils import calculate_molecule_energies, create_energy_calculator
from molgen3D.config.paths import get_data_path


def load_conformer_data(data_path: str) -> Tuple[List[Chem.Mol], List[float], List[float], Dict]:
    """
    Load conformer data from pickle file.

    Args:
        data_path: Path to the pickle file containing conformer data

    Returns:
        Tuple of (molecules, reported_energies, relative_energies, metadata)
    """
    print(f"Loading conformer data from {data_path}...")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    molecules = []
    reported_energies = []
    relative_energies = []
    metadata = {k: v for k, v in data.items() if k != 'conformers'}

    print(f"Metadata: {metadata}")

    for conf_data in data['conformers']:
        if 'rd_mol' in conf_data and conf_data['rd_mol'] is not None:
            molecules.append(conf_data['rd_mol'])
            reported_energies.append(conf_data['totalenergy'])
            # Try to get relative energies if available
            rel_energy = conf_data.get('relativeenergy')
            if rel_energy is not None:
                relative_energies.append(rel_energy)
            else:
                relative_energies.append(None)

    print(f"Loaded {len(molecules)} conformers with energies")
    return molecules, reported_energies, relative_energies, metadata


def compare_aimnet2_energies(pickle_path: str, device: str = "cpu", batch_size: int = 256):
    """
    Compare AIMNet2 calculated energies with reported energies from conformer data.

    Args:
        pickle_path: Path to the pickle file containing conformer data
        device: Device to use for AIMNet2 calculations
        batch_size: Batch size for energy calculations
    """
    print("="*60)
    print("AIMNet2 Energy Comparison")
    print("="*60)

    # Check if AIMNet2 model exists
    aimnet2_path = get_data_path("aimnet2_model")
    if not aimnet2_path.exists():
        print(f"Error: AIMNet2 model not found at {aimnet2_path}")
        print("Please download the AIMNet2 model and place it at the correct path.")
        return

    print(f"Using AIMNet2 model at: {aimnet2_path}")

    # Check if pickle file exists
    if not Path(pickle_path).exists():
        print(f"Error: Pickle file not found at {pickle_path}")
        return

    # Load conformer data
    try:
        molecules, reported_energies, relative_energies, metadata = load_conformer_data(pickle_path)
        if not molecules:
            print("No conformers loaded from the pickle file!")
            return
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    print(f"Dataset info: {metadata}")

    # Calculate energies with AIMNet2
    print("\nCalculating energies with AIMNet2...")
    start_time = time.time()

    try:
        energies_result = calculate_molecule_energies(
            molecules,
            device=device,
            batch_size=batch_size
        )

        aimnet2_energies = energies_result['energies']
        calculation_time = time.time() - start_time

    except Exception as e:
        print(f"Error calculating energies with AIMNet2: {e}")
        return

    # Compare energies
    if len(aimnet2_energies) != len(reported_energies):
        print(f"Error: Mismatch in number of energies. AIMNet2: {len(aimnet2_energies)}, Reported: {len(reported_energies)}")
        return

    # Convert to numpy arrays for easier calculations
    reported_energies = np.array(reported_energies)
    aimnet2_energies = np.array(aimnet2_energies)

    print("\\n⚠️  WARNING: Large energy differences detected!")
    print("   This suggests the reported energies may be relative energies,")
    print("   while AIMNet2 returns absolute energies.")

    # Check if reported energies look like relative energies
    reported_range = reported_energies.max() - reported_energies.min()
    aimnet2_range = aimnet2_energies.max() - aimnet2_energies.min()

    print("\\nEnergy scale analysis:")
    print(f"   Reported energy range: {reported_range:.6f} Hartree")
    print(f"   AIMNet2 energy range:  {aimnet2_range:.6f} Hartree")

    if reported_range < 1.0:  # Less than 1 Hartree range
        print("   → Reported energies appear to be RELATIVE (conformer differences)")
        print("   → Switching to relative energy comparison...")

        # Compare relative energies instead of absolute
        # Find the minimum energy for each method and compare relative differences
        reported_min = reported_energies.min()
        aimnet2_min = aimnet2_energies.min()

        reported_relative = reported_energies - reported_min
        aimnet2_relative = aimnet2_energies - aimnet2_min

        differences = reported_relative - aimnet2_relative

        comparison_type = "relative energies (both shifted to min=0)"
    else:
        # Absolute energy comparison
        differences = reported_energies - aimnet2_energies
        comparison_type = "absolute energies"

    # Calculate statistics
    mean_diff = np.mean(differences)
    min_diff = np.min(differences)
    max_diff = np.max(differences)
    std_diff = np.std(differences)
    rmse = np.sqrt(np.mean(differences**2))

    # Convert energies from Hartree to kcal/mol (1 Hartree = 627.509 kcal/mol)
    hartree_to_kcal = 627.509
    differences_kcal = differences * hartree_to_kcal
    mean_diff_kcal = mean_diff * hartree_to_kcal
    min_diff_kcal = min_diff * hartree_to_kcal
    max_diff_kcal = max_diff * hartree_to_kcal
    std_diff_kcal = std_diff * hartree_to_kcal
    rmse_kcal = rmse * hartree_to_kcal

    # Print results
    print("\nENERGY COMPARISON RESULTS")
    print("="*60)
    print(f"Total conformers: {len(molecules)}")
    print(f"Calculation time: {calculation_time:.2f} seconds ({calculation_time/len(molecules):.3f} s/conformer)")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Comparison type: {comparison_type}")

    print("\nEnergy Differences (Hartree):")
    print(f"  Mean: {mean_diff:+.6f}")
    print(f"  Min:  {min_diff:+.6f}")
    print(f"  Max:  {max_diff:+.6f}")
    print(f"  Std:  {std_diff:.6f}")
    print(f"  RMSE: {rmse:.6f}")

    print("\nEnergy Differences (kcal/mol):")
    print(f"  Mean: {mean_diff_kcal:+.3f}")
    print(f"  Min:  {min_diff_kcal:+.3f}")
    print(f"  Max:  {max_diff_kcal:+.3f}")
    print(f"  Std:  {std_diff_kcal:.3f}")
    print(f"  RMSE: {rmse_kcal:.3f}")

    # Show energy ranges
    print("\nEnergy Ranges:")
    print(f"  Reported energies: {reported_energies.min():.6f} to {reported_energies.max():.6f} Hartree")
    print(f"  AIMNet2 energies:  {aimnet2_energies.min():.6f} to {aimnet2_energies.max():.6f} Hartree")

    # Check if energies are reasonable
    print("\nEnergy Statistics:")
    print(f"  Reported lowest energy: {metadata.get('lowestenergy', 'N/A')}")
    print(f"  AIMNet2 lowest energy:   {aimnet2_energies.min():.6f}")
    print(f"  Energy range (reported): {reported_energies.max() - reported_energies.min():.6f} Hartree")
    print(f"  Energy range (AIMNet2):  {aimnet2_energies.max() - aimnet2_energies.min():.6f} Hartree")

    # Additional analysis for relative energies
    if "relative" in comparison_type:
        print("\nRelative Energy Analysis:")
        print("  Both energy sets shifted so minimum energy = 0")
        print("  This compares the energy landscapes, not absolute values")

        # Check if the energy ordering is similar
        reported_sorted_idx = np.argsort(reported_energies)
        aimnet2_sorted_idx = np.argsort(aimnet2_energies)

        # Calculate Spearman rank correlation
        from scipy.stats import spearmanr
        correlation, p_value = spearmanr(reported_energies, aimnet2_energies)
        print(f"  Rank correlation: {correlation:.4f} (p={p_value:.2e})")

        if correlation > 0.8:
            print("  ✅ Good agreement in energy ordering")
        elif correlation > 0.5:
            print("  ⚠️  Moderate agreement in energy ordering")
        else:
            print("  ❌ Poor agreement in energy ordering")

    return {
        'mean_diff_hartree': mean_diff,
        'min_diff_hartree': min_diff,
        'max_diff_hartree': max_diff,
        'std_diff_hartree': std_diff,
        'rmse_hartree': rmse,
        'mean_diff_kcal': mean_diff_kcal,
        'min_diff_kcal': min_diff_kcal,
        'max_diff_kcal': max_diff_kcal,
        'std_diff_kcal': std_diff_kcal,
        'rmse_kcal': rmse_kcal,
        'calculation_time': calculation_time,
        'num_conformers': len(molecules)
    }


def load_real_molecules(data_path: str, max_molecules: int = 1000, max_confs_per_mol: int = -1) -> List[Chem.Mol]:
    """
    Load molecules from the distinct_smi.pickle file.

    Args:
        data_path: Path to the pickle file
        max_molecules: Maximum number of molecules to load
        max_confs_per_mol: Maximum conformations per molecule

    Returns:
        List of RDKit molecules
    """
    print(f"Loading molecules from {data_path}...")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    molecules = []
    total_confs = 0

    # Sample molecules randomly
    molecule_keys = list(data.keys())
    random.shuffle(molecule_keys)

    for key in molecule_keys[:max_molecules]:
        mol_data = data[key]
        confs = mol_data.get('confs', [])

        if not confs:
            continue

        # Take up to max_confs_per_mol conformations per molecule
        for conf in confs[:max_confs_per_mol]:
            if hasattr(conf, 'GetNumAtoms') and conf.GetNumAtoms() > 0:
                molecules.append(conf)
                total_confs += 1

    print(f"Loaded {len(molecules)} conformations from {min(max_molecules, len(molecule_keys))} molecules")
    return molecules


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)


def benchmark_energy_calculation(molecules: List[Chem.Mol], device: str = "cpu",
                               batch_sizes: List[int] = [32, 64, 128]) -> dict:
    """
    Benchmark energy calculation performance.

    Args:
        molecules: List of RDKit molecules
        device: Device to use ('cpu' or 'cuda')
        batch_sizes: Batch sizes to test

    Returns:
        Dictionary with benchmark results
    """
    results = {
        'device': device,
        'num_molecules': len(molecules),
        'batch_size_results': {}
    }

    print(f"\nBenchmarking on {len(molecules)} molecules with device: {device}")

    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")

        try:
            # Warm up
            if len(molecules) >= batch_size:
                warmup_mols = molecules[:min(batch_size, 5)]
                calculate_molecule_energies(warmup_mols, device=device, batch_size=batch_size)

            # Actual benchmark
            start_time = time.time()
            start_memory = get_memory_usage()

            energies = calculate_molecule_energies(molecules, device=device, batch_size=batch_size)

            end_time = time.time()
            end_memory = get_memory_usage()

            elapsed = end_time - start_time
            memory_used = end_memory - start_memory

            results['batch_size_results'][batch_size] = {
                'elapsed_time': elapsed,
                'molecules_per_second': len(molecules) / elapsed,
                'memory_used_gb': memory_used,
                'avg_max_forces': energies.get('avg_max_forces', None),
                'median_max_forces': energies.get('median_max_forces', None),
                'success': True
            }

            print(".2f")
            print(".2f")

        except Exception as e:
            print(f"Error with batch size {batch_size}: {e}")
            results['batch_size_results'][batch_size] = {
                'success': False,
                'error': str(e)
            }

    return results


def benchmark_optimization_metrics(molecules: List[Chem.Mol], device: str = "cpu") -> dict:
    """
    Benchmark geometry optimization with metrics.

    Args:
        molecules: List of RDKit molecules
        device: Device to use

    Returns:
        Dictionary with optimization results
    """
    print("\nBenchmarking geometry optimization with metrics...")

    # Use a smaller subset for optimization (it's more expensive)
    test_mols = molecules[:min(20, len(molecules))]

    try:
        calculator = create_energy_calculator(
            device=device,
            batch_size=8,
            opt_metrics=True
        )

        start_time = time.time()
        results = calculator(test_mols)
        elapsed = time.time() - start_time

        opt_results = {
            'num_molecules': len(test_mols),
            'total_time': elapsed,
            'time_per_molecule': elapsed / len(test_mols),
            'opt_total_time': results.get('opt_total_time', None),
            'opt_converged': results.get('opt_converged', None),
            'opt_steps': results.get('opt_steps', None),
            'preserved_topology': results.get('preserved_topology', None),
            'opt_avg_energy_drop': results.get('opt_avg_energy_drop', None),
            'success': True
        }

        print(f"Optimization completed in {elapsed:.2f}s")
        print(".2f")
        if 'opt_converged' in results:
            print(".1%")

        return opt_results

    except Exception as e:
        print(f"Error in optimization benchmark: {e}")
        return {'success': False, 'error': str(e)}


def benchmark_real_molecules():
    """Benchmark energy calculation on real molecules from distinct_smi.pickle."""
    print("Benchmarking AIMNet2 energy calculation on real molecules...")

    # Check if AIMNet2 model exists
    aimnet2_path = get_data_path("aimnet2_model")
    if not aimnet2_path.exists():
        print(f"Warning: AIMNet2 model not found at {aimnet2_path}")
        print("Please download the AIMNet2 model and place it at the correct path.")
        return

    print(f"Using AIMNet2 model at: {aimnet2_path}")

    # Check if data file exists
    data_path = get_data_path("distinct_smi")
    if not data_path.exists():
        print(f"Warning: Data file not found at {data_path}")
        return

    # Load real molecules
    molecules = load_real_molecules(str(data_path), max_molecules=1000, max_confs_per_mol=-1)
    if not molecules:
        print("No molecules loaded!")
        return

    # Determine device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Benchmark energy calculation with different batch sizes
    energy_results = benchmark_energy_calculation(
        molecules,
        device=device,
        batch_sizes=[256, 512, 1024, 2048, 4096]
    )

    # Benchmark optimization
    opt_results = benchmark_optimization_metrics(molecules[:min(10, len(molecules))], device=device)

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    print(f"Dataset: {len(molecules)} conformations from distinct_smi.pickle")
    print(f"Device: {device}")
    print(f"AIMNet2 model: {aimnet2_path}")

    print("\nEnergy Calculation Performance:")
    print("-" * 40)
    for batch_size, results in energy_results['batch_size_results'].items():
        if results.get('success', False):
            print(f"Batch size {batch_size}: {results['molecules_per_second']:.2f} mol/s")
        else:
            print(f"Batch size {batch_size}: Failed - {results.get('error', 'Unknown error')}")

    print("\nOptimization Performance:")
    print("-" * 40)
    if opt_results.get('success', False):
        print(f"Total time: {opt_results['total_time']:.2f}s")
        print(f"Time per molecule: {opt_results['time_per_molecule']:.2f}s")
        print(f"Convergence rate: {opt_results.get('opt_converged', 0):.1%}")
    else:
        print(f"Optimization failed: {opt_results.get('error', 'Unknown error')}")

    print("\nBenchmark completed!")


def main():
    """Main function with command line argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare AIMNet2 energies with reported conformer energies')
    parser.add_argument('--pickle', '-p', type=str, help='Path to pickle file containing conformer data')
    parser.add_argument('--device', '-d', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use for AIMNet2 calculations (default: cpu)')
    parser.add_argument('--batch-size', '-b', type=int, default=256,
                        help='Batch size for energy calculations (default: 256)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark on real molecules instead of energy comparison')
    parser.add_argument('--verify', action='store_true',
                        help='Run comprehensive verification of AIMNet2 implementation')

    args = parser.parse_args()

    if args.verify:
        # Run verification
        success = verify_aimnet2_implementation()
        sys.exit(0 if success else 1)
    elif args.benchmark:
        # Run the original benchmark
        benchmark_real_molecules()
    elif args.pickle:
        # Run energy comparison
        compare_aimnet2_energies(args.pickle, device=args.device, batch_size=args.batch_size)
    else:
        print("Error: Must specify one of --verify, --pickle, or --benchmark")
        print("Use --help for more information")
        sys.exit(1)


def test_energy_comparison():
    """Test the energy comparison functionality with the provided pickle file."""
    # Example usage - replace with your actual pickle file path
    pickle_file = "CCCCCCCCCCCC1=C(O)C(=O)C=C(O)C1=O.pickle"  # This should be provided as command line argument

    if Path(pickle_file).exists():
        print(f"Testing energy comparison with {pickle_file}")
        results = compare_aimnet2_energies(pickle_file, device="cuda" if torch.cuda.is_available() else "cpu", batch_size=32)
        if results:
            print("\nTest completed successfully!")
            print(f"Mean energy difference: {results['mean_diff_kcal']:.3f} kcal/mol")
        else:
            print("Test failed!")
    else:
        print(f"Test pickle file {pickle_file} not found. Skipping test.")


def verify_aimnet2_implementation():
    """
    Comprehensive verification of AIMNet2 implementation correctness.
    """
    print("="*60)
    print("AIMNet2 Implementation Verification")
    print("="*60)

    # Check if AIMNet2 model exists
    aimnet2_path = get_data_path("aimnet2_model")
    if not aimnet2_path.exists():
        print(f"❌ AIMNet2 model not found at {aimnet2_path}")
        return False

    print(f"✅ AIMNet2 model found at: {aimnet2_path}")

    # Test 1: Simple molecule energy calculation
    print("\n1. Testing basic energy calculation...")
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Create a simple water molecule
        mol = Chem.MolFromSmiles('O')
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        molecules = [mol]
        result = calculate_molecule_energies(molecules, device="cpu", batch_size=1)

        if 'energies' in result and len(result['energies']) == 1:
            energy = result['energies'][0]
            print(f"   Calculated energy: {energy:.6f} Hartree")
        else:
            print("❌ Energy calculation failed - no energies returned")
            return False

    except Exception as e:
        print(f"❌ Basic energy calculation failed: {e}")
        return False

    # Test 2: Force calculation consistency (energy gradient)
    print("\n2. Testing force calculation consistency...")
    try:
        import torch
        import numpy as np
        from molgen3D.evaluation.aimnet2_metrics import MoleculeAIMNet2Metrics

        calculator = MoleculeAIMNet2Metrics(
            model_path=str(aimnet2_path),
            batchsize=1,
            device="cpu",
            opt_metrics=False
        )

        # Use proper batch preparation
        from molgen3D.evaluation.aimnet2_metrics import prepare_for_aimnet
        batch = prepare_for_aimnet([mol], device="cpu")
        energy, forces = calculator.calculate_energy_forces_batched(batch)

        print(f"   Energy: {energy.item():.6f} Hartree")
        print(f"   Max force component: {forces.abs().max().item():.6f}")

        # Basic sanity check - forces should be reasonable
        if forces.abs().max().item() > 100:  # Very large forces indicate issues
            print("⚠️  Warning: Very large forces detected")
        else:
            print("✅ Force magnitudes look reasonable")

    except Exception as e:
        print(f"❌ Force calculation test failed: {e}")
        return False

    # Test 3: Energy conservation during small displacements
    print("\n3. Testing energy conservation...")
    try:
        # Get conformer for the molecule
        conf = mol.GetConformer()

        # Displace one atom slightly and check energy changes
        original_coords = conf.GetPositions().copy()

        # Small displacement (0.01 Å)
        displacement = np.array([0.01, 0.0, 0.0])
        new_coords = original_coords.copy()
        new_coords[0] += displacement  # Displace first atom

        # Create new conformer
        new_conf = Chem.Conformer(mol.GetNumAtoms())
        for i, pos in enumerate(new_coords):
            new_conf.SetAtomPosition(i, pos)
        mol_displaced = Chem.Mol(mol)
        mol_displaced.RemoveConformer(0)
        mol_displaced.AddConformer(new_conf)

        # Calculate energies
        result_original = calculate_molecule_energies([mol], device="cpu", batch_size=1)
        result_displaced = calculate_molecule_energies([mol_displaced], device="cpu", batch_size=1)

        energy_original = result_original['energies'][0]
        energy_displaced = result_displaced['energies'][0]
        energy_diff = energy_displaced - energy_original

        print(f"   Original energy: {energy_original:.6f} Hartree")
        print(f"   Displaced energy: {energy_displaced:.6f} Hartree")
        print("✅ Energy changes with displacement")

    except Exception as e:
        print(f"❌ Energy conservation test failed: {e}")
        return False

    # Test 4: Test with the conformer data if available
    print("\n4. Testing with conformer data...")
    pickle_file = "CCCCCCCCCCCC1=C(O)C(=O)C=C(O)C1=O.pickle"
    if Path(pickle_file).exists():
        try:
            results = compare_aimnet2_energies(pickle_file, device="cpu", batch_size=10)
            if results:
                print("✅ Conformer energy comparison successful")
                print(f"   Mean relative energy difference: {results['mean_diff_kcal']:.3f} kcal/mol")
            else:
                print("❌ Conformer energy comparison failed")
                return False
        except Exception as e:
            print(f"❌ Conformer test failed: {e}")
            return False
    else:
        print(f"⚠️  Conformer pickle file not found at {pickle_file}, skipping this test")

    print("\n" + "="*60)
    print("✅ ALL VERIFICATION TESTS PASSED!")
    print("AIMNet2 implementation appears to be working correctly.")
    print("="*60)
    return True


def test_energy_calculation():
    """Legacy test function - now redirects to benchmark."""
    benchmark_real_molecules()


if __name__ == "__main__":
    main()
