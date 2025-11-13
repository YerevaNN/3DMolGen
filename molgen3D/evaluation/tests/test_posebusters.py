import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

pytest.importorskip("posebusters")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molgen3D.evaluation.posebusters_check import bust_cpu, bust_full_gens


@pytest.fixture(scope="module")
def fixture_smiles_to_confs() -> Dict[str, List[Chem.Mol]]:
    """Load the deterministic PoseBusters fixture used across tests."""
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    fixture_path = fixture_dir / "posebusters_fixture.pkl"
    with fixture_path.open("rb") as handle:
        data = pickle.load(handle)
    assert isinstance(data, dict)
    return data


def _clone_mols(mols: List[Chem.Mol], repeats: int = 1) -> List[Chem.Mol]:
    """Return deep copies of RDKit molecules to avoid shared state."""
    clones: List[Chem.Mol] = []
    for _ in range(repeats):
        for mol in mols:
            clones.append(Chem.Mol(mol))
    return clones


def test_bust_cpu_returns_expected_lengths(
    fixture_smiles_to_confs: Dict[str, List[Chem.Mol]]
) -> None:
    """bust_cpu should return per-conformer pass rates matching the input length."""
    # Arrange
    conformers: List[Chem.Mol] = []
    for mols in fixture_smiles_to_confs.values():
        conformers.extend(_clone_mols(mols))

    # Act
    rates_avg, avg_avg = bust_cpu(
        conformers,
        num_workers=1,
        aggregation_type="avg",
        full_report=False,
    )
    rates_bool, avg_bool = bust_cpu(
        conformers,
        num_workers=1,
        aggregation_type="bool",
        full_report=False,
    )

    # Assert
    assert len(rates_avg) == len(conformers)
    assert len(rates_bool) == len(conformers)

    assert all(0.0 <= rate <= 1.0 for rate in rates_avg)
    assert all(rate in (0.0, 1.0) for rate in rates_bool)

    assert pytest.approx(np.mean(rates_avg), rel=1e-6) == avg_avg
    assert pytest.approx(np.mean(rates_bool), rel=1e-6) == avg_bool


def test_bust_cpu_worker_invariance(
    fixture_smiles_to_confs: Dict[str, List[Chem.Mol]]
) -> None:
    """Changing num_workers should not change pass-rate outputs.
    Making sure task splitting and aggregation logic are correct for different worker counts.
    """
    
    # Arrange
    conformers: List[Chem.Mol] = []
    for idx, mols in enumerate(fixture_smiles_to_confs.values(), start=1):
        conformers.extend(_clone_mols(mols, repeats=idx))

    # Act
    single_rates, single_avg = bust_cpu(
        conformers,
        num_workers=1,
        aggregation_type="bool",
        full_report=False,
    )
    multi_rates, multi_avg = bust_cpu(
        conformers,
        num_workers=4,
        aggregation_type="bool",
        full_report=False,
    )

    # Assert
    assert multi_rates == single_rates
    assert pytest.approx(single_avg, rel=1e-6) == multi_avg


def test_bust_full_gens_summary_and_overall_pass(
    fixture_smiles_to_confs: Dict[str, List[Chem.Mol]]
) -> None:
    """bust_full_gens should produce per-SMILES pass summaries with consistent averages.
    Making sure the summary and overall pass rates are correct for different worker counts as well.
    """
    # Arrange
    expected_counts: Dict[str, int] = {}
    test_payload: Dict[str, List[Chem.Mol]] = {}
    for idx, (smiles, mols) in enumerate(fixture_smiles_to_confs.items(), start=1):
        clones = _clone_mols(mols, repeats=idx)
        test_payload[smiles] = clones
        expected_counts[smiles] = len(clones)

    # Act
    summary_single, overall_single = bust_full_gens(
        test_payload,
        num_workers=1,
        config="mol",
        full_report=False,
    )
    summary_multi, overall_multi = bust_full_gens(
        test_payload,
        num_workers=3,
        config="mol",
        full_report=False,
    )

    # Assert
    pd.testing.assert_frame_equal(summary_single, summary_multi)
    assert pytest.approx(overall_single, rel=1e-6) == overall_multi

    expected_columns = {"smiles", "num_conformers", "num_passed", "pass_pct"}
    assert set(summary_single.columns) == expected_columns

    for _, row in summary_single.iterrows():
        expected = expected_counts[row["smiles"]]
        assert row["num_conformers"] == expected
        assert 0.0 <= row["pass_pct"] <= 1.0
        assert row["num_passed"] <= expected

    weighted_total = 0.0
    total_confs = 0
    for _, row in summary_single.iterrows():
        weighted_total += row["pass_pct"] * row["num_conformers"]
        total_confs += row["num_conformers"]
    assert total_confs > 0
    assert pytest.approx(weighted_total / total_confs, rel=1e-6) == overall_single

