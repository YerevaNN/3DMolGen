from __future__ import annotations

from collections import OrderedDict
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if PROJECT_ROOT.name != "3DMolGen":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molgen3D.evaluation import posebusters_check as pb

THIS_DIR = Path(__file__).resolve().parent
EVAL_DIR = THIS_DIR.parent
SAMPLE_BY_PATH = EVAL_DIR / "sample_by_smiles_df.csv"
SAMPLE_SUMMARY_PATH = EVAL_DIR / "sample_summary_df.csv"
SAMPLE_BY_COLUMNS = pd.read_csv(SAMPLE_BY_PATH, nrows=0).columns.tolist()
SAMPLE_SUMMARY_COLUMNS = pd.read_csv(SAMPLE_SUMMARY_PATH, nrows=0).columns.tolist()
SAMPLE_BOOL_COLUMNS = SAMPLE_BY_COLUMNS[:-4]  # omit pass_percentage, smiles, num_of_conformers, error


def test_bust_full_gens_uses_conformer_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure overall pass rate is computed across conformers, not SMILES averages."""

    fake_report = pd.DataFrame(
        {
            "conformer_index": [0, 1, 2, 3, 4, 5],
            "check_a": [False, True, True, False, False, False],
            "check_b": [False, True, True, False, False, False],
        }
    )

    def fake_collect(
        rd_confs,
        *,
        num_workers: int,
        full_report: bool,
        config: str,
        task_chunk_size: Optional[int] = None,
        log_progress: bool = False,
    ) -> pd.DataFrame:
        assert len(rd_confs) == len(fake_report)
        return fake_report.copy()

    monkeypatch.setattr(pb, "_collect_posebusters_report", fake_collect)

    smiles_to_confs = OrderedDict(
        {
            "smile_1": [object(), object()],
            "smile_2": [object(), object(), object(), object()],
        }
    )

    per_smiles_df, summary, overall = pb.bust_full_gens(
        smiles_to_confs,
        num_workers=1,
        fail_threshold=0.0,
    )

    # Two conformers pass (indices 1 and 2) out of six → 2/6 ≈ 0.33̅.
    assert pytest.approx(overall, rel=1e-9) == pytest.approx(2.0 / 6.0)
    assert summary.attrs["per_smiles_df"].equals(per_smiles_df)

    row1 = per_smiles_df.loc[per_smiles_df["smiles"] == "smile_1"].iloc[0]
    row2 = per_smiles_df.loc[per_smiles_df["smiles"] == "smile_2"].iloc[0]

    assert pytest.approx(row1["pass_percentage"], rel=1e-9) == 50.0
    assert pytest.approx(row2["pass_percentage"], rel=1e-9) == 25.0

    # Averaging the per-SMILES pass percentages (50 + 25) / 2 = 37.5
    # would overstate the true conformer-level mean (33.3%). The function
    # keeps these distinct and returns the conformer-wise average separately.
    assert pytest.approx(per_smiles_df["pass_percentage"].mean(), rel=1e-9) == 37.5

    summary_row = summary.iloc[0]
    assert summary_row["smiles"] == "ALL"
    assert summary_row["num_smiles"] == pytest.approx(2.0)
    assert summary_row["num_conformers"] == pytest.approx(6.0)
    assert pytest.approx(summary_row["pass_percentage"], rel=1e-9) == overall * 100.0

    # Check that users can filter per_smiles_df by pass_percentage to get failed SMILES
    fail_smiles = per_smiles_df.loc[per_smiles_df["pass_percentage"] < 100.0, "smiles"].tolist()
    assert set(fail_smiles) == {"smile_1", "smile_2"}  # both have < 100% pass rate


def test_bust_full_gens_matches_sample_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure per-smiles and summary DataFrames match the documented CSV schemas."""

    def fake_collect(
        rd_confs,
        *,
        num_workers: int,
        full_report: bool,
        config: str,
        task_chunk_size: Optional[int] = None,
        log_progress: bool = False,
    ) -> pd.DataFrame:
        rows = []
        for idx in range(len(rd_confs)):
            row = {"conformer_index": idx}
            for col_idx, col in enumerate(SAMPLE_BOOL_COLUMNS):
                row[col] = (idx + col_idx) % 2 == 0
            row["error"] = ""
            rows.append(row)
        columns = ["conformer_index", *SAMPLE_BOOL_COLUMNS, "error"]
        return pd.DataFrame(rows, columns=columns)

    monkeypatch.setattr(pb, "_collect_posebusters_report", fake_collect)

    smiles_to_confs = OrderedDict(
        {
            "a": [object(), object()],
            "b": [object()],
        }
    )

    per_smiles_df, summary_df, overall = pb.bust_full_gens(
        smiles_to_confs,
        num_workers=1,
        fail_threshold=0.0,
    )

    assert per_smiles_df.columns.tolist() == SAMPLE_BY_COLUMNS
    assert summary_df.columns.tolist() == SAMPLE_SUMMARY_COLUMNS
    assert summary_df.attrs["per_smiles_df"].equals(per_smiles_df)
    assert overall == pytest.approx(summary_df.iloc[0]["pass_percentage"] / 100.0)

