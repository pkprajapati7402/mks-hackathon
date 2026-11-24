"""
Sanity and Compliance Tests for Final Competition Submission File.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.config import OUTPUTS_DIR, PROJECT_ROOT, RAW_TEST

def test_submission_format_and_integrity():
    sub_path = OUTPUTS_DIR / "submission.csv"
    root_sub_path = PROJECT_ROOT / "submission.csv"

    assert sub_path.exists(), "outputs/submission.csv must exist!"
    assert root_sub_path.exists(), "Root submission.csv must exist!"

    sub = pd.read_csv(sub_path)
    test_raw = pd.read_csv(RAW_TEST, low_memory=False)

    # 1. Exact Row Count Check (58,219 rows)
    assert len(sub) == len(test_raw), f"Submission row count {len(sub)} must equal test.csv row count {len(test_raw)} (58,219 rows)!"
    assert len(sub) == 58219

    # 2. Column Names Check
    expected_cols = ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]
    assert list(sub.columns) == expected_cols, f"Columns must exactly match {expected_cols}"

    # 3. No Nulls Check
    assert sub.isnull().sum().sum() == 0, "Submission must contain 0 null values!"

    # 4. Non-Negative Integer Check
    assert (sub["Units_Sold_Forecast"] >= 0).all(), "Units_Sold_Forecast must be non-negative!"
    assert (sub["Units_Sold_Forecast"] == sub["Units_Sold_Forecast"].round()).all(), "Units_Sold_Forecast must be integers!"

    # 5. Check Reconciliation Discrepancy
    recon_path = OUTPUTS_DIR / "hierarchical_reconciliation.csv"
    if recon_path.exists():
        recon_df = pd.read_csv(recon_path)
        assert (recon_df["Discrepancy"] == 0).all(), "Hierarchical reconciliation must have 0 discrepancy!"
