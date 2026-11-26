"""
Unit Tests for Forecasting Models, Quantile Monotonicity, and Lifecycle Rules (Workstream 4).
"""

import pytest
import pandas as pd
import numpy as np
from src.models.intermittent import croston_sba
from src.models.cold_start import apply_lifecycle_and_cold_start

def test_croston_sba():
    # Demand series with sporadic non-zeros
    s = pd.Series([0, 10, 0, 0, 20, 0, 15, 0])
    rate = croston_sba(s, alpha=0.1, method="sba")
    assert rate > 0.0
    assert rate < 20.0

def test_lifecycle_rules():
    sku_master_sample = pd.DataFrame({
        "SKU_ID": ["EOL-001", "LAUNCH-001", "NORM-001"],
        "Launch_Date": ["2019-01", "2027-01", "2019-01"],
        "EOL_Date": ["2026-12", "", ""],
        "Replaces_SKU": ["", "", ""]
    })

    forecast_sample = pd.DataFrame({
        "Date": ["2027-01", "2026-11", "2027-01"],
        "SKU_ID": ["EOL-001", "LAUNCH-001", "NORM-001"],
        "Region": ["North America", "North America", "North America"],
        "Units_Sold_Forecast": [100.0, 50.0, 75.0],
        "P10": [80.0, 40.0, 60.0],
        "P50": [100.0, 50.0, 75.0],
        "P90": [120.0, 60.0, 90.0]
    })

    hist_sample = pd.DataFrame({
        "SKU_ID": ["NORM-001"],
        "Region": ["North America"],
        "Units_Sold_Unconstrained": [75.0]
    })

    adjusted = apply_lifecycle_and_cold_start(forecast_sample, sku_master_sample, hist_sample)

    # EOL-001 in 2027-01 is past EOL (2026-12) -> should be 0
    eol_row = adjusted[adjusted["SKU_ID"] == "EOL-001"]
    assert eol_row["Units_Sold_Forecast"].iloc[0] == 0.0
    assert eol_row["P50"].iloc[0] == 0.0

    # LAUNCH-001 in 2026-11 is before Launch (2027-01) -> should be 0
    launch_row = adjusted[adjusted["SKU_ID"] == "LAUNCH-001"]
    assert launch_row["Units_Sold_Forecast"].iloc[0] == 0.0
    assert launch_row["P50"].iloc[0] == 0.0

    # NORM-001 should remain 75
    norm_row = adjusted[adjusted["SKU_ID"] == "NORM-001"]
    assert norm_row["Units_Sold_Forecast"].iloc[0] == 75.0


def test_cold_start_replaces_sku_ramp():
    sku_master = pd.DataFrame({
        "SKU_ID": ["OLD-001", "NEW-001"],
        "Launch_Date": ["2020-01", "2027-01"],
        "EOL_Date": ["2027-03", ""],
        "Replaces_SKU": ["", "OLD-001"],
    })
    hist = pd.DataFrame({
        "SKU_ID": ["OLD-001"] * 3,
        "Region": ["North America"] * 3,
        "Product_Family": ["Powertrain Components"] * 3,
        "Units_Sold_Unconstrained": [100.0, 100.0, 100.0],
    })
    fcst = pd.DataFrame({
        "Date": ["2026-12", "2027-01", "2027-07"],
        "SKU_ID": ["NEW-001", "NEW-001", "NEW-001"],
        "Region": ["North America", "North America", "North America"],
        "Product_Family": ["Powertrain Components"] * 3,
        "Units_Sold_Forecast": [30.0, 30.0, 30.0],
        "P10": [20.0, 20.0, 20.0],
        "P50": [30.0, 30.0, 30.0],
        "P90": [45.0, 45.0, 45.0],
    })
    out = apply_lifecycle_and_cold_start(fcst, sku_master, hist)
    # Pre-launch (2026-12) must be 0
    assert out.loc[out["Date"] == "2026-12", "P50"].iloc[0] == 0.0
    # Launch month m=0 (2027-01): 0.30*30 + 0.70*(100*0.15) = ~20
    assert 15.0 <= out.loc[out["Date"] == "2027-01", "P50"].iloc[0] <= 25.0
    # Mature handoff m=6 (2027-07): 0.30*30 + 0.70*(100*1.00) = ~79
    assert out.loc[out["Date"] == "2027-07", "P50"].iloc[0] >= 75.0
    assert (out["P10"] <= out["P50"]).all() and (out["P50"] <= out["P90"]).all()
