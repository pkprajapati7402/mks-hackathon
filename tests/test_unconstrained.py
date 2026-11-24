"""
Unit Tests for Unconstrained Demand Reconstruction (Workstream 2).
"""

import pytest
import pandas as pd
import numpy as np
from src.anomalies.unconstrained import reconstruct_unconstrained_demand

def test_unconstrained_reconstruction():
    # Synthetic transactions
    train_sample = pd.DataFrame({
        "Record_ID": ["R1", "R2", "R3"],
        "Date": ["2021-05", "2021-05", "2021-05"],
        "SKU_ID": ["PTC-001", "PTC-001", "PTC-002"],
        "Region": ["North America", "North America", "Europe"],
        "Customer_Name": ["Cust A", "Cust B", "Cust C"],
        "Units_Sold": [40, 60, 100],
        "Unit_Price_USD": [10.0, 10.0, 15.0],
        "Revenue_USD": [400.0, 600.0, 1500.0]
    })

    # Synthetic stockout for PTC-001 in NA (Requested = 200, Shipped = 100)
    stockouts_sample = pd.DataFrame({
        "Date": ["2021-05"],
        "SKU_ID": ["PTC-001"],
        "Region": ["North America"],
        "Requested_Units": [200],
        "Carried_In_Units": [0],
        "Shipped_Units": [100],
        "Backordered_Units": [0],
        "Lost_Units": [100],
        "Fill_Rate_Pct": [50.0],
        "Reason": ["Semiconductor shortage"]
    })

    result = reconstruct_unconstrained_demand(train_sample, stockouts_sample)

    # PTC-001 total unconstrained should be 200 (Cust A = 80, Cust B = 120)
    ptc1 = result[result["SKU_ID"] == "PTC-001"]
    assert ptc1["Units_Sold_Unconstrained"].sum() == 200
    assert ptc1.loc[ptc1["Customer_Name"] == "Cust A", "Units_Sold_Unconstrained"].iloc[0] == 80
    assert ptc1.loc[ptc1["Customer_Name"] == "Cust B", "Units_Sold_Unconstrained"].iloc[0] == 120

    # PTC-002 (no stockout) should remain 100
    ptc2 = result[result["SKU_ID"] == "PTC-002"]
    assert ptc2["Units_Sold_Unconstrained"].iloc[0] == 100
