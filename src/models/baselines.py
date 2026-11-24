"""
Baseline Forecasting Models (Workstream 4 Core Tier).
Implements Seasonal Naive and Historical Moving Average.
High performance vectorized implementation.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

def forecast_seasonal_naive(
    history: pd.DataFrame,
    test_keys: pd.DataFrame,
    target_col: str = "Units_Sold_Unconstrained"
) -> pd.Series:
    """
    Seasonal Naive: Forecast equals actual demand from the same month in the prior year (t - 12).
    Falls back to series median if t - 12 is unavailable.
    """
    hist_indexed = history.set_index(["SKU_ID", "Region", "Month"])[target_col].to_dict()
    medians = history.groupby(["SKU_ID", "Region"])[target_col].median().to_dict()
    family_medians = history.groupby("Product_Family")[target_col].median().to_dict()

    skus = test_keys["SKU_ID"].values
    regs = test_keys["Region"].values
    months = test_keys["Date"].str[5:7].astype(int).values
    fams = test_keys.get("Product_Family", pd.Series([""]*len(test_keys))).values

    preds = np.array([
        max(0.0, float(hist_indexed.get((s, r, m), medians.get((s, r), family_medians.get(f, 10.0)))))
        for s, r, m, f in zip(skus, regs, months, fams)
    ], dtype=float)

    return pd.Series(preds, index=test_keys.index, name="Forecast_Seasonal_Naive")

def forecast_moving_average(
    history: pd.DataFrame,
    test_keys: pd.DataFrame,
    window: int = 12,
    target_col: str = "Units_Sold_Unconstrained"
) -> pd.Series:
    """
    Moving Average: Forecast equals the rolling average over the trailing window months.
    """
    recent_history = history.sort_values("Date").groupby(["SKU_ID", "Region"]).tail(window)
    ma_dict = recent_history.groupby(["SKU_ID", "Region"])[target_col].mean().to_dict()
    family_ma = history.groupby("Product_Family")[target_col].mean().to_dict()

    skus = test_keys["SKU_ID"].values
    regs = test_keys["Region"].values
    fams = test_keys.get("Product_Family", pd.Series([""]*len(test_keys))).values

    preds = np.array([
        max(0.0, float(ma_dict.get((s, r), family_ma.get(f, 10.0))))
        for s, r, f in zip(skus, regs, fams)
    ], dtype=float)

    return pd.Series(preds, index=test_keys.index, name="Forecast_Moving_Average")
