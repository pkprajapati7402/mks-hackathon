"""
Unit Tests for Feature Engineering & Intermittency (Workstream 3).
"""

import pytest
import pandas as pd
import numpy as np
from src.features.intermittency import classify_series_intermittency
from src.features.calendar_events import add_calendar_and_event_features

def test_intermittency_classification():
    # Smooth: steady non-zero demand
    smooth_series = pd.Series([100, 105, 98, 102, 101, 99, 104, 100])
    smooth_res = classify_series_intermittency(smooth_series)
    assert smooth_res["category"] == "Smooth"
    assert smooth_res["adi"] < 1.32
    assert smooth_res["cv2"] < 0.49

    # Intermittent: zero demand with regular non-zero spikes
    interm_series = pd.Series([0, 0, 10, 0, 0, 10, 0, 0, 10, 0])
    interm_res = classify_series_intermittency(interm_series)
    assert interm_res["category"] == "Intermittent"
    assert interm_res["adi"] >= 1.32

def test_calendar_features():
    df = pd.DataFrame({
        "Date": ["2023-01", "2023-08", "2023-12"],
        "Region": ["North America", "Europe", "Asia Pacific"],
        "SKU_ID": ["BSS-001", "BSS-001", "BSS-001"]
    })
    out = add_calendar_and_event_features(df)

    assert "month_sin" in out.columns
    assert "month_cos" in out.columns
    assert out.loc[out["Date"] == "2023-08", "is_europe_summer_shutdown"].iloc[0] == 1
    assert out.loc[out["Date"] == "2023-01", "is_europe_summer_shutdown"].iloc[0] == 0
    assert out.loc[out["Date"] == "2023-12", "is_pre_price_hike_month"].iloc[0] == 1
